"""T1（v2.6）：全 NMR 与 Iron-Core 域内 R² 的受试者层面置换检验。

协议（统一主协议，与 §3.3 regression_biomarkers.py 的 NMR 层完全一致）：
log1p → 逐特征集 StandardScaler → 外层 GroupKFold(5) + 内层 GroupKFold(3)
选 Ridge alpha ∈ {0.1,…,10000}（scoring=r2）；subject 一律 astype(str)。

置换：受试者层面（group_level=True）：置换"受试者→均值标签"映射，受试者内
中心化残差保留；n_perm=1000，seed=42；p=(1+sum(null>=observed))/(1+n_valid)。

模型：full_nmr_19（19 项）与 iron_core_5（frozen_selector.json 冻结核心 5 项）。
预期 observed：全 NMR ≈0.196（§3.3 复现口径）；Iron-Core 统一协议 ≈0.080
（选择器原评估协议无标准化，为 0.0720——差异源于标准化是否纳入评估，DECISION_LOG 记录）。

输出：IBS/cellomics/permutation_r2_results.csv
      IBS/figs/fig12_permutation_r2.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'IBS/pipeline'))
from supervised_biomarkers import build_features  # noqa: E402
from permutation_utils import (fit_predict_ridge, fit_predict_ridge_fast,  # noqa: E402
                               oof_metrics, permutation_test)

FIGS = ROOT / 'IBS/figs'
CE = ROOT / 'IBS/cellomics'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
    sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
    sel = json.load(open(ROOT / 'IBS/jacobs_validation/frozen_selector.json',
                         encoding='utf-8'))
    iron5 = sel['selected_features']

    tr = nmr.join(sm[['bristol', 'subject']]).dropna()
    idx = tr.index
    y = tr['bristol'].values.astype(float)
    g = tr['subject'].astype(str).values
    print(f'分析集 n={len(idx)} | 受试者={len(set(g))} | Iron-Core={iron5}',
          flush=True)

    X_full, cols_full = build_features(nmr.loc[idx, list(nmr.columns)], 0.10, 'log')
    X_iron, cols_iron = build_features(nmr.loc[idx, iron5], 0.10, 'log')
    assert X_full.shape[1] == 19 and X_iron.shape[1] == 5, '特征数异常'

    Xs = {'full_nmr_19': X_full, 'iron_core_5': X_iron}

    print('\n[协议一致性] 快速路径 vs GridSearchCV 单折拟合：', flush=True)
    for k, X in Xs.items():
        r_fast = oof_metrics(X, y, g, fit_predict_ridge_fast)
        r_grid = oof_metrics(X, y, g, fit_predict_ridge)
        print(f'  {k}: fast R²={r_fast["r2"]:.4f} (ρ={r_fast["rho"]:.4f}) | '
              f'grid R²={r_grid["r2"]:.4f} (ρ={r_grid["rho"]:.4f}) | '
              f'Δ={abs(r_fast["r2"] - r_grid["r2"]):.2e}', flush=True)
        assert abs(r_fast['r2'] - r_grid['r2']) < 1e-9, '快速路径与主协议不一致'

    t0 = time.time()
    res = permutation_test(Xs, y, g, n_perm=args.n_perm, seed=args.seed,
                           group_level=True)
    elapsed = time.time() - t0

    rows = []
    for k, X in Xs.items():
        r = res[k]
        rows.append({
            'model': k, 'n_features': X.shape[1],
            'observed_r2': round(r['observed_r2'], 6),
            'null_mean': round(r['null_mean'], 6),
            'null_sd': round(r['null_sd'], 6),
            'p_value': round(r['p_value'], 6),
            'n_valid': r['n_valid'], 'n_perm': args.n_perm,
        })
    df = pd.DataFrame(rows)
    out_csv = CE / 'permutation_r2_results.csv'
    df.to_csv(out_csv, index=False)

    print(f'\n置换检验（n_perm={args.n_perm}, seed={args.seed}, '
          f'group_level=True, 耗时 {elapsed:.1f}s）:', flush=True)
    for k in Xs:
        r = res[k]
        print(f"  {k}: observed R²={r['observed_r2']:.4f} ρ={r['observed_rho']:.4f} | "
              f"null mean={r['null_mean']:.4f} sd={r['null_sd']:.4f} | "
              f"p={r['p_value']:.4f} (n_valid={r['n_valid']})", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    meta = [('full_nmr_19', 'A  Full NMR (19 features)', '#2f6db3'),
            ('iron_core_5', 'B  Iron-Core (5 features)', '#b3562f')]
    for ax, (k, title, col) in zip(axes, meta):
        r = res[k]
        ax.hist(r['null'], bins=50, color=col, alpha=0.75)
        ax.axvline(r['observed_r2'], color='crimson', lw=1.8)
        ax.set_title(title)
        ax.set_xlabel('Subject-level permutation $R^2$ (null)')
        ax.set_ylabel('Count')
        ax.text(0.98, 0.95,
                f"observed $R^2$={r['observed_r2']:.3f}\n"
                f"p={r['p_value']:.4f} ({args.n_perm} perms)",
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))
    fig.suptitle(f'Permutation tests of in-domain OOF $R^2$ '
                 f'(GroupKFold(5), subject-level, {args.n_perm} permutations)',
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_fig = FIGS / 'fig12_permutation_r2.png'
    fig.savefig(out_fig, dpi=300)
    print(f'\n输出: {out_csv}\n输出: {out_fig}', flush=True)

    ok = abs(res['full_nmr_19']['observed_r2'] - 0.196) < 0.01
    print(f'\n[T1.2 验收] 全 NMR observed 与 0.196 差异 <0.01: {ok} | '
          f"p_full={res['full_nmr_19']['p_value']:.4f} | "
          f"p_iron={res['iron_core_5']['p_value']:.4f} | "
          f"null 均值 full={res['full_nmr_19']['null_mean']:.4f} "
          f"iron={res['iron_core_5']['null_mean']:.4f}", flush=True)


if __name__ == '__main__':
    main()
