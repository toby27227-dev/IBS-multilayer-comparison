"""T2（v2.6）：NMR 内部 SCFA vs 氨基酸的 Shapley 分解与 ΔR² 置换检验。

协议（统一主协议，与 §3.3 NMR 层一致；逐特征集 log1p+StandardScaler、
外层 GroupKFold(5)、内层 GroupKFold(3) 选 Ridge alpha；subject astype(str)）。

特征分组（T2.1，直接读 nmr.csv 实际列名，不凭正文推断）：
  SCFA = 直链/支链短链脂肪酸（5 项）
  AminoAcid = 游离氨基酸 + BCAA（7 项）
  Other = 核酸/碳源/有机酸/醇类等（7 项）
模型（六项；M_other 为 Shapley 基线，T2.3 公式中 R²(Other) 所必需）：
  M_full(19), M_scfa(5), M_aa(7), M_other(7), M_no_scfa(14), M_no_aa(12)

Shapley（Other 恒为基线）：
  φ_SCFA = 0.5[(R²(SCFA∪Other) − R²(Other)) + (R²(full) − R²(AA∪Other))]
  φ_AA   = 0.5[(R²(AA∪Other) − R²(Other)) + (R²(full) − R²(SCFA∪Other))]
  φ_interaction = R²(full) − R²(Other) − φ_SCFA − φ_AA

ΔR² 受试者层面置换（n_perm=1000, seed=42）：每次置换共享同一 y_perm，
同时重跑 full 与对应 no_* 模型；Δ = R²(full) − R²(no_*)。
p = (1 + sum(null_Δ >= observed_Δ)) / (1 + n_valid)。

输出：IBS/cellomics/nmr_group_decomposition.csv
      IBS/figs/fig13_nmr_group_decomposition.png
"""
import argparse
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
from permutation_utils import oof_metrics, permutation_test  # noqa: E402

CE = ROOT / 'IBS/cellomics'
FIGS = ROOT / 'IBS/figs'

GROUPS = {
    'SCFA': ['Acetate', 'Propionate', 'Butyrate', 'Isovalerate', '2-Methylbutyrate'],
    'AminoAcid': ['Alanine', 'Glycine', 'Leucine', 'Valine', 'Isoleucine',
                  'Lysine', 'Tyrosine'],
    'Other': ['Lactate', 'Succinate', 'b_arabinose', 'b_xylose', 'Glucose',
              'Uracil', 'Hypoxanthine'],
}
FEATURE_SETS = {
    'M_full': GROUPS['SCFA'] + GROUPS['AminoAcid'] + GROUPS['Other'],
    'M_scfa': GROUPS['SCFA'],
    'M_aa': GROUPS['AminoAcid'],
    'M_other': GROUPS['Other'],
    'M_no_scfa': GROUPS['AminoAcid'] + GROUPS['Other'],
    'M_no_aa': GROUPS['SCFA'] + GROUPS['Other'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
    sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)

    # ---------- T2.1 分组校验（实际列名） ----------
    cols = list(nmr.columns)
    all_groups = GROUPS['SCFA'] + GROUPS['AminoAcid'] + GROUPS['Other']
    assert len(cols) == 19, f'nmol 列数 {len(cols)} != 19'
    assert set(all_groups) == set(cols), \
        f'分组与实际列名不一致: {set(all_groups) ^ set(cols)}'
    assert not (set(GROUPS['SCFA']) & set(GROUPS['AminoAcid'])
                or set(GROUPS['SCFA']) & set(GROUPS['Other'])
                or set(GROUPS['AminoAcid']) & set(GROUPS['Other'])), '分组重叠'
    print('T2.1 分组校验通过（19 项 = 5 SCFA + 7 AA + 7 Other）：', flush=True)
    for gname, gcols in GROUPS.items():
        print(f'  {gname}: {gcols}', flush=True)

    tr = nmr.join(sm[['bristol', 'subject']]).dropna()
    idx = tr.index
    y = tr['bristol'].values.astype(float)
    g = tr['subject'].astype(str).values
    print(f'分析集 n={len(idx)} | 受试者={len(set(g))}', flush=True)

    # ---------- 六模型 OOF（统一主协议） ----------
    Xs, rows = {}, []
    for mname, fcols in FEATURE_SETS.items():
        Xm, built = build_features(nmr.loc[idx, fcols], 0.10, 'log')
        assert len(built) == len(fcols), f'{mname} 特征过滤异常'
        m = oof_metrics(Xm, y, g)
        Xs[mname] = Xm
        rows.append({'kind': 'model', 'model': mname,
                     'features': '|'.join(fcols), 'n_features': Xm.shape[1],
                     'oof_r2': round(m['r2'], 6), 'oof_rho': round(m['rho'], 6)})
        print(f"  {mname:9s} ({len(fcols):2d} 项): R²={m['r2']:.4f} ρ={m['rho']:.4f}",
              flush=True)

    r2 = {r['model']: r['oof_r2'] for r in rows}
    assert abs(r2['M_full'] - 0.196) < 0.01, f"M_full R²={r2['M_full']} 偏离 0.196"

    # ---------- T2.3 Shapley（Other 为基线） ----------
    phi_scfa = 0.5 * ((r2['M_no_aa'] - r2['M_other']) +
                      (r2['M_full'] - r2['M_no_scfa']))
    phi_aa = 0.5 * ((r2['M_no_scfa'] - r2['M_other']) +
                    (r2['M_full'] - r2['M_no_aa']))
    phi_inter = r2['M_full'] - r2['M_other'] - phi_scfa - phi_aa
    gain = r2['M_full'] - r2['M_other']
    err = abs((phi_scfa + phi_aa + phi_inter) - gain)
    assert err < 1e-6, f'Shapley 加和误差 {err}'
    print(f'\nShapley（基线 R²(Other)={r2["M_other"]:.4f}，总增益 {gain:.4f}）:', flush=True)
    print(f'  φ_SCFA={phi_scfa:.4f} | φ_AA={phi_aa:.4f} | '
          f'φ_interaction={phi_inter:.4f} | 加和误差={err:.2e}', flush=True)

    # ---------- T2.4 ΔR² 受试者层面置换（共享置换序列） ----------
    t0 = time.time()
    res = permutation_test({'full': Xs['M_full'], 'no_scfa': Xs['M_no_scfa'],
                            'no_aa': Xs['M_no_aa']}, y, g,
                           n_perm=args.n_perm, seed=args.seed, group_level=True)
    elapsed = time.time() - t0

    d_scfa = res['full']['observed_r2'] - res['no_scfa']['observed_r2']
    d_aa = res['full']['observed_r2'] - res['no_aa']['observed_r2']
    null_dS = res['full']['null'] - res['no_scfa']['null']
    null_dA = res['full']['null'] - res['no_aa']['null']

    def p_of(null, obs):
        valid = np.isfinite(null)
        return (1 + int((null[valid] >= obs).sum())) / (1 + int(valid.sum()))

    p_scfa, p_aa = p_of(null_dS, d_scfa), p_of(null_dA, d_aa)
    print(f'\nΔR² 置换（n_perm={args.n_perm}, seed={args.seed}, 共享序列, '
          f'耗时 {elapsed:.1f}s）:', flush=True)
    print(f'  去 SCFA: Δ={d_scfa:.4f} | null mean={np.nanmean(null_dS):.4f} '
          f'sd={np.nanstd(null_dS):.4f} | p={p_scfa:.4f}', flush=True)
    print(f'  去 AA  : Δ={d_aa:.4f} | null mean={np.nanmean(null_dA):.4f} '
          f'sd={np.nanstd(null_dA):.4f} | p={p_aa:.4f}', flush=True)

    # ---------- 输出 CSV ----------
    rows.append({'kind': 'shapley', 'r2_full': round(r2['M_full'], 6),
                 'r2_other': round(r2['M_other'], 6),
                 'phi_scfa': round(phi_scfa, 6), 'phi_aa': round(phi_aa, 6),
                 'phi_interaction': round(phi_inter, 6)})
    rows.append({'kind': 'delta', 'model': 'M_no_scfa',
                 'delta_no_scfa': round(d_scfa, 6), 'p_no_scfa': round(p_scfa, 6)})
    rows.append({'kind': 'delta', 'model': 'M_no_aa',
                 'delta_no_aa': round(d_aa, 6), 'p_no_aa': round(p_aa, 6)})
    df = pd.DataFrame(rows)
    df['n_perm'] = args.n_perm
    df['seed'] = args.seed
    col_order = ['kind', 'model', 'features', 'n_features', 'oof_r2', 'oof_rho',
                 'phi_scfa', 'phi_aa', 'phi_interaction', 'r2_full', 'r2_other',
                 'delta_no_scfa', 'delta_no_aa', 'p_no_scfa', 'p_no_aa',
                 'n_perm', 'seed']
    df = df.reindex(columns=col_order)
    out_csv = CE / 'nmr_group_decomposition.csv'
    df.to_csv(out_csv, index=False)

    # ---------- 图13 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    ax = axes[0]
    order = ['M_full', 'M_no_scfa', 'M_no_aa', 'M_scfa', 'M_aa', 'M_other']
    vals = [r2[m] for m in order]
    colors = ['#2f6db3', '#5a8fc9', '#8fb3dd', '#b3562f', '#d98d5f', '#999999']
    ax.bar(range(len(order)), vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['full', 'no\nSCFA', 'no\nAA', 'SCFA\nonly', 'AA\nonly',
                        'Other\nonly'], fontsize=8)
    ax.set_ylabel('OOF $R^2$')
    ax.set_title('A  Model $R^2$ (GroupKFold(5))')

    ax = axes[1]
    segs = [('Other (baseline)', r2['M_other'], '#999999'),
            ('$\\varphi_{SCFA}$', phi_scfa, '#b3562f'),
            ('$\\varphi_{AA}$', phi_aa, '#2f6db3'),
            ('interaction', phi_inter, '#7aa66a')]
    bot = 0.0
    for lab, v, c in segs:
        ax.bar([0], [v], bottom=bot, color=c, alpha=0.9, label=f'{lab} = {v:+.3f}')
        bot += v
    ax.axhline(r2['M_full'], color='crimson', ls='--', lw=1.2)
    ax.text(0.55, r2['M_full'] + 0.004, f"$R^2$(full)={r2['M_full']:.3f}",
            color='crimson', fontsize=8.5)
    ax.set_ylim(0, r2['M_full'] + 0.035)
    ax.set_xlim(-0.6, 0.9)
    ax.set_xticks([])
    ax.legend(fontsize=7.5, frameon=False, loc='center right')
    ax.set_ylabel('$R^2$ contribution (stacked)')
    ax.set_title('B  Shapley decomposition')

    ax = axes[2]
    ax.hist(null_dS, bins=50, color='#b3562f', alpha=0.55,
            label=f'null Δ (no SCFA)\np={p_scfa:.4f}')
    ax.hist(null_dA, bins=50, color='#2f6db3', alpha=0.55,
            label=f'null Δ (no AA)\np={p_aa:.4f}')
    ax.axvline(d_scfa, color='#7a2f14', lw=1.8, ls='--',
               label=f'observed Δ(no SCFA)={d_scfa:.3f}')
    ax.axvline(d_aa, color='#1a3a52', lw=1.8, ls=':',
               label=f'observed Δ(no AA)={d_aa:.3f}')
    ax.set_xlabel('$\\Delta R^2$ = full − without group')
    ax.set_ylabel('Count')
    ax.set_title('C  $\\Delta R^2$ permutation (subject-level)')
    ax.legend(fontsize=7.2, frameon=False, loc='upper left')

    fig.suptitle('NMR internal SCFA vs amino-acid decomposition '
                 f'({args.n_perm} subject-level permutations)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_fig = FIGS / 'fig13_nmr_group_decomposition.png'
    fig.savefig(out_fig, dpi=300)
    print(f'\n输出: {out_csv}\n输出: {out_fig}', flush=True)

    print('\n[T2.6 决策依据] '
          f'φ_SCFA={phi_scfa:.4f} vs φ_AA={phi_aa:.4f} | '
          f'p_no_scfa={p_scfa:.4f} p_no_aa={p_aa:.4f}', flush=True)


if __name__ == '__main__':
    main()
