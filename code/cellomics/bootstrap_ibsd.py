"""IBS-D 与 IBS-C 亚型内部 ρ 的受试者聚类 bootstrap（B=2000）。

v2.4：原版本仅覆盖 IBS-D（方法学审查指出不对称）。本版本对两个亚型同时
执行（按受试者整簇放回重抽样，重算 Spearman ρ 与 OOF R²，输出 95% 百分位
CI 与 P(ρ≥0)），并重绘 fig10（A：IBS-D；B：IBS-C）。
输出：IBS/cellomics/results_bristol_nmr_ibsd/bootstrap_rho.csv（长表，
      subtype/b/rho/r2 两亚型并列）
      IBS/figs/fig10_ibsd_rho_bootstrap.png
"""
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
FIG = ROOT / 'IBS/figs'
OUT = CE / 'results_bristol_nmr_ibsd'
B = 2000

SPECS = [
    ('IBS-D', CE / 'results_bristol_nmr_ibsd' / 'predictions.csv',
     CE / 'sample_map_ibsd.csv', '#4c72b0'),
    ('IBS-C', CE / 'results_bristol_nmr_ibsc' / 'predictions.csv',
     CE / 'sample_map_ibsc.csv', '#55a868'),
]

store, frames = {}, []
for name, pred_p, smap_p, col in SPECS:
    pred = pd.read_csv(pred_p, index_col=0)
    smb = pd.read_csv(smap_p, index_col=0)
    subj = smb['subject'].reindex(pred.index).astype(str).values
    obs = pred['observed'].values.astype(float)
    prd = pred['predicted'].values.astype(float)
    subjects = np.unique(subj)
    idx_by_subject = {s: np.where(subj == s)[0] for s in subjects}

    rng = np.random.default_rng(42)
    rhos, r2s = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([idx_by_subject[s] for s in pick])
        rhos[b] = spearmanr(obs[idx], prd[idx]).statistic
        r2s[b] = 1 - ((obs[idx] - prd[idx]) ** 2).sum() / \
            ((obs[idx] - obs[idx].mean()) ** 2).sum()

    point = spearmanr(obs, prd).statistic
    r2_point = 1 - ((obs - prd) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    ci_rho = np.percentile(rhos, [2.5, 97.5])
    ci_r2 = np.percentile(r2s, [2.5, 97.5])
    p_ge0 = float((rhos >= 0).mean())
    print(f'{name}: n={len(obs)} samples / {len(subjects)} subjects | '
          f'rho={point:+.4f} 95% CI [{ci_rho[0]:+.3f}, {ci_rho[1]:+.3f}] '
          f'P(rho>=0)={p_ge0:.3f}')
    print(f'{name}: R2={r2_point:+.4f} 95% CI [{ci_r2[0]:+.3f}, {ci_r2[1]:+.3f}]')
    store[name] = dict(rhos=rhos, r2s=r2s, ci_rho=ci_rho, ci_r2=ci_r2,
                       p_ge0=p_ge0, point=point, r2_point=r2_point, col=col,
                       n=len(obs), ns=len(subjects))
    frames.append(pd.DataFrame({'subtype': name, 'bootstrap': np.arange(B),
                                'rho': rhos, 'r2': r2s}))

pd.concat(frames, ignore_index=True).to_csv(OUT / 'bootstrap_rho.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
for ax, name in zip(axes, ['IBS-D', 'IBS-C']):
    s = store[name]
    ax.hist(s['rhos'], bins=60, color=s['col'], alpha=0.85,
            edgecolor='white', lw=0.3)
    for v in s['ci_rho']:
        ax.axvline(v, c='#d62728', ls='--', lw=1)
    ax.axvline(0, c='k', lw=0.8)
    ax.axvline(s['point'], c='#2ca02c', lw=1.2)
    ax.set_xlabel(f'cluster-bootstrap Spearman ρ ({name})')
    panel = 'A' if name == 'IBS-D' else 'B'
    ax.set_title(f'{panel}  ρ (n={s["n"]}, {s["ns"]} subj)')
    ax.text(0.02, 0.95,
            f'ρ={s["point"]:+.3f} [{s["ci_rho"][0]:+.3f}, {s["ci_rho"][1]:+.3f}]\n'
            f'P(ρ≥0)={s["p_ge0"]:.3f}\n'
            f'R² CI [{s["ci_r2"][0]:+.3f}, {s["ci_r2"][1]:+.3f}]',
            transform=ax.transAxes, va='top', fontsize=8, zorder=6,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                      alpha=0.9))

fig.suptitle('Subtype-internal NMR→Bristol: subject-cluster bootstrap '
             '(B=2000; exploratory only)', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / 'fig10_ibsd_rho_bootstrap.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nsaved: bootstrap_rho.csv -> {OUT} | fig10 -> {FIG}')
