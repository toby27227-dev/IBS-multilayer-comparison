"""稳定性阈值扫描：Iron-Core 规模与组成的平台性分析。

方法学审查落地（v2.4）：5 项核心依赖预先设定阈值 (π_uni_cell≥0.8, KS<0.15)；
本脚本在 πt ∈ [0.70, 0.95] × KSt ∈ [0.10, 0.20] 网格内重放三门判定
（门值来自 frozen_selector.json 的逐特征 π_uni_cell 与 KS，选择阶段无标签），
输出核心规模/组成的阈值敏感性，检验 (0.80, 0.15) 是否位于平台区。
输出：threshold_sweep.csv、IBS/figs/fig11_threshold_sweep.png
"""
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

ROOT = Path.cwd()
OUT = ROOT / 'IBS/jacobs_validation'
FIG = ROOT / 'IBS/figs'

sel = json.load(open(OUT / 'frozen_selector.json', encoding='utf-8'))
feats = sel['features']
PI_TS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
KS_TS = [0.10, 0.12, 0.15, 0.18, 0.20]


def core_at(pt, kt):
    out = []
    for f, d in feats.items():
        ks = d['ks']
        if not d['in_target']:
            continue
        if d['pi_uni_cell'] >= pt and ks == ks and ks < kt:
            out.append(f)
    return sorted(out)


rows = []
for pt in PI_TS:
    for kt in KS_TS:
        core = core_at(pt, kt)
        rows.append({'pi_thr': pt, 'ks_thr': kt, 'n_core': len(core),
                     'members': '/'.join(core)})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'threshold_sweep.csv', index=False)
print(res.to_string(index=False))

M = res.pivot(index='pi_thr', columns='ks_thr', values='n_core').loc[PI_TS, KS_TS]

plateau = res[res['n_core'] == 5]
pl_pi = sorted(plateau['pi_thr'].unique())
pl_ks = sorted(plateau['ks_thr'].unique())
print(f'\n5 项平台区: pi_thr ∈ {pl_pi}; ks_thr ∈ {pl_ks}')

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

ax = axes[0]
im = ax.imshow(M.values, cmap='YlGnBu', vmin=0, vmax=max(8, int(M.values.max())),
               aspect='auto')
for i in range(len(PI_TS)):
    for j in range(len(KS_TS)):
        v = int(M.values[i, j])
        ax.text(j, i, str(v), ha='center', va='center',
                fontsize=10, color='white' if v >= 6 else '#1a3a52')
ax.set_xticks(range(len(KS_TS)), [f'{k:.2f}' for k in KS_TS])
ax.set_yticks(range(len(PI_TS)), [f'{p:.2f}' for p in PI_TS])
ax.set_xlabel('KS threshold')
ax.set_ylabel('π_uni_cell threshold')
i0, j0 = PI_TS.index(0.80), KS_TS.index(0.15)
ax.add_patch(Rectangle((j0 - .5, i0 - .5), 1, 1, fill=False,
                       edgecolor='#d62728', lw=2))
ax.set_title('A  Iron-Core size across thresholds')
fig.colorbar(im, ax=ax, shrink=.85, pad=.03, label='Iron-Core size')

ax = axes[1]
col = res[res['ks_thr'] == 0.15]
ax.step(col['pi_thr'], col['n_core'], where='mid', marker='o', color='#4c72b0')
for _, r in col.iterrows():
    ax.annotate(f"{r['n_core']}", (r['pi_thr'], r['n_core']),
                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=9)
ax.set_xlabel('π_uni_cell threshold (KS=0.15)')
ax.set_ylabel('Iron-Core size')
ax.set_ylim(0, 8)
ax.set_title('B  π sweep')

ax = axes[2]
row = res[res['pi_thr'] == 0.80]
ax.step(row['ks_thr'], row['n_core'], where='mid', marker='o', color='#55a868')
for _, r in row.iterrows():
    ax.annotate(f"{r['n_core']}", (r['ks_thr'], r['n_core']),
                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=9)
ax.set_xlabel('KS threshold (π=0.80)')
ax.set_ylabel('Iron-Core size')
ax.set_ylim(0, 8)
ax.set_title('C  KS sweep')

fig.suptitle('Threshold sensitivity of the source-closed Iron-Core '
             '(red box = prespecified 0.80 / 0.15)', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / 'fig11_threshold_sweep.png', dpi=300)
plt.close(fig)
print(f'\nsaved: threshold_sweep.csv -> {OUT} | fig11 -> {FIG}')
