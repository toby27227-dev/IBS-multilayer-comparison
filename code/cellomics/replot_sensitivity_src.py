# -*- coding: utf-8 -*-
"""从 sensitivity_src.csv 重绘 fig9：英文标签、标注置于 CI 上方，避免重叠。"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
FIG = ROOT / 'IBS/figs'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

res = pd.read_csv(CE / 'results_bristol_nmr' / 'sensitivity_src.csv')
res = res.set_index('subset').loc[
    ['all', 'real_w3w7', 'w3', 'w7', 'fallback']].reset_index()
labels = ['All', 'Measured\n(w3+w7)', 'w3\n(±3d)', 'w7\n(±7d)',
          'Subject mean']
cols_ = ['#4c72b0', '#55a868', '#55a868', '#55a868', '#d62728']

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
ax = axes[0]
err = np.vstack([(res['oof_r2'] - res['r2_ci_lo']).clip(0),
                 (res['r2_ci_hi'] - res['oof_r2']).clip(0)])
b = ax.bar(labels, res['oof_r2'], color=cols_, alpha=0.88, yerr=err,
           capsize=3, error_kw=dict(ecolor='#444', lw=1))
for r, v, n, hi in zip(b, res['oof_r2'], res['n'], res['r2_ci_hi']):
    ax.text(r.get_x() + r.get_width() / 2, hi + 0.015,
            f'{v:.3f}\n(n={n})', ha='center', va='bottom', fontsize=8)
ax.axhline(0, c='k', lw=0.6)
ax.set_ylim(-0.20, 0.50)
ax.set_ylabel('OOF $R^2$ (NMR → Bristol)')
ax.set_title('A  Prediction by Bristol source')

ax = axes[1]
err = np.vstack([(res['spearman'] - res['rho_ci_lo']).clip(0),
                 (res['rho_ci_hi'] - res['spearman']).clip(0)])
b = ax.bar(labels, res['spearman'], color=cols_, alpha=0.88, yerr=err,
           capsize=3, error_kw=dict(ecolor='#444', lw=1))
for r, v, hi in zip(b, res['spearman'], res['rho_ci_hi']):
    ax.text(r.get_x() + r.get_width() / 2, hi + 0.015, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, 0.68)
ax.set_ylabel('OOF Spearman ρ')
ax.set_title('B  Rank correlation by Bristol source')

fig.suptitle('Sensitivity: NMR→Bristol performance vs Bristol target provenance\n'
             '(GroupKFold by subject; error bars = 95% subject-cluster bootstrap)',
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / 'fig9_bristol_src_sensitivity.png', dpi=300,
            bbox_inches='tight')
plt.close(fig)
print('saved fig9')
