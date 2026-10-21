"""图8：两样本 MR forest plot（属 → IBS 风险）。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
ROOT = R.parents[1] / 'IBS'
df = pd.read_csv(R / 'results' / 'mr_results.csv').sort_values('ivw_or').reset_index(drop=True)
fdr_min = df['ivw_fdr'].min()

fig, ax = plt.subplots(figsize=(9.5, 8.8))
y = np.arange(len(df))
for i, r in df.iterrows():
    ax.plot([r['ivw_or_lo'], r['ivw_or_hi']], [i, i], color='0.55', lw=1.1, zorder=1)
colors = ['#c0392b' if o > 1 else '#2471a3' for o in df['ivw_or']]
sizes = [64 if p < 0.05 else (52 if p < 0.10 else 40) for p in df['ivw_p']]
ax.scatter(df['ivw_or'], y, c=colors, s=sizes, zorder=2, edgecolors='k', linewidths=0.8)
ax.axvline(1.0, color='0.3', lw=1.0, ls='--')
ax.set_yticks(y)
ax.set_yticklabels(df['taxon'], fontsize=8.5)
ax.set_xscale('log')
ax.set_xlim(0.78, 1.34)
ticks = [0.8, 0.9, 1.0, 1.1, 1.2]
ax.set_xticks(ticks)
ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=9)
ax.set_xlabel('IVW OR (95% CI) per 1-SD higher genus abundance', fontsize=9.5)
for i, r in df.iterrows():
    ax.text(1.36, i, f"{int(r['n_snp']):>2d}", ha='right', va='center',
            fontsize=7.5, family='monospace', clip_on=False)
    pstr = f"{r['ivw_p']:.3f}" if r['ivw_p'] >= 1e-3 else f"{r['ivw_p']:.1e}"
    ax.text(1.52, i, pstr, ha='right', va='center', fontsize=7.5, family='monospace',
            color='#c0392b' if r['ivw_p'] < 0.05 else 'black', clip_on=False)
ax.text(1.36, len(df) - 0.1, 'nSNP', ha='right', fontsize=8, family='monospace',
        weight='bold', clip_on=False)
ax.text(1.52, len(df) - 0.1, 'P(IVW)', ha='right', fontsize=8, family='monospace',
        weight='bold', clip_on=False)
ax.set_title('Two-sample MR: gut genus (MiBioGen) → IBS risk (GCST90016564)\n'
             '27 genera, 259 harmonized SNPs; no taxon survives BH-FDR (min FDR=%.2f)' % fdr_min,
             fontsize=10.5)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.subplots_adjust(left=0.22, right=0.78, top=0.92, bottom=0.07)
out = ROOT / 'figs' / 'fig8_mr_forest.png'
fig.savefig(out, dpi=300)
print('saved:', out)
