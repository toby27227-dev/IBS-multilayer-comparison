"""Paper/汇报用图：火山图、关联热图、拟合散点、层间对比。输出到 IBS/figs/。"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from label_utils import place_labels

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
FIG = ROOT / 'IBS/figs'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})
COLORS = {'IBS-D': '#d62728', 'IBS-C': '#1f77b4', 'Healthy': '#2ca02c'}
POS, NEG, NS = '#d62728', '#1f77b4', '#aaaaaa'


def stars(q):
    return '***' if q < 1e-3 else '**' if q < 0.01 else '*' if q < 0.05 else ''


b = pd.read_csv(CE / 'results_bristol_nmr' / 'biomarkers.csv')
nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)

# ---------- fig1: volcano ----------
b['neglogq'] = -np.log10(b['q'].clip(lower=1e-18))
LABEL_OFF = {
    'Acetate': (8, 0), 'Butyrate': (-8, 8), 'Lysine': (8, -2),
    'Tyrosine': (8, 0), 'Propionate': (8, -2), 'Succinate': (8, -2),
    'Uracil': (0, 16), 'Glucose': (12, 2), 'Leucine': (-46, 6),
    'Valine': (8, 2), 'Isoleucine': (8, -14), 'Alanine': (-40, -10),
    'Hypoxanthine': (-56, 6), 'Lactate': (10, -6), 'Glycine': (-24, 4),
    'b_arabinose': (7, -12), 'b_xylose': (0, -14),
    'Isovalerate': (8, 0), '2-Methylbutyrate': (8, 0),
}
fig, ax = plt.subplots(figsize=(6.8, 5.2))
for _, r in b.iterrows():
    c = NS if r['q'] >= 0.05 else (POS if r['spearman'] > 0 else NEG)
    ax.scatter(r['spearman'], r['neglogq'], s=42, c=c, edgecolors='white',
               linewidths=0.6, zorder=3)
ax.axhline(-np.log10(0.05), color='gray', ls='--', lw=0.8)
ax.text(-0.50, -np.log10(0.05) + 0.25, 'q = 0.05', fontsize=8, color='gray')
ax.set_xlabel('Spearman ρ (metabolite vs Bristol)')
ax.set_ylabel('−log10(q)')
ax.set_title(f'NMR metabolites vs Bristol (n={len(pd.read_csv(CE / "results_bristol_nmr" / "predictions.csv"))})')
ax.set_xlim(-0.55, 0.58)
ax.set_ylim(-0.4, 18.2)
fig.tight_layout()
place_labels(ax, [(r['taxon'], r['spearman'], r['neglogq'])
                  for _, r in b.iterrows()], LABEL_OFF, verbose=True)
fig.savefig(FIG / 'fig1_volcano_nmr_bristol.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- fig2: correlation heatmap (16 sig + Bristol) ----------
sig = b[b['q'] < 0.05]['taxon'].tolist()
order = sorted(sig, key=lambda c: -b.set_index('taxon').loc[c, 'spearman'])
df = nmr[order].join(sm['bristol']).dropna()
mat = np.log1p(df[order].clip(lower=0)).copy()
mat['Bristol'] = df['bristol']
C = mat.corr(method='spearman')
idx = order + ['Bristol']
C = C.loc[idx, idx]
vmax = float(np.abs(C.values).max())

fig, ax = plt.subplots(figsize=(9.4, 8.0))
im = ax.imshow(C.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               interpolation='nearest')
ax.set_xticks(range(len(idx)))
ax.set_xticklabels(idx, rotation=50, ha='right', fontsize=8.2)
ax.set_yticks(range(len(idx)))
ax.set_yticklabels(idx, fontsize=8.2)
ax.set_xticks(np.arange(-.5, len(idx), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(idx), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.0)
ax.tick_params(which='minor', length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
qmap = b.set_index('taxon')['q']
j = len(idx) - 1
for i, name in enumerate(idx[:-1]):
    st = stars(qmap[name])
    ax.text(j, i, f'{C.iloc[i, j]:.2f}' + (f'\n{st}' if st else ''),
            ha='center', va='center', fontsize=6.5, fontweight='bold')
    ax.text(i, j, f'{C.iloc[i, j]:.2f}', ha='center', va='center',
            fontsize=6.5)
for t in ax.texts:
    x, y = t.get_position()
    v = C.iloc[int(round(y)), int(round(x))]
    t.set_color('white' if abs(v) > 0.6 * vmax else 'black')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Spearman ρ')
ax.set_title('NMR metabolites vs Bristol — Spearman correlation\n'
             '(* q<0.05, ** q<0.01, *** q<0.001)', fontsize=10)
fig.tight_layout()
fig.savefig(FIG / 'fig2_heatmap_nmr_bristol.png', dpi=300, bbox_inches='tight')
plt.close(fig)


# ---------- fig3: OOF scatter ----------
def scatter_fit(pred_path, title, out, color_by_cohort):
    pred = pd.read_csv(pred_path, index_col=0)
    obs, prd = pred['observed'].values, pred['predicted'].values
    r2 = r2_score(obs, prd)
    rho = spearmanr(obs, prd).statistic
    rng = np.random.default_rng(0)
    xj = obs + rng.uniform(-0.13, 0.13, len(obs))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if color_by_cohort:
        lab = sm.reindex(pred.index)['Cohort_label']
        for g in ['IBS-D', 'IBS-C', 'Healthy']:
            m = (lab == g).values
            ax.scatter(xj[m], prd[m], s=28, alpha=0.8, c=COLORS[g],
                       label=f'{g} (n={m.sum()})', edgecolors='white',
                       linewidths=0.3)
        ax.legend(fontsize=8.5, loc='upper left', frameon=False)
        xcol = '#333333'
    else:
        ax.scatter(xj, prd, s=28, alpha=0.8, c='#1f77b4',
                   edgecolors='white', linewidths=0.3)
        xcol = '#1f77b4'
    k, d = np.polyfit(obs, prd, 1)
    xs = np.array([obs.min(), obs.max()])
    ax.plot(xs, k * xs + d, color='black', lw=1.6,
            label=f'fit: y={k:.2f}x+{d:.2f}')
    lo, hi = min(obs.min(), prd.min()), max(obs.max(), prd.max())
    ax.plot([lo, hi], [lo, hi], ls='--', color='gray', lw=0.9,
            label='y = x')
    ax.text(0.03, 0.97, f'$R^2$ = {r2:.3f}\nSpearman ρ = {rho:.3f}',
            transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))
    ax.legend(fontsize=8.5, loc='lower right', frameon=False)
    ax.set_xlabel('Observed Bristol')
    ax.set_ylabel('Predicted Bristol (out-of-fold)')
    ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=300)
    plt.close(fig)
    return r2, rho


r2a, rhoa = scatter_fit(CE / 'results_bristol_nmr' / 'predictions.csv',
                        'NMR → Bristol: out-of-fold prediction\n'
                        '(subject-grouped CV, n=424)',
                        'fig3_scatter_nmr_bristol.png', True)
r2b, rhob = scatter_fit(CE / 'results_bristol_nmr_ibsc' / 'predictions.csv',
                        'NMR → Bristol within IBS-C\n'
                        '(subject-grouped CV, n=128)',
                        'fig3b_scatter_nmr_ibsc.png', False)

# ---------- fig4: layer comparison ----------
rb = pd.read_csv(CE / 'summary_bristol.csv')
rc = pd.read_csv(CE / 'summary_cohort.csv')
rb = rb[~rb['task'].str.contains('内部')]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, d, val, xlabel in [
        (axes[0], rb, 'R2_OOF', 'OOF $R^2$ (Bristol)'),
        (axes[1], rc, 'balanced_acc', 'balanced accuracy (IBS-D/C/Healthy)')]:
    d = d.copy()
    d['label'] = (d['task'].str.replace('Bristol: ', '').str.replace('Cohort: ', '')
                  .str.replace('胆汁酸', 'Bile acids')
                  .str.replace('色氨酸', 'Tryptophan')
                  .str.replace('KEGG通路', 'KEGG pathways')
                  .str.replace('属', 'Genus').str.replace('种', 'Species'))
    d = d.sort_values(val)
    colors = ['#ff7f0e' if l.startswith('NMR(') else
              '#9467bd' if l.startswith('NMR+') else '#4c72b0'
              for l in d['label']]
    ax.barh(d['label'], d[val], color=colors, alpha=0.9)
    for y, (v, s) in enumerate(zip(d[val], d.get('std', pd.Series([0] * len(d))).fillna(0))):
        ax.text(v + (0.005 if v >= 0 else -0.005), y, f'{v:.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
    if val == 'R2_OOF':
        ax.axvline(0, color='gray', lw=0.8)
    else:
        ax.axvline(1 / 3, color='gray', ls='--', lw=0.9)
        ax.text(1 / 3 + 0.006, len(d) - 0.25, 'chance', fontsize=8,
                color='gray', va='top')
    ax.set_xlim(0, d[val].max() * 1.10)
    ax.set_ylim(-0.6, len(d) - 0.10)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.tick_params(labelsize=9)
fig.suptitle('Layer comparison (subject-grouped CV)', fontsize=11.5)
fig.tight_layout()
fig.savefig(FIG / 'fig4_layer_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print('saved:')
for p in sorted(FIG.glob('*.png')):
    print(' -', p.name)
print(f'fig3: R2={r2a:.3f} rho={rhoa:.3f} | fig3b (IBS-C): R2={r2b:.3f} rho={rhob:.3f}')
