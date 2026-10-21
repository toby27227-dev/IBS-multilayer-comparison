"""跨层验证：丁酸代谢/氨基酸降解/胆汁酸代谢 KO 与通路 × NMR 代谢物标志物。

功能特征（宏基因组 KO/通路）与代谢物（NMR/胆汁酸）做 Spearman 关联，
BH-FDR 校正，输出多组学关联热图 IBS/figs/fig5_crosslayer_correlation.png。
"""
import warnings
from itertools import chain
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
FIG = ROOT / 'IBS/figs'
FIG.mkdir(exist_ok=True)


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = prev
    return q


ko = pd.read_csv(CE / 'ko_stool.csv', index_col=0)
pw = pd.read_csv(CE / 'kegg_pathways_ko.csv', index_col=0)
nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
ba = pd.read_csv(CE / 'bile_acids.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)

b16 = pd.read_csv(CE / 'results_bristol_nmr' / 'biomarkers.csv')
sig16 = b16[b16['q'] < 0.05].sort_values('spearman', ascending=False)['taxon'].tolist()
bba = pd.read_csv(CE / 'results_bristol_bileacids' / 'biomarkers.csv')
sigBA = bba.sort_values('q').head(6)['taxon'].tolist()

BUTYRATE_GENES = ['K00625', 'K00925', 'K00929', 'K00634', 'K01034', 'K01035', 'K01895']
feats = {}
feats['Butyrate synthesis genes (pta/ackA/buk/ptb/CoA-T/acs)'] = ko[BUTYRATE_GENES].sum(axis=1)
feats['Butyryl-CoA dehydrogenase (bcd)'] = ko['K00248']
feats['Butyraldehyde dehydrogenase (adhE)'] = ko['K04072']
feats['KEGG ko00650 Butanoate metabolism'] = pw['ko00650']
feats['KEGG ko00280 BCAA degradation'] = pw['ko00280']
feats['KEGG ko00350 Tyrosine metabolism'] = pw['ko00350']
feats['KEGG ko00310 Lysine degradation'] = pw['ko00310']
feats['KEGG ko00380 Tryptophan metabolism'] = pw['ko00380']
feats['KEGG ko00120 Primary bile acid biosynthesis'] = pw['ko00120']
feats['KEGG ko00121 Secondary bile acid biosynthesis'] = pw['ko00121']
feats['Bile salt hydrolase (K01442)'] = ko['K01442']
F = pd.DataFrame(feats)

X = (F.join(nmr[sig16], how='inner').join(ba[sigBA], how='inner')
      .join(sm['bristol'].rename('Bristol'), how='inner').dropna())
n = len(X)
feat_names = list(F.columns)
mat_cols = sig16 + sigBA + ['Bristol']

R = pd.DataFrame(index=feat_names, columns=mat_cols, dtype=float)
Pv = pd.DataFrame(index=feat_names, columns=mat_cols, dtype=float)
for f in feat_names:
    for c in mat_cols:
        rho, p = spearmanr(X[f], X[c])
        R.loc[f, c] = rho
        Pv.loc[f, c] = p
Q = pd.DataFrame(bh(Pv.values.flatten()).reshape(Pv.shape),
                 index=Pv.index, columns=Pv.columns)
R.to_csv(CE / 'crosslayer_spearman.csv')
Q.to_csv(CE / 'crosslayer_q.csv')

SHORT = {
    'Butyrate synthesis genes (pta/ackA/buk/ptb/CoA-T/acs)': 'Butyrate synthesis genes (7 KOs)',
    'Butyryl-CoA dehydrogenase (bcd)': 'bcd (K00248)',
    'Butyraldehyde dehydrogenase (adhE)': 'adhE (K04072)',
    'KEGG ko00650 Butanoate metabolism': 'ko00650 Butanoate metabolism',
    'KEGG ko00280 BCAA degradation': 'ko00280 BCAA degradation',
    'KEGG ko00350 Tyrosine metabolism': 'ko00350 Tyrosine metabolism',
    'KEGG ko00310 Lysine degradation': 'ko00310 Lysine degradation',
    'KEGG ko00380 Tryptophan metabolism': 'ko00380 Tryptophan metabolism',
    'KEGG ko00120 Primary bile acid biosynthesis': 'ko00120 Primary BA biosynthesis',
    'KEGG ko00121 Secondary bile acid biosynthesis': 'ko00121 Secondary BA biosynthesis',
    'Bile salt hydrolase (K01442)': 'BSH (K01442)',
}


def mkpanel(ax, cols, xlabels, xticks, fs):
    sub = R[cols].astype(float).values
    im = ax.imshow(sub, cmap='RdBu_r', vmin=-VMAX, vmax=VMAX, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=fs)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(feat_names), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', length=0)
    ax.set_axisbelow(False)
    for i, f in enumerate(feat_names):
        for j, c in enumerate(cols):
            rho = R.loc[f, c]
            q = Q.loc[f, c]
            st = '***' if q < 1e-3 else '**' if q < 0.01 else '*' if q < 0.05 else ''
            ax.text(j, i, f'{rho:.2f}{st}', ha='center', va='center', fontsize=fs,
                    color='white' if abs(rho) > 0.6 * VMAX else 'black')
    if xticks:
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels([SHORT[f] for f in feat_names], fontsize=9)
    else:
        ax.set_yticks([])
    return im


VMAX = float(np.nanmax(np.abs(R.values)))
fig = plt.figure(figsize=(12.8, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[16, 6, 1], wspace=0.03)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
im = mkpanel(ax0, sig16, sig16, True, 6.2)
mkpanel(ax1, sigBA, sigBA, False, 6.2)
mkpanel(ax2, ['Bristol'], ['Bristol'], False, 7)
ax0.set_title('NMR metabolites (n=16 sig)', fontsize=10)
ax1.set_title('Bile acids (top 6)', fontsize=10)
ax2.set_title('Ref', fontsize=10)
cax = fig.add_axes([0.952, 0.30, 0.011, 0.46])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Spearman ρ', fontsize=9, labelpad=8)
fig.suptitle(f'Cross-layer correlation: microbial functional potential vs metabolite pools '
             f'(n={n}; * q<0.05, ** q<0.01, *** q<0.001)', fontsize=10.5, y=0.97)
fig.subplots_adjust(left=0.26, right=0.93, top=0.84, bottom=0.25, wspace=0.035)
fig.savefig(FIG / 'fig5_crosslayer_correlation.png', dpi=300,
            bbox_inches='tight')
plt.close(fig)

print(f'n = {n}')
print()
print('NMR 16 significant markers:')
print(b16[b16['q'] < 0.05].sort_values('spearman', ascending=False)
      [['taxon', 'spearman', 'q']].round(4).to_string(index=False))
print()
print('Bile acid top-6 markers:')
print(bba.sort_values('q').head(6)[['taxon', 'spearman', 'q']].round(4).to_string(index=False))
print()
for f in ['Butyrate synthesis genes (pta/ackA/buk/ptb/CoA-T/acs)',
          'Butyryl-CoA dehydrogenase (bcd)',
          'Butyraldehyde dehydrogenase (adhE)',
          'KEGG ko00650 Butanoate metabolism',
          'KEGG ko00280 BCAA degradation',
          'KEGG ko00380 Tryptophan metabolism',
          'Bile salt hydrolase (K01442)']:
    print(f'ROW: {SHORT[f]}')
    print(pd.DataFrame({'rho': R.loc[f], 'q': Q.loc[f]}).round(4).to_string())
    print()
KEYS = [
    ('Butyrate synthesis genes (pta/ackA/buk/ptb/CoA-T/acs)', ['Butyrate', 'Acetate', 'Propionate']),
    ('KEGG ko00650 Butanoate metabolism', ['Butyrate', 'Acetate', 'Propionate']),
    ('KEGG ko00280 BCAA degradation', ['Leucine', 'Valine', 'Isoleucine']),
    ('KEGG ko00350 Tyrosine metabolism', ['Tyrosine']),
    ('KEGG ko00310 Lysine degradation', ['Lysine']),
    ('Bile salt hydrolase (K01442)', sigBA),
    ('KEGG ko00121 Secondary bile acid biosynthesis', sigBA),
    ('KEGG ko00120 Primary bile acid biosynthesis', sigBA),
]
for f, cols in KEYS:
    print(f'-- {f}')
    for c in cols:
        if c in R.columns:
            print(f'   {c:36s} rho={R.loc[f, c]:+.3f}  q={Q.loc[f, c]:.2e}')
print()
print('Strongest cross-layer couplings (q<0.001):')
pairs = [(f, c, R.loc[f, c], Q.loc[f, c]) for f in feat_names for c in mat_cols
         if Q.loc[f, c] < 1e-3]
pairs.sort(key=lambda t: -abs(t[2]))
for f, c, rho, q in pairs[:15]:
    print(f'   {f[:48]:48s} ~ {c:32s} rho={rho:+.3f} q={q:.2e}')
