"""敏感性分析：Bristol 目标来源（实测 vs 兜底）对 NMR 层预测力的影响。

背景：build_bristol_target.py 为每个 run 构建 Bristol 目标——±3 天（w3）/±7 天（w7）
窗口内实测日记均值；若窗口内无记录则回退为受试者个人均值（subject 兜底）。
此前论文报告的 R²=0.196 使用全部来源样本；本脚本在完全相同的监督协议
（regression_biomarkers.py：log1p → 标准化 → GroupKFold(5, 按受试者) →
内层 GroupKFold(3) Ridge 调参 → 合并 OOF）下分层重跑：

  all            全体（复现口径）
  real_w3w7      仅实测（w3+w7）
  w3            仅 ±3 天实测
  w7            仅 ±7 天实测
  fallback       仅受试者均值兜底

输出: IBS/cellomics/results_bristol_nmr/sensitivity_src.csv
      IBS/figs/fig9_bristol_src_sensitivity.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
FIG = ROOT / 'IBS/figs'
sys.path.insert(0, str(ROOT / 'IBS/pipeline'))
from supervised_biomarkers import build_features

nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
mb = pd.read_csv(ROOT / 'IBS/gmrepo/metadata_bristol.csv', index_col=0)

df = sm.join(mb[['bristol_src']], how='inner')
df = df[df['bristol'].notna() & df.index.isin(nmr.index)]
X, cols = build_features(nmr.loc[df.index], 0.10, 'log')
y = df['bristol'].values.astype(float)
g = df['subject'].astype(str).values
src = df['bristol_src'].values

SUBSETS = [
    ('all', np.ones(len(y), bool)),
    ('real_w3w7', np.isin(src, ['w3', 'w7'])),
    ('w3', src == 'w3'),
    ('w7', src == 'w7'),
    ('fallback', src == 'subject'),
]

B_BOOT = 2000


def run_subset(mask):
    Xm, ym, gm = X[mask], y[mask], g[mask]
    oof = np.full(len(ym), np.nan)
    for tr, te in GroupKFold(5).split(Xm, ym, gm):
        gs = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10, 100, 1000, 10000]},
                          cv=GroupKFold(3), scoring='r2')
        gs.fit(Xm[tr], ym[tr], groups=gm[tr])
        oof[te] = gs.predict(Xm[te])
    rng = np.random.default_rng(42)
    subs_u = np.unique(gm)
    idm = {s: np.where(gm == s)[0] for s in subs_u}
    rhos, r2s = np.empty(B_BOOT), np.empty(B_BOOT)
    for b in range(B_BOOT):
        pick = rng.choice(subs_u, size=len(subs_u), replace=True)
        ii = np.concatenate([idm[s] for s in pick])
        rhos[b] = spearmanr(ym[ii], oof[ii]).statistic
        r2s[b] = 1 - ((ym[ii] - oof[ii]) ** 2).sum() / \
            ((ym[ii] - ym[ii].mean()) ** 2).sum()
    return (r2_score(ym, oof), spearmanr(ym, oof).statistic,
            len(ym), len(set(gm)), float(np.mean(ym)), oof,
            np.percentile(r2s, [2.5, 97.5]), np.percentile(rhos, [2.5, 97.5]))


rows, oof_store = [], {}
for tag, mask in SUBSETS:
    if mask.sum() < 40 or len(set(g[mask])) < 5:
        print(f'skip {tag}: n={int(mask.sum())} 过小')
        continue
    r2, rho, n, ns, m, oof, ci_r2, ci_rho = run_subset(mask)
    oof_store[tag] = (mask, oof)
    rows.append({'subset': tag, 'n': n, 'n_subjects': ns,
                 'mean_bristol': round(m, 3),
                 'oof_r2': round(r2, 4), 'spearman': round(rho, 4),
                 'r2_ci_lo': round(ci_r2[0], 4), 'r2_ci_hi': round(ci_r2[1], 4),
                 'rho_ci_lo': round(ci_rho[0], 4), 'rho_ci_hi': round(ci_rho[1], 4)})
    print(f'{tag:12s} n={n:4d} subj={ns:3d} mean={m:.2f} '
          f'R2={r2:+.4f} [{ci_r2[0]:+.3f},{ci_r2[1]:+.3f}] '
          f'rho={rho:+.4f} [{ci_rho[0]:+.3f},{ci_rho[1]:+.3f}]')

res = pd.DataFrame(rows)
res.to_csv(CE / 'results_bristol_nmr' / 'sensitivity_src.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
names = res['subset'].tolist()
labels = ['全体\n(all)', '仅实测\n(w3+w7)', 'w3\n(±3d)', 'w7\n(±7d)', '兜底\n(subject)'][:len(names)]
cols_ = ['#4c72b0', '#55a868', '#55a868', '#55a868', '#d62728'][:len(names)]

ax = axes[0]
err = np.vstack([(res['oof_r2'] - res['r2_ci_lo']).clip(0),
                 (res['r2_ci_hi'] - res['oof_r2']).clip(0)])
b = ax.bar(labels, res['oof_r2'], color=cols_, alpha=0.88, yerr=err,
           capsize=3, error_kw=dict(ecolor='#444', lw=1))
for r, v, n in zip(b, res['oof_r2'], res['n']):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.025, f'{v:.3f}\n(n={n})',
            ha='center', fontsize=8)
ax.axhline(0, c='k', lw=0.6)
ax.set_ylabel('OOF $R^2$ (NMR → Bristol)')
ax.set_title('A  Prediction by Bristol source')

ax = axes[1]
err = np.vstack([(res['spearman'] - res['rho_ci_lo']).clip(0),
                 (res['rho_ci_hi'] - res['spearman']).clip(0)])
b = ax.bar(labels, res['spearman'], color=cols_, alpha=0.88, yerr=err,
           capsize=3, error_kw=dict(ecolor='#444', lw=1))
for r, v in zip(b, res['spearman']):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.04, f'{v:.3f}',
            ha='center', fontsize=8)
ax.axhline(0, c='k', lw=0.6)
ax.set_ylabel('OOF Spearman ρ')
ax.set_title('B  Rank correlation by Bristol source')

fig.suptitle('Sensitivity: NMR→Bristol performance vs Bristol target provenance '
             '(GroupKFold by subject; error bars = 95% subject-cluster bootstrap)',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / 'fig9_bristol_src_sensitivity.png', dpi=300)
plt.close(fig)
print(f'\nsaved: sensitivity_src.csv -> {CE / "results_bristol_nmr"} | fig9 -> {FIG}')
