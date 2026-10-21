"""冻结选择器的留出零样本评估（held-out zero-shot evaluation）。

- 读取 frozen_selector.json（源域冻结：特征/缩放/系数）
- 读取 Jacobs 2023 特征矩阵；BH 标签仅在评估末端引入
- 对照：从 frozen_model_metadata.json 载入 R11(shared11)/F16(full16) 冻结模型
  （均在 Cell 2020 训练冻结；非共享特征按训练均值 z 插补 = 0）
- 指标：AUC(D vs C+M，与既有分析同口径)、BH 序数 Spearman
- 配对 bootstrap（B=2000，样本级重抽样）：各模型 AUC 95% CI、
  ΔAUC(Iron−R11)、ΔAUC(Iron−F16) 的 95% CI 与 p=mean(Δ≤0)
- 重绘图7：A 源域稳定性×平台偏移（label-blind）；B 平台 KS；C 零样本 AUC（含 95% CI）
输出：zero_shot_results.csv、figs/fig7_stability_selection.png
"""
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

ROOT = Path.cwd()
NW = ROOT / 'IBS/jacobs2023'
OUT = ROOT / 'IBS/jacobs_validation'
FIG = ROOT / 'IBS/figs'
B = 2000
RNG = np.random.default_rng(42)

sel = json.load(open(OUT / 'frozen_selector.json', encoding='utf-8'))
IRON = sel['selected_features']
SHARED = sel['shared']
FMAP = sel['feature_map']

met = pd.read_csv(NW / 'jacobs2023_metabolites.csv', index_col=0)
meta = pd.read_csv(NW / 'jacobs2023_metadata.csv').set_index('Patient')
idx = met.index.intersection(meta.index)
met, meta = met.loc[idx], meta.loc[idx]
ordinal = {'Constipation': 0, 'Mixed': 1, 'Diarrhea': 2}
keep = meta['BH'].isin(ordinal)
n = int(keep.sum())
Xj_shared = np.log1p(met.loc[keep, [FMAP[f] for f in SHARED]].values.astype(float))
y = (meta.loc[keep, 'BH'] == 'Diarrhea').astype(int).values
ordv = meta.loc[keep, 'BH'].map(ordinal).values
print(f'Jacobs evaluation set: n={n} (C={int((meta.loc[keep, "BH"] == "Constipation").sum())}, '
      f'M={int((meta.loc[keep, "BH"] == "Mixed").sum())}, '
      f'D={int((meta.loc[keep, "BH"] == "Diarrhea").sum())})')

# Iron-Core 冻结得分
Xi = np.log1p(met.loc[keep, [FMAP[f] for f in IRON]].values.astype(float))
Zi = (Xi - np.array(sel['scaler_mean'])) / np.array(sel['scaler_scale'])
score_iron = Zi @ np.array(sel['ridge_coef']) + sel['ridge_intercept']

# R11 / F16 冻结对照（源域已冻结，不调参）
frozen = json.load(open(OUT / 'frozen_model_metadata.json', encoding='utf-8'))


def score_frozen(fm):
    feats = fm['features']
    Z = np.zeros((n, len(feats)))
    for j, f in enumerate(feats):
        if f in SHARED:
            k = SHARED.index(f)
            Z[:, j] = (Xj_shared[:, k] - fm['mean'][j]) / fm['sd'][j]
        else:
            Z[:, j] = 0.0
    return Z @ np.array(fm['coef']) + fm['intercept']


models = {'Iron-Core': score_iron,
          'R11': score_frozen(frozen['shared11']),
          'F16': score_frozen(frozen['full16'])}

point = {k: roc_auc_score(y, v) for k, v in models.items()}
for k, v in models.items():
    print(f'{k:10s} AUC(D vs C+M)={point[k]:.3f} '
          f'rho_BH={spearmanr(ordv, v).statistic:+.3f}')

keys = ['Iron-Core', 'R11', 'F16']
boot = {k: [] for k in keys}
d_ir, d_if = [], []
for b in range(B):
    ii = RNG.choice(n, n, replace=True)
    if len(np.unique(y[ii])) < 2:
        continue
    a = {k: roc_auc_score(y[ii], models[k][ii]) for k in keys}
    for k in keys:
        boot[k].append(a[k])
    d_ir.append(a['Iron-Core'] - a['R11'])
    d_if.append(a['Iron-Core'] - a['F16'])
boot = {k: np.array(v) for k, v in boot.items()}
d_ir, d_if = np.array(d_ir), np.array(d_if)


def ci(a):
    return np.percentile(a, [2.5, 97.5])


rows = []
for k in keys:
    lo, hi = ci(boot[k])
    rows.append({'model': k, 'auc': round(point[k], 4),
                 'auc_ci_lo': round(lo, 4), 'auc_ci_hi': round(hi, 4),
                 'rho_bh_ordinal': round(spearmanr(ordv, models[k]).statistic, 4)})
for tag, d in [('Iron-Core - R11', d_ir), ('Iron-Core - F16', d_if)]:
    lo, hi = ci(d)
    rows.append({'model': tag, 'delta_auc': round(float(d.mean()), 4),
                 'delta_ci_lo': round(lo, 4), 'delta_ci_hi': round(hi, 4),
                 'p_delta_le_0': round(float((d <= 0).mean()), 4)})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'zero_shot_results.csv', index=False)
print('\n' + res.to_string(index=False))

# ---------- 图7 重绘（A/B label-blind；C 评估含 95% CI） ----------
feat = sel['features']
names = list(sel['sig16'])
xs = np.array([feat[f]['pi_uni_cell'] for f in names])
ys = np.array([feat[f]['ks'] for f in names], dtype=float)
yplot = np.where(np.isnan(ys), -0.05, ys)
is_sel = np.array([f in IRON for f in names])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
ax = axes[0]
ax.scatter(xs[~is_sel], yplot[~is_sel], c='#aaaaaa', s=42, alpha=.85,
           edgecolors='white', lw=.6)
ax.scatter(xs[is_sel], yplot[is_sel], c='#2ca02c', s=52, alpha=.9,
           edgecolors='white', lw=.6)
ax.set_xlim(0.55, 1.16)
ax.set_ylim(-0.18, 0.27)
LAB = {
    'Hypoxanthine': (8, 4), 'Succinate': (-58, 6),
    'Isoleucine': (8, 4), 'Alanine': (-48, 4),
    'Valine': (8, 4), 'Glucose': (-44, 12),
    'Uracil': (10, -14), 'Leucine': (-46, -14),
    'Lysine': (8, 8), 'Tyrosine': (8, -12),
    'Lactate': (-40, 6), '2-Methylbutyrate': (8, -14),
    'Acetate': (8, -14), 'Butyrate': (8, 8),
    'Propionate': (-56, 8), 'Isovalerate': (8, -14),
}
for f, x, yv in zip(names, xs, yplot):
    ax.annotate(f, (x, yv), textcoords='offset points',
                xytext=LAB.get(f, (8, 4)), fontsize=7)
ax.text(0.02, 0.02, 'SCFA with undefined KS plotted at −0.05',
        transform=ax.transAxes, fontsize=6.5, color='#666666')
ax.axvline(0.8, ls='--', c='gray', lw=.8)
ax.axhline(0.15, ls='--', c='gray', lw=.8)
ax.set_xlabel('π_uni_cell (Cell 2020 bootstrap, label-blind)')
ax.set_ylabel('KS (Cell vs Jacobs X, label-blind)')
ax.set_title('A  Source-closed selection')

ax = axes[1]
order = np.argsort(yplot)
nm = np.array(names)[order]
vals = yplot[order]
ax.barh(nm, vals, color=['#d62728' if v >= .15 else '#55a868' for v in vals])
ax.axvline(0.15, ls='--', c='gray', lw=.9)
ax.set_xlabel('KS statistic (platform shift)')
ax.set_title('B  Platform sensitivity')

ax = axes[2]
kk = ['F16', 'R11', 'Iron-Core']
vals = [point[k] for k in kk]
lo_hi = [ci(boot[k]) for k in kk]
err = [[v - lo for v, (lo, hi) in zip(vals, lo_hi)],
       [hi - v for v, (lo, hi) in zip(vals, lo_hi)]]
lbl = ['F16\n(16 feat)', 'R11\n(11 feat)', f'Iron-Core\n({len(IRON)} feat)']
bars = ax.bar(lbl, vals, color=['#d62728', '#ff7f0e', '#2ca02c'], alpha=.85,
              yerr=err, capsize=4)
for r, v, (lo, hi) in zip(bars, vals, lo_hi):
    ax.text(r.get_x() + r.get_width() / 2, hi + .010, f'{v:.3f}',
            ha='center', fontsize=10)
ax.axhline(0.5, ls='--', c='gray', lw=.8)
ax.set_ylim(0.45, 0.75)
ax.set_ylabel('Zero-shot AUC (D vs C+M, Jacobs)')
ax.set_title('C  Held-out zero-shot (95% CI)')
fig.suptitle('Source-closed stability selection & held-out zero-shot evaluation',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / 'fig7_stability_selection.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nsaved: zero_shot_results.csv -> {OUT} | fig7 -> {FIG}')
