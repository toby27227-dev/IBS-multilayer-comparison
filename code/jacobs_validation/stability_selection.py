"""跨队列特征稳定性筛选（Stability Selection / Cross-Cohort Intersection）。

针对 F16 坍塌警示，剔除"对技术平台高度敏感、易被插补毒化"的脆弱特征，
锁定高频稳定、方向一致的核心集（Iron-Core）。

三重筛选（预注册标准；v2.3 修正：选择在源域内完全闭环，目标域标签不参与）：
  ① 源域内稳定性：Cell 2020 分组 bootstrap B=300（按受试者重抽样 50%），
     统计 (a) 单变量 Spearman p<0.05 且方向与全样本一致频率 π_uni_cell；
     (b) Lasso 多变量非零选择频率 π_multi_cell（固定 alpha 来自 Cell 全数据 LassoCV）。
     标准：π_uni_cell ≥ 0.8。
  ② 平台稳健：Cell vs Jacobs 自标准化分布 KS < 0.15（无监督、不使用标签）。
  ③ 共同可测：特征同时存在于两队列面板（in_target）。

目标域标签（Jacobs BH 亚型）仅用于冻结模型的**事后评估**（零样本 AUC/ρ）；
π_uni_jac / π_multi_jac 作为诊断列报告（解释旧协议为何过拟合目标域），
**不参与特征选择**——避免目标域标签泄漏。
（v2.2 及之前协议将 π_uni_jac ≥ 0.8 并入选择判据；本版保留审计列
iron_legacy_leaky 以便对照。）

输出：stability_results.csv / iron_set.json / fig7；并以 Iron-Core 重训冻结模型
复测零样本（对照 R11 / F16）。
"""
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
NW = ROOT / 'IBS/jacobs2023'
OUT = ROOT / 'IBS/jacobs_validation'
FIG = ROOT / 'IBS/figs'
FIG.mkdir(exist_ok=True)
B = 300
RNG = np.random.default_rng(42)

nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
b16 = pd.read_csv(CE / 'results_bristol_nmr' / 'biomarkers.csv')
SIG16 = b16[b16['q'] < 0.05].sort_values('spearman', ascending=False)['taxon'].tolist()
FEATURE_MAP = {
    'Lysine': 'lysine', 'Tyrosine': 'tyrosine', 'Leucine': 'leucine',
    'Valine': 'valine', 'Isoleucine': 'isoleucine', 'Alanine': 'alanine',
    'Glucose': 'glucose', 'Succinate': 'succinate', 'Uracil': 'uracil',
    'Hypoxanthine': 'hypoxanthine', 'Lactate': 'lactate',
}
SHARED = [f for f in SIG16 if f in FEATURE_MAP]

# ---------- Cell 2020 训练域数据（源域） ----------
tr = nmr.join(sm[['bristol', 'subject']]).dropna()
Xc_log = np.log1p(tr[SIG16].values.astype(float))
yc = tr['bristol'].values
gc = tr['subject'].values
ref_sign_cell = {f: np.sign(spearmanr(tr[f], tr['bristol']).statistic) for f in SIG16}

# ---------- Jacobs 2023 目标域数据（仅评估；标签不用于选择） ----------
met = pd.read_csv(NW / 'jacobs2023_metabolites.csv', index_col=0)
meta = pd.read_csv(NW / 'jacobs2023_metadata.csv').set_index('Patient')
idx = met.index.intersection(meta.index)
met, meta = met.loc[idx], meta.loc[idx]
ordinal = {'Constipation': 0, 'Mixed': 1, 'Diarrhea': 2}
keep_j = meta['BH'].isin(ordinal)
Xj_log = np.log1p(met.loc[keep_j, [FEATURE_MAP[f] for f in SHARED]].values.astype(float))
yj = meta.loc[keep_j, 'BH'].map(ordinal).values
ref_sign_jac = {f: np.sign(spearmanr(Xj_log[:, k], yj).statistic)
                for k, f in enumerate(SHARED)}

# ---------- Lasso 参考 alpha（源域全数据） ----------
alpha_cell = LassoCV(cv=5, random_state=0, max_iter=20000).fit(
    StandardScaler().fit_transform(Xc_log), yc).alpha_
alpha_jac = LassoCV(cv=5, random_state=0, max_iter=20000).fit(
    StandardScaler().fit_transform(Xj_log), yj).alpha_
print(f'LassoCV alpha: cell={alpha_cell:.4f}, jacobs={alpha_jac:.4f}')

subjects = tr['subject'].unique()
feat_rows = {}
for f in SIG16:
    feat_rows[f] = {'pi_uni_cell': 0, 'pi_multi_cell': 0}
for f in SHARED:
    feat_rows[f].update({'pi_uni_jac': 0, 'pi_multi_jac': 0})

# ---------- 分组 bootstrap ----------
for b in range(B):
    pick = RNG.choice(subjects, size=int(0.5 * len(subjects)), replace=False)
    m = np.isin(gc, pick)
    Xb, yb = Xc_log[m], yc[m]
    for j, f in enumerate(SIG16):
        rho, p = spearmanr(Xb[:, j], yb)
        if p < 0.05 and np.sign(rho) == ref_sign_cell[f]:
            feat_rows[f]['pi_uni_cell'] += 1
    sel = Lasso(alpha=alpha_cell, max_iter=20000).fit(
        StandardScaler().fit_transform(Xb), yb).coef_ != 0
    for j, f in enumerate(SIG16):
        feat_rows[f]['pi_multi_cell'] += int(sel[j])

idx_j = np.arange(len(yj))
for b in range(B):
    pick = RNG.choice(idx_j, size=int(0.5 * len(idx_j)), replace=False)
    Xb, yb = Xj_log[pick], yj[pick]
    for j, f in enumerate(SHARED):
        rho, p = spearmanr(Xb[:, j], yb)
        if p < 0.05 and np.sign(rho) == ref_sign_jac[f]:
            feat_rows[f]['pi_uni_jac'] += 1
    sel = Lasso(alpha=alpha_jac, max_iter=20000).fit(
        StandardScaler().fit_transform(Xb), yb).coef_ != 0
    for j, f in enumerate(SHARED):
        feat_rows[f]['pi_multi_jac'] += int(sel[j])

for f, d in feat_rows.items():
    for k in list(d):
        d[k] /= B

# ---------- 平台稳健性（KS，无监督） ----------
for f in SIG16:
    if f in FEATURE_MAP:
        xc = np.log1p(tr[f].values.astype(float))
        xj = np.log1p(met[FEATURE_MAP[f]].values.astype(float))
        zc = (xc - xc.mean()) / xc.std()
        zj = (xj - xj.mean()) / xj.std()
        feat_rows[f]['ks'] = ks_2samp(zc, zj).statistic
    else:
        feat_rows[f]['ks'] = np.nan

res = pd.DataFrame(feat_rows).T
res['in_target'] = res.index.isin(SHARED)
res['dir_conc'] = [ref_sign_cell[f] == ref_sign_jac.get(f, np.nan)
                   for f in res.index]

KRIT = dict(pi_uni=0.8, ks=0.15)
res['pass_stability'] = (res['pi_uni_cell'] >= KRIT['pi_uni'])
res['pass_stability_jac'] = res['pi_uni_jac'].fillna(0) >= KRIT['pi_uni']
res['pass_ks'] = res['ks'].fillna(0) < KRIT['ks']
# 主判据（v2.3 修正）：选择仅在源域（Cell 2020）内闭环；目标域标签不参与
res['iron'] = (res['in_target'] & res['pass_stability'] & res['pass_ks'])
# 审计列：v2.2 及之前协议（曾将 Jacobs 标签稳定性并入选择，存在目标域标签泄漏）
res['iron_legacy_leaky'] = (res['in_target'] & res['pass_stability'] &
                            res['pass_stability_jac'] & res['pass_ks'])
IRON = [f for f in res.index if res.loc[f, 'iron']]
IRON_LEGACY = [f for f in res.index if res.loc[f, 'iron_legacy_leaky']]
print(f'\nIron-Core set (source-closed, n={len(IRON)}):', IRON)
print(f'legacy (target-label leakage, n={len(IRON_LEGACY)}):', IRON_LEGACY)
print(res[['pi_uni_cell', 'pi_multi_cell', 'pi_uni_jac', 'pi_multi_jac',
           'ks', 'in_target', 'dir_conc', 'pass_stability_jac', 'iron']]
      .round(3).to_string())
if not IRON:
    raise SystemExit('Iron-Core 为空：源域闭环筛选未选出任何特征，请检查阈值。')
res.to_csv(OUT / 'stability_results.csv')
json.dump({'iron_set': IRON, 'iron_set_legacy_leaky': IRON_LEGACY,
           'criteria': KRIT, 'B': B,
           'selection_protocol': ('source-closed (Cell 2020 only); '
                                  'Jacobs labels used for post-hoc evaluation only')},
          open(OUT / 'iron_set.json', 'w'), indent=2)

# ---------- Iron-Core 零样本复测 ----------
Xi_log = Xc_log[:, [SIG16.index(f) for f in IRON]]
Xji_log = np.log1p(met.loc[keep_j, [FEATURE_MAP[f] for f in IRON]]
                   .values.astype(float))


def frozen_ridge(features_idx, tag):
    X, y, g = Xc_log[:, features_idx], yc, gc
    grid = [0.1, 1, 10, 100, 1000]
    best = None
    for a in grid:
        oof = np.zeros(len(y))
        for tri, tei in GroupKFold(5).split(X, y, g):
            oof[tei] = Ridge(alpha=a).fit(X[tri], y[tri]).predict(X[tei])
        r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if best is None or r2 > best[1]:
            best = (a, r2)
    a = best[0]
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=a).fit(sc.transform(X), y)
    Zj = (Xji_log - sc.mean_) / sc.scale_
    score = Zj @ m.coef_ + m.intercept_
    return score, best, tag


y_ord = (meta.loc[keep_j, 'BH'] == 'Diarrhea').astype(int).values
score_iron, oof_iron, _ = frozen_ridge([SIG16.index(f) for f in IRON], 'iron')
auc_iron = roc_auc_score(y_ord, score_iron)
rho_iron = spearmanr(meta.loc[keep_j, 'BH'].map(ordinal), score_iron).statistic
print(f'\nIron zero-shot: in-domain OOF R2={oof_iron[1]:.3f} | '
      f'Jacobs AUC(D vs C)={auc_iron:.3f}, rho={rho_iron:+.3f}')

# 复算 R11 / F16 对照（与 zero_shot_validation.py 同协议, alpha=1/10 已固化）
def repro_score(features):
    Xi = Xc_log[:, [SIG16.index(f) for f in features]]
    a = 10 if len(features) == 16 else 1
    sc = StandardScaler().fit(Xi)
    m = Ridge(alpha=a).fit(sc.transform(Xi), yc)
    Z = np.zeros((len(yj), len(features)))
    for j, f in enumerate(features):
        if f in SHARED:
            k = SHARED.index(f)
            Z[:, j] = (Xj_log[:, k] - sc.mean_[j]) / sc.scale_[j]
    return Z @ m.coef_ + m.intercept_


auc_r11 = roc_auc_score(y_ord, repro_score(SHARED))
auc_f16 = roc_auc_score(y_ord, repro_score(SIG16))
print(f'controls: F16 AUC={auc_f16:.3f}, R11 AUC={auc_r11:.3f}')

# ---------- 图（fig7） ----------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
ax = axes[0]
x = res['pi_uni_cell'] + RNG.uniform(-0.015, 0.015, len(res))
ypt = res['pi_uni_jac'].fillna(-0.05) + RNG.uniform(-0.015, 0.015, len(res))
colors = ['#d62728' if i else '#aaaaaa' for i in res['in_target']]
ax.scatter(x, ypt, c=colors, s=42, alpha=0.85, edgecolors='white', linewidths=0.6)
for f, xi, yi in zip(res.index, x, ypt):
    ax.annotate(f, (xi, yi), textcoords='offset points', xytext=(5, 3), fontsize=7)
ax.axhline(KRIT['pi_uni'], ls='--', c='gray', lw=0.8)
ax.axvline(KRIT['pi_uni'], ls='--', c='gray', lw=0.8)
ax.set_xlabel('π_uni (Cell 2020 vs Bristol)')
ax.set_ylabel('π_uni (Jacobs 2023 vs BH; diagnostic only)')
ax.set_title('A  Bootstrap stability (B=300)')

ax = axes[1]
s = res.sort_values('ks')
ax.barh(s.index, s['ks'].fillna(0), color=['#d62728' if v >= KRIT['ks'] else '#55a868'
                                           for v in s['ks'].fillna(0)])
ax.axvline(KRIT['ks'], ls='--', c='gray', lw=0.9)
ax.set_xlabel('KS statistic (platform shift)')
ax.set_title('B  Platform sensitivity')

ax = axes[2]
vals = [auc_f16, auc_r11, auc_iron]
names = ['F16\n(+SCFA imputed)', 'R11\n(shared)', f'Iron-Core\n(n={len(IRON)})']
bars = ax.bar(names, vals, color=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.85)
for r, v in zip(bars, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.008, f'{v:.3f}',
            ha='center', fontsize=10)
ax.axhline(0.5, ls='--', c='gray', lw=0.8)
ax.set_ylim(0.45, 0.72)
ax.set_ylabel('Zero-shot AUC (D vs C, Jacobs)')
ax.set_title('C  Zero-shot performance')
fig.suptitle('Stability selection (source-closed, Cell 2020) & Iron-Core feature set',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / 'fig7_stability_selection.png', dpi=300)
plt.close(fig)
print(f'\nsaved: stability_results.csv, iron_set.json -> {OUT} | fig7 -> {FIG}')
