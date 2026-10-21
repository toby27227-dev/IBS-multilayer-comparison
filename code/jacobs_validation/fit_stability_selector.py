"""源域闭环稳定性特征选择器（label-blind transductive feature selection）。

v2.3 工程硬隔离协议：
- 输入：Cell 2020（NMR 特征 + Bristol 标签 + 受试者 ID）；
  Jacobs 2023 的**无标签特征矩阵 X**（仅用于无监督平台偏移诊断 KS）。
- 本脚本严禁读取 Jacobs 的临床标签（jacobs2023_metadata*.csv / BH 列）。
- 三门筛选：
  ① π_uni_cell ≥ 0.8：Cell 分组 bootstrap（B=300，按受试者 50% 无放回），
     单变量 Spearman p<0.05 且方向与全样本一致的频率；
  ② KS(X_cell, X_jacobs) < 0.15（自标准化 log1p，无监督，label-blind）；
  ③ 两组共同可测（in_target，按特征名交集判定，无标签）。
- 冻结：在 Cell 内 GroupKFold(5) 选 α，拟合 StandardScaler + Ridge；
  缩放参数与系数随特征集一并导出，供 held-out 零样本评估直接使用。
输出：frozen_selector.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
NW = ROOT / 'IBS/jacobs2023'
OUT = ROOT / 'IBS/jacobs_validation'
B = 300
SEED = 42
RNG = np.random.default_rng(SEED)
KRIT = dict(pi_uni=0.8, ks=0.15)

FEATURE_MAP = {
    'Lysine': 'lysine', 'Tyrosine': 'tyrosine', 'Leucine': 'leucine',
    'Valine': 'valine', 'Isoleucine': 'isoleucine', 'Alanine': 'alanine',
    'Glucose': 'glucose', 'Succinate': 'succinate', 'Uracil': 'uracil',
    'Hypoxanthine': 'hypoxanthine', 'Lactate': 'lactate',
}

# ---------- 源域（Cell 2020）：特征 + 标签 ----------
nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
b16 = pd.read_csv(CE / 'results_bristol_nmr' / 'biomarkers.csv')
SIG16 = b16[b16['q'] < 0.05].sort_values('spearman', ascending=False)['taxon'].tolist()
SHARED = [f for f in SIG16 if f in FEATURE_MAP]

tr = nmr.join(sm[['bristol', 'subject']]).dropna()
Xc_log = np.log1p(tr[SIG16].values.astype(float))
yc = tr['bristol'].values
gc = tr['subject'].values
ref_sign_cell = {f: np.sign(spearmanr(tr[f], tr['bristol']).statistic) for f in SIG16}
print(f'Cell 2020: n={len(tr)} | subjects={len(set(gc))} | SIG16={len(SIG16)} | shared={len(SHARED)}')

# ---------- 目标域 X（无标签；严禁读取 metadata） ----------
met = pd.read_csv(NW / 'jacobs2023_metabolites.csv', index_col=0)
missing_cols = [FEATURE_MAP[f] for f in SHARED if FEATURE_MAP[f] not in met.columns]
assert not missing_cols, f'Jacobs X 缺少列: {missing_cols}'
print(f'Jacobs 2023 X: n={len(met)} | 可测共享特征={len(SHARED)}')

# ---------- π_uni_cell（源域 bootstrap，B=300） ----------
alpha_cell = LassoCV(cv=5, random_state=0, max_iter=20000).fit(
    StandardScaler().fit_transform(Xc_log), yc).alpha_
subjects = tr['subject'].unique()
feat_rows = {f: {'pi_uni_cell': 0, 'pi_multi_cell': 0} for f in SIG16}
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
for f, d in feat_rows.items():
    d['pi_uni_cell'] /= B
    d['pi_multi_cell'] /= B

# ---------- 平台偏移 KS（无监督，仅用 X） ----------
for f in SIG16:
    if f in FEATURE_MAP:
        xc = np.log1p(tr[f].values.astype(float))
        xj = np.log1p(met[FEATURE_MAP[f]].values.astype(float))
        zc = (xc - xc.mean()) / xc.std()
        zj = (xj - xj.mean()) / xj.std()
        feat_rows[f]['ks'] = float(ks_2samp(zc, zj).statistic)
    else:
        feat_rows[f]['ks'] = np.nan
    feat_rows[f]['in_target'] = f in FEATURE_MAP

# ---------- 三门判定与冻结模型 ----------
SELECTED = [f for f in SIG16 if feat_rows[f]['in_target']
            and feat_rows[f]['pi_uni_cell'] >= KRIT['pi_uni']
            and feat_rows[f]['ks'] < KRIT['ks']]
print(f'\nIron-Core (source-closed, n={len(SELECTED)}):', SELECTED)

Xi = Xc_log[:, [SIG16.index(f) for f in SELECTED]]
grid = [0.1, 1, 10, 100, 1000]
best = None
for a in grid:
    oof = np.zeros(len(yc))
    for tri, tei in GroupKFold(5).split(Xi, yc, gc):
        oof[tei] = Ridge(alpha=a).fit(Xi[tri], yc[tri]).predict(Xi[tei])
    r2 = 1 - ((yc - oof) ** 2).sum() / ((yc - yc.mean()) ** 2).sum()
    rho = spearmanr(yc, oof).statistic
    if best is None or r2 > best[1]:
        best = (a, r2, rho)
alpha, oof_r2, oof_rho = best
scaler = StandardScaler().fit(Xi)
model = Ridge(alpha=alpha).fit(scaler.transform(Xi), yc)
print(f'frozen Ridge: alpha={alpha} | in-domain OOF R2={oof_r2:.4f} rho={oof_rho:.4f}')

for f in SELECTED:
    d = feat_rows[f]
    print(f"  {f:14s} pi_uni={d['pi_uni_cell']:.3f} pi_multi={d['pi_multi_cell']:.3f} KS={d['ks']:.3f}")

payload = {
    'version': 'v2.3-source-closed',
    'protocol': ('label-blind transductive feature selection: '
                 'pi_uni_cell (Cell-only grouped bootstrap, B=300), '
                 'KS(X_cell, X_jacobs) < 0.15 unsupervised, in_target by name'),
    'seed': SEED,
    'n_cell': int(len(tr)), 'n_subjects_cell': int(len(set(gc))),
    'n_jacobs_x': int(len(met)),
    'sig16': SIG16, 'shared': SHARED, 'feature_map': FEATURE_MAP,
    'criteria': KRIT,
    'lasso_alpha_cell': float(alpha_cell),
    'features': feat_rows,
    'selected_features': SELECTED,
    'alpha': float(alpha),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'ridge_coef': model.coef_.tolist(),
    'ridge_intercept': float(model.intercept_),
    'in_domain_oof_r2': float(oof_r2),
    'in_domain_oof_rho': float(oof_rho),
}
json.dump(payload, open(OUT / 'frozen_selector.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print(f'\nsaved: frozen_selector.json -> {OUT}')
