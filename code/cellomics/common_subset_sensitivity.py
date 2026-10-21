"""八层共同完整子集敏感性分析（v2.6 复审 T-A）。

背景：§3.3 八层比较的样本量各层不同（Trp 259 → KO 437，建模口径），
"NMR 最强"可能部分由样本构成差异驱动。本脚本取八层**共同完整样本**
（交集 ∩ Bristol 可用 / 标签可用），以同一公平比较协议重跑回归与分类，
检验排序是否保持。

协议（与 §2.3/§2.4 一致）：
- 属/种：prev=0.10，CLR；KO：prev=0.30，CLR；KEGG：prev=0.50，CLR；
  NMR/BA/Trp：prev=0.10，log1p；combined：预变换拼接（none，prev=0）。
- 回归：外层 GroupKFold(5) + 内层 GroupKFold(3) 选 Ridge alpha
  （fit_predict_ridge_fast，与 GridSearchCV 数值等价，T0.2 已验证）。
- 分类：StratifiedGroupKFold(5/3, shuffle, seed=42) + L2 logistic
  （C∈{0.01,0.1,1,10}，class_weight=balanced）。
输出：common_subset_sensitivity.csv、IBS/figs/fig14_common_subset.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'IBS' / 'pipeline'))
sys.path.insert(0, str(ROOT / 'IBS' / 'cellomics'))
from supervised_biomarkers import build_features, load_table
from permutation_utils import oof_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})

CE = ROOT / 'IBS' / 'cellomics'
GM = ROOT / 'IBS' / 'gmrepo'
FIG = ROOT / 'IBS' / 'figs'

# 层名, 文件, prevalence, transform, summary 行名
LAYERS = [
    ('NMR', CE / 'nmr.csv', 0.10, 'log', 'NMR(19)'),
    ('Genus', GM / 'abundance_genus.csv', 0.10, 'clr', '属(144)'),
    ('Species', GM / 'abundance_species.csv', 0.10, 'clr', '种(328)'),
    ('Bile acids', CE / 'bile_acids.csv', 0.10, 'log', '胆汁酸(12)'),
    ('Tryptophan', CE / 'tryptophan.csv', 0.10, 'log', '色氨酸(12)'),
    ('KO', CE / 'ko_stool.csv', 0.30, 'clr', 'KO(~5.3k)'),
    ('KEGG', CE / 'kegg_pathways_ko.csv', 0.50, 'clr', 'KEGG通路(327)'),
    ('NMR+Genus', CE / 'combined_genus_nmr.csv', 0.0, 'none', 'NMR+属(163)'),
]

meta = load_table(CE / 'sample_map.csv')
tabs = {name: load_table(path) for name, path, *_ in LAYERS}
inter = set.intersection(*[set(t.index) for t in tabs.values()])
idx = meta.index[meta.index.isin(inter)]
print(f'八层共同完整样本: {len(idx)} | 受试者: {meta.loc[idx, "subject"].nunique()}', flush=True)

reg_mask = meta.loc[idx, 'bristol'].notna().values
cls_mask = meta.loc[idx, 'Cohort_label'].notna().values
print(f'回归可用: {int(reg_mask.sum())} 样本 / '
      f'{meta.loc[idx][reg_mask]["subject"].nunique()} 受试者 | '
      f'分类可用: {int(cls_mask.sum())} 样本', flush=True)

y_reg = pd.to_numeric(meta.loc[idx, 'bristol'], errors='coerce').values
raw_cls = meta.loc[idx, 'Cohort_label'].astype(str).values
le = LabelEncoder()
y_cls = le.fit_transform(raw_cls)
groups = meta.loc[idx, 'subject'].astype(str).values
print(f'分类类别: {list(le.classes_)}', flush=True)

reg_full = pd.read_csv(CE / 'summary_bristol.csv')
cls_full = pd.read_csv(CE / 'summary_cohort.csv')

rows = []
for name, path, prev, transform, key in LAYERS:
    abund = tabs[name]
    print(f'\n=== {name} ===', flush=True)

    # ---------- 回归（共同子集） ----------
    ab = abund.loc[idx][reg_mask]
    X, feats = build_features(ab, prev, transform)
    m = oof_metrics(X, y_reg[reg_mask], groups[reg_mask])
    ref = reg_full[reg_full['task'] == f'Bristol: {key}'].iloc[0]
    rows.append({'layer': name, 'task': 'regression', 'scope': 'common_subset',
                 'n_samples': int(reg_mask.sum()),
                 'n_subjects': int(meta.loc[idx][reg_mask]['subject'].nunique()),
                 'n_features': X.shape[1], 'r2': round(m['r2'], 4),
                 'rho': round(m['rho'], 4),
                 'r2_full': round(float(ref['R2_OOF']), 4),
                 'rho_full': round(float(ref['rho_OOF']), 4),
                 'delta_r2': round(m['r2'] - float(ref['R2_OOF']), 4)})
    print(f'  回归: features={X.shape[1]} R2={m["r2"]:.4f} '
          f'(full {ref["R2_OOF"]:.3f}) rho={m["rho"]:.3f}', flush=True)

    # ---------- 分类（共同子集） ----------
    ab = abund.loc[idx][cls_mask]
    Xc, featsc = build_features(ab, prev, transform)
    yc, gc = y_cls[cls_mask], groups[cls_mask]
    n_cls = len(abund.index.intersection(meta.index))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    pipe = Pipeline([('clf', LogisticRegression(
        max_iter=5000, class_weight='balanced', solver='lbfgs'))])
    grid = {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    pred = np.empty_like(yc)
    fold_ba = []
    for tr, te in outer.split(Xc, yc, groups=gc):
        gs = GridSearchCV(pipe, grid, cv=inner, scoring='balanced_accuracy', n_jobs=1)
        gs.fit(Xc[tr], yc[tr], groups=gc[tr])
        p = gs.predict(Xc[te])
        pred[te] = p
        fold_ba.append(balanced_accuracy_score(yc[te], p))
    ba_mean, ba_sd = float(np.mean(fold_ba)), float(np.std(fold_ba))
    ba_pooled = float(balanced_accuracy_score(yc, pred))
    cref = cls_full[cls_full['task'] == f'Cohort: {key}'].iloc[0]
    rows.append({'layer': name, 'task': 'classification', 'scope': 'common_subset',
                 'n_samples': int(cls_mask.sum()), 'n_subjects': len(np.unique(gc)),
                 'n_features': Xc.shape[1],
                 'bal_acc_mean': round(ba_mean, 4), 'bal_acc_sd': round(ba_sd, 4),
                 'bal_acc_pooled': round(ba_pooled, 4),
                 'bal_acc_mean_full': round(float(cref['balanced_acc']), 4),
                 'bal_acc_sd_full': round(float(cref['std']), 4),
                 'n_full_cls': int(n_cls),
                 'delta_ba': round(ba_mean - float(cref['balanced_acc']), 4)})
    print(f'  分类: features={Xc.shape[1]} BA={ba_mean:.3f}±{ba_sd:.3f} '
          f'(pooled {ba_pooled:.3f}; full {cref["balanced_acc"]:.3f})', flush=True)

res = pd.DataFrame(rows)
res.to_csv(CE / 'common_subset_sensitivity.csv', index=False)

reg = res[res['task'] == 'regression'].sort_values('r2', ascending=False)
cls = res[res['task'] == 'classification'].sort_values('bal_acc_mean', ascending=False)
print('\n===== 共同子集回归排序 =====')
print(reg[['layer', 'n_features', 'r2', 'rho', 'r2_full', 'delta_r2']].to_string(index=False))
print('\n===== 共同子集分类排序 =====')
print(cls[['layer', 'n_features', 'bal_acc_mean', 'bal_acc_sd',
           'bal_acc_mean_full', 'delta_ba']].to_string(index=False))

print('\nNMR 回归首位保持:', bool(reg.iloc[0]['layer'] == 'NMR'),
      '| 末位:', reg.iloc[-1]['layer'], end=' | ')
print('分类首位:', cls.iloc[0]['layer'])

# ---------- 图14 ----------
order = list(reg['layer'])
reg = reg.set_index('layer').loc[order]
cls = cls.set_index('layer').loc[order]
ypos = np.arange(len(order))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
ax = axes[0]
for i, name in enumerate(order):
    r = reg.loc[name]
    ax.plot([r['r2_full'], r['r2']], [ypos[i], ypos[i]], color='#bbbbbb', lw=1.4)
    ax.scatter(r['r2_full'], ypos[i], facecolors='white', edgecolors='#777777',
               s=34, zorder=3, label='Full sample (per-layer)' if i == 0 else None)
    ax.scatter(r['r2'], ypos[i], color='#4c72b0', s=40, zorder=4,
               label='Common complete subset (n=255)' if i == 0 else None)
    ax.annotate(f"{r['r2']:.3f}", (r['r2'], ypos[i]), textcoords='offset points',
                xytext=(8, 6), fontsize=7.5, color='#4c72b0')
ax.set_yticks(ypos, order)
ax.axvline(0, color='#999999', lw=.8, ls=':')
ax.set_xlabel('Bristol OOF R²')
ax.set_title('A  Regression (Bristol)')
ax.margins(x=0.10)

ax = axes[1]
for i, name in enumerate(order):
    r = cls.loc[name]
    ax.plot([r['bal_acc_mean_full'], r['bal_acc_mean']], [ypos[i], ypos[i]],
            color='#bbbbbb', lw=1.4)
    ax.scatter(r['bal_acc_mean_full'], ypos[i], facecolors='white',
               edgecolors='#777777', s=34, zorder=3)
    ax.scatter(r['bal_acc_mean'], ypos[i], color='#55a868', s=40, zorder=4)
    ax.annotate(f"{r['bal_acc_mean']:.3f}", (r['bal_acc_mean'], ypos[i]),
                textcoords='offset points', xytext=(8, 6), fontsize=7.5,
                color='#2e7d4f')
ax.axvline(1 / 3, color='#999999', lw=.8, ls=':')
ax.set_yticks(ypos, order)
ax.set_xlabel('Balanced accuracy (3-class)')
ax.set_title('B  Classification (IBS-D/C/Healthy)')
ax.margins(x=0.10)

handles = [plt.Line2D([], [], marker='o', ls='', markerfacecolor='white',
                      markeredgecolor='#777777', label='Full sample (per-layer)'),
           plt.Line2D([], [], marker='o', ls='', color='#4c72b0',
                      label='Common subset, regression'),
           plt.Line2D([], [], marker='o', ls='', color='#55a868',
                      label='Common subset, classification')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False)
fig.suptitle('Common complete-subset sensitivity: 268 samples / 75 subjects '
             '(regression n=255 / 67 subjects)', fontsize=10)
fig.tight_layout(rect=[0, 0.08, 1, 0.93])
fig.savefig(FIG / 'fig14_common_subset.png', dpi=300)
plt.close(fig)
print(f'\nsaved: common_subset_sensitivity.csv | fig14 -> {FIG}')
