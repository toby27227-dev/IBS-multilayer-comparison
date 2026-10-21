"""零样本跨队列域泛化验证：Cell 2020 (NMR→Bristol) → Jacobs 2023 (BH 亚型)。

严格协议（防泄漏）：
1) 特征对齐：预注册名匹配映射；未匹配特征在 z 空间按训练均值插补（等价 z=0）。
2) 零样本：模型仅在 Cell 2020 上训练（分组 CV 选 α），冻结权重与标准化统计，
   直接应用于 Jacobs 2023 —— 目标队列不做任何训练/调参/特征选择。
3) 失败模式分析：域偏移诊断（KS / Cohen's d）、分批测、混淆因素检查、
   单变量方向性复现（11 个共享特征）。

目标队列无 Bristol；用 BH 亚型的肠动力轴作为代理终点：
  预注册假设：预测 Bristol 分数 Diarrhea > Mixed > Constipation（方向单调递增）。
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ks_2samp
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score, roc_curve
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
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sys.path.append(str(CE))
from label_utils import place_labels

# ---------- 1. 特征对齐（预注册） ----------
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
MISSING = [f for f in SIG16 if f not in FEATURE_MAP]
print('SIG16:', SIG16)
print('SHARED (11):', SHARED)
print('MISSING in target (5, 全部 SCFA):', MISSING)

# ---------- 2. 在 Cell 2020 上训练并冻结 ----------
tr = nmr.join(sm[['bristol', 'subject']]).dropna()
print(f'\ntrain domain: n={len(tr)}, subjects={tr["subject"].nunique()}')


def fit_frozen(features, tag):
    X = np.log1p(tr[features].values.astype(float))
    y = tr['bristol'].values
    g = tr['subject'].values
    grid = [0.1, 1, 10, 100, 1000, 10000]
    best, rows = None, []
    for a in grid:
        oof = np.zeros(len(y))
        for tri, tei in GroupKFold(5).split(X, y, g):
            m = Ridge(alpha=a).fit(X[tri], y[tri])
            oof[tei] = m.predict(X[tei])
        r2 = r2_score(y, oof)
        rho = spearmanr(y, oof).statistic
        rows.append((a, r2, rho))
        if best is None or r2 > best[1]:
            best = (a, r2, rho, oof)
    alpha, r2, rho, oof = best
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    frozen = {
        'tag': tag, 'features': features, 'alpha': alpha,
        'mean': scaler.mean_.tolist(), 'sd': scaler.scale_.tolist(),
        'coef': model.coef_.tolist(), 'intercept': float(model.intercept_),
        'oof_r2': float(r2), 'oof_rho': float(rho),
    }
    print(f'[{tag}] features={len(features)} alpha={alpha} '
          f'| in-domain OOF: R2={r2:.3f} rho={rho:.3f}')
    for a, r, s in rows:
        print(f'    alpha={a:<7g} OOF R2={r:+.3f} rho={s:+.3f}')
    return frozen


frozen_F = fit_frozen(SIG16, 'full16(z-imputed)')
frozen_R = fit_frozen(SHARED, 'shared11')

with open(OUT / 'frozen_models.pkl', 'wb') as f:
    pickle.dump({'full16': frozen_F, 'shared11': frozen_R}, f)
json.dump({'full16': frozen_F, 'shared11': frozen_R},
          open(OUT / 'frozen_model_metadata.json', 'w'), indent=2)

# ---------- 3. 目标域（Jacobs 2023）零样本应用 ----------
met = pd.read_csv(NW / 'jacobs2023_metabolites.csv', index_col=0)
meta = pd.read_csv(NW / 'jacobs2023_metadata.csv').set_index('Patient')
idx = met.index.intersection(meta.index)
met, meta = met.loc[idx], meta.loc[idx]
print(f'\ntarget domain: n={len(idx)}')

Xj_log = np.log1p(met[[FEATURE_MAP[f] for f in SHARED]].values.astype(float))


def score_target(frozen, Xlog_shared):
    feats = frozen['features']
    Z = np.zeros((Xlog_shared.shape[0], len(feats)))
    for j, f in enumerate(feats):
        if f in SHARED:
            k = SHARED.index(f)
            Z[:, j] = (Xlog_shared[:, k] - frozen['mean'][j]) / frozen['sd'][j]
        else:
            Z[:, j] = 0.0
    return Z @ np.array(frozen['coef']) + frozen['intercept']


pred = pd.DataFrame({'score_F16': score_target(frozen_F, Xj_log),
                     'score_R11': score_target(frozen_R, Xj_log)}, index=idx)
pred = pred.join(meta[['Group', 'BH', 'Batch_metabolomics', 'Age', 'Sex', 'BMI']])

ordinal = {'Constipation': 0, 'Mixed': 1, 'Diarrhea': 2}
isub = pred[pred['BH'].isin(ordinal)].copy()
isub['ord'] = isub['BH'].map(ordinal)

metrics = {}
for col in ['score_F16', 'score_R11']:
    d = isub[isub['BH'] == 'Diarrhea'][col]
    c = isub[isub['BH'] == 'Constipation'][col]
    y = (isub['BH'] == 'Diarrhea').astype(int)
    auc = roc_auc_score(y, isub[col])
    u, p = mannwhitneyu(d, c, alternative='greater')
    rho, prho = spearmanr(isub['ord'], isub[col])
    h, pk = kruskal(*[isub[isub['BH'] == g][col] for g in ordinal])
    acc = balanced_accuracy_score(y, (isub[col] >= 0).astype(int))
    metrics[col] = {'auc_D_vs_C': auc, 'mwu_p': p, 'rho_vs_BH_ordinal': rho,
                    'rho_p': prho, 'kruskal_p': pk, 'bacc_thresh0': acc}
    print(f'\n== {col} ==')
    print(f'  AUC(D vs C)      = {auc:.3f}   (MWU p={p:.2e})')
    print(f'  rho vs BH ordinal= {rho:+.3f}  (p={prho:.2e})')
    print(f'  Kruskal (C/M/D)  = p={pk:.2e}')
    print(f'  bACC(阈值0)      = {acc:.3f}')

# 对照与 IBS 整体
for col in ['score_F16', 'score_R11']:
    ctrl = pred[pred['Group'] == 'Control'][col]
    ibs = pred[pred['Group'] == 'IBS'][col]
    u, p = mannwhitneyu(ibs, ctrl, alternative='two-sided')
    print(f'\n{col}: IBS vs Control (双侧) p={p:.3f} | '
          f'IBS mean={ibs.mean():+.2f}, Control mean={ctrl.mean():+.2f}')

# 分批 A未C
batch_rows = []
for b, g in isub.groupby('Batch_metabolomics'):
    if g['BH'].isin(['Diarrhea']).sum() >= 10 and g['BH'].isin(['Constipation']).sum() >= 10:
        y = (g['BH'] == 'Diarrhea').astype(int)
        batch_rows.append({'batch': b, 'n': len(g),
                           'auc_F16': roc_auc_score(y, g['score_F16']),
                           'auc_R11': roc_auc_score(y, g['score_R11'])})
        print(f'batch {b}: n={len(g)} AUC_F16={batch_rows[-1]["auc_F16"]:.3f} '
              f'AUC_R11={batch_rows[-1]["auc_R11"]:.3f}')

# 混淆因素
conf_rows = []
for col in ['score_F16', 'score_R11']:
    r_age, p_age = spearmanr(isub['Age'], isub[col])
    r_bmi, p_bmi = spearmanr(isub['BMI'], isub[col])
    mw = mannwhitneyu(isub[isub['Sex'] == 'Female'][col],
                      isub[isub['Sex'] == 'Male'][col])
    conf_rows.append({'score': col, 'rho_age': r_age, 'p_age': p_age,
                      'rho_bmi': r_bmi, 'p_bmi': p_bmi, 'p_sex': mw.pvalue})
    print(f'{col}: rho_age={r_age:+.3f}(p={p_age:.2f}) '
          f'rho_bmi={r_bmi:+.3f}(p={p_bmi:.2f}) sex_p={mw.pvalue:.3f}')

# ---------- 4. 域偏移诊断 ----------
q = b16.set_index('taxon')
shift_rows = []
for f in SHARED:
    xc = np.log1p(tr[f].values.astype(float))
    xj = np.log1p(met[FEATURE_MAP[f]].values.astype(float))
    zc = (xc - xc.mean()) / xc.std()
    zj = (xj - xj.mean()) / xj.std()
    ks, pks = ks_2samp(zc, zj)
    shift_rows.append({'feature': f, 'ks_stat': ks, 'ks_p': pks,
                       'median_ratio(log)': np.median(xj) - np.median(xc)})
shift = pd.DataFrame(shift_rows).sort_values('ks_stat', ascending=False)
print('\n域偏移 (KS, 各域自标准化后形状差异):')
print(shift.round(3).to_string(index=False))

# ---------- 5. 单变量方向性复现 ----------
repl_rows = []
met_sub = met.loc[isub.index]
for f in SHARED:
    xj = np.log1p(met_sub[FEATURE_MAP[f]].values.astype(float))
    rho_j, p_j = spearmanr(isub['ord'], xj)
    repl_rows.append({'feature': f, 'rho_cell2020': q.loc[f, 'spearman'],
                      'rho_jacobs': rho_j, 'p_jacobs': p_j,
                      'concordant': np.sign(q.loc[f, 'spearman']) == np.sign(rho_j)})
repl = pd.DataFrame(repl_rows)
n_conc = int(repl['concordant'].sum())
print(f'\n方向性复现: {n_conc}/{len(repl)} 共享特征与 Cell 2020 方向一致')
print(repl.round(3).to_string(index=False))
from scipy.stats import combine_pvalues
fisher = combine_pvalues(repl['p_jacobs'].clip(lower=1e-300))
print(f'Fisher 合并 p = {fisher.pvalue:.2e}')

# ---------- 6. 图 ----------
COLORS = {'Control': '#7f7f7f', 'Constipation': '#1f77b4',
          'Mixed': '#9467bd', 'Diarrhea': '#d62728'}
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))

ax = axes[0, 0]
pred['score_R11_z'] = ((pred['score_R11'] - pred['score_R11'].mean())
                       / pred['score_R11'].std())
grp_bh = {'Control': 'Normal', 'Constipation': 'Constipation',
          'Mixed': 'Mixed', 'Diarrhea': 'Diarrhea'}
order = ['Control', 'Constipation', 'Mixed', 'Diarrhea']
data = [pred[pred['BH'] == grp_bh[g]]['score_R11_z'] for g in order]
bp = ax.boxplot(data, tick_labels=[f'{g}\n(n={len(d)})' for g, d in zip(order, data)],
                patch_artist=True, showfliers=False)
for patch, g in zip(bp['boxes'], order):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.55)
for i, d in enumerate(data, 1):
    ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(d)), d,
               s=9, alpha=0.55, c=COLORS[order[i - 1]], edgecolors='none')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_ylabel('Zero-shot predicted Bristol\n(within-target z, label-blind)')
ax.set_title('A  Jacobs 2023: R11 score by bowel-habit group')

ax = axes[0, 1]
y = (isub['BH'] == 'Diarrhea').astype(int)
for col, c in [('score_F16', '#d62728'), ('score_R11', '#ff7f0e')]:
    fpr, tpr, _ = roc_curve(y, isub[col])
    ax.plot(fpr, tpr, color=c, lw=1.8,
            label=f'{col} AUC={metrics[col]["auc_D_vs_C"]:.3f}')
ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=0.8)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('B  Zero-shot ROC: Diarrhea vs Constipation')
ax.legend(fontsize=8, loc='lower right', frameon=False)

ax = axes[1, 0]
ax.scatter(repl['rho_cell2020'], repl['rho_jacobs'], s=46, c='#4c72b0',
           edgecolors='white', linewidths=0.7)
lim = [min(repl.min()['rho_cell2020'], repl.min()['rho_jacobs']) - 0.07,
       max(repl.max()['rho_cell2020'], repl.max()['rho_jacobs']) + 0.07]
ax.plot(lim, lim, ls='--', color='gray', lw=0.8)
place_labels(ax, [(r['feature'], r['rho_cell2020'], r['rho_jacobs'])
                  for _, r in repl.iterrows()], fontsize=7.2, marker_s=46,
             verbose=True)
ax.set_xlabel('Spearman ρ (Cell 2020, metabolite vs Bristol)')
ax.set_ylabel('Spearman ρ (Jacobs 2023, vs BH ordinal)')
ax.set_title(f'C  Directional replication: {n_conc}/{len(repl)} concordant')

ax = axes[1, 1]
s = shift.sort_values('ks_stat')
ax.barh(s['feature'], s['ks_stat'], color='#55a868', alpha=0.85)
ax.set_xlabel('KS statistic (standardized distributions)')
ax.set_title('D  Domain shift by feature (Cell 2020 vs Jacobs 2023)')
fig.suptitle('Zero-shot cross-cohort validation: Cell 2020 (NMR→Bristol) → Jacobs 2023',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG / 'fig6_zeroshot_jacobs.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 7. 落盘 ----------
pred.to_csv(OUT / 'predictions_jacobs.csv')
shift.to_csv(OUT / 'domain_shift.csv', index=False)
repl.to_csv(OUT / 'directional_replication.csv', index=False)
json.dump({
    'protocol': {
        'shared_features': SHARED, 'missing_features': MISSING,
        'n_train': int(len(tr)), 'n_target': int(len(idx)),
        'pre_registered_hypothesis': 'Diarrhea > Mixed > Constipation in predicted Bristol',
    },
    'metrics': metrics,
    'batch_auc': batch_rows,
    'confounders': conf_rows,
    'directional_replication': {
        'concordant': n_conc, 'total': len(repl), 'fisher_p': float(fisher.pvalue)},
    'frozen_models': {'full16': {k: frozen_F[k] for k in ('alpha', 'oof_r2', 'oof_rho')},
                      'shared11': {k: frozen_R[k] for k in ('alpha', 'oof_r2', 'oof_rho')}},
}, open(OUT / 'zero_shot_metrics.json', 'w'), indent=2)
print('\nsaved to', OUT)
