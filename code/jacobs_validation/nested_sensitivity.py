"""Fully Nested 预筛选偏差敏感性（v2.3 规则 5）。

主分析的 16 项标志物来自全数据 BH q<0.05 预筛选；本脚本将预筛选
"下沉"到分组 bootstrap 内层：每个重抽样（B=300，按受试者 50% 无放回）
内对全部 19 项 NMR 特征重跑单变量 Spearman + BH-FDR（q<0.05 且方向与
全数据一致）→ 统计筛选存活频率；再与固定闸门（π_uni_cell ≥ 0.8、
KS < 0.15、in_target，读自 frozen_selector.json）组合，输出逐特征的
nested 选择频率与 nested 核心集。
输出：nested_sensitivity.csv
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
OUT = ROOT / 'IBS/jacobs_validation'
B = 300
RNG = np.random.default_rng(42)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
tr = nmr.join(sm[['bristol', 'subject']]).dropna()
FEATS = list(nmr.columns)
X = np.log1p(tr[FEATS].values.astype(float))
y = tr['bristol'].values
g = tr['subject'].values
full_sign = {f: np.sign(spearmanr(tr[f], tr['bristol']).statistic) for f in FEATS}
print(f'Cell 2020: n={len(tr)} | subjects={len(set(g))} | features={len(FEATS)}')

subjects = tr['subject'].unique()
surv = {f: 0 for f in FEATS}
for b in range(B):
    pick = RNG.choice(subjects, size=int(0.5 * len(subjects)), replace=False)
    m = np.isin(g, pick)
    Xb, yb = X[m], y[m]
    stats = [spearmanr(Xb[:, j], yb) for j in range(Xb.shape[1])]
    rhos = np.array([s.statistic for s in stats])
    ps = np.array([s.pvalue for s in stats])
    qs = bh_fdr(ps)
    for j, f in enumerate(FEATS):
        if qs[j] < 0.05 and np.sign(rhos[j]) == full_sign[f]:
            surv[f] += 1

sel = json.load(open(OUT / 'frozen_selector.json', encoding='utf-8'))
feat = sel['features']
rows = []
for f in FEATS:
    sf = surv[f] / B
    pi = feat.get(f, {}).get('pi_uni_cell', np.nan)
    ks = feat.get(f, {}).get('ks', np.nan)
    it = bool(feat.get(f, {}).get('in_target', False))
    nested = (it and sf >= 0.8 and pi == pi and pi >= 0.8 and ks == ks and ks < 0.15)
    rows.append({'feature': f, 'screen_survival': round(sf, 3),
                 'pi_uni_cell': round(pi, 3) if pi == pi else np.nan,
                 'ks': round(ks, 3) if ks == ks else np.nan,
                 'in_target': it, 'nested_selected': bool(nested)})
res = pd.DataFrame(rows).sort_values(['nested_selected', 'screen_survival'],
                                     ascending=[False, False])
res.to_csv(OUT / 'nested_sensitivity.csv', index=False)
print('\n' + res.to_string(index=False))

nested_set = res.loc[res['nested_selected'], 'feature'].tolist()
print(f'\nNested core (n={len(nested_set)}):', nested_set)
print('intersection with {Uracil, Glucose}:',
      sorted(set(nested_set) & {'Uracil', 'Glucose'}))
print('intersection with frozen selector:',
      sorted(set(nested_set) & set(sel['selected_features'])))
print(f'\nsaved: nested_sensitivity.csv -> {OUT}')
