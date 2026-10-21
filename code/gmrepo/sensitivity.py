"""伪计数/CLR 敏感性检验 + Bootstrap 泄漏对照（GMrepo 属水平 IBS）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS

mat = pd.read_csv(HERE / 'abundance_genus.csv', index_col=0)
meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
common = mat.index.intersection(meta.index)
mat, meta = mat.loc[common], meta.loc[common]
mask = (meta['health_state'] == 'IBS').values
X_raw = mat.loc[mask]
cov = meta.loc[mask, ['host_age', 'sex', 'BMI']].rename(columns={'host_age': 'Age', 'sex': 'Sex'})

configs = [
    ('CLR auto(min/2)', dict()),
    ('CLR 1e-6', dict(pseudo=1e-6)),
    ('CLR min_rel', dict(pseudo='min_rel')),
    ('rCLR', dict(clr_method='rclr')),
]

print('=== 伪计数 / rCLR 敏感性 ===')
emb = {}
for name, kw in configs:
    X, pipe = IBS.fit_features(X_raw, cov, **kw)
    metrics, labels_by_k = IBS.evaluate_k_metrics(X)
    k, comp = IBS.select_best_metric(metrics)
    emb[name] = labels_by_k[k]
    row = metrics[metrics['K'] == k].iloc[0]
    print(f'{name:18s} -> K={k} | sil={row["silhouette"]:.4f} | '
          f'CH={row["calinski_harabasz"]:.1f} | DB={row["davies_bouldin"]:.3f} | '
          f'composite={comp:.3f}')

print('\n=== 不同 CLR 设定之间聚类标签的一致性 (ARI) ===')
names = list(emb)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = IBS.adjusted_rand_score(np.asarray(emb[names[i]]), np.asarray(emb[names[j]]))
        print(f'  {names[i]:18s} vs {names[j]:18s} ARI={a:.3f}')

print('\n=== Bootstrap：旧（在 X_pcs 上重采样）vs 新（完整重跑流水线）===')
X_default, _ = IBS.fit_features(X_raw, cov)
metrics, labels_by_k = IBS.evaluate_k_metrics(X_default)
k, _ = IBS.select_best_metric(metrics)
reference = labels_by_k[k]
factory = lambda: IBS.ALGORITHMS['Agglomerative'](k)

rng = np.random.default_rng(IBS.RANDOM_STATE)
n = len(X_default)
old = []
for _ in range(50):
    idx = rng.integers(0, n, n)
    lab = factory().fit_predict(X_default[idx])
    first = {}
    for pos, s in enumerate(idx):
        first.setdefault(int(s), pos)
    uniq = np.array(list(first))
    old.append(IBS.adjusted_rand_score(reference[uniq], lab[[first[i] for i in uniq]]))

old_mean, old_std = float(np.mean(old)), float(np.std(old))
saved = IBS.N_BOOTSTRAP
IBS.N_BOOTSTRAP = 50
new_mean, new_std = IBS.stability_score(
    X_raw, reference, factory, lambda df, cv: IBS.fit_features(df, cv), covariates=cov)
IBS.N_BOOTSTRAP = saved
print(f'  旧（泄漏）: {old_mean:.3f} ± {old_std:.3f}')
print(f'  新（完整重跑）: {new_mean:.3f} ± {new_std:.3f}')
print(f'  高估幅度: {old_mean - new_mean:+.3f}')
