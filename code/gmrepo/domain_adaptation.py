"""批次解耦与跨队列泛化。

1. MMUPHin 联合校正 vs 基线（残差化对照）
2. CORAL 闭式域对齐（无 torch 环境下替代 DANN/GRL 的可解释域适应）
3. 跨队列泛化：PRJEB37924 发现集训练 -> PRJNA705217 / PRJEB3227 不重拟合投影
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS

LEVEL = 'genus'
N_BOOT_CORAL = 30


def load():
    mat = pd.read_csv(HERE / f'abundance_{LEVEL}.csv', index_col=0)
    meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    common = mat.index.intersection(meta.index)
    mat, meta = mat.loc[common], meta.loc[common]
    cov = pd.DataFrame({
        'Age': meta['host_age'], 'Sex': meta['sex'], 'BMI': meta['BMI'],
        'Project': meta['project_id'], 'Country': meta['country'],
        'Health': meta['health_state'],
    }, index=meta.index)
    return mat, meta, cov


def assoc_p(labels, meta_sub, col):
    tab = pd.crosstab(labels, meta_sub[col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    return float(chi2_contingency(tab)[1])


def eval_feature(name, X_raw, cov, meta_sub, **kw):
    X, _ = IBS.fit_features(X_raw, cov, **kw)
    metrics, labels_by_k = IBS.evaluate_k_metrics(X)
    k, comp = IBS.select_best_metric(metrics)
    labels = labels_by_k[k]
    sil = float(metrics.loc[metrics['K'] == k, 'silhouette'].iloc[0])
    feat = lambda df, cv: IBS.fit_features(df, cv, **kw)  # noqa: E731
    ari, _ = IBS.stability_score(
        X_raw, labels, lambda: IBS.ALGORITHMS['Agglomerative'](k), feat, covariates=cov)
    row = {'config': name, 'n': len(X_raw), 'K': k, 'silhouette': sil,
           'composite': comp, 'bootstrap_ARI': ari,
           'p_project': assoc_p(labels, meta_sub, 'project_id'),
           'p_country': assoc_p(labels, meta_sub, 'country')}
    print(f"{name:<26s} n={row['n']:3d} K={k} sil={sil:.3f} comp={comp:.3f} "
          f"ARI={ari:.3f} p(proj)={row['p_project']:.2e} p(ctry)={row['p_country']:.2e}")
    return row


def _sqrtm(M, inv=False):
    w, V = np.linalg.eigh(M)
    if inv:
        d = 1.0 / np.sqrt(np.clip(w, 1e-12, None))
    else:
        d = np.sqrt(np.clip(w, 0, None))
    return V @ np.diag(d) @ V.T


def coral_align(X, domains, reference=None, lam=1.0):
    """把每个域的二阶统计量（均值+协方差）对齐到参考域（CORAL）。"""
    X = np.asarray(X, dtype=float)
    domains = np.asarray(domains).astype(str)
    if reference is None:
        reference = pd.Series(domains).value_counts().idxmax()
    d = X.shape[1]
    Xr = X[domains == reference]
    mr = Xr.mean(axis=0)
    Cr = np.cov(Xr, rowvar=False) + np.eye(d) * lam
    out = X.copy()
    for dom in pd.unique(domains):
        if dom == reference:
            continue
        idx = domains == dom
        Xd = X[idx]
        md = Xd.mean(axis=0)
        Cd = np.cov(Xd, rowvar=False) + np.eye(d) * lam
        A = _sqrtm(Cd, inv=True) @ _sqrtm(Cr)
        out[idx] = (Xd - md) @ A + mr
    return out


def eval_coral(name, X_raw, cov, meta_sub, domain_col='Project', n_boot=N_BOOT_CORAL):
    X, _ = IBS.fit_features(X_raw, cov)
    domains = cov[domain_col].values
    Xa = coral_align(X, domains)
    metrics, labels_by_k = IBS.evaluate_k_metrics(Xa)
    k, comp = IBS.select_best_metric(metrics)
    labels = labels_by_k[k]
    sil = float(metrics.loc[metrics['K'] == k, 'silhouette'].iloc[0])

    # CORAL + 完整重跑流水线的 Bootstrap
    rng = np.random.default_rng(IBS.RANDOM_STATE)
    n = len(X_raw)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            Xb, _ = IBS.fit_features(X_raw.iloc[idx], cov.iloc[idx])
            Xab = coral_align(Xb, cov.iloc[idx][domain_col].values)
            lb = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xab)
        except Exception:
            continue
        first = {}
        for pos, s in enumerate(idx):
            first.setdefault(int(s), pos)
        uniq = np.array(list(first))
        scores.append(IBS.adjusted_rand_score(labels[uniq],
                                              lb[[first[i] for i in uniq]]))
    ari = float(np.mean(scores)) if scores else np.nan
    row = {'config': name, 'n': len(X_raw), 'K': k, 'silhouette': sil,
           'composite': comp, 'bootstrap_ARI': ari,
           'p_project': assoc_p(labels, meta_sub, 'project_id'),
           'p_country': assoc_p(labels, meta_sub, 'country')}
    print(f"{name:<26s} n={row['n']:3d} K={k} sil={sil:.3f} comp={comp:.3f} "
          f"ARI={ari:.3f} p(proj)={row['p_project']:.2e}")
    return row


def cross_cohort(mat, meta, cov, discovery='PRJEB37924',
                 validations=('PRJNA705217', 'PRJEB3227')):
    """发现集（仅 IBS 病例）拟合 -> 验证集不重拟合投影。"""
    disc = mat.index[(meta['project_id'] == discovery) &
                     (meta['health_state'] == 'IBS')]
    Xd, pipe = IBS.fit_features(mat.loc[disc], cov.loc[disc])
    metrics, labels_by_k = IBS.evaluate_k_metrics(Xd)
    k, _ = IBS.select_best_metric(metrics)
    ref = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xd)
    centers = np.vstack([Xd[ref == c].mean(axis=0) for c in range(k)])

    # DMM 发现集模型
    counts_d = IBS.rarefy_counts(mat.loc[disc], depth=IBS.DMM_TARGET_DEPTH)
    _, dmm_k, dmm_models = IBS.evaluate_dmm(counts_d.values, IBS.K_RANGE)
    dmm_model = dmm_models[dmm_k]
    print(f'\n[跨队列] 发现集 {discovery}: n={len(disc)} | '
          f'Agglomerative K={k} | DMM K={dmm_k}')

    rows = []
    for proj in validations:
        idx = mat.index[meta['project_id'] == proj]
        if len(idx) == 0:
            continue

        # (a) 特征空间最近质心
        Xv = IBS.apply_features(mat.loc[idx], pipe, covariates=cov.loc[idx])
        dist = ((Xv[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist.argmin(axis=1)
        d1 = dist.min(axis=1)
        d2 = np.sort(dist, axis=1)[:, 1] if k > 1 else np.full(len(dist), np.nan)
        conf = 1.0 - d1 / np.clip(d2, 1e-12, None)

        # (b) DMM 后验投影
        counts_v = IBS.rarefy_counts(mat.loc[idx], depth=IBS.DMM_TARGET_DEPTH)
        resp = IBS.dmm_predict_proba(counts_v.values, dmm_model)
        assign_d = resp.argmax(axis=1)
        conf_d = resp.max(axis=1)

        p_health = assoc_p(assign, meta.loc[idx], 'health_state')
        p_health_d = assoc_p(assign_d, meta.loc[idx], 'health_state')
        print(f"  验证集 {proj}: n={len(idx)}")
        print(f"    [Agg] 分布={dict(pd.Series(assign).value_counts().sort_index())} "
              f"置信度={np.nanmean(conf):.3f} health p={p_health:.2e}")
        print(f"    [DMM] 分布={dict(pd.Series(assign_d).value_counts().sort_index())} "
              f"置信度={np.nanmean(conf_d):.3f} health p={p_health_d:.2e}")
        rows.append({'method': 'Agg', 'validation': proj, 'n': len(idx),
                     'discovery_K': k, 'mean_transfer_conf': float(np.nanmean(conf)),
                     'p_health_state': p_health,
                     'subtype_counts': str(dict(pd.Series(assign).value_counts().sort_index()))})
        rows.append({'method': 'DMM', 'validation': proj, 'n': len(idx),
                     'discovery_K': dmm_k, 'mean_transfer_conf': float(np.nanmean(conf_d)),
                     'p_health_state': p_health_d,
                     'subtype_counts': str(dict(pd.Series(assign_d).value_counts().sort_index()))})
    return rows


def main():
    mat, meta, cov = load()
    ibs = meta.index[meta['health_state'] == 'IBS']
    X_ibs = mat.loc[ibs]
    cov_ibs = cov.loc[ibs]
    meta_ibs = meta.loc[ibs]

    print('=== 一、批次解耦对比（IBS, genus）===')
    rows = [
        eval_feature('baseline', X_ibs, cov_ibs, meta_ibs),
        eval_feature('MMUPHin: Project', X_ibs, cov_ibs, meta_ibs,
                     batch_correct='mmuphin', batch_cols=['Project']),
        eval_feature('MMUPHin: Project+Country', X_ibs, cov_ibs, meta_ibs,
                     batch_correct='mmuphin', batch_cols=['Project', 'Country']),
        eval_coral('CORAL: Project', X_ibs, cov_ibs, meta_ibs),
    ]

    print('\n=== 二、跨队列泛化 ===')
    rows += cross_cohort(mat, meta.copy(), cov.copy())

    pd.DataFrame(rows).to_csv(HERE / 'domain_adaptation.csv', index=False)
    print('\n已保存: domain_adaptation.csv')


if __name__ == '__main__':
    main()
