"""批次/项目混杂控制：残差化 vs 单项目分层。

背景：GMrepo IBS 亚型与 project/country 强绑定（chi2 p≈1e-20）。
本脚本比较：
  1. baseline（不控制）
  2. 残差化掉 Project（one-hot 回归残差）
  3. 残差化掉 Country
  4. 单项目内分层聚类（PRJEB37924 / PRJNA705217），检验亚型是否可复现
所有稳定性均使用"重采样 + 完整重跑预处理"的 ARI。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS

LEVEL = 'genus'


def load():
    mat = pd.read_csv(HERE / f'abundance_{LEVEL}.csv', index_col=0)
    meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    common = mat.index.intersection(meta.index)
    mat, meta = mat.loc[common], meta.loc[common]
    cov = pd.DataFrame({
        'Age': meta['host_age'], 'Sex': meta['sex'], 'BMI': meta['BMI'],
        'Project': meta['project_id'], 'Country': meta['country'],
    }, index=meta.index)
    return mat, meta, cov


def assoc_p(labels, meta_sub, col):
    tab = pd.crosstab(labels, meta_sub[col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    return float(chi2_contingency(tab)[1])


def run_config(name, X_raw, cov, meta_sub, use_residual=False, resid=None,
               do_boot=True):
    X, _ = IBS.fit_features(X_raw, cov, use_residual=use_residual,
                            resid_covariates=resid or ['Sex', 'Age'])
    metrics, labels_by_k = IBS.evaluate_k_metrics(X)
    k, comp = IBS.select_best_metric(metrics)
    labels = labels_by_k[k]
    sil = float(metrics.loc[metrics['K'] == k, 'silhouette'].iloc[0])
    p_proj = assoc_p(labels, meta_sub, 'project_id')
    p_ctry = assoc_p(labels, meta_sub, 'country')

    ari = np.nan
    if do_boot and len(X_raw) >= 20:
        feat = lambda df, cv: IBS.fit_features(  # noqa: E731
            df, cv, use_residual=use_residual, resid_covariates=resid or ['Sex', 'Age'])
        ari, _ = IBS.stability_score(
            X_raw, labels, lambda: IBS.ALGORITHMS['Agglomerative'](k),
            feat, covariates=cov)

    print(f'{name:<26s} n={len(X_raw):3d} | K={k} | sil={sil:.3f} | '
          f'comp={comp:.3f} | ARI={ari if np.isnan(ari) else round(ari, 3)} | '
          f'p(project)={p_proj:.2e} | p(country)={p_ctry:.2e}')
    return {'config': name, 'n': len(X_raw), 'K': k, 'silhouette': sil,
            'composite': comp, 'bootstrap_ARI': ari,
            'p_project': p_proj, 'p_country': p_ctry}


def main():
    mat, meta, cov = load()
    ibs_mask = (meta['health_state'] == 'IBS').values
    X_ibs = mat.loc[ibs_mask]
    cov_ibs = cov.loc[ibs_mask]
    meta_ibs = meta.loc[ibs_mask]

    print('=== 批次控制对比（IBS 患者内, genus）===')
    rows = [
        run_config('baseline', X_ibs, cov_ibs, meta_ibs),
        run_config('resid: Project', X_ibs, cov_ibs, meta_ibs,
                   use_residual=True, resid=['Project']),
        run_config('resid: Country', X_ibs, cov_ibs, meta_ibs,
                   use_residual=True, resid=['Country']),
    ]

    print('\n=== 单项目分层（避免跨项目批次）===')
    for proj, idx in meta_ibs.groupby('project_id').groups.items():
        idx = list(idx)
        if len(idx) < 20:
            print(f'{proj:<26s} n={len(idx):3d} | 样本过少，跳过稳定性')
            continue
        rows.append(run_config(f'stratum: {proj}', mat.loc[idx], cov.loc[idx],
                               meta.loc[idx], do_boot=True))

    pd.DataFrame(rows).to_csv(HERE / 'batch_control.csv', index=False)
    print('\n已保存: batch_control.csv')


if __name__ == '__main__':
    main()
