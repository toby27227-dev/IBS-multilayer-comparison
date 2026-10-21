"""在 GMrepo IBS 队列上应用现有分型方法（IBS.py 的孪生脚本）。

方法链与 IBS.py 一致：CLR -> (可选)协变量残差化 -> SPCA 核心菌属 -> PCA -> 层次聚类 -> 稳定性/验证。
数据来自 build_dataset.py 生成的属/种相对丰度矩阵与 gmrepo_metadata.csv。

配置隔离（Configuration Isolation）：预设由 IBS.py 顶部 PRESET 统一控制（单一事实来源）。
  'baseline'    → CLUSTER_SCOPE='ALL'，预处理与主流水线对齐（无 winsorize 等）；输出 figures/baseline/
  'exploration' → CLUSTER_SCOPE='IBS'，允许宽松参数，仅作探索性分析（功效约束，论文 §3.5/§4.7）；输出 figures/exploration/

公平比较协议（Fair Comparison Protocol）：本脚本产出的结构指标（轮廓/CH/DB/Gap/ARI）
仅用于簇结构诊断，不得与监督预测指标（OOF R²/balanced accuracy）跨尺度比较；
属/种矩阵的层间对照必须经主流水线（IBS/pipeline/regression_biomarkers.py）统一协议重算（论文 §2.4）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS

ABUNDANCE_LEVEL = 'genus'   # 'genus' | 'species'

# 预设继承：与 IBS.py 保持同一 PRESET，避免基准对照与子集探索参数混用
PRESET = IBS.PRESET
CLUSTER_SCOPE = 'ALL' if PRESET == 'baseline' else 'IBS'


def load_data():
    mat = pd.read_csv(HERE / f'abundance_{ABUNDANCE_LEVEL}.csv', index_col=0)
    meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    common = mat.index.intersection(meta.index)
    mat, meta = mat.loc[common], meta.loc[common]
    health_state = meta['health_state']
    covariates = pd.DataFrame({
        'Age': meta['host_age'],
        'Sex': meta['sex'],
        'BMI': meta['BMI'],
        'Project': meta['project_id'],
        'Country': meta['country'],
    }, index=meta.index)
    return mat, health_state, covariates


def main():
    fig_dir = HERE / 'figures' / PRESET
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f'[配置隔离] PRESET={PRESET} | scope={CLUSTER_SCOPE} | '
          f'prevalence={IBS.MIN_PREVALENCE} | outlier={IBS.OUTLIER_METHOD} | 输出 → {fig_dir}')
    if PRESET == 'exploration':
        print('[警告] exploration 结果仅作探索性分析（功效约束），不得进入基准对照表。')

    mat, health_state, covariates = load_data()
    print(f'丰度层级: {ABUNDANCE_LEVEL} | 样本: {mat.shape[0]} | 特征: {mat.shape[1]}')
    print(health_state.value_counts().to_string())

    ibs_mask = (health_state == 'IBS').values
    is_ibs_only = CLUSTER_SCOPE == 'IBS'
    sub_mask = ibs_mask if is_ibs_only else np.ones(len(health_state), dtype=bool)

    sub_cov = covariates.loc[sub_mask]
    X_pcs, pipe = IBS.fit_features(mat.loc[sub_mask], sub_cov)

    # 计数空间：稀释到固定深度，供 DMM 使用
    counts = IBS.rarefy_counts(mat.loc[sub_mask], depth=IBS.DMM_TARGET_DEPTH)
    print(f'DMM 计数矩阵: {counts.shape[0]} 样本 × {counts.shape[1]} 单元'
          f'（深度={IBS.DMM_TARGET_DEPTH}）')

    # 统一评估：特征空间（Agglomerative 多指标）+ 计数空间（DMM/BIC）
    res = IBS.evaluate_all_models(X_pcs, counts.values)
    feat, dmm = res['feature'], res['dmm']
    dmm['table'].to_csv(HERE / f'dmm_scan_{PRESET}.csv', index=False)
    print(f'\n最优 K：特征空间={feat["best_k"]} | DMM={dmm["best_k"]}')

    if IBS.CLUSTER_METHOD == 'DMM':
        algo = 'DMM'
        k = dmm['best_k']
        model = dmm['models'][k]
        labels = model['labels']
        score = float('nan')
        resp = model['resp']
        pd.DataFrame(resp, index=counts.index,
                     columns=[f'subtype_{c}' for c in range(k)]).to_csv(
            HERE / f'dmm_posteriors_{PRESET}.csv')
        pd.Series([f'S{c}' for c in labels], index=counts.index,
                  name='subtype').to_csv(HERE / f'dmm_labels_{PRESET}.csv')
        print(f'最终分型: DMM, K={k}, '
              f'后验置信度中位数={np.median(resp.max(axis=1)):.3f}')
    else:
        algo = 'Agglomerative'
        k = feat['best_k']
        labels = feat['labels_by_k'][k]
        score = float(feat['table'].loc[feat['table']['K'] == k, 'silhouette'].iloc[0])
        print(f'最终分型: Agglomerative, K={k}')

    if algo == 'Agglomerative':
        mean_ari, std_ari = IBS.stability_score(
            mat.loc[sub_mask], labels, lambda: IBS.ALGORITHMS[algo](k),
            lambda df, cov: IBS.fit_features(df, cov), covariates=sub_cov)
        print(f'Bootstrap 稳定性 (ARI): {mean_ari:.3f} ± {std_ari:.3f}')
    else:
        print('DMM 稳定性以后验置信度评估（见上）。')

    IBS.validate_vs_covariates(labels, sub_cov)
    if is_ibs_only:
        health_pcs = IBS.apply_features(mat.loc[~ibs_mask], pipe)
        IBS.subtype_vs_health(X_pcs, labels, health_pcs)
    else:
        print(f'\n聚类 × 疾病状态 ARI = '
              f'{IBS.adjusted_rand_score(health_state.values, labels):.4f}')

    if algo == 'Agglomerative':
        IBS.plot_silhouette(X_pcs, labels, score, f'Silhouette ({algo}, K={k})',
                            fig_dir / 'silhouette.png')
    IBS.plot_pca(X_pcs, labels, f'PCA by {algo} subtype ({ABUNDANCE_LEVEL}, K={k})',
                 fig_dir / 'pca_subtype.png')
    print(f'\n图已保存至: {fig_dir}')
    plt.show()


if __name__ == '__main__':
    main()
