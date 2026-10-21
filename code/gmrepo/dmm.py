"""Dirichlet-Multinomial Mixture (DMM) 用于 IBS 肠道微生态分型。

复用 IBS.py 中的实现（fit_dmm / evaluate_dmm / align_dmm_by_size）：
  1. 相对丰度 -> 稀释到固定深度的整数计数矩阵
  2. K=1..5 拟合，BIC（负对数边际似然的拉普拉斯型近似）选 K
  3. 输出每个样本属于各亚型的后验概率（软分配），编号按簇大小确定
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS

LEVEL = 'genus'          # 属水平
SCOPE = 'IBS'            # 只对 IBS 患者分型
K_RANGE = range(1, 6)


def build_counts(level=LEVEL):
    abund = pd.read_csv(HERE / f'abundance_{level}.csv', index_col=0)
    meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    common = abund.index.intersection(meta.index)
    counts = IBS.rarefy_counts(abund.loc[common], depth=IBS.DMM_TARGET_DEPTH)
    return counts, meta.loc[common]


def main():
    fig_dir = HERE / 'figures'
    fig_dir.mkdir(exist_ok=True)

    counts_df, meta = build_counts()
    mask = (meta['health_state'] == 'IBS').values if SCOPE == 'IBS' else np.ones(len(meta), bool)
    counts = counts_df.values[mask]
    ids = counts_df.index[mask]
    meta_sub = meta.loc[mask]
    print(f'DMM 输入: {counts.shape[0]} 样本 × {counts.shape[1]} 单元'
          f'（{LEVEL}, scope={SCOPE}, 深度={IBS.DMM_TARGET_DEPTH}）')

    scores, best_K, models = IBS.evaluate_dmm(counts, K_RANGE)
    scores.to_csv(HERE / 'dmm_scores.csv', index=False)
    print(f'\n最优 K (BIC 最小) = {best_K}')

    res = models[best_K]
    post = pd.DataFrame(res['resp'], index=ids,
                        columns=[f'subtype_{k}' for k in range(best_K)])
    post.to_csv(HERE / 'dmm_posteriors.csv')
    hard = pd.Series([f'S{k}' for k in res['labels']], index=ids, name='subtype')
    hard.to_csv(HERE / 'dmm_labels.csv')
    print('各亚型样本数:'); print(hard.value_counts().sort_index().to_string())
    print('软分配置信度中位数: %.3f' % np.median(res['resp'].max(axis=1)))

    IBS.validate_vs_covariates(
        res['labels'],
        meta_sub[['host_age', 'sex', 'BMI']].rename(
            columns={'host_age': 'Age', 'sex': 'Sex'}))

    # BIC 曲线
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scores['K'], scores['BIC'], marker='o', color='steelblue', lw=2)
    ax.axvline(best_K, color='red', ls='--', label=f'best K={best_K}')
    ax.set_xlabel('Number of subtypes (K)')
    ax.set_ylabel('BIC (lower is better)')
    ax.set_title('DMM Model Selection')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / 'dmm_bic.png', dpi=150); plt.close(fig)

    # PCA（CLR 空间）
    P = counts_df.loc[ids].values.astype(float)
    pseudo = IBS.resolve_pseudo(P, IBS.PSEUDOCOUNT)
    Xc = IBS.clr_transform(P, pseudo, IBS.CLR_METHOD)
    Z = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)
    pcs = PCA(n_components=2, random_state=IBS.RANDOM_STATE).fit_transform(Z)
    IBS.plot_pca(pcs, hard.values, f'DMM subtypes (K={best_K}), PCA of CLR',
                 fig_dir / 'dmm_pca.png')
    print(f'\n输出: dmm_scores.csv / dmm_posteriors.csv / dmm_labels.csv；图: {fig_dir}')


if __name__ == '__main__':
    main()
