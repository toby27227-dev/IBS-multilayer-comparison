"""Leave-One-Domain-Out (LODO) 域泛化评估。

每次留出一个 project 作为完全未见的测试域；在其余项目上训练：
  - PCA + 最近质心（基线）
  - ERM（λ=0，仅亚型头）
  - DANN（λ=1，域对抗）
伪标签在每折训练集内部重新用 DMM 生成，避免跨折泄漏。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import IBS
import dann

LEVEL = 'genus'


def fit_transform(train_df, test_df):
    """在训练集上拟合 prevalence/CLR/scaler，并变换训练/测试。"""
    Ptr = train_df.values.astype(float)
    Pte = test_df.values.astype(float)
    keep = (Ptr > 0).mean(axis=0) >= IBS.MIN_PREVALENCE
    pseudo = IBS.resolve_pseudo(Ptr, IBS.PSEUDOCOUNT)
    Xtr = IBS.clr_transform(Ptr, pseudo, IBS.CLR_METHOD)[:, keep]
    Xte = IBS.clr_transform(Pte, pseudo, IBS.CLR_METHOD)[:, keep]
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte)


def train_pseudo_labels(train_df):
    counts = IBS.rarefy_counts(train_df, depth=IBS.DMM_TARGET_DEPTH)
    scores, k, models = IBS.evaluate_dmm(counts.values, IBS.K_RANGE)
    return models[k]['labels'], k


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_fold(heldout, mat, meta, ibs_mask):
    tr_ibs = ibs_mask & (meta['project_id'] != heldout).values
    te_all = (meta['project_id'] == heldout).values

    train_df = mat.loc[tr_ibs]
    test_df = mat.loc[te_all]
    meta_te = meta.loc[te_all]
    Xtr, Xte = fit_transform(train_df, test_df)

    ytr, k = train_pseudo_labels(train_df)
    dtr = pd.factorize(meta.loc[tr_ibs, 'project_id'])[0]
    n_dom = int(dtr.max() + 1) if len(dtr) else 1

    rows = []
    for method, lam in [('ERM', 0.0), ('DANN', 1.0)]:
        model = dann.train_dann(Xtr, ytr, dtr, k, max(2, n_dom), lam_max=lam)
        Ztr = dann.latent_of(model, Xtr)
        Zte = dann.latent_of(model, Xte)
        lab_tr, m_tr = dann.cluster_eval(Ztr, k)
        assign, conf = dann.assign_centroids(Ztr, lab_tr, Zte)
        rows.append({
            'heldout': heldout, 'method': method, 'n_train': len(Xtr),
            'n_test': len(Xte), 'K': k,
            'train_silhouette': m_tr['silhouette'],
            'domain_acc': dann.domain_accuracy(Ztr, dtr),
            'transfer_conf': float(np.nanmean(conf)),
            'dist_entropy': entropy(np.bincount(assign, minlength=k) / len(assign)),
            'health_p': dann.health_p(assign, meta_te),
        })

    # PCA 基线
    pca = PCA(n_components=min(16, Xtr.shape[1]), random_state=IBS.RANDOM_STATE).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    lab_p, m_p = dann.cluster_eval(Ptr, k)
    a_p, c_p = dann.assign_centroids(Ptr, lab_p, Pte)
    rows.append({
        'heldout': heldout, 'method': 'PCA', 'n_train': len(Xtr),
        'n_test': len(Xte), 'K': k, 'train_silhouette': m_p['silhouette'],
        'domain_acc': dann.domain_accuracy(Ptr, dtr),
        'transfer_conf': float(np.nanmean(c_p)),
        'dist_entropy': entropy(np.bincount(a_p, minlength=k) / len(a_p)),
        'health_p': dann.health_p(a_p, meta_te),
    })
    return rows


def main():
    mat = pd.read_csv(HERE / f'abundance_{LEVEL}.csv', index_col=0)
    meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    common = mat.index.intersection(meta.index)
    mat, meta = mat.loc[common], meta.loc[common]
    ibs_mask = (meta['health_state'] == 'IBS').values

    rows = []
    for heldout in sorted(meta['project_id'].unique()):
        n_tr = int((ibs_mask & (meta['project_id'] != heldout).values).sum())
        n_te = int((meta['project_id'] == heldout).sum())
        if n_tr < 20:
            print(f'[skip] heldout={heldout}: 训练集仅 {n_tr} 例')
            continue
        print(f'\n=== LODO: heldout={heldout} (train {n_tr} IBS, test {n_te}) ===')
        fold_rows = run_fold(heldout, mat, meta, ibs_mask)
        for r in fold_rows:
            print(f"  {r['method']:<5s} K={r['K']} train_sil={r['train_silhouette']:.3f} "
                  f"domain_acc={r['domain_acc']:.3f} conf={r['transfer_conf']:.3f} "
                  f"entropy={r['dist_entropy']:.3f} health_p={r['health_p']}")
        rows += fold_rows

    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'domain_generalization.csv', index=False)
    print('\n=== 汇总（按方法均值）===')
    print(df.groupby('method')[['train_silhouette', 'domain_acc', 'transfer_conf',
                                'dist_entropy']].mean().round(3).to_string())
    print('\n已保存: domain_generalization.csv')


if __name__ == '__main__':
    main()
