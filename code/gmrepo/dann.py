"""DANN：域对抗隐空间重构（IBS 多中心 batch-invariant 表示）。

结构：Encoder G_f + 亚型头 G_y（DMM 伪标签）+ 域判别器 G_d（GRL）。
实验：
  A. 全部 IBS（3 域）训练，lambda 消融，评估域可预测性与亚型结构
  B. 零样本：PRJEB37924 + PRJEB3227 训练，冻结 Encoder 投影 PRJNA705217（IBS+健康）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Function

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS

LEVEL = 'genus'
LATENT = 16
EPOCHS = 400
BATCH = 64
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


def grl(x, lam):
    return GradientReversal.apply(x, lam)


class DANN(nn.Module):
    def __init__(self, input_dim, latent_dim, n_subtypes, n_domains):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
        )
        self.subtype = nn.Linear(latent_dim, n_subtypes)
        self.domain = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, n_domains),
        )

    def forward(self, x, lam):
        z = self.encoder(x)
        return z, self.subtype(z), self.domain(grl(z, lam))


def prepare():
    mat = pd.read_csv(HERE / f'abundance_{LEVEL}.csv', index_col=0)
    meta = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    labels = pd.read_csv(HERE / 'dmm_labels.csv', index_col=0).iloc[:, 0]
    labels.index = labels.index.astype(str)
    common = mat.index.intersection(meta.index)
    mat, meta = mat.loc[common], meta.loc[common]

    P = mat.values.astype(float)
    keep = (P > 0).mean(axis=0) >= IBS.MIN_PREVALENCE
    pseudo = IBS.resolve_pseudo(P, IBS.PSEUDOCOUNT)
    X = IBS.clr_transform(P, pseudo, IBS.CLR_METHOD)[:, keep]
    ibs_mask = (meta['health_state'] == 'IBS').values
    scaler = StandardScaler().fit(X[ibs_mask])
    Xs = scaler.transform(X)

    y = pd.Series(-1, index=meta.index, dtype=int)
    y.loc[labels.index] = labels.str.replace('S', '').astype(int).values
    return Xs, y.values, meta, ibs_mask, scaler, keep


def train_dann(Xs, y, d, n_sub, n_dom, lam_max=1.0, epochs=EPOCHS):
    X = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dt = torch.tensor(d, dtype=torch.long)
    model = DANN(Xs.shape[1], LATENT, n_sub, n_dom)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    counts = np.bincount(y, minlength=n_sub).astype(float)
    weights = torch.tensor(counts.sum() / (n_sub * np.clip(counts, 1, None)),
                           dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=weights)
    ce_dom = nn.CrossEntropyLoss()
    n = len(X)
    bs = min(BATCH, n)
    for ep in range(epochs):
        lam = lam_max * min(1.0, ep / max(1, 0.3 * epochs))
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            _, sub_logits, dom_logits = model(X[idx], lam)
            loss = ce(sub_logits, yt[idx]) + ce_dom(dom_logits, dt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def latent_of(model, Xs):
    with torch.no_grad():
        return model.encoder(torch.tensor(Xs, dtype=torch.float32)).numpy()


def domain_accuracy(Z, d):
    counts = np.bincount(d)
    counts = counts[counts > 0]
    cv = int(min(5, counts.min())) if counts.size else 0
    if cv < 2:
        return np.nan
    return float(cross_val_score(LogisticRegression(max_iter=1000), Z, d, cv=cv).mean())


def cluster_eval(Z, K):
    labels = AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(Z)
    return labels, IBS.cluster_metrics(Z, labels)


def assign_centroids(Z_tr, lab_tr, Z_te):
    cs = np.unique(lab_tr)
    centers = np.vstack([Z_tr[lab_tr == c].mean(axis=0) for c in cs])
    dist = ((Z_te[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    a = dist.argmin(axis=1)
    d1 = dist.min(axis=1)
    d2 = np.sort(dist, axis=1)[:, 1] if len(cs) > 1 else np.full(len(dist), np.nan)
    return a, 1.0 - d1 / np.clip(d2, 1e-12, None)


def health_p(assign, meta_sub):
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(assign, meta_sub['health_state'])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    return float(chi2_contingency(tab)[1])


def main():
    Xs, y, meta, ibs_mask, scaler, keep = prepare()
    Xa, ya = Xs[ibs_mask], y[ibs_mask]
    da = pd.factorize(meta.loc[ibs_mask, 'project_id'])[0]
    n_sub, n_dom = ya.max() + 1, int(da.max() + 1)
    print(f'DANN 输入: {Xa.shape} | 亚型={n_sub} | 域={n_dom} | 域分布={np.bincount(da).tolist()}')

    rows = []
    print('\n=== A. lambda 消融（全 IBS，3 域）===')
    print(f"{'lambda':>7} {'域acc(input)':>13} {'域acc(latent)':>14} {'sil':>7} {'ARI_vs_DMM':>11}")
    for lam in [0.0, 0.3, 1.0, 3.0]:
        model = train_dann(Xa, ya, da, n_sub, n_dom, lam_max=lam)
        Z = latent_of(model, Xa)
        dom_in = domain_accuracy(Xa, da)
        dom_lat = domain_accuracy(Z, da)
        lab, m = cluster_eval(Z, n_sub)
        ari = IBS.adjusted_rand_score(ya, lab)
        print(f'{lam:>7.1f} {dom_in:>13.3f} {dom_lat:>14.3f} '
              f'{m["silhouette"]:>7.3f} {ari:>11.3f}')
        rows.append({'experiment': f'A_lambda_{lam}', 'n': len(Xa),
                     'domain_acc_input': dom_in, 'domain_acc_latent': dom_lat,
                     'K': n_sub, 'silhouette': m['silhouette'],
                     'CH': m['calinski_harabasz'], 'DB': m['davies_bouldin'],
                     'ARI_vs_DMM': ari})
        if lam == 1.0:
            pd.DataFrame(Z, index=meta.index[ibs_mask]).to_csv(HERE / 'dann_latent_all.csv')

    # ---------- B. 零样本 ----------
    heldout = 'PRJNA705217'
    tr = ibs_mask & (meta['project_id'] != heldout).values
    te = (meta['project_id'] == heldout).values
    Xtr, ytr = Xs[tr], y[tr]
    dtr = pd.factorize(meta.loc[tr, 'project_id'])[0]
    Xte, meta_te = Xs[te], meta.loc[te]

    model = train_dann(Xtr, ytr, dtr, n_sub, int(dtr.max() + 1), lam_max=1.0)
    Ztr, Zte = latent_of(model, Xtr), latent_of(model, Xte)
    lab_tr, m_tr = cluster_eval(Ztr, n_sub)
    a_d, c_d = assign_centroids(Ztr, lab_tr, Zte)

    pca = PCA(n_components=LATENT, random_state=SEED).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    lab_p, m_p = cluster_eval(Ptr, n_sub)
    a_p, c_p = assign_centroids(Ptr, lab_p, Pte)

    print(f'\n=== B. 零样本：训练 {sorted(meta.loc[tr, "project_id"].unique())} '
          f'-> 测试 {heldout}（n={int(te.sum())}, 含健康对照）===')
    print(f'[DANN] 分布={dict(pd.Series(a_d).value_counts().sort_index())} '
          f'置信度={np.nanmean(c_d):.3f} health p={health_p(a_d, meta_te):.2e} '
          f'| 训练 latent sil={m_tr["silhouette"]:.3f} 域acc={domain_accuracy(Ztr, dtr):.3f}')
    print(f'[PCA ] 分布={dict(pd.Series(a_p).value_counts().sort_index())} '
          f'置信度={np.nanmean(c_p):.3f} health p={health_p(a_p, meta_te):.2e} '
          f'| 训练 PCA sil={m_p["silhouette"]:.3f}')

    rows.append({'experiment': 'B_zero_shot_dann', 'n': int(te.sum()),
                 'domain_acc_input': np.nan, 'domain_acc_latent': domain_accuracy(Ztr, dtr),
                 'K': n_sub, 'silhouette': float(np.nanmean(c_d)), 'CH': np.nan,
                 'DB': np.nan, 'ARI_vs_DMM': np.nan, 'health_p': health_p(a_d, meta_te)})
    rows.append({'experiment': 'B_zero_shot_pca', 'n': int(te.sum()),
                 'domain_acc_input': np.nan, 'domain_acc_latent': np.nan, 'K': n_sub,
                 'silhouette': float(np.nanmean(c_p)), 'CH': np.nan, 'DB': np.nan,
                 'ARI_vs_DMM': np.nan, 'health_p': health_p(a_p, meta_te)})

    pd.DataFrame(rows).to_csv(HERE / 'dann_results.csv', index=False)
    print('\n已保存: dann_latent_all.csv / dann_results.csv')


if __name__ == '__main__':
    main()
