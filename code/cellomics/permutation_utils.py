"""受试者层面置换检验通用工具（v2.6 T0.1/T0.2）。

协议（统一主协议，与主分析 IBS/pipeline/regression_biomarkers.py 的 NMR 层一致）：
- 特征构建阶段完成 log1p 与逐特征集 StandardScaler（build_features(..., 'log') 口径），
  本模块接收已完成该变换的 X；
- 外层 GroupKFold(5)、内层 GroupKFold(3) 选 Ridge alpha ∈ {0.1, 1, 10, 100, 1000, 10000}
  （scoring='r2'），合并 OOF；
- subject 一律 astype(str)（分组标签类型会改变 GroupKFold 划分，必须与主分析一致）。

置换语义（group_level=True，默认）：
- 置换单位 = 受试者；置换"受试者→均值标签"映射：受试者 s 的样本获得受试者 π(s)
  的均值移位，受试者内中心化残差保留（保持样本级目标分布与受试者内结构不变）；
- 禁止样本级 np.random.permutation(y)（会破坏受试者内自相关，虚增显著性）。

p 值：p = (1 + sum(null >= observed)) / (1 + n_valid)。
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
N_OUTER = 5
N_INNER = 3
SEED_DEFAULT = 42
N_PERM_DEFAULT = 1000


def fit_predict_ridge(X_train, y_train, X_test, groups_train):
    """主分析同款单折拟合：内层 GroupKFold(3) 调 alpha → Ridge 全训练折重拟合 → 测试折预测。"""
    gs = GridSearchCV(Ridge(), {'alpha': RIDGE_ALPHAS},
                      cv=GroupKFold(N_INNER), scoring='r2')
    gs.fit(X_train, y_train, groups=groups_train)
    return gs.predict(X_test)


def fit_predict_ridge_fast(X_train, y_train, X_test, groups_train):
    """与 fit_predict_ridge 数值等价的快速实现（去 GridSearchCV 开销；用于 1000× 置换）。"""
    inner = list(GroupKFold(N_INNER).split(X_train, y_train, groups_train))
    best_a, best_s = None, None
    for a in RIDGE_ALPHAS:
        s = 0.0
        for ia, ib in inner:
            m = Ridge(alpha=a).fit(X_train[ia], y_train[ia])
            s += r2_score(y_train[ib], m.predict(X_train[ib]))
        s /= len(inner)
        if best_s is None or s > best_s:
            best_s, best_a = s, a
    return Ridge(alpha=best_a).fit(X_train, y_train).predict(X_test)


def oof_predict(X, y, groups, model_fn=fit_predict_ridge_fast):
    """外层 GroupKFold(5) 合并 OOF 预测。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups).astype(str)
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(N_OUTER).split(X, y, groups):
        pred[te] = model_fn(X[tr], y[tr], X[te], groups[tr])
    return pred


def oof_metrics(X, y, groups, model_fn=fit_predict_ridge_fast):
    """合并 OOF 的 R² 与 Spearman ρ。"""
    y = np.asarray(y, dtype=float)
    pred = oof_predict(X, y, groups, model_fn)
    return {'r2': float(r2_score(y, pred)),
            'rho': float(spearmanr(y, pred).statistic), 'pred': pred}


def _permute_subject_labels(y, idm, uniq, rng):
    """受试者层面置换：均值映射置换 + 受试者内中心化残差保留。"""
    perm = rng.permutation(len(uniq))
    means = {s: float(y[idm[s]].mean()) for s in uniq}
    out = np.empty_like(y)
    for s, t in zip(uniq, uniq[perm]):
        ii = idm[s]
        out[ii] = means[t] + (y[ii] - means[s])
    return out


def permutation_test(Xs, y, groups, n_perm=N_PERM_DEFAULT, seed=SEED_DEFAULT,
                     group_level=True, model_fn=None):
    """受试者层面 R² 置换检验。

    参数
    ----
    Xs : ndarray 或 {名称: ndarray}；多模型共享同一置换序列（沿用同一 rng 流），
         便于同序列计算 ΔR² = full − no_* 的零分布。
    y, groups : 样本级目标与受试者 ID（内部 astype(str)）。
    n_perm, seed : 置换次数与随机种子。
    group_level : True=受试者层面（默认）；False=样本级（仅对比用，不建议）。
    model_fn : 单折拟合函数，默认快速路径（与 fit_predict_ridge 数值等价）。

    返回 {name: {'observed_r2','observed_rho','null','null_mean','null_sd',
                'p_value','n_valid'}, 'n_perm','seed','group_level'}
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups).astype(str)
    if model_fn is None:
        model_fn = fit_predict_ridge_fast
    if not isinstance(Xs, dict):
        Xs = {'model': Xs}
    names = list(Xs)
    Xs = {k: np.asarray(v, dtype=float) for k, v in Xs.items()}

    observed = {k: oof_metrics(Xs[k], y, groups, model_fn) for k in names}

    uniq = np.unique(groups)
    idm = {s: np.where(groups == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    null = {k: np.empty(n_perm) for k in names}
    for b in range(n_perm):
        if group_level:
            y_p = _permute_subject_labels(y, idm, uniq, rng)
        else:
            y_p = y[rng.permutation(len(y))]
        for k in names:
            pred = oof_predict(Xs[k], y_p, groups, model_fn)
            null[k][b] = r2_score(y_p, pred)

    out = {'n_perm': int(n_perm), 'seed': int(seed), 'group_level': bool(group_level)}
    for k in names:
        arr = null[k]
        valid = np.isfinite(arr)
        n_valid = int(valid.sum())
        n_ge = int(np.sum(arr[valid] >= observed[k]['r2']))
        out[k] = {
            'observed_r2': observed[k]['r2'],
            'observed_rho': observed[k]['rho'],
            'null': arr,
            'null_mean': float(np.mean(arr[valid])) if n_valid else float('nan'),
            'null_sd': float(np.std(arr[valid])) if n_valid else float('nan'),
            'p_value': (1 + n_ge) / (1 + n_valid),
            'n_valid': n_valid,
        }
    return out
