"""基于肠道菌群的数据驱动 IBS 亚型重构。

流程：
  1. 载入 IBS 患者 + 健康对照的菌群丰度矩阵
  2. CLR（中心对数比）变换，处理组成型数据
  3. 核心菌属筛选（方差 / 稀疏主成分 SPCA），剔除背景噪音菌
  4. 生活方式/临床协变量的线性回归残差化（正交分离）
  5. PCA 降维，缓解高维稀疏
  6. 层次聚类（Agglomerative）在不同簇数下比较并选最优
  7. Bootstrap 稳定性检验（ARI）
  8. 与临床协变量及健康对照做生物学验证

说明：默认只对 IBS 患者聚类以发现亚型，健康对照作为外部参照
（计算各亚型到健康群体中心的距离，判断是否有亚型更接近健康态）。
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import chi2_contingency, kruskal
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score, silhouette_samples)
from sklearn.preprocessing import StandardScaler

# ==========================================
# 配置区：所有可调参数集中在此
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'Project-master' / 'data' / 'raw'
IBS_ABUNDANCE = DATA_DIR / 'final_bacteria_ibs_141.csv'
HEALTH_ABUNDANCE = DATA_DIR / 'final_bacteria_health.csv'
IBS_STAT = DATA_DIR / 'final_ibs_141_statistic.xlsx'
# 原始 Project-master 数据缺失时的回退数据源（GMrepo 汇总队列）
GMREPO_ABUNDANCE = SCRIPT_DIR / 'gmrepo' / 'abundance_genus.csv'
GMREPO_METADATA = SCRIPT_DIR / 'gmrepo' / 'gmrepo_metadata.csv'
FIGURE_DIR = SCRIPT_DIR / 'figures'

CLUSTER_SCOPE = 'IBS'   # 'IBS' 只对患者聚类找亚型；'ALL' 对 IBS+健康一起聚类
CLR_METHOD = 'clr'      # 'clr'（伪计数）| 'rclr'（仅非零值几何均值，稳健）
PSEUDOCOUNT = None      # None=自动(最小正值/2)；float=固定值(如 1e-6)；'min_rel'=最小正值
MIN_PREVALENCE = 0.10   # 保留在至少 10% 样本中检出的菌属
N_METRIC_REFS = 20      # Gap statistic 参考数据集数量
OUTLIER_METHOD = 'winsorize'  # 'none' | 'winsorize'：按分位数把极端值替换为边界值
OUTLIER_LOW = 0.01      # 下分位
OUTLIER_HIGH = 0.99     # 上分位
N_PCS = 20              # PCA 保留的主成分数
K_RANGE = range(2, 7)   # 候选簇数
N_BOOTSTRAP = 50        # 稳定性检验重采样次数（完整重跑流水线，故取 50）
RANDOM_STATE = 42

# 最终分型方法：'Agglomerative'（特征空间）| 'DMM'（计数空间）
CLUSTER_METHOD = 'Agglomerative'
# DMM 参数
DMM_TARGET_DEPTH = 10000  # 稀释到的固定测序深度（None=不稀释）
DMM_N_INIT = 5
DMM_MAX_ITER = 300
DMM_TOL = 1e-6

# VIF 与核心菌属筛选是两种替代性的降噪手段；本数据上二者叠加会过度删减，
# 默认关闭 VIF、交给 SPCA 做核心菌筛选（见实验：SPCA 单独轮廓更高）。
USE_VIF = False
VIF_THRESHOLD = 10.0    # VIF 上限，超过则视为共线并剔除
VIF_TOP_N = 100         # 参与 VIF 的候选特征上限（须 < 样本数）

# 核心菌属（Driver Taxa）筛选：'none' | 'variance' | 'spca'
CORE_TAXA_METHOD = 'spca'
N_CORE_TAXA = 30
SPCA_ALPHA = 1.0
SPCA_N_COMPONENTS = 10
SPCA_MAX_FEATURES = 300  # SPCA 前的方差预筛选，控制计算量
SPCA_METHOD = 'cd'       # 'lars'(精确但慢) | 'cd'(坐标下降，快，适合 bootstrap)
SPCA_MAX_ITER = 100
SPCA_TOL = 1e-3

# 生活方式/临床协变量正交分离（线性回归残差法）
# 注意：本数据 Age 仅 44%、BMI 仅 13% 非缺失，大比例填补会引入噪声，
# 实测残差化反而拉低轮廓（0.23 -> 0.09~0.17），故默认关闭，可按需开启。
RESIDUALIZE = False
RESID_COVARIATES = ['Sex', 'Age']

# MMUPHin 式联合批次校正（在 CLR 空间拟合，仅扣除批次系数）
BATCH_CORRECT = 'none'     # 'none' | 'mmuphin'
BATCH_COLS = ['Project']
BIO_COLS = []              # 需要保护的生物学变量（如 ['health_state']）

ALGORITHMS = {
    'Agglomerative': lambda k: AgglomerativeClustering(n_clusters=k, linkage='ward'),
}

COVARIATE_CAT = ['Sex', 'Physical_activity', 'Social_status', 'Smoking', 'Diet_type']
COVARIATE_NUM = ['Age', 'BMI']

# ==========================================
# 用途预设（Configuration Isolation）：两个角色的参数绝不混用
#   'baseline'    基准对照：CLUSTER_SCOPE 强制 'ALL'；预处理与主流水线
#                 （IBS/pipeline/regression_biomarkers.py）100% 对齐——
#                 prevalence=0.10、CLR（同一 resolve_pseudo/clr_transform 约定）、
#                 无 winsorize、无残差化、无批次校正；输出 → figures/baseline/
#   'exploration' 亚型内/子集探索：独立参数（winsorize 等），受样本量功效约束，
#                 结论仅可标注为探索性（论文 §3.5/§4.7）；输出 → figures/exploration/
# 纪律：无监督结构指标（轮廓/CH/DB/Gap/ARI）仅用于簇结构诊断，
#       不得与监督预测指标（OOF R²/balanced accuracy）跨尺度比较；
#       层间对照数字必须经主流水线统一协议重算（论文 §2.4 公平比较协议）。
# ==========================================
PRESET = 'baseline'  # 'baseline' | 'exploration'

PRESETS = {
    'baseline': dict(CLUSTER_SCOPE='ALL', MIN_PREVALENCE=0.10, OUTLIER_METHOD='none'),
    'exploration': dict(CLUSTER_SCOPE='IBS', MIN_PREVALENCE=0.10, OUTLIER_METHOD='winsorize'),
}
_preset_cfg = PRESETS[PRESET]
CLUSTER_SCOPE = _preset_cfg['CLUSTER_SCOPE']
MIN_PREVALENCE = _preset_cfg['MIN_PREVALENCE']
OUTLIER_METHOD = _preset_cfg['OUTLIER_METHOD']
RESIDUALIZE = False
BATCH_CORRECT = 'none'
if PRESET == 'baseline':
    assert CLUSTER_SCOPE == 'ALL', '基准对照必须全量对齐（CLUSTER_SCOPE=ALL）'
FIGURE_DIR = SCRIPT_DIR / 'figures' / PRESET


# ==========================================
# 步骤 1：导入特征数据
# ==========================================
def load_data():
    """载入并合并 IBS 与健康对照的菌群丰度矩阵，返回 (X_df, health_state, covariates)。"""
    ibs = pd.read_csv(IBS_ABUNDANCE, index_col=0)
    health = pd.read_csv(HEALTH_ABUNDANCE, index_col=0)
    if list(ibs.columns) != list(health.columns):
        raise ValueError('两个队列的菌属列不一致，无法直接合并')

    X_df = pd.concat([ibs, health], axis=0).apply(pd.to_numeric, errors='coerce')
    health_state = pd.Series(
        ['IBS'] * len(ibs) + ['Health'] * len(health),
        index=X_df.index, name='health_state',
    )

    stat = pd.read_excel(IBS_STAT).set_index('patient_ID')
    covariates = stat.reindex(X_df.index)
    return X_df, health_state, covariates


def load_gmrepo_data():
    """回退载入 GMrepo 汇总队列（564 样本：379 IBS / 185 Health）。

    仅当 Project-master 原始数据缺失时使用；结果对应 GMrepo 汇总队列，
    不等同于原始 141-IBS 队列，输出仅作无监督结构诊断。
    """
    if not GMREPO_ABUNDANCE.exists() or not GMREPO_METADATA.exists():
        raise FileNotFoundError(
            '缺少输入数据：请恢复 IBS/Project-master/data/raw/ 下的三个原始文件，'
            f'或提供 {GMREPO_ABUNDANCE} 与 {GMREPO_METADATA}。'
        )
    abundance = pd.read_csv(GMREPO_ABUNDANCE, index_col=0)
    meta = pd.read_csv(GMREPO_METADATA).set_index('run_id').reindex(abundance.index)
    if meta['health_state'].isna().any():
        missing = int(meta['health_state'].isna().sum())
        raise ValueError(f'GMrepo 元数据缺少 {missing} 个样本的 health_state，无法分组')

    X_df = abundance.apply(pd.to_numeric, errors='coerce')
    health_state = meta['health_state'].astype(str).rename('health_state')
    covariates = pd.DataFrame({
        'Sex': meta['sex'].replace({'female': 'Female', 'male': 'Male'}),
        'Age': pd.to_numeric(meta['host_age'], errors='coerce'),
        'BMI': pd.to_numeric(meta['BMI'], errors='coerce'),
    }, index=X_df.index)
    covariates.index.name = 'patient_ID'
    return X_df, health_state, covariates


def resolve_pseudo(X, pseudo):
    """解析伪计数配置：None=最小正值/2；'min_rel'=最小正值；float=固定值。"""
    pos = X[X > 0]
    if pseudo is None:
        return float(pos.min() / 2) if pos.size else 1e-6
    if isinstance(pseudo, str) and pseudo == 'min_rel':
        return float(pos.min()) if pos.size else 1e-6
    return float(pseudo)


def clr_transform(X, pseudo=None, method=CLR_METHOD):
    """组成型数据的对数比变换。

    - 'clr' : log(x+p) - mean(log(x+p))；p 越小，0 值变换后越极端，
              容易让 PCA 捕捉“存在/缺失”而非真实丰度差异。
    - 'rclr': 稳健 CLR，只用非零值计算几何均值，不引入伪计数；零值置 0。
    """
    if method == 'rclr':
        out = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            nz = X[i] > 0
            if nz.any():
                g = np.exp(np.log(X[i, nz]).mean())
                out[i, nz] = np.log(X[i, nz] / g)
        return out
    p = resolve_pseudo(X, pseudo)
    log_x = np.log(X + p)
    return log_x - log_x.mean(axis=1, keepdims=True)


def replace_outliers(X_df, low=OUTLIER_LOW, high=OUTLIER_HIGH):
    """按特征分位数 winsorize：把 1%/99% 之外的极端值替换为边界值。

    返回 (替换后的矩阵, 被替换的单元格数)。不删除样本，避免丢失病例。
    """
    lo = X_df.quantile(low)
    hi = X_df.quantile(high)
    clipped = X_df.clip(lower=lo, upper=hi, axis=1)
    n_replaced = int((clipped != X_df).to_numpy().sum())
    return clipped, n_replaced


def build_design(cov_df, cov_cols, model=None):
    """构造协变量设计矩阵 D（含截距）。

    数值列中位数填补；类别列众数填补 + one-hot(drop_first)。
    model 为 None 时拟合填补参数，否则复用同一套参数处理新数据。
    """
    fit = model is None
    n = len(cov_df)
    if fit:
        model = {'cov_cols': list(cov_cols), 'numeric_fill': {},
                 'cat_fill': {}, 'cat_dummies': {}, 'coef': None}
    blocks = []
    for c in model['cov_cols']:
        present = c in cov_df.columns
        if c in model['numeric_fill'] or (
                fit and present and pd.api.types.is_numeric_dtype(cov_df[c])):
            v = pd.to_numeric(cov_df[c], errors='coerce') if present else pd.Series(np.nan, index=range(n))
            if fit:
                model['numeric_fill'][c] = v.median()
            blocks.append(v.fillna(model['numeric_fill'][c]).values.reshape(-1, 1).astype(float))
        else:
            v = cov_df[c].astype('object') if present else pd.Series(np.nan, index=range(n))
            if fit:
                mode = v.mode()
                model['cat_fill'][c] = mode.iloc[0] if not mode.empty else 'NA'
            v = v.fillna(model['cat_fill'][c]).astype(str)
            dm = pd.get_dummies(v, prefix=c, drop_first=True)
            if fit:
                model['cat_dummies'][c] = list(dm.columns)
            else:
                dm = dm.reindex(columns=model['cat_dummies'][c], fill_value=0)
            blocks.append(dm.values.astype(float))
    D = np.hstack(blocks) if blocks else np.empty((n, 0))
    return np.column_stack([np.ones(n), D]), model


def residualize_matrix(X, cov_df, cov_cols):
    """对每个菌属特征回归协变量并取残差，实现协变量效应的正交分离。"""
    D, model = build_design(cov_df, cov_cols)
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    model['coef'] = coef
    return X - D @ coef, model


def apply_residual(X, model, cov_df=None):
    """对新数据套用训练时拟合的残差化；协变量缺失时用训练集中位数/众数。"""
    n = X.shape[0]
    if cov_df is None:
        cov_df = pd.DataFrame(index=range(n))
    D, _ = build_design(cov_df, model['cov_cols'], model)
    return X - D @ model['coef']


def _build_named_design(cov_df, cols, model=None):
    """构造带列名的协变量设计块（数值中位数 + 类别 one-hot），可复用于新数据。

    每列若已在 model 中则复用其填补/哑变量结构；否则在本次数据上拟合。
    这样可以在同一次拟合中先放生物学块、再放批次块。
    """
    n = len(cov_df)
    if model is None:
        model = {'numeric_fill': {}, 'cat_levels': {}, 'dummy_names': {}}
    blocks, names = [], []
    for c in cols:
        present = c in cov_df.columns
        col = cov_df[c] if present else pd.Series(np.nan, index=range(n))
        if c in model['numeric_fill']:
            v = pd.to_numeric(col, errors='coerce')
            blocks.append(v.fillna(model['numeric_fill'][c]).values.reshape(-1, 1).astype(float))
            names.append(c)
        elif c in model['cat_levels']:
            v = col.astype('object').fillna(model['cat_levels'][c]).astype(str)
            dm = pd.get_dummies(v, prefix=c, drop_first=True)
            dm = dm.reindex(columns=model['dummy_names'][c], fill_value=0)
            blocks.append(dm.values.astype(float))
            names.extend(dm.columns.tolist())
        elif present and pd.api.types.is_numeric_dtype(col):
            v = pd.to_numeric(col, errors='coerce')
            model['numeric_fill'][c] = v.median()
            blocks.append(v.fillna(model['numeric_fill'][c]).values.reshape(-1, 1).astype(float))
            names.append(c)
        else:
            v = col.astype('object')
            mode = v.mode()
            model['cat_levels'][c] = mode.iloc[0] if not mode.empty else 'NA'
            v = v.fillna(model['cat_levels'][c]).astype(str)
            dm = pd.get_dummies(v, prefix=c, drop_first=True)
            model['dummy_names'][c] = list(dm.columns)
            blocks.append(dm.values.astype(float))
            names.extend(dm.columns.tolist())
    D = np.hstack(blocks) if blocks else np.empty((n, 0))
    return D, names, model


def fit_mmuphin(X_clr, metadata_df, batch_cols, bio_cols=()):
    """MMUPHin 式联合多变量批次校正（CLR 空间的最小二乘实现）。

    设计矩阵 = [截距, 生物学保护块, 批次块]；拟合后**只扣除批次块**的贡献，
    从而在保留生物学方差的同时对齐不同 project 的特征分布。

    返回 (校正后矩阵, 可复用于新数据的模型)。
    """
    X = np.asarray(X_clr, dtype=float)
    D_bio, bio_names, model = _build_named_design(metadata_df, list(bio_cols))
    D_bat, bat_names, model = _build_named_design(metadata_df, list(batch_cols), model)
    D = np.column_stack([np.ones(len(X)), D_bio, D_bat])
    coefs, *_ = np.linalg.lstsq(D, X, rcond=None)
    start = 1 + D_bio.shape[1]
    corrected = X - D[:, start:] @ coefs[start:]
    model.update({'batch_cols': list(batch_cols), 'bio_cols': list(bio_cols),
                  'batch_names': bat_names, 'bio_names': bio_names, 'coefs': coefs,
                  'batch_start': start})
    print(f'[MMUPHin] n={len(X)}, 保护生物学自由度={D_bio.shape[1]}, '
          f'剥离批次自由度={D_bat.shape[1]}')
    return corrected, model


def apply_mmuphin(X_clr, metadata_df, model):
    """把训练时拟合的 MMUPHin 批次系数应用到新数据（不重拟合）。"""
    X = np.asarray(X_clr, dtype=float)
    if metadata_df is None:
        metadata_df = pd.DataFrame(index=range(len(X)))
    D_bio, _, _ = _build_named_design(metadata_df, model['bio_cols'], model)
    D_bat, _, _ = _build_named_design(metadata_df, model['batch_cols'], model)
    D = np.column_stack([np.ones(len(X)), D_bio, D_bat])
    start = 1 + D_bio.shape[1]
    return X - D[:, start:] @ model['coefs'][start:]


def select_core_taxa(X, cols, method, n_core):
    """按方差或稀疏主成分载荷挑选核心菌属，返回 (列索引, 列名)。"""
    cols = np.asarray(cols)
    if method == 'none':
        return np.arange(len(cols)), list(cols)
    if method == 'variance':
        score = X.var(axis=0)
    elif method == 'spca':
        Xs = StandardScaler().fit_transform(X)
        pre = None
        if Xs.shape[1] > SPCA_MAX_FEATURES:
            pre = np.argsort(-Xs.var(axis=0))[:SPCA_MAX_FEATURES]
            Xs = Xs[:, pre]
        spca = SparsePCA(n_components=min(SPCA_N_COMPONENTS, Xs.shape[1]),
                         alpha=SPCA_ALPHA, method=SPCA_METHOD,
                         max_iter=SPCA_MAX_ITER, tol=SPCA_TOL,
                         random_state=RANDOM_STATE)
        spca.fit(Xs)
        score = np.abs(spca.components_).sum(axis=0)
        if pre is not None:
            full = np.zeros(len(cols))
            full[pre] = score
            score = full
    else:
        raise ValueError(f'未知的 CORE_TAXA_METHOD: {method}')
    n = min(n_core, len(cols))
    idx = np.sort(np.argsort(-score)[:n])
    return idx, list(cols[idx])


def compute_vif(X_df):
    """逐特征对其余特征做线性回归，返回按 VIF 降序排列的序列。

    VIF_j = 1 / (1 - R2_j)，R2_j 为用其余特征预测第 j 个特征的拟合优度。
    要求样本数 > 特征数，否则无法估计（共线性无法被唯一辨识）。
    """
    if X_df.shape[1] >= X_df.shape[0]:
        raise ValueError('VIF 需要样本数大于特征数')
    X = X_df.values.astype(float)
    vif = {}
    for j, col in enumerate(X_df.columns):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        r2 = min(max(r2, 0.0), 1 - 1e-10)
        vif[col] = 1.0 / (1.0 - r2)
    return pd.Series(vif).sort_values(ascending=False)


def report_vif(X_df, title, min_coverage=0.3):
    """打印 VIF 诊断表（用于临床协变量等低维矩阵）。

    临床字段普遍缺失，先保留覆盖率 >= min_coverage 的列，再按均值填补。
    """
    X_df = X_df.loc[:, X_df.notna().mean() >= min_coverage]
    X_df = X_df.loc[:, X_df.std(axis=0) > 0].fillna(X_df.mean())
    if X_df.shape[1] < 2 or X_df.shape[1] >= X_df.shape[0]:
        return
    vif = compute_vif(X_df)
    print(f'\n--- VIF 诊断：{title}（n={X_df.shape[0]}）---')
    for col, val in vif.items():
        flag = '  <-- 高共线' if val >= VIF_THRESHOLD else ''
        print(f'{col:>14}: VIF={val:8.2f}{flag}')


def select_by_vif(X_df, threshold=VIF_THRESHOLD, top_n=VIF_TOP_N):
    """先按方差取 top_n 个候选中，再迭代剔除 VIF 超阈值的特征。

    返回 (保留后的特征矩阵, 被剔除的特征名列表)。
    """
    X_df = X_df.loc[:, X_df.std(axis=0) > 0]
    top_n = min(top_n, X_df.shape[0] - 1)
    if X_df.shape[1] > top_n:
        top = X_df.var(axis=0).sort_values(ascending=False).index[:top_n]
        X_df = X_df[top]
    dropped = []
    while X_df.shape[1] > 2:
        vif = compute_vif(X_df)
        if vif.iloc[0] <= threshold:
            break
        dropped.append(vif.index[0])
        X_df = X_df.drop(columns=[vif.index[0]])
    return X_df, dropped


def fit_features(X_df, covariates=None, *, use_vif=USE_VIF,
                 core_method=CORE_TAXA_METHOD, n_core=N_CORE_TAXA,
                 use_residual=RESIDUALIZE, resid_covariates=RESID_COVARIATES,
                 outlier_method=OUTLIER_METHOD, clr_method=CLR_METHOD,
                 pseudo=PSEUDOCOUNT, batch_correct=BATCH_CORRECT,
                 batch_cols=BATCH_COLS, bio_cols=BIO_COLS):
    """异常值替换 -> 过滤低丰度菌属 -> VIF -> CLR -> 残差化 -> 核心菌属筛选 -> PCA。

    返回 (X_pcs, 可复用的变换流水线)。
    """
    if outlier_method == 'winsorize':
        X_df, n_out = replace_outliers(X_df)
        print(f'异常值替换(winsorize {OUTLIER_LOW:.0%}-{OUTLIER_HIGH:.0%}): {n_out} 个单元格')

    prevalence = (X_df > 0).mean(axis=0)
    X_df = X_df[prevalence[prevalence >= MIN_PREVALENCE].index]

    if use_vif:
        kept, dropped = select_by_vif(X_df)
        print(f'VIF 剔除高共线菌属: {X_df.shape[1]} -> {kept.shape[1]}'
              f'（剔除 {len(dropped)} 个, 阈值={VIF_THRESHOLD}）')
        X_df = kept
    all_cols = list(X_df.columns)

    pseudo_val = resolve_pseudo(X_df.values, pseudo)
    X = clr_transform(X_df.values, pseudo_val, clr_method)

    batch_model = None
    if batch_correct == 'mmuphin' and covariates is not None:
        X, batch_model = fit_mmuphin(X, covariates, batch_cols, bio_cols)

    resid_model = None
    if use_residual and covariates is not None:
        X, resid_model = residualize_matrix(X, covariates, resid_covariates)
        print(f'协变量残差化: 已剥离 {resid_covariates} 的线性效应')
    elif use_residual:
        print('!! 未提供协变量，跳过残差化')

    core_idx, core_cols = select_core_taxa(X, all_cols, core_method, n_core)
    X = X[:, core_idx]
    print(f'核心菌属筛选({core_method}): {len(all_cols)} -> {len(core_cols)}')

    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=min(N_PCS, X.shape[0], X.shape[1]),
              random_state=RANDOM_STATE).fit(scaler.transform(X))
    X_pcs = pca.transform(scaler.transform(X))
    print(f'PCA 前 {X_pcs.shape[1]} 个主成分累计解释方差: '
          f'{pca.explained_variance_ratio_.sum():.1%}')
    return X_pcs, {'all_cols': all_cols, 'core_idx': core_idx, 'cols': core_cols,
                   'pseudo': pseudo_val, 'clr_method': clr_method, 'scaler': scaler,
                   'pca': pca, 'resid_model': resid_model, 'batch_model': batch_model}


def apply_features(X_df, pipe, covariates=None):
    """用已拟合的流水线变换新数据（如健康对照）。"""
    X = clr_transform(X_df[pipe['all_cols']].values, pipe['pseudo'],
                      pipe.get('clr_method', 'clr'))
    if pipe.get('batch_model') is not None:
        X = apply_mmuphin(X, covariates, pipe['batch_model'])
    if pipe['resid_model'] is not None:
        X = apply_residual(X, pipe['resid_model'], covariates)
    X = X[:, pipe['core_idx']]
    return pipe['pca'].transform(pipe['scaler'].transform(X))


# ==========================================
# 步骤 2：多算法聚类比较
# ==========================================
def evaluate_algorithms(X):
    """遍历算法与簇数，返回 {算法: {k: (轮廓系数, 标签)}}。"""
    results = {}
    for name, factory in ALGORITHMS.items():
        results[name] = {}
        for k in K_RANGE:
            labels = factory(k).fit_predict(X)
            score = silhouette_score(X, labels)
            results[name][k] = (score, labels)
            print(f'{name:>14} | K={k} | 平均轮廓系数 = {score:.4f}')
    return results


def select_best(results):
    """按轮廓系数选出最优 (算法, k, 分数, 标签)。"""
    return max(
        ((name, k, score, labels)
         for name, ks in results.items() for k, (score, labels) in ks.items()),
        key=lambda t: t[2],
    )


def bh_fdr(p):
    """Benjamini-Hochberg FDR 校正，返回 q 值。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def cluster_metrics(X, labels):
    """轮廓（越高越好）、Calinski-Harabasz（越高越好）、Davies-Bouldin（越低越好）。"""
    return {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
    }


def _dispersion(X, labels):
    w = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        w += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return w


def gap_statistic(X, k_range, n_refs=N_METRIC_REFS):
    """Gap statistic（越高越好），用于评估最优簇数。"""
    rng = np.random.default_rng(RANDOM_STATE)
    lo, hi = X.min(axis=0), X.max(axis=0)
    gaps, sks = [], []
    for k in k_range:
        lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
        log_w = np.log(_dispersion(X, lab) + 1e-12)
        refs = []
        for _ in range(n_refs):
            Xr = rng.uniform(lo, hi, size=X.shape)
            lr = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xr)
            refs.append(np.log(_dispersion(Xr, lr) + 1e-12))
        gaps.append(float(np.mean(refs) - log_w))
        sks.append(float(np.std(refs) * np.sqrt(1 + 1.0 / n_refs)))
    return np.array(gaps), np.array(sks)


def evaluate_k_metrics(X, k_range=K_RANGE):
    """综合轮廓、CH、DB、Gap 评估每个 K（几何结构 + 稳定性视角）。"""
    rows, labels_by_k = [], {}
    gaps, sks = gap_statistic(X, k_range)
    for i, k in enumerate(k_range):
        lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
        labels_by_k[k] = lab
        m = cluster_metrics(X, lab)
        rows.append({'K': k, **m, 'gap': gaps[i], 'gap_sk': sks[i]})
        print(f'K={k} | silhouette={m["silhouette"]:.4f} | '
              f'CH={m["calinski_harabasz"]:.1f} | DB={m["davies_bouldin"]:.3f} | '
              f'gap={gaps[i]:.3f}')
    df = pd.DataFrame(rows)

    def norm(s, higher_better=True):
        span = s.max() - s.min()
        z = (s - s.min()) / span if span > 0 else np.ones(len(s))
        return z if higher_better else 1 - z

    df['composite'] = np.mean([
        norm(df['silhouette']), norm(df['calinski_harabasz']),
        norm(df['davies_bouldin'], higher_better=False), norm(df['gap']),
    ], axis=0)
    return df, labels_by_k


def select_best_metric(df):
    """按综合得分选 K。"""
    best = df.loc[df['composite'].idxmax()]
    return int(best['K']), float(best['composite'])


# ==========================================
# 步骤 2b：计数空间模型 DMM
# ==========================================
def rarefy_counts(X_df, depth=DMM_TARGET_DEPTH, seed=RANDOM_STATE):
    """把相对丰度矩阵按比例稀释为固定深度的整数计数矩阵。"""
    P = np.clip(np.asarray(X_df, dtype=float), 0, None)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    if depth is None:
        return pd.DataFrame(P, index=X_df.index, columns=X_df.columns)
    rng = np.random.default_rng(seed)
    C = np.vstack([rng.multinomial(int(depth), p) for p in P]).astype(np.int64)
    cols = np.asarray(X_df.columns)[C.sum(axis=0) > 0]
    return pd.DataFrame(C[:, C.sum(axis=0) > 0], index=X_df.index, columns=cols)


def _dmm_component_loglik(counts, alpha):
    """对称 Dirichlet 分量的对数边际似然（省略与 k 无关的常数）。"""
    n, D = counts.shape
    N = counts.sum(axis=1)
    return (gammaln(alpha * D) - gammaln(N + alpha * D)
            + (gammaln(counts + alpha) - gammaln(alpha)).sum(axis=1))


def fit_dmm(counts, K, n_init=DMM_N_INIT, max_iter=DMM_MAX_ITER,
            tol=DMM_TOL, seed=RANDOM_STATE):
    """EM 拟合 DMM，返回最优初始化的结果字典。

    counts: 样本 × 分类单元的整数计数矩阵（非负）。
    采用对称 Dirichlet 浓度 alpha（每个分量一个标量），与 DMM 包一致。
    """
    counts = np.asarray(counts)
    n, D = counts.shape
    Xc = np.log(counts + 0.5)
    Xc -= Xc.mean(axis=1, keepdims=True)

    best = None
    for init in range(max(1, n_init)):
        if K == 1:
            r = np.ones((n, 1))
        else:
            lab = AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(Xc)
            r = np.full((n, K), 0.05)
            r[np.arange(n), lab] += 0.9
            r += np.random.default_rng(seed + init).uniform(0, 0.02, r.shape)
            r /= r.sum(axis=1, keepdims=True)

        prev = -np.inf
        ll = -np.inf
        for _ in range(max_iter):
            pi = r.mean(axis=0)
            alpha = np.empty(K)
            for k in range(K):
                def neg(a):
                    return -(r[:, k] * _dmm_component_loglik(counts, float(a))).sum()
                alpha[k] = minimize_scalar(neg, bounds=(1e-3, 1e3), method='bounded',
                                           options={'xatol': 1e-4}).x
            loglik = np.column_stack([_dmm_component_loglik(counts, alpha[k])
                                      for k in range(K)])
            logw = np.log(pi + 1e-12) + loglik
            log_norm = logsumexp(logw, axis=1, keepdims=True)
            r = np.exp(logw - log_norm)
            ll = float(log_norm.sum())
            if np.isfinite(prev) and abs(ll - prev) < tol * max(1.0, abs(prev)):
                break
            prev = ll

        if best is None or ll > best['loglik']:
            best = {'loglik': ll, 'pi': pi, 'alpha': alpha, 'resp': r,
                    'labels': r.argmax(axis=1)}

    best['BIC'] = -2.0 * best['loglik'] + (2 * K - 1) * np.log(n)
    return best


def align_dmm_by_size(res):
    """按簇大小升序重排 DMM 分量，得到确定性的 S0/S1/S2... 编号。"""
    labels = np.asarray(res['labels'])
    sizes = np.bincount(labels, minlength=res['resp'].shape[1])
    order = np.argsort(sizes)
    remap = {old: new for new, old in enumerate(order)}
    return {**res,
            'labels': np.array([remap[int(x)] for x in labels]),
            'resp': res['resp'][:, order],
            'alpha': np.asarray(res['alpha'])[order],
            'pi': np.asarray(res['pi'])[order]}


def evaluate_dmm(counts, k_range=K_RANGE):
    """K=1..K 拟合 DMM，按 BIC（越小越好）选 K；返回 (指标表, 最优 K, 各 K 结果)。"""
    rows, models = [], {}
    for K in k_range:
        res = align_dmm_by_size(fit_dmm(counts, K))
        models[K] = res
        rows.append({'K': K, 'loglik': res['loglik'], 'BIC': res['BIC'],
                     'alpha': ';'.join(f'{a:.3f}' for a in res['alpha']),
                     'pi': ';'.join(f'{p:.3f}' for p in res['pi'])})
        print(f'DMM K={K} | loglik={res["loglik"]:.1f} | BIC={res["BIC"]:.1f}')
    df = pd.DataFrame(rows)
    best_k = int(df.loc[df['BIC'].idxmin(), 'K'])
    return df, best_k, models


def dmm_predict_proba(counts, model):
    """把已拟合的 DMM 应用到新计数矩阵，返回后验概率（软分配）。"""
    counts = np.asarray(counts)
    K = len(model['alpha'])
    loglik = np.column_stack([_dmm_component_loglik(counts, model['alpha'][k])
                              for k in range(K)])
    logw = np.log(np.asarray(model['pi']) + 1e-12) + loglik
    return np.exp(logw - logsumexp(logw, axis=1, keepdims=True))


def evaluate_all_models(X_pcs=None, counts=None, k_range=K_RANGE):
    """统一评估：特征空间（Agglomerative 多指标）+ 计数空间（DMM/BIC）。"""
    out = {}
    if X_pcs is not None:
        print('\n--- 特征空间聚类（Agglomerative）---')
        df, labels_by_k = evaluate_k_metrics(X_pcs, k_range)
        k, comp = select_best_metric(df)
        out['feature'] = {'table': df, 'best_k': k, 'composite': comp,
                          'labels_by_k': labels_by_k}
        print(f'特征空间综合最优 K={k}（composite={comp:.3f}）')

    if counts is not None:
        print('\n--- 计数空间模型（DMM）---')
        df, best_k, models = evaluate_dmm(counts, k_range)
        out['dmm'] = {'table': df, 'best_k': best_k, 'models': models}
        print(f'DMM 最优 K={best_k}（BIC 最小）')

    return out


# ==========================================
# 步骤 3：Bootstrap 稳定性检验
# ==========================================
def stability_score(X_raw, reference, cluster_factory, feature_fn, covariates=None):
    """Bootstrap 稳定性：每轮对原始样本重采样并**完整重跑预处理流水线**。

    与旧实现（只在已降维的 X_pcs 上重采样）不同，这里每轮都重新做
    CLR -> 筛选/SPCA -> 标准化 -> PCA，避免全量样本统计信息泄漏进重采样，
    从而给出更保守、无偏的 ARI 估计。

    参数
    - X_raw:          原始丰度矩阵（样本 × 特征，未降维）
    - reference:      全量数据得到的参考聚类标签
    - cluster_factory: 返回未拟合聚类器（给定 K）
    - feature_fn:     (X_raw_boot, cov_boot) -> (X_pcs_boot, pipe)
    - covariates:     与 X_raw 行对齐的协变量（可选）
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_raw)
    scores = []
    failed = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            boot_cov = covariates.iloc[idx] if covariates is not None else None
            with contextlib.redirect_stdout(io.StringIO()):
                X_boot, _ = feature_fn(X_raw.iloc[idx], boot_cov)
            boot_labels = cluster_factory().fit_predict(X_boot)
        except Exception:
            failed += 1
            continue
        first_pos = {}
        for pos, sample_i in enumerate(idx):
            first_pos.setdefault(int(sample_i), pos)
        uniq = np.array(list(first_pos))
        boot_uniq = boot_labels[[first_pos[i] for i in uniq]]
        scores.append(adjusted_rand_score(reference[uniq], boot_uniq))
    if failed:
        print(f'  (Bootstrap 中有 {failed}/{N_BOOTSTRAP} 轮失败，已跳过)')
    return float(np.mean(scores)), float(np.std(scores))


# ==========================================
# 步骤 4：生物学验证
# ==========================================
def validate_vs_covariates(labels, covariates):
    """检验亚型与临床协变量是否相关；所有 p 值统一做 BH-FDR 校正。"""
    labels = np.asarray(labels)
    clu = pd.Series(labels, index=covariates.index, name='subtype')
    tests = []

    for col in COVARIATE_CAT:
        if col not in covariates or covariates[col].notna().sum() == 0:
            continue
        table = pd.crosstab(clu, covariates[col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        stat, p, _, _ = chi2_contingency(table)
        tests.append({'variable': col, 'test': 'chi2', 'stat': stat, 'p': p, 'table': table})

    for col in COVARIATE_NUM:
        if col not in covariates:
            continue
        groups, means = [], {}
        for c in np.unique(labels):
            vals = covariates[col].values[labels == c].astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 1:
                groups.append(vals)
                means[int(c)] = round(float(np.mean(vals)), 2)
        if len(groups) > 1:
            stat, p = kruskal(*groups)
            tests.append({'variable': col, 'test': 'kruskal', 'stat': stat, 'p': p,
                          'means': means})

    if not tests:
        print('  （无可用协变量）')
        return pd.DataFrame()

    q = bh_fdr([t['p'] for t in tests])
    print('\n--- 亚型 × 临床协变量（p 经 BH-FDR 校正）---')
    for t, qi in zip(tests, q):
        print(f"{t['variable']:<16s} {t['test']:<7s} stat={t['stat']:8.3f} "
              f"p={t['p']:.3g}  q={qi:.3g}")
        if 'table' in t:
            print(t['table'])
        if 'means' in t:
            print(f"  各亚型均值: {t['means']}")
    return pd.DataFrame([{'variable': t['variable'], 'test': t['test'],
                          'stat': t['stat'], 'p': t['p'], 'q': qi}
                         for t, qi in zip(tests, q)])


def subtype_vs_health(X_ibs_pcs, labels, health_pcs):
    """计算各 IBS 亚型到健康群体中心的距离，判断是否有亚型更接近健康态。"""
    centroid = health_pcs.mean(axis=0)
    dist = np.linalg.norm(X_ibs_pcs - centroid, axis=1)
    print('\n--- 各亚型到健康群体中心的平均距离（越小越接近健康）---')
    for c in np.unique(labels):
        print(f'亚型 {c}: {dist[labels == c].mean():.3f} '
              f'(n={np.sum(labels == c)})')


# ==========================================
# 可视化
# ==========================================
def plot_silhouette(X, labels, score, title, out_path):
    n_clusters = len(np.unique(labels))
    vals = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    for i in range(n_clusters):
        ith = np.sort(vals[labels == i])
        y_upper = y_lower + ith.shape[0]
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * ith.shape[0], str(i))
        y_lower = y_upper + 10
    ax.axvline(x=score, color='red', linestyle='--', label=f'Average: {score:.2f}')
    ax.set_title(title)
    ax.set_xlabel('The silhouette coefficient values')
    ax.set_ylabel('Cluster label')
    ax.legend(loc='upper right')
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_pca(X_pcs, color, title, out_path, categorical=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        for c in np.unique(color):
            m = np.asarray(color) == c
            ax.scatter(X_pcs[m, 0], X_pcs[m, 1], s=25, alpha=0.8, label=str(c))
        ax.legend(title='group')
    else:
        sc = ax.scatter(X_pcs[:, 0], X_pcs[:, 1], c=color, s=25, cmap='viridis')
        fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


# ==========================================
# 主流程
# ==========================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[配置隔离] PRESET={PRESET} | scope={CLUSTER_SCOPE} | '
          f'prevalence={MIN_PREVALENCE} | outlier={OUTLIER_METHOD} | 输出 → {FIGURE_DIR}')
    if PRESET == 'exploration':
        print('[警告] exploration 结果仅作探索性分析（功效约束），不得进入基准对照表。')

    legacy_ready = (IBS_ABUNDANCE.exists() and HEALTH_ABUNDANCE.exists()
                    and IBS_STAT.exists())
    if legacy_ready:
        X_df, health_state, covariates = load_data()
        data_source = 'Project-master/data/raw（原始队列）'
    else:
        print('[数据回退] 未找到 IBS/Project-master/data/raw 原始数据，'
              '改用 IBS/gmrepo 汇总队列（379 IBS / 185 Health）。')
        print('[数据回退] 该结果不等同于原始 141-IBS 队列，仅用于无监督结构诊断。')
        X_df, health_state, covariates = load_gmrepo_data()
        data_source = 'IBS/gmrepo（汇总队列回退）'
        (FIGURE_DIR / 'DATA_SOURCE.txt').write_text(
            '数据源: IBS/gmrepo/abundance_genus.csv + gmrepo_metadata.csv\n'
            '样本: 564（379 IBS / 185 Health）\n'
            '说明: 原始 Project-master/141-IBS 队列缺失时的回退数据；\n'
            '      结果仅用于无监督结构诊断，不得作为论文主结果。\n',
            encoding='utf-8')
    print(f'数据源: {data_source}')
    print(f'总样本数: {X_df.shape[0]} | 原始菌属数: {X_df.shape[1]}')

    ibs_mask = (health_state == 'IBS').values
    is_ibs_only = CLUSTER_SCOPE == 'IBS'
    sub_mask = ibs_mask if is_ibs_only else np.ones(len(health_state), dtype=bool)

    sub_covariates = covariates.loc[sub_mask]
    X_pcs, pipe = fit_features(X_df.loc[sub_mask], sub_covariates)
    report_vif(sub_covariates.select_dtypes('number'), '临床数值协变量')

    print('\n--- 多指标评估不同 K（轮廓 / Calinski-Harabasz / Davies-Bouldin / Gap）---')
    metrics, labels_by_k = evaluate_k_metrics(X_pcs)
    k, composite = select_best_metric(metrics)
    algo = 'Agglomerative'
    labels = labels_by_k[k]
    score = float(metrics.loc[metrics['K'] == k, 'silhouette'].iloc[0])
    print(f'\n综合最优 K={k}（composite={composite:.3f}, silhouette={score:.4f}）')

    mean_ari, std_ari = stability_score(
        X_df.loc[sub_mask], labels, lambda: ALGORITHMS[algo](k),
        lambda df, cov: fit_features(df, cov), covariates=sub_covariates)
    print(f'Bootstrap 稳定性 (ARI): {mean_ari:.3f} ± {std_ari:.3f}')

    validate_vs_covariates(labels, sub_covariates)
    if is_ibs_only:
        health_pcs = apply_features(X_df.loc[~ibs_mask], pipe)
        subtype_vs_health(X_pcs, labels, health_pcs)
    else:
        print(f'\n聚类 × 疾病状态的 ARI = '
              f'{adjusted_rand_score(health_state.values, labels):.4f}')

    plot_silhouette(X_pcs, labels, score,
                    f'Silhouette ({algo}, K={k})',
                    FIGURE_DIR / 'silhouette.png')
    plot_pca(X_pcs, labels, f'PCA colored by subtype ({algo}, K={k})',
             FIGURE_DIR / 'pca_subtype.png')
    print(f'\n图已保存至: {FIGURE_DIR}')
    plt.show()


if __name__ == '__main__':
    main()
