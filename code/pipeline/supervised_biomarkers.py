"""有监督表型驱动生物标志物挖掘。

输入：
  --abundance  样本 × taxa 的相对丰度/计数表（CSV/TSV，首列样本 ID）
  --metadata   样本元数据（CSV/TSV，首列样本 ID；含 --label 指定列）
  --label      监督标签列（如 Bristol 分级、IBS 亚型、health_state）
  --task       binary | multiclass（默认按标签类别数自动判断）

流程：
  低检出过滤 -> CLR -> 标准化 -> 嵌套交叉验证（LogisticRegression, L1/L2）
  -> 因果解释：|coef| 重要性 + 单变量 log-ratio 检验（BH-FDR）
输出：
  cv_results.csv, biomarkers.csv, roc_pr.png, importance.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from scipy.stats import mannwhitneyu, kruskal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS


def load_table(path):
    sep = '\t' if path.suffix.lower() in ('.tsv', '.txt') else ','
    return pd.read_csv(path, sep=sep, index_col=0, low_memory=False)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def build_features(abund, min_prev, transform='clr'):
    P = abund.values.astype(float)
    keep = (P > 0).mean(axis=0) >= min_prev
    cols = np.asarray(abund.columns)[keep]
    if transform == 'clr':
        pseudo = IBS.resolve_pseudo(P, IBS.PSEUDOCOUNT)
        X = IBS.clr_transform(P, pseudo, IBS.CLR_METHOD)[:, keep]
    elif transform == 'log':
        X = np.log1p(np.clip(P, 0, None))[:, keep]
    elif transform == 'none':
        X = np.clip(P, 0, None)[:, keep]
    else:
        raise ValueError(f'unknown transform: {transform}')
    Z = StandardScaler().fit_transform(X)
    return Z, list(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abundance', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--min-prevalence', type=float, default=0.10)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--group', default=None,
                    help='受试者 ID 列名；提供时用 StratifiedGroupKFold 防止同一受试者跨折泄漏')
    ap.add_argument('--transform', default='clr', choices=['clr', 'log', 'none'],
                    help='clr 适合组成型丰度；log 适合非组成型强度；none 用于已预先变换的输入')
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    abund = load_table(Path(args.abundance))
    meta = load_table(Path(args.metadata))
    common = abund.index.intersection(meta.index)
    abund, meta = abund.loc[common], meta.loc[common]
    y_raw = meta[args.label].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    binary = len(classes) == 2
    print(f'样本={len(common)} | 标签={classes} | 任务={"binary" if binary else "multiclass"}')

    X, taxa = build_features(abund, args.min_prevalence, args.transform)
    print(f'特征={X.shape[1]} | CV folds={args.folds} | transform={args.transform}')

    # ---------- 嵌套交叉验证（可按受试者分组） ----------
    group = meta[args.group].astype(str).values if args.group else None
    if binary:
        solver, grid = 'liblinear', {'clf__C': [0.01, 0.1, 1.0, 10.0],
                                     'clf__penalty': ['l1', 'l2']}
    else:
        # liblinear 不支持多分类；lbfgs 支持多项 logistic
        solver, grid = 'lbfgs', {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    if group is not None:
        outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                     random_state=IBS.RANDOM_STATE)
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=IBS.RANDOM_STATE)
        splits = outer.split(X, y, groups=group)
        print(f'按受试者分组 CV: {len(set(group))} 个受试者')
    else:
        outer = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                random_state=IBS.RANDOM_STATE)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=IBS.RANDOM_STATE)
        splits = outer.split(X, y)
    pipe = Pipeline([('clf', LogisticRegression(max_iter=5000, class_weight='balanced',
                                                solver=solver))])
    scoring = 'roc_auc' if binary else 'balanced_accuracy'

    rows, probs, trues = [], [], []
    for fold, (tr, te) in enumerate(splits, 1):
        gs = GridSearchCV(pipe, grid, cv=inner, scoring=scoring, n_jobs=1)
        if group is not None:
            gs.fit(X[tr], y[tr], groups=group[tr])
        else:
            gs.fit(X[tr], y[tr])
        pred = gs.predict(X[te])
        if binary:
            pr = gs.predict_proba(X[te])[:, 1]
            auc_v = roc_auc_score(y[te], pr)
            probs.append(pr); trues.append(y[te])
        else:
            auc_v = np.nan
        rows.append({'fold': fold, 'best_params': gs.best_params_,
                     'balanced_acc': balanced_accuracy_score(y[te], pred),
                     'f1_macro': f1_score(y[te], pred, average='macro'),
                     'roc_auc': auc_v})
        print(f'  fold {fold}: {rows[-1]}')

    cv = pd.DataFrame(rows)
    cv.to_csv(out / 'cv_results.csv', index=False)
    print('\nCV 汇总:')
    print(cv[['balanced_acc', 'f1_macro', 'roc_auc']].agg(['mean', 'std']).round(3).to_string())

    # ---------- 最终模型 + 重要性 ----------
    gs = GridSearchCV(pipe, grid, cv=inner, scoring=scoring, n_jobs=1)
    if group is not None:
        gs.fit(X, y, groups=group)
    else:
        gs.fit(X, y)
    coef = gs.best_estimator_.named_steps['clf'].coef_
    importance = np.abs(coef).max(axis=0) if coef.ndim > 1 else np.abs(coef)
    imp = pd.Series(importance, index=taxa, name='abs_coef').sort_values(ascending=False)

    # ---------- 单变量 log-ratio 差异检验 (BH) ----------
    pvals, effects = [], []
    for j, t in enumerate(taxa):
        groups = [X[y == c, j] for c in np.unique(y)]
        if binary:
            _, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            eff = float(groups[1].mean() - groups[0].mean())
        else:
            _, p = kruskal(*groups)
            eff = float(np.max([g.mean() for g in groups]) - np.min([g.mean() for g in groups]))
        pvals.append(p); effects.append(eff)
    q = bh_fdr(pvals)
    da = pd.DataFrame({'taxon': taxa, 'clr_effect': effects, 'p': pvals, 'q': q})
    da = da.merge(imp.rename('abs_coef'), left_on='taxon', right_index=True, how='left')
    da = da.sort_values(['q', 'abs_coef'], ascending=[True, False])
    da.to_csv(out / 'biomarkers.csv', index=False)
    print('\nTop biomarkers (q<0.05):')
    print(da[da['q'] < 0.05].head(15).to_string(index=False))
    print(f"\n显著 taxa: {int((da['q'] < 0.05).sum())}/{len(da)}")

    # ---------- 图 ----------
    if binary:
        fpr, tpr, _ = roc_curve(np.concatenate(trues), np.concatenate(probs))
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, lw=2, label=f'AUC={auc(fpr, tpr):.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('Nested-CV ROC')
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out / 'roc_pr.png', dpi=150); plt.close(fig)

    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color='steelblue')
    ax.set_xlabel('|coef| (L1/L2 logistic)'); ax.set_title('Top biomarker taxa')
    fig.tight_layout(); fig.savefig(out / 'importance.png', dpi=150); plt.close(fig)
    print(f'\n输出目录: {out}')


if __name__ == '__main__':
    main()
