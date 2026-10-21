"""定量表型回归生物标志物挖掘（如 Bristol 评分）。

输入: --abundance --metadata --target [--group --folds --min-prevalence --model]
流程: 低检出过滤 -> CLR -> 标准化 -> 嵌套 KFold/GroupKFold (Ridge/ElasticNet)
     -> 单变量 Spearman + BH-FDR
输出: cv_results.csv, predictions.csv, biomarkers.csv, scatter.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supervised_biomarkers import build_features, load_table, bh_fdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abundance', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--min-prevalence', type=float, default=0.10)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--group', default=None)
    ap.add_argument('--model', default='ridge', choices=['ridge', 'elasticnet'])
    ap.add_argument('--transform', default='clr', choices=['clr', 'log', 'none'],
                    help='clr 适合组成型丰度；log 适合非组成型强度；none 用于已预先变换的输入')
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    abund = load_table(Path(args.abundance))
    meta = load_table(Path(args.metadata))
    common = abund.index.intersection(meta.index)
    abund, meta = abund.loc[common], meta.loc[common]
    y_all = pd.to_numeric(meta[args.target], errors='coerce')
    mask = y_all.notna().values
    abund, meta, y = abund[mask], meta[mask], y_all.values[mask]
    group = meta[args.group].astype(str).values if args.group else None

    print(f'样本={len(abund)} | 目标={args.target} (mean={y.mean():.2f}, sd={y.std():.2f})',
          flush=True)
    if group is not None:
        print(f'分组 CV: {len(set(group))} 个受试者', flush=True)

    X, taxa = build_features(abund, args.min_prevalence, args.transform)
    print(f'特征={X.shape[1]} | folds={args.folds} | transform={args.transform}', flush=True)

    if args.model == 'ridge':
        est = Ridge()
        grid = {'reg__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    else:
        est = ElasticNet(max_iter=20000)
        grid = {'reg__alpha': [0.01, 0.1, 1.0, 10.0],
                'reg__l1_ratio': [0.2, 0.5, 0.8]}
    pipe = Pipeline([('reg', est)])

    if group is not None:
        splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=group))
        inner = GroupKFold(n_splits=3)
    else:
        splits = list(KFold(n_splits=args.folds, shuffle=True,
                            random_state=42).split(X, y))
        inner = KFold(n_splits=3, shuffle=True, random_state=42)

    rows, preds = [], np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(splits, 1):
        gs = GridSearchCV(pipe, grid, cv=inner, scoring='r2', n_jobs=1)
        if group is not None:
            gs.fit(X[tr], y[tr], groups=group[tr])
        else:
            gs.fit(X[tr], y[tr])
        p = gs.predict(X[te])
        preds[te] = p
        r2 = r2_score(y[te], p)
        rho = spearmanr(y[te], p).statistic
        rows.append({'fold': fold, 'best_params': gs.best_params_, 'r2': r2,
                     'spearman': rho, 'mae': mean_absolute_error(y[te], p)})
        print(f'  fold {fold}: R2={r2:.3f} rho={rho:.3f}', flush=True)

    cv = pd.DataFrame(rows)
    cv.to_csv(out / 'cv_results.csv', index=False)
    print('\nCV 汇总:', flush=True)
    print(cv[['r2', 'spearman', 'mae']].agg(['mean', 'std']).round(3).to_string(), flush=True)
    r2_oof = r2_score(y, preds)
    rho_oof = spearmanr(y, preds).statistic
    print(f'合并 OOF: R2={r2_oof:.3f} Spearman={rho_oof:.3f}', flush=True)
    pd.DataFrame({'observed': y, 'predicted': preds}, index=meta.index)\
        .to_csv(out / 'predictions.csv')

    brows = []
    for j, t in enumerate(taxa):
        rho_j, p_j = spearmanr(X[:, j], y)
        brows.append({'taxon': t, 'spearman': rho_j, 'p': p_j})
    b = pd.DataFrame(brows)
    b['q'] = bh_fdr(b['p'].values)
    b = b.sort_values('q')
    b.to_csv(out / 'biomarkers.csv', index=False)
    sig = b[b['q'] < 0.05]
    print(f'单变量显著 taxa (q<0.05): {len(sig)}/{len(b)}', flush=True)
    print(sig.head(15).round(4).to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, preds, s=12, alpha=.5)
    lo = min(y.min(), preds.min()) - .3
    hi = max(y.max(), preds.max()) + .3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted (OOF)')
    ax.set_title(f'{args.target}: R2={r2_oof:.3f}, Spearman={rho_oof:.3f}')
    fig.tight_layout()
    fig.savefig(out / 'scatter.png', dpi=150)
    print('输出目录:', out, flush=True)


if __name__ == '__main__':
    main()
