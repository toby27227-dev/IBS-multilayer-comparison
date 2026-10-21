"""P0 revision: three-class and IBS-SSS uncertainty / permutation tests.

Outputs:
  p0_classification.csv      per-layer OOF balanced accuracy, macro-F1, fold sd,
                             subject-cluster bootstrap 95% CI, permutation p, confusion matrix
  p0_classification_oof.csv  sample-level OOF predictions

Protocol matches the main analysis: StratifiedGroupKFold(5 outer / 3 inner, seed=42);
multiclass L2 logistic (C grid, class_weight=balanced); binary L1/L2 logistic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'IBS' / 'pipeline'))
sys.path.insert(0, str(ROOT / 'IBS' / 'cellomics'))
from supervised_biomarkers import build_features, load_table  # noqa: E402

SEED = 42
CE = ROOT / 'IBS' / 'cellomics'
GM = ROOT / 'IBS' / 'gmrepo'
OUT = ROOT / 'IBS' / 'revision_p0'
OUT.mkdir(exist_ok=True)

LAYERS = [
    ('NMR', CE / 'nmr.csv', 0.10, 'log'),
    ('Genus', GM / 'abundance_genus.csv', 0.10, 'clr'),
    ('Species', GM / 'abundance_species.csv', 0.10, 'clr'),
    ('Bile acids', CE / 'bile_acids.csv', 0.10, 'log'),
    ('Tryptophan', CE / 'tryptophan.csv', 0.10, 'log'),
    ('KO', CE / 'ko_stool.csv', 0.30, 'clr'),
    ('KEGG', CE / 'kegg_pathways_ko.csv', 0.50, 'clr'),
    ('NMR+Genus', CE / 'combined_genus_nmr.csv', 0.0, 'none'),
]


def make_pipeline(binary):
    if binary:
        solver, grid = 'liblinear', {'clf__C': [0.01, 0.1, 1.0, 10.0],
                                     'clf__penalty': ['l1', 'l2']}
    else:
        solver, grid = 'lbfgs', {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    pipe = Pipeline([('clf', LogisticRegression(max_iter=5000,
                                                class_weight='balanced',
                                                solver=solver))])
    return pipe, grid, solver


def run_oof(X, y, groups, binary, fixed_c=None):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), dtype=int)
    prob = np.full(len(y), np.nan)
    fold_bacc = []
    if fixed_c is not None:
        solver = 'liblinear' if binary else 'lbfgs'
        for tr, te in outer.split(X, y, groups=groups):
            clf = LogisticRegression(max_iter=500, class_weight='balanced',
                                     solver=solver, C=fixed_c)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
            if binary:
                prob[te] = clf.predict_proba(X[te])[:, 1]
            fold_bacc.append(balanced_accuracy_score(y[te], pred[te]))
        return pred, prob, np.asarray(fold_bacc)

    pipe, grid, _ = make_pipeline(binary)
    scoring = 'roc_auc' if binary else 'balanced_accuracy'
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for tr, te in outer.split(X, y, groups=groups):
        gs = GridSearchCV(pipe, grid, cv=inner, scoring=scoring, n_jobs=1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        pred[te] = gs.predict(X[te])
        if binary:
            prob[te] = gs.predict_proba(X[te])[:, 1]
        fold_bacc.append(balanced_accuracy_score(y[te], pred[te]))
    return pred, prob, np.asarray(fold_bacc)


def boot_ci(y, pred, groups, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_subject = {s: np.where(groups == s)[0] for s in uniq}
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_subject[s] for s in samp])
        vals.append(balanced_accuracy_score(y[idx], pred[idx]))
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def subject_perm_p(X, y, groups, observed, binary, n_perm, seed=SEED, fixed_c=None):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    subj_label = {s: int(np.bincount(y[groups == s]).argmax()) for s in uniq}
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(uniq)
        yp = y.copy()
        for s, t in zip(uniq, perm):
            yp[groups == s] = subj_label[t]
        pred, _, _ = run_oof(X, yp, groups, binary, fixed_c=fixed_c)
        null.append(balanced_accuracy_score(yp, pred))
    null = np.asarray(null)
    p = (1 + int(np.sum(null >= observed))) / (1 + len(null))
    return p, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', default='all')
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--task', choices=['cohort', 'severity'], default='cohort')
    ap.add_argument('--fixed-c-perm', type=float, default=None,
                    help='use fixed C in permutation loop (skip nested tuning)')
    args = ap.parse_args()

    if args.task == 'cohort':
        meta = load_table(CE / 'sample_map.csv')
        label_col = 'Cohort_label'
    else:
        meta = load_table(GM / 'metadata_severity.csv')
        label_col = 'severity_bin'

    rows = []
    oof_frames = []
    for name, path, prev, transform in LAYERS:
        if args.layer != 'all' and name != args.layer:
            continue
        abund = load_table(path)
        common = abund.index.intersection(meta.index)
        abund, md = abund.loc[common], meta.loc[common]
        y_raw = md[label_col].astype(str).values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        groups = md['subject'].astype(str).values
        binary = len(le.classes_) == 2
        X, feats = build_features(abund, prev, transform)
        print(f'=== {name} | n={len(y)} subjects={len(np.unique(groups))} '
              f'features={X.shape[1]} classes={list(le.classes_)} ===', flush=True)

        pred, prob, fold_bacc = run_oof(X, y, groups, binary)
        bacc = float(balanced_accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average='macro'))
        lo, hi = boot_ci(y, pred, groups, args.n_boot)
        cm = confusion_matrix(y, pred, labels=np.arange(len(le.classes_)))
        auc = float(roc_auc_score(y, prob)) if binary else float('nan')
        pval, null = subject_perm_p(X, y, groups, bacc, binary, args.n_perm,
                                    fixed_c=args.fixed_c_perm)
        print(f'  bacc={bacc:.4f} (fold {fold_bacc.mean():.4f}±{fold_bacc.std(ddof=1):.4f}) '
              f'F1={f1:.4f} CI=[{lo:.4f},{hi:.4f}] perm_p={pval:.4f}', flush=True)

        rows.append({
            'task': args.task, 'layer': name, 'n_samples': len(y),
            'n_subjects': int(len(np.unique(groups))), 'n_features': X.shape[1],
            'classes': '|'.join(le.classes_),
            'bacc_pooled': round(bacc, 4),
            'bacc_fold_mean': round(float(fold_bacc.mean()), 4),
            'bacc_fold_sd': round(float(fold_bacc.std(ddof=1)), 4),
            'macro_f1': round(f1, 4), 'auc': round(auc, 4) if binary else '',
            'bacc_ci_lo': round(lo, 4), 'bacc_ci_hi': round(hi, 4),
            'perm_p': round(pval, 4), 'n_perm': args.n_perm,
            'perm_fixed_c': args.fixed_c_perm if args.fixed_c_perm is not None else '',
            'confusion': json.dumps(cm.tolist()),
            'null_mean': round(float(np.mean(null)), 4),
            'null_sd': round(float(np.std(null)), 4),
        })
        oof_frames.append(pd.DataFrame({
            'task': args.task, 'layer': name, 'subject': groups,
            'y': y, 'pred': pred, 'prob': prob}))

    if rows:
        out_csv = OUT / f'p0_{args.task}.csv'
        if out_csv.exists():
            old = pd.read_csv(out_csv)
            old = old[~old['layer'].isin([r['layer'] for r in rows])]
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out_csv, index=False)
        else:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        oof_csv = OUT / f'p0_{args.task}_oof.csv'
        new_oof = pd.concat(oof_frames, ignore_index=True)
        if oof_csv.exists():
            old = pd.read_csv(oof_csv)
            old = old[~old['layer'].isin(new_oof['layer'].unique())]
            pd.concat([old, new_oof], ignore_index=True).to_csv(oof_csv, index=False)
        else:
            new_oof.to_csv(oof_csv, index=False)
        print(f'saved {out_csv} | {oof_csv}')


if __name__ == '__main__':
    main()
