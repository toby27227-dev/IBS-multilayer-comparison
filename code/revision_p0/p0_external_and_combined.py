"""P0 revision: external baselines + fair combined-model sensitivity.

A. Cell 2020 -> Jacobs 2023 zero-shot:
   - observed 11 shared metabolites (R11)
   - 200 random 11-feature models (same Ridge/GroupKFold protocol)
   - demographics-only model (Age, BMI, Sex)
B. NMR+Genus combined model with the same genus prevalence filter (>=0.10)
   on the common Bristol subset, compared with 163-dim unfiltered model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
CE = ROOT / 'IBS' / 'cellomics'
GM = ROOT / 'IBS' / 'gmrepo'
NW = ROOT / 'IBS' / 'jacobs2023'
OUT = ROOT / 'IBS' / 'revision_p0'
OUT.mkdir(exist_ok=True)
SEED = 42
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]

FEATURE_MAP = {
    'Lysine': 'lysine', 'Tyrosine': 'tyrosine', 'Leucine': 'leucine',
    'Valine': 'valine', 'Isoleucine': 'isoleucine', 'Alanine': 'alanine',
    'Glucose': 'glucose', 'Succinate': 'succinate', 'Uracil': 'uracil',
    'Hypoxanthine': 'hypoxanthine', 'Lactate': 'lactate',
}


def fit_ridge_oof(X, y, groups):
    best = None
    for a in ALPHAS:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            oof[te] = Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[te])
        r2 = r2_score(y, oof)
        if best is None or r2 > best[0]:
            best = (r2, a, spearmanr(y, oof).statistic)
    r2, a, rho = best
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=a).fit(sc.transform(X), y)
    return m, sc, r2, rho, a


def score(model, sc, X):
    return model.predict(sc.transform(X))


def external_baselines(n_rep=200):
    nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
    sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
    clin = pd.read_csv(GM / 'metadata_clinical.csv', index_col=0)
    tr = nmr.join(sm[['bristol', 'subject']]).dropna()
    b16 = pd.read_csv(CE / 'results_bristol_nmr' / 'biomarkers.csv')
    sig16 = b16[b16['q'] < 0.05].sort_values('spearman', ascending=False)['taxon'].tolist()
    shared = [f for f in sig16 if f in FEATURE_MAP]
    print(f'train n={len(tr)} subjects={tr["subject"].nunique()} shared={len(shared)}')

    met = pd.read_csv(NW / 'jacobs2023_metabolites.csv', index_col=0)
    meta = pd.read_csv(NW / 'jacobs2023_metadata.csv').set_index('Patient')
    idx = met.index.intersection(meta.index)
    met, meta = met.loc[idx], meta.loc[idx]
    keep = meta['BH'].isin(['Constipation', 'Mixed', 'Diarrhea'])
    met, meta = met.loc[keep], meta.loc[keep]
    yj = (meta['BH'] == 'Diarrhea').astype(int).values
    print(f'target D vs C+M n={len(yj)} D={int(yj.sum())} C+M={int((1-yj).sum())}')

    Xj = np.log1p(met[[FEATURE_MAP[f] for f in shared]].values.astype(float))
    Xtr = np.log1p(tr[shared].values.astype(float))
    ytr = tr['bristol'].values
    gtr = tr['subject'].values
    m, sc, r2, rho, alpha = fit_ridge_oof(Xtr, ytr, gtr)
    pred_j = score(m, sc, Xj)
    obs_auc = roc_auc_score(yj, pred_j)
    dc = meta['BH'].isin(['Diarrhea', 'Constipation']).values
    obs_auc_dc = roc_auc_score(yj[dc], pred_j[dc])
    print(f'R11: R2={r2:.3f} rho={rho:.3f} alpha={alpha} '
          f'AUC(D vs C+M)={obs_auc:.4f} AUC(D vs C)={obs_auc_dc:.4f}')

    rng = np.random.default_rng(SEED)
    null = []
    for b in range(n_rep):
        Xr = rng.normal(size=Xtr.shape)
        Xrj = rng.normal(size=Xj.shape)
        mr, scr, _, _, _ = fit_ridge_oof(Xr, ytr, gtr)
        null.append(roc_auc_score(yj, score(mr, scr, Xrj)))
    null = np.asarray(null)
    p_rand = (1 + int(np.sum(null >= obs_auc))) / (1 + len(null))
    print(f'random 11-feature: mean={null.mean():.4f} '
          f'95%=[{np.percentile(null,2.5):.4f},{np.percentile(null,97.5):.4f}] p={p_rand:.4f}')

    # demographics
    dtr = clin.loc[tr.index, ['Age', 'BMI', 'Gender']].copy()
    dtr['Sex_M'] = (dtr['Gender'].astype(str).str.upper().str[0] == 'M').astype(float)
    dtr = dtr[['Age', 'BMI', 'Sex_M']].astype(float)
    dte = meta[['Age', 'BMI', 'Sex']].copy()
    dte['Sex_M'] = (dte['Sex'].astype(str).str.upper().str[0] == 'M').astype(float)
    dte = dte[['Age', 'BMI', 'Sex_M']].astype(float)
    print('demographics missing train/target:', int(dtr.isna().sum().sum()), int(dte.isna().sum().sum()))
    md, scd, r2d, rhod, alphad = fit_ridge_oof(dtr.values, ytr, gtr)
    pred_d = score(md, scd, dte.values)
    demo_auc = roc_auc_score(yj, pred_d)
    demo_auc_dc = roc_auc_score(yj[dc], pred_d[dc])
    boot = []
    rng2 = np.random.default_rng(SEED)
    for _ in range(2000):
        ii = rng2.integers(0, len(yj), len(yj))
        if len(np.unique(yj[ii])) < 2:
            continue
        boot.append(roc_auc_score(yj[ii], pred_d[ii]))
    dlo, dhi = np.percentile(boot, [2.5, 97.5])
    print(f'demographics: Bristol R2={r2d:.3f} '
          f'AUC(D vs C+M)={demo_auc:.4f} AUC(D vs C)={demo_auc_dc:.4f} '
          f'95%=[{dlo:.4f},{dhi:.4f}]')

    delta = []
    rng3 = np.random.default_rng(SEED)
    for _ in range(2000):
        ii = rng3.integers(0, len(yj), len(yj))
        if len(np.unique(yj[ii])) < 2:
            continue
        delta.append(roc_auc_score(yj[ii], pred_j[ii]) -
                     roc_auc_score(yj[ii], pred_d[ii]))
    delta = np.asarray(delta)
    d_lo, d_hi = np.percentile(delta, [2.5, 97.5])
    p_delta = float(np.mean(delta <= 0))
    print(f'delta AUC (R11 - demographics)={delta.mean():+.4f} '
          f'95%=[{d_lo:+.4f},{d_hi:+.4f}] p(<=0)={p_delta:.4f}')

    rows = [
        {'baseline': 'R11_observed', 'n_features': len(shared),
         'auc_D_vs_CM': round(obs_auc, 4), 'auc_D_vs_C': round(obs_auc_dc, 4),
         'ci_lo': '', 'ci_hi': '', 'p_vs_random': '', 'note': f'Bristol R2={r2:.3f}'},
        {'baseline': 'random_11', 'n_features': 11,
         'auc_D_vs_CM': round(float(null.mean()), 4), 'auc_D_vs_C': '',
         'ci_lo': round(float(np.percentile(null, 2.5)), 4),
         'ci_hi': round(float(np.percentile(null, 97.5)), 4),
         'p_vs_random': round(p_rand, 4), 'note': f'B={n_rep}'},
        {'baseline': 'demographics', 'n_features': 3,
         'auc_D_vs_CM': round(demo_auc, 4), 'auc_D_vs_C': round(demo_auc_dc, 4),
         'ci_lo': round(float(dlo), 4), 'ci_hi': round(float(dhi), 4),
         'p_vs_random': '', 'note': f'Age/BMI/Sex; Bristol R2={r2d:.3f}'},
    ]
    pd.DataFrame(rows).to_csv(OUT / 'p0_external_baselines.csv', index=False)
    json.dump({'endpoint': 'D vs C+M (primary); D vs C (sensitivity)',
               'obs_auc': float(obs_auc), 'obs_auc_dc': float(obs_auc_dc),
               'null_mean': float(null.mean()),
               'null_lo': float(np.percentile(null, 2.5)),
               'null_hi': float(np.percentile(null, 97.5)), 'p_random': float(p_rand),
               'demo_auc': float(demo_auc), 'demo_auc_dc': float(demo_auc_dc),
               'demo_ci': [float(dlo), float(dhi)],
               'delta_auc_r11_minus_demo': float(delta.mean()),
               'delta_auc_ci': [float(d_lo), float(d_hi)],
               'delta_auc_p': p_delta,
               'r11_r2': float(r2), 'r11_rho': float(rho)},
              open(OUT / 'p0_external_baselines.json', 'w'), indent=2)


def combined_filtered():
    nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
    genus = pd.read_csv(GM / 'abundance_genus.csv', index_col=0)
    sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
    idx = nmr.index.intersection(genus.index).intersection(sm.dropna(subset=['bristol']).index)
    Xn = np.log1p(nmr.loc[idx].values.astype(float))
    g = genus.loc[idx]
    keep = (g > 0).mean(axis=0) >= 0.10
    g = g.loc[:, keep]
    Xg = np.log(g.replace(0, np.nan)).fillna(0)  # CLR-like; zeros -> 0
    Xg = Xg.sub(Xg.mean(axis=1), axis=0).values
    y = sm.loc[idx, 'bristol'].values.astype(float)
    grp = sm.loc[idx, 'subject'].astype(str).values
    print(f'combined subset n={len(idx)} subjects={len(np.unique(grp))} '
          f'genus={g.shape[1]} total={Xn.shape[1]+Xg.shape[1]}')

    out = []
    oofs = {}
    for tag, X in [('NMR19', Xn), ('Genus67', Xg), ('Combined86', np.hstack([Xn, Xg]))]:
        sc = StandardScaler().fit(X)
        Z = sc.transform(X)
        r2, a, rho = None, None, None
        best = None
        for aa in ALPHAS:
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(Z, y, grp):
                oof[te] = Ridge(alpha=aa).fit(Z[tr], y[tr]).predict(Z[te])
            rr = r2_score(y, oof)
            if best is None or rr > best[0]:
                best = (rr, aa, spearmanr(y, oof).statistic, oof)
        r2, a, rho, oof = best
        oofs[tag] = oof
        print(f'{tag}: R2={r2:.4f} rho={rho:.4f} alpha={a}')
        out.append({'model': tag, 'n': len(y), 'subjects': len(np.unique(grp)),
                    'n_features': X.shape[1], 'oof_r2': round(r2, 4),
                    'oof_rho': round(rho, 4), 'alpha': a})

    rng = np.random.default_rng(SEED)
    subjects = np.unique(grp)
    by_subject = {s: np.where(grp == s)[0] for s in subjects}
    for tag in ['Combined86', 'Genus67']:
        d = []
        for _ in range(2000):
            samp = rng.choice(subjects, len(subjects), replace=True)
            ii = np.concatenate([by_subject[s] for s in samp])
            d.append(r2_score(y[ii], oofs[tag][ii]) - r2_score(y[ii], oofs['NMR19'][ii]))
        d = np.asarray(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f'delta R2 ({tag} - NMR19)={d.mean():+.4f} '
              f'95%=[{lo:+.4f},{hi:+.4f}] p(<=0)={np.mean(d <= 0):.4f}')
        for row in out:
            if row['model'] == tag:
                row['delta_r2_vs_NMR'] = round(float(d.mean()), 4)
                row['delta_ci_lo'] = round(float(lo), 4)
                row['delta_ci_hi'] = round(float(hi), 4)
                row['delta_p_le0'] = round(float(np.mean(d <= 0)), 4)
    pd.DataFrame(out).to_csv(OUT / 'p0_combined_filtered.csv', index=False)


if __name__ == '__main__':
    external_baselines()
    combined_filtered()
