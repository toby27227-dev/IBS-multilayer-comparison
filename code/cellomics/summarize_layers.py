"""汇总各层（分类学/功能/代谢组）的 Bristol 回归与亚型分类对比。"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path.cwd()
GM = ROOT / 'IBS/gmrepo'
CE = ROOT / 'IBS/cellomics'

reg_dirs = [
    ('Bristol: 属(144)', GM / 'results_bristol_genus'),
    ('Bristol: 种(328)', GM / 'results_bristol_species'),
    ('Bristol: NMR(19)', CE / 'results_bristol_nmr'),
    ('Bristol: 胆汁酸(12)', CE / 'results_bristol_bileacids'),
    ('Bristol: 色氨酸(12)', CE / 'results_bristol_tryptophan'),
    ('Bristol: KO(~5.3k)', CE / 'results_bristol_ko'),
    ('Bristol: KEGG通路(327)', CE / 'results_bristol_kegg_pathways'),
    ('Bristol: NMR+属(163)', CE / 'results_bristol_combined'),
    ('Bristol: NMR IBS-D内部', CE / 'results_bristol_nmr_ibsd'),
    ('Bristol: NMR IBS-C内部', CE / 'results_bristol_nmr_ibsc'),
]
rows = []
for name, d in reg_dirs:
    cv = pd.read_csv(d / 'cv_results.csv')
    pred = pd.read_csv(d / 'predictions.csv', index_col=0)
    b = pd.read_csv(d / 'biomarkers.csv')
    rows.append({
        'task': name, 'n': len(pred),
        'R2_fold_mean': round(cv['r2'].mean(), 3),
        'R2_OOF': round(r2_score(pred['observed'], pred['predicted']), 3),
        'rho_OOF': round(spearmanr(pred['observed'], pred['predicted']).statistic, 3),
        'sig_taxa': f"{(b['q'] < 0.05).sum()}/{len(b)}"})
reg = pd.DataFrame(rows)

cls_dirs = [
    ('Cohort: 属(144)', GM / 'results_clinical_cohort'),
    ('Cohort: 种(328)', GM / 'results_clinical_cohort_species'),
    ('Cohort: NMR(19)', CE / 'results_cohort_nmr'),
    ('Cohort: 胆汁酸(12)', CE / 'results_cohort_bileacids'),
    ('Cohort: 色氨酸(12)', CE / 'results_cohort_tryptophan'),
    ('Cohort: KO(~5.3k)', CE / 'results_cohort_ko'),
    ('Cohort: KEGG通路(327)', CE / 'results_cohort_kegg_pathways'),
    ('Cohort: NMR+属(163)', CE / 'results_cohort_combined'),
]
rows = []
for name, d in cls_dirs:
    cv = pd.read_csv(d / 'cv_results.csv')
    b = pd.read_csv(d / 'biomarkers.csv')
    rows.append({
        'task': name,
        'balanced_acc': round(cv['balanced_acc'].mean(), 3),
        'std': round(cv['balanced_acc'].std(), 3),
        'f1': round(cv['f1_macro'].mean(), 3),
        'sig_taxa': f"{(b['q'] < 0.05).sum()}/{len(b)}"})
cls = pd.DataFrame(rows)

print('===== Bristol 回归（受试者分组 CV） =====')
print(reg.to_string(index=False))
print()
print('===== IBS-D/C/Healthy 分类（受试者分组 CV） =====')
print(cls.to_string(index=False))
reg.to_csv(CE / 'summary_bristol.csv', index=False)
cls.to_csv(CE / 'summary_cohort.csv', index=False)

nmr_b = pd.read_csv(CE / 'results_bristol_nmr' / 'biomarkers.csv')
ba_b = pd.read_csv(CE / 'results_bristol_bileacids' / 'biomarkers.csv')
print()
print('===== NMR  metabolite → Bristol 单变量 Top 12 =====')
print(nmr_b.sort_values('q').head(12)[['taxon', 'spearman', 'q']].round(4).to_string(index=False))
print()
print('===== 胆汁酸 → Bristol 单变量 Top 6 =====')
print(ba_b.sort_values('q').head(6)[['taxon', 'spearman', 'q']].round(4).to_string(index=False))
