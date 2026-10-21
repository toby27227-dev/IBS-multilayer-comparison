"""把 Cell 2020 Table S1 的真实临床表型映射到 GMrepo run_id。

Cell SampleID 形如 1.T.0（Participant 1, Sample 0）；
GMrepo original_sample_description 形如 "Participant 1 Sample 0"。
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CELL_XLSX = ROOT / 'cell2020_mmc1.xlsx'

info = pd.read_csv(HERE / 'sample_to_run_info.txt.gz', sep='\t', dtype=str, low_memory=False)
sub = info[info['project_id'] == 'PRJEB37924'].copy()

pat = re.compile(r'Participant\s+(\d+)\s+Sample\s+(\d+)')


def to_cell_id(s):
    m = pat.search(str(s))
    return f'{int(m.group(1))}.T.{int(m.group(2))}' if m else None


sub['cell_SampleID'] = sub['original_sample_description'].map(to_cell_id)
print('PRJEB37924 rows:', len(sub), '| parsed cell IDs:', sub['cell_SampleID'].notna().sum())
print('GMrepo disease_stage:', sub['disease_stage'].value_counts(dropna=False).to_dict())

cell = pd.read_excel(CELL_XLSX, sheet_name='stool metadata')
cell['SampleID'] = cell['SampleID'].astype(str)
keep = ['SampleID', 'Cohort', 'Age', 'BMI', 'Gender',
        'abdominal_pain_severity_1_100', 'Pain_over_10_days_0_100',
        'abdominal_distension_severity_0_100_', 'bowel_habit_dissatisfaction_0_100',
        'Effect_on_life_in_general_0_100', 'IBS_symptom_severity_1_500',
        'HAD_Anxiety_score', 'HAD_Depression_score',
        'Fatigue_0_10_10_greastest_possible_fatigue', 'Flare', 'Timepoint']
cell = cell[[c for c in keep if c in cell.columns]].copy()

merged = sub.merge(cell, left_on='cell_SampleID', right_on='SampleID', how='inner')
merged = merged.set_index('run_id')
merged.to_csv(HERE / 'cell_phenotypes_merged.csv')
print('\nmapped runs:', len(merged))
print('Cohort:', merged['Cohort'].value_counts(dropna=False).to_dict())
print('Cohort x GMrepo disease_stage:')
print(pd.crosstab(merged['Cohort'], merged['disease_stage'].fillna('NA')))
print('\nseverity summary by Cohort:')
print(merged.groupby('Cohort')['IBS_symptom_severity_1_500'].describe().round(1))
print('\nhead:')
print(merged[['cell_SampleID', 'Cohort', 'IBS_symptom_severity_1_500', 'Age', 'BMI']].head(8).to_string())
