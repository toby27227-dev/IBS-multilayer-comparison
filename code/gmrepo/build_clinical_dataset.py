"""用 Cell 2020 Table S1 的真实临床标签构建监督数据集。

输出：
  abundance_genus_clinical.csv / abundance_species_clinical.csv / metadata_clinical.csv
      -> 标签 Cohort_label: IBS-D / IBS-C / Healthy（464 个 run）
  abundance_genus_severity.csv / metadata_severity.csv
      -> 标签 severity_bin: High / Low（IBS 且 IBS-SSS 非缺失）
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

cl = pd.read_csv(HERE / 'cell_phenotypes_merged.csv', index_col=0)
cl.index = cl.index.astype(str)

# subject / timepoint from cell_SampleID = X.T.Y
pat = re.compile(r'(\d+)\.T\.(\d+)')
cl['subject'] = cl['cell_SampleID'].map(lambda s: pat.match(str(s)).group(1))
cl['Timepoint'] = cl['cell_SampleID'].map(lambda s: int(pat.match(str(s)).group(2)))

cohort_map = {'D': 'IBS-D', 'C': 'IBS-C', 'H': 'Healthy'}
meta = pd.DataFrame({
    'run_id': cl.index,
    'subject': cl['subject'].values,
    'Timepoint': cl['Timepoint'].values,
    'Cohort_label': cl['Cohort'].map(cohort_map).values,
    'IBS_symptom_severity': pd.to_numeric(cl['IBS_symptom_severity_1_500'], errors='coerce').values,
    'pain': pd.to_numeric(cl['abdominal_pain_severity_1_100'], errors='coerce').values,
    'distension': pd.to_numeric(cl['abdominal_distension_severity_0_100_'], errors='coerce').values,
    'HAD_anxiety': pd.to_numeric(cl['HAD_Anxiety_score'], errors='coerce').values,
    'HAD_depression': pd.to_numeric(cl['HAD_Depression_score'], errors='coerce').values,
    'Age': pd.to_numeric(cl['Age'], errors='coerce').values,
    'BMI': pd.to_numeric(cl['BMI_y'], errors='coerce').values,
    'Gender': cl['Gender'].values,
}).set_index('run_id')
meta = meta.dropna(subset=['Cohort_label'])

gen = pd.read_csv(HERE / 'abundance_genus.csv', index_col=0)
spe = pd.read_csv(HERE / 'abundance_species.csv', index_col=0)
common = meta.index.intersection(gen.index)
meta = meta.loc[common]
gen = gen.loc[common]
spe = spe.loc[common]

meta.to_csv(HERE / 'metadata_clinical.csv')
gen.to_csv(HERE / 'abundance_genus_clinical.csv')
spe.to_csv(HERE / 'abundance_species_clinical.csv')
print('clinical cohort:', gen.shape, meta['Cohort_label'].value_counts().to_dict())
print('subjects:', meta['subject'].nunique())

# ---- severity subset (IBS only, severity available) ----
sev = meta[meta['Cohort_label'].isin(['IBS-D', 'IBS-C']) &
           meta['IBS_symptom_severity'].notna()].copy()
med = sev['IBS_symptom_severity'].median()
sev['severity_bin'] = (sev['IBS_symptom_severity'] >= med).map({True: 'High', False: 'Low'})
sev_gen = gen.loc[sev.index]
sev.to_csv(HERE / 'metadata_severity.csv')
sev_gen.to_csv(HERE / 'abundance_genus_severity.csv')
print('severity subset:', sev_gen.shape, sev['severity_bin'].value_counts().to_dict(),
      '| median IBS-SSS =', med)
