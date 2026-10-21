"""整合 Cell 2020 (Mars et al.) Mendeley 多组学数据。

输入（项目根目录）:
  mendeley_NMR.xlsx              粪便 NMR 代谢组 (ID_on_tube + Time point)
  mendeley_BileAcid.xlsx         粪便胆汁酸 (SampleID)
  mendeley_Tryptophan.xlsx       粪便色氨酸通路代谢物 (SampleID)
  mendeley_microbiome.xlsx       含 'kegg orthology stool' (KOs × samples)

输出 (IBS/cellomics/):
  nmr.csv / bile_acids.csv / tryptophan.csv / ko_stool.csv   run_id × features
  sample_map.csv                  映射与标签: run_id, cell_SampleID, subject, Cohort_label, bristol
  缺失值: 每个特征以列最小值的一半填补 (min/2)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GMREPO = HERE.parent / 'gmrepo'

cl = pd.read_csv(GMREPO / 'cell_phenotypes_merged.csv', index_col=0)
cl.index = cl.index.astype(str)
sid2run = dict(zip(cl['cell_SampleID'].astype(str), cl.index))

meta_b = pd.read_csv(GMREPO / 'metadata_bristol.csv', index_col=0)
bristol = meta_b['bristol'] if 'bristol' in meta_b.columns else None

report = []

# ---------- NMR ----------
nmr = pd.read_excel(ROOT / 'mendeley_NMR.xlsx')
nmr['SampleID'] = nmr['ID_on_tube'].astype(int).astype(str) + '.T.' + \
                  nmr['Time point'].astype(int).astype(str)
nmr_meta_cols = ['Matlab_no', 'ID_on_tube', 'Time point', 'Gender', 'Gender code',
                 'Diagnosis', 'Group']
X = nmr.drop(columns=nmr_meta_cols + ['SampleID']).set_index(nmr['SampleID'])
X = X.apply(pd.to_numeric, errors='coerce')
X = X[~X.index.duplicated(keep='first')]
report.append(f'NMR: {X.shape[0]} samples x {X.shape[1]} metabolites | '
              f'mapped to runs: {sum(s in sid2run for s in X.index)}')

# ---------- Bile acids ----------
ba = pd.read_excel(ROOT / 'mendeley_BileAcid.xlsx')
ba_meta = ['ID_on_tube', 'Time point', 'Flare', 'Group', 'Age', 'BMI', 'Gender', 'SampleID']
BA = ba.drop(columns=ba_meta).set_index(ba['SampleID'].astype(str))
BA = BA.apply(pd.to_numeric, errors='coerce')
BA = BA[~BA.index.duplicated(keep='first')]
report.append(f'BileAcid: {BA.shape[0]} samples x {BA.shape[1]} metabolites | '
              f'mapped to runs: {sum(s in sid2run for s in BA.index)}')

# ---------- Tryptophan ----------
trp = pd.read_excel(ROOT / 'mendeley_Tryptophan.xlsx')
trp_meta = ['BIOME_ID_nr', 'Cohort', 'SubjectID', 'time_point', 'alt_name',
            'ID_on_tube', 'SampleID']
TRP = trp.drop(columns=trp_meta).set_index(trp['SampleID'].astype(str))
TRP = TRP.apply(pd.to_numeric, errors='coerce')
TRP = TRP[~TRP.index.duplicated(keep='first')]
report.append(f'Tryptophan: {TRP.shape[0]} samples x {TRP.shape[1]} metabolites | '
              f'mapped to runs: {sum(s in sid2run for s in TRP.index)}')

# ---------- KO (SHOGUN) ----------
ko = pd.read_excel(ROOT / 'mendeley_microbiome.xlsx',
                   sheet_name='kegg orthology stool')
ko = ko.rename(columns={ko.columns[0]: 'KO'}).set_index('KO')
ko = ko.apply(pd.to_numeric, errors='coerce').fillna(0.0)
KO = ko.T
KO.index = KO.index.astype(str)
report.append(f'KO: {KO.shape[0]} samples x {KO.shape[1]} KOs | '
              f'mapped to runs: {sum(s in sid2run for s in KO.index)}')

# ---------- 统一到 run_id 并保存 ----------
def to_runs(df, name):
    keep = [s for s in df.index if s in sid2run]
    out = df.loc[keep].copy()
    out.index = [sid2run[s] for s in keep]
    out = out[~out.index.duplicated(keep='first')]
    nan_frac = out.isna().mean().mean()
    if out.isna().any().any():
        out = out.fillna(out.min() / 2)
    out.to_csv(HERE / f'{name}.csv')
    return out, nan_frac

for df, name in [(X, 'nmr'), (BA, 'bile_acids'), (TRP, 'tryptophan'), (KO, 'ko_stool')]:
    out, nanf = to_runs(df, name)
    report.append(f'{name}: saved {out.shape} | pre-impute NaN mean fraction {nanf:.3f}')

smap = pd.DataFrame({'cell_SampleID': cl['cell_SampleID'].astype(str),
                     'subject': cl['cell_SampleID'].astype(str).str.split('.').str[0],
                     'Cohort_label': cl['Cohort'].map({'D': 'IBS-D', 'C': 'IBS-C', 'H': 'Healthy'})},
                    index=cl.index)
if bristol is not None:
    smap['bristol'] = bristol.reindex(smap.index)
smap.to_csv(HERE / 'sample_map.csv')

print('\n'.join(report))
print('\nsample_map:', smap.shape)
print(smap['Cohort_label'].value_counts().to_dict())
