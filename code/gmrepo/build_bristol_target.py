"""构建每个 run 的 Bristol 定量表型。

- survey_event_date (Excel 序列号) = 样本采集日期
- daily stool diary: Subject ID + Date + 'Mean consistency/total BM' = 每日平均 Bristol
- 目标: 采集日期 ±3 天窗口内日记日均值的平均; 不足则 ±7 天; 仍不足用受试者全期均值
验证: 按 Cohort 分组的 Bristol 均值应呈 IBS-D > Healthy > IBS-C。
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

cl = pd.read_csv(HERE / 'cell_phenotypes_merged.csv', index_col=0)
cl.index = cl.index.astype(str)

x = pd.ExcelFile(ROOT / 'cell2020_mmc1.xlsx')
sl = x.parse('stool metadata', usecols=['SampleID', 'study_id', 'Timepoint',
                                        'survey_event_date', 'Cohort'])
sl['survey_event_date'] = pd.to_datetime(
    pd.to_numeric(sl['survey_event_date'], errors='coerce'), unit='D', origin='1899-12-30')

diary = x.parse('daily stool diary',
                usecols=['Subject ID', 'Date', 'Mean consistency/total BM'])
diary = diary.rename(columns={'Subject ID': 'subject', 'Date': 'date',
                              'Mean consistency/total BM': 'bristol_day'})
diary['date'] = pd.to_datetime(diary['date']).dt.normalize()
diary['bristol_day'] = pd.to_numeric(diary['bristol_day'], errors='coerce')
diary = diary.dropna(subset=['subject', 'bristol_day'])
subj_mean = diary.groupby('subject')['bristol_day'].mean()

def window_mean(subject, d, win):
    days = diary[diary['subject'] == subject]
    near = days[(days['date'] - d).abs() <= pd.Timedelta(days=win)]
    return near['bristol_day'].mean() if len(near) else np.nan

rows = []
for _, r in sl.iterrows():
    sid = r['study_id']
    if pd.isna(sid) or pd.isna(r['survey_event_date']):
        continue
    sid = int(sid)
    d = r['survey_event_date']
    val, src = window_mean(sid, d, 3), 'w3'
    if pd.isna(val):
        val, src = window_mean(sid, d, 7), 'w7'
    if pd.isna(val) and sid in subj_mean.index:
        val, src = subj_mean[sid], 'subject'
    if pd.notna(val):
        rows.append({'SampleID': r['SampleID'], 'bristol': round(float(val), 3),
                     'bristol_src': src, 'bristol_subject_mean': subj_mean.get(sid, np.nan)})

bri = pd.DataFrame(rows)
print('samples with Bristol target:', len(bri))
print('source breakdown:', bri['bristol_src'].value_counts().to_dict())

cl2 = cl.reset_index().rename(columns={'index': 'run_id'})
cl2 = cl2.merge(bri, left_on='cell_SampleID', right_on='SampleID', how='inner')
cl2 = cl2.set_index('run_id')

meta = pd.read_csv(HERE / 'metadata_clinical.csv', index_col=0)
meta = meta.join(cl2[['bristol', 'bristol_src', 'bristol_subject_mean']])
print('\nclinical metadata with bristol:', meta['bristol'].notna().sum(), '/', len(meta))
print('\nBristol by Cohort (sample-level):')
print(meta.groupby('Cohort_label')['bristol'].agg(['count', 'mean', 'std', 'median']).round(2))
sm = meta.groupby(['Cohort_label', 'subject'])['bristol_subject_mean'].first().reset_index()
print('\nBristol by Cohort (subject-level mean):')
print(sm.groupby('Cohort_label')['bristol_subject_mean'].agg(['count', 'mean', 'std']).round(2))

meta.to_csv(HERE / 'metadata_bristol.csv')
gen = pd.read_csv(HERE / 'abundance_genus_clinical.csv', index_col=0)
spe = pd.read_csv(HERE / 'abundance_species_clinical.csv', index_col=0)
meta_b = meta[meta['bristol'].notna()]
meta_b.to_csv(HERE / 'metadata_bristol.csv')
gen.loc[meta_b.index].to_csv(HERE / 'abundance_genus_bristol.csv')
spe.loc[meta_b.index].to_csv(HERE / 'abundance_species_bristol.csv')
print('\nsaved:', gen.loc[meta_b.index].shape, '| subjects:', meta_b['subject'].nunique())
