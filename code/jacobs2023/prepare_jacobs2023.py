"""整理 Jacobs 2023 队列 (Microbiome 2023, PRJNA812699) 补充材料：
- jacobs2023_metabolites_raw.csv (原 MOESM2): 代谢物强度矩阵（metabolite × sample） -> 转置为 sample × metabolite
- jacobs2023_metadata_raw.csv (原 MOESM3): 表型元数据（Patient, Group, BH, Age, Sex, BMI, HAD...）
输出: jacobs2023_metabolites.csv, jacobs2023_metadata.csv
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

# 代谢物矩阵
raw = pd.read_csv(HERE / 'jacobs2023_metabolites_raw.csv')
raw = raw.set_index('Metabolite')
raw = raw.apply(pd.to_numeric, errors='coerce')
# 同名代谢物取均值，避免重复列
if raw.index.duplicated().any():
    raw = raw.groupby(level=0).mean()
X = raw.T
X.index = X.index.astype(str)
X.index.name = 'Patient'
X = X.fillna(0)

# 表型
meta = pd.read_csv(HERE / 'jacobs2023_metadata_raw.csv')
meta['Patient'] = meta['Patient'].astype(str)
meta = meta.set_index('Patient')
meta = meta.reindex(X.index)

X.to_csv(HERE / 'jacobs2023_metabolites.csv')
meta.to_csv(HERE / 'jacobs2023_metadata.csv')
print('metabolites:', X.shape, '| metadata:', meta.shape)
print('Group:', meta['Group'].value_counts().to_dict())
print('BH:', meta['BH'].value_counts().to_dict())
