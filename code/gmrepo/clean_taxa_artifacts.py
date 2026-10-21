"""清洗 GMrepo 丰度矩阵中的非生物学伪影列（如 '-1'）。

规则：列名不以字母或 '[' 开头（如 '-1'、纯数字），或命中已知伪影名单。
输出：覆盖原矩阵文件，并写 clean_report.txt 记录被删除的列。
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACT_NAMES = {'-1', 'unclassified', 'unknown', 'na', 'nan', ''}
NAME_OK = re.compile(r"^[A-Za-z\[]")

report = []
for fname in ['abundance_genus.csv', 'abundance_species.csv']:
    path = HERE / fname
    df = pd.read_csv(path, index_col=0)
    drop = [c for c in df.columns
            if str(c).strip().lower() in ARTIFACT_NAMES or not NAME_OK.match(str(c))]
    report.append(f'{fname}: {df.shape[1]} 列 -> 删除 {len(drop)}: {drop}')
    if drop:
        mean_ab = df[drop].mean().round(3).to_dict()
        nz = (df[drop] > 0).mean().round(3).to_dict()
        report.append(f'  删除列统计: mean={mean_ab} | nonzero_frac={nz}')
        df = df.drop(columns=drop)
        df.to_csv(path)
    report.append(f'  清洗后: {df.shape[1]} 列')

txt = '\n'.join(report)
(HERE / 'clean_report.txt').write_text(txt, encoding='utf-8')
print(txt)
