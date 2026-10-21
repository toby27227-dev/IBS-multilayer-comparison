"""把 KO 丰度聚合为 KEGG 通路丰度。

输入: ko_stool.csv (run_id × KO), ko2pathway.tsv (KEGG REST link/pathway/ko),
      kegg_pathway_name_map.tsv (KEGG REST list/pathway，通路 ID→名称)
输出: kegg_pathways_750.csv (run_id × 750 条通路；map/ko 双前缀形式),
      kegg_pathway_ids.txt (通路 ID 清单)
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

ko = pd.read_csv(HERE / 'ko_stool.csv', index_col=0)

m = pd.read_csv(HERE / 'ko2pathway.tsv', sep='\t', header=None, names=['ko', 'path'])
m['ko'] = m['ko'].str.replace('ko:', '', regex=False)
m['path'] = m['path'].str.replace('path:', '', regex=False)
m = m.drop_duplicates()

names = pd.read_csv(HERE / 'kegg_pathway_name_map.tsv', sep='\t', header=None,
                    names=['path', 'name'])
names['path'] = names['path'].str.replace('path:', '', regex=False)
name_map = dict(zip(names['path'], names['name']))

pairs = m[m['ko'].isin(ko.columns)]
print('mapped KOs:', pairs['ko'].nunique(), '/', ko.shape[1],
      '| pathways:', pairs['path'].nunique())

agg = {}
for path_id, grp in pairs.groupby('path'):
    cols = [c for c in grp['ko'] if c in ko.columns]
    if cols:
        agg[path_id] = ko[cols].sum(axis=1)
P = pd.DataFrame(agg)
P = P.loc[:, P.sum(axis=0) > 0]
P.to_csv(HERE / 'kegg_pathways_750.csv')

with open(HERE / 'kegg_pathway_ids.txt', 'w', encoding='utf-8') as f:
    for pid in P.columns:
        f.write(f'{pid}\t{name_map.get(pid, pid)}\n')

print('pathway matrix:', P.shape)
print('top pathways by mean abundance:')
print(P.mean().sort_values(ascending=False).head(8).round(4).to_string())
