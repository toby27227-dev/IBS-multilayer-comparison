"""全部 Cell 2020 标志物（Bristol 属层 + 严重度层）与 MiBioGen 属级交集表。"""
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
MBG = ROOT / 'IBS/mr/data/MBG.allHits.p1e4.txt'
GM = ROOT / 'IBS/gmrepo'
OUT = ROOT / 'IBS/mr/data'

bris = pd.read_csv(GM / 'results_bristol_genus/biomarkers.csv')
sev = pd.read_csv(GM / 'results_clinical_severity/biomarkers.csv')
cands = list(dict.fromkeys(list(bris['taxon']) + list(sev['taxon'])))
print('unique candidate names:', len(cands))


def norm(s):
    return s.lower().replace('-', '_').replace(' ', '_').replace('.', '')


by_gen = defaultdict(lambda: [0, 0, 0])
with open(MBG, encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        tax = row['bac']
        if not tax.startswith('genus.'):
            continue
        g = tax[len('genus.'):].split('.id.')[0]
        p = float(row['P.weightedSumZ'])
        h = by_gen[g]
        h[0] += 1
        if p < 1e-5:
            h[1] += 1
        if p < 5e-8:
            h[2] += 1

gn = {norm(g): g for g in by_gen}
print('MBG genus taxa:', len(by_gen))
(OUT / 'mbg_genus_taxa.txt').write_text('\n'.join(sorted(by_gen)), encoding='utf-8')

rows = []
hits = {}
for name in cands:
    cn = norm(name)
    match = sorted(set(g for k, g in gn.items() if k == cn or k.startswith(cn + '_')))
    hits[name] = match
    for g in match:
        p4, p5, p8 = by_gen[g]
        rows.append({'cell_name': name, 'mbg_genus': g, 'p_lt_1e4': p4,
                     'p_lt_1e5': p5, 'p_lt_5e8': p8})
df = pd.DataFrame(rows).sort_values(['p_lt_1e5', 'p_lt_5e8'], ascending=False)
df.to_csv(OUT / 'candidate_taxa_matches.csv', index=False)
print('\nmatched pairs:', len(df), '| matched candidates:', int(df['cell_name'].nunique()) if len(df) else 0)
print(df.to_string(index=False))
missing = [n for n in cands if not hits[n]]
print('\nno MBG match (%d):' % len(missing))
print(', '.join(missing))
