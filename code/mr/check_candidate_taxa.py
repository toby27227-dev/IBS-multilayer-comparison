"""候选菌属（Cell 2020 标志物）与 MiBioGen 工具变量库的交集与可行性检查。

读取 Bristol 属层与严重度层的 top 标志物，与 MBG.allHits.p1e4.txt 的属级 taxa
做名称归一化匹配（支持 _group/sensu_stricto 等后缀），统计每候选属的
SNP 可用量（p<1e-4 / 1e-5 / 5e-8）与最小 p 值。
"""
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
MBG = ROOT / 'IBS/mr/data/MBG.allHits.p1e4.txt'
GM = ROOT / 'IBS/gmrepo'

bris = pd.read_csv(GM / 'results_bristol_genus/biomarkers.csv')
sev = pd.read_csv(GM / 'results_clinical_severity/biomarkers.csv')

pairs = []
for t in bris.sort_values('spearman', key=abs, ascending=False)['taxon'].head(25):
    pairs.append(('bristol', t))
for t in sev.sort_values('clr_effect', key=abs, ascending=False)['taxon'].head(16):
    pairs.append(('severity', t))
seen = set()
names = []
for src, t in pairs:
    if t not in seen:
        seen.add(t)
        names.append((src, t))
print('candidates:', len(names))


def norm(s):
    return s.lower().replace('-', '_').replace(' ', '_').replace('.', '')


cand_norm = {t: norm(t) for src, t in names}
stats = defaultdict(lambda: {'p4': 0, 'p5': 0, 'p8': 0, 'minp': 1.0})
mapping = defaultdict(set)
alltaxa = set()
nrows = 0
with open(MBG, encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        nrows += 1
        tax = row['bac']
        alltaxa.add(tax)
        if not tax.startswith('genus.'):
            continue
        gname = norm(tax[len('genus.'):].split('.id.')[0])
        p = float(row['P.weightedSumZ'])
        for t, cn in cand_norm.items():
            if gname == cn or gname.startswith(cn + '_'):
                s = stats[t]
                s['p4'] += 1
                if p < 1e-5:
                    s['p5'] += 1
                if p < 5e-8:
                    s['p8'] += 1
                if p < s['minp']:
                    s['minp'] = p
                mapping[t].add(tax)

gen = [t for t in alltaxa if t.startswith('genus.')]
print('MBG rows:', nrows, '| total taxa:', len(alltaxa), '| genus-level taxa:', len(gen))
print()
print('%-28s %7s %7s %7s %10s  %s' % ('candidate(src)', 'p<1e-4', 'p<1e-5', 'p<5e-8', 'minP', 'MBG taxon matches'))
for src, t in names:
    s = stats[t]
    tag = '%s(%s)' % (t, src[:4])
    if s['p4']:
        print('%-28s %7d %7d %7d %10.1e  %s' % (tag, s['p4'], s['p5'], s['p8'], s['minp'], '; '.join(sorted(mapping[t]))))
    else:
        print('%-28s %7d %7d %7d %10s  (no match)' % (tag, 0, 0, 0, '-'))
