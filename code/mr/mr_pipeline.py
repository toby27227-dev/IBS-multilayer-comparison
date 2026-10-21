"""两样本孟德尔随机化（Two-sample MR）：MiBioGen 菌属 → IBS 风险 (GCST90016564)。

暴露（instruments）：MBG.allHits.p1e4.txt 中属级 p<1e-5 SNPs；
  无 LD 参考面板 → 距离法剪枝（±1 Mb 窗口保留最小 p，近似去 LD，见局限说明）。
结局：GWAS Catalog harmonised (34741163-GCST90016564-EFO_0000555.h.tsv.gz)；
  hm_rsid 匹配 + 等位基因协调（含链翻转；回文 SNP 保守剔除）。
方法：IVW（随机效应，delta 法 SE，Cochran Q 检验异质性）/ 加权中位数（bootstrap SE）/
  MR-Egger（截距 = 水平多效性检验）/ leave-one-out 稳健性 / F 统计量。
输出：results/mr_results.csv、results/mr_instruments_harmonized.csv
"""
import csv
import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path.cwd()
MR = ROOT / 'IBS/mr'
DATA = MR / 'data'
OUT = MR / 'results'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)

P_EXP = 1e-5
CLUMP_KB = 1000
COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

EXPOSURES = [
    'Clostridiumsensustricto1', 'Roseburia', 'Blautia', 'Eubacteriumrectalegroup',
    'Eubacteriumhalliigroup', 'Bifidobacterium', 'Desulfovibrio', 'Anaerostipes',
    'Methanobrevibacter', 'Faecalibacterium', 'Barnesiella', 'Dorea', 'Romboutsia',
    'Coprococcus1', 'Coprococcus2', 'Coprococcus3', 'Alistipes', 'Fusicatenibacter',
    'Parabacteroides', 'Gordonibacter', 'Adlercreutzia', 'Lachnospira',
    'Lachnoclostridium', 'Oscillibacter', 'Dialister', 'Bacteroides', 'Akkermansia',
]
CELL_NAME = {
    'Clostridiumsensustricto1': 'Clostridium', 'Roseburia': 'Roseburia', 'Blautia': 'Blautia',
    'Eubacteriumrectalegroup': 'Eubacterium', 'Eubacteriumhalliigroup': 'Eubacterium',
    'Bifidobacterium': 'Bifidobacterium', 'Desulfovibrio': 'Desulfovibrio',
    'Anaerostipes': 'Anaerostipes', 'Methanobrevibacter': 'Methanobrevibacter',
    'Faecalibacterium': 'Faecalibacterium', 'Barnesiella': 'Barnesiella', 'Dorea': 'Dorea',
    'Romboutsia': 'Romboutsia', 'Coprococcus1': 'Coprococcus', 'Coprococcus2': 'Coprococcus',
    'Coprococcus3': 'Coprococcus', 'Alistipes': 'Alistipes',
    'Fusicatenibacter': 'Fusicatenibacter', 'Parabacteroides': 'Parabacteroides',
    'Gordonibacter': 'Gordonibacter', 'Adlercreutzia': 'Adlercreutzia',
    'Lachnospira': 'Lachnospira', 'Lachnoclostridium': 'Lachnoclostridium',
    'Oscillibacter': 'Oscillibacter', 'Dialister': 'Dialister', 'Bacteroides': 'Bacteroides',
    'Akkermansia': 'Akkermansia',
}

print('=== 1. 读取 MiBioGen 工具变量 ===')
rows = []
with open(DATA / 'MBG.allHits.p1e4.txt', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        b = row['bac']
        if not b.startswith('genus.'):
            continue
        nm = b[len('genus.'):].split('.id.')[0].lstrip('.')
        if nm not in EXPOSURES:
            continue
        rows.append(dict(taxon=nm, chr=int(row['chr']), bp=int(row['bp']),
                         rsid=row['rsID'], ea=row['eff.allele'], oa=row['ref.allele'],
                         beta=float(row['beta']), se=float(row['SE']),
                         p=float(row['P.weightedSumZ']), n=int(row['N'])))
inst = pd.DataFrame(rows)
print('p<1e-4 命中:', len(inst), '| taxa:', inst['taxon'].nunique())
inst = inst[inst['p'] < P_EXP].copy()
inst['f_stat'] = inst['beta'] ** 2 / inst['se'] ** 2
print('p<1e-5:', len(inst), '| F>10 占比: %.1f%%' % (100 * (inst['f_stat'] > 10).mean()))


def prune(df, kb=CLUMP_KB):
    keep = []
    for _, g in df.sort_values('p').groupby('chr', sort=False):
        taken = []
        for _, r in g.iterrows():
            if all(abs(r['bp'] - bp) > kb * 1000 for bp in taken):
                taken.append(r['bp'])
                keep.append(r['rsid'])
    return df[df['rsid'].isin(keep)]


inst = prune(inst)
print('距离剪枝后:', len(inst), 'SNPs |', inst.groupby('taxon')['rsid'].count().to_dict())

print('\n=== 2. 读取 IBS 结局 (harmonised) ===')
OUTCOME = DATA / 'GCST90016564_harmonised.tsv.gz'
with gzip.open(OUTCOME, 'rt', encoding='utf-8') as f:
    head = pd.read_csv(f, sep='\t', nrows=5, low_memory=False)
print('columns:', list(head.columns))
lc = {c.lower(): c for c in head.columns}


def col(*names):
    for n in names:
        if n in lc:
            return lc[n]
    return None


rs_c = col('hm_rsid', 'rsid')
ea_c = col('hm_effect_allele', 'effect_allele')
oa_c = col('hm_other_allele', 'other_allele')
b_c = col('hm_beta', 'beta')
se_c = col('hm_standard_error', 'standard_error', 'se')
or_c = col('hm_odds_ratio', 'odds_ratio')
eaf_c = col('hm_effect_allele_frequency', 'effect_allele_frequency', 'hm_effect_allele_freq')
print('detected:', rs_c, ea_c, oa_c, b_c, se_c, or_c, eaf_c)
need = list(dict.fromkeys([c for c in [rs_c, ea_c, oa_c, b_c, se_c, or_c, eaf_c] if c]))
keep_rs = set(inst['rsid'])
print('需匹配 rsID:', len(keep_rs))
parts = []
with gzip.open(OUTCOME, 'rt', encoding='utf-8') as f:
    reader = pd.read_csv(f, sep='\t', usecols=need, chunksize=2_000_000, low_memory=False)
    for ch in reader:
        ch = ch[ch[rs_c].isin(keep_rs)]
        if len(ch):
            parts.append(ch)
out2 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=need)
print('结局命中行:', len(out2))
out2 = out2.dropna(subset=[rs_c])
out2 = out2.drop_duplicates(subset=[rs_c])
if b_c is None and or_c is not None:
    out2['_b'] = np.log(out2[or_c])
    b_c = '_b'
    se_c = col('hm_standard_error', 'standard_error', 'se')
m = inst.merge(out2, left_on='rsid', right_on=rs_c, how='inner')
print('暴露-结局 rsID 共同:', len(m), '/', len(inst))

print('\n=== 3. 等位基因协调 ===')
records = []
for _, r in m.iterrows():
    ea_o, oa_o = str(r[ea_c]).upper(), str(r[oa_c]).upper()
    bx, by, sy = r['beta'], r[b_c], r[se_c]
    if by is None or not np.isfinite(by) or not np.isfinite(sy):
        continue
    if ea_o == r['ea'] and oa_o == r['oa']:
        pass
    elif ea_o == r['oa'] and oa_o == r['ea']:
        by = -by
    elif ea_o == COMP.get(r['ea']) and oa_o == COMP.get(r['oa']):
        pass
    elif ea_o == COMP.get(r['oa']) and oa_o == COMP.get(r['ea']):
        by = -by
    else:
        continue
    palindromic = (ea_o in 'AT' and oa_o in 'AT') or (ea_o in 'CG' and oa_o in 'CG')
    if palindromic:
        continue
    records.append(dict(taxon=r['taxon'], rsid=r['rsid'], chr=r['chr'], bp=r['bp'],
                        ea=r['ea'], oa=r['oa'], bx=bx, sx=r['se'], by=by, sy=sy,
                        p_exp=r['p'], n_exp=r['n'], f_stat=r['f_stat']))
h = pd.DataFrame(records)
print('协调后:', len(h), 'SNPs |', h.groupby('taxon')['rsid'].count().to_dict())
h.to_csv(OUT / 'mr_instruments_harmonized.csv', index=False)


def ratio_stats(bx, by, sx, sy):
    b = by / bx
    se = np.sqrt(sy ** 2 / bx ** 2 + by ** 2 * sx ** 2 / bx ** 4)
    return b, se


def ivw(bx, by, sx, sy):
    r, rse = ratio_stats(bx, by, sx, sy)
    w = 1 / rse ** 2
    b = float(np.sum(w * r) / np.sum(w))
    se_fixed = float(np.sqrt(1 / np.sum(w)))
    q = float(np.sum(w * (r - b) ** 2))
    dof = len(r) - 1
    q_p = float(1 - stats.chi2.cdf(q, dof)) if dof > 0 else np.nan
    se = se_fixed * float(np.sqrt(max(1.0, q / dof))) if dof > 0 else se_fixed
    return b, se, q, q_p, r, rse


def weighted_median(bx, by, sx, sy, nboot=2000):
    r, rse = ratio_stats(bx, by, sx, sy)
    w = 1 / rse ** 2
    order = np.argsort(r)
    r_s, w_s = r[order], w[order]
    cum = np.cumsum(w_s) / w_s.sum()
    k = int(np.searchsorted(cum, 0.5))
    b = float(r_s[min(k, len(r_s) - 1)])
    boots = np.empty(nboot)
    for i in range(nboot):
        idx = RNG.integers(0, len(r), len(r))
        rb, wb = r[idx], w[idx]
        o = np.argsort(rb)
        cb = np.cumsum(wb[o]) / wb.sum()
        kk = int(np.searchsorted(cb, 0.5))
        boots[i] = rb[o][min(kk, len(rb) - 1)]
    se = float(boots.std(ddof=1))
    p = float(2 * (1 - stats.norm.cdf(abs(b / se)))) if se > 0 else np.nan
    return b, se, p


def egger(bx, by, sx, sy):
    r, rse = ratio_stats(bx, by, sx, sy)
    w = 1 / rse ** 2
    X = np.column_stack([np.ones_like(bx), bx])
    beta = np.linalg.lstsq(X.T @ (w[:, None] * X), X.T @ (w * by), rcond=None)[0]
    dof = len(bx) - 2
    if dof <= 0:
        return (np.nan,) * 6
    resid = by - X @ beta
    sigma2 = float((w * resid ** 2).sum() / dof)
    cov = sigma2 * np.linalg.inv(X.T @ (w[:, None] * X))
    se_s, se_i = np.sqrt(cov[1, 1]), np.sqrt(cov[0, 0])
    p_s = float(2 * (1 - stats.t.cdf(abs(beta[1] / se_s), dof)))
    p_i = float(2 * (1 - stats.t.cdf(abs(beta[0] / se_i), dof)))
    return float(beta[1]), float(se_s), p_s, float(beta[0]), float(se_i), p_i


print('\n=== 4. MR 估计 ===')
res = []
for taxon, g in h.groupby('taxon'):
    bx, by = g['bx'].values, g['by'].values
    sx, sy = g['sx'].values, g['sy'].values
    n = len(g)
    rec = dict(taxon=taxon, cell_name=CELL_NAME.get(taxon, taxon), n_snp=n,
               f_mean=float(np.mean(g['f_stat'])))
    if n >= 2:
        b, se, q, qp, r, rse = ivw(bx, by, sx, sy)
        rec.update(ivw_beta=b, ivw_se=se, ivw_or=float(np.exp(b)),
                   ivw_or_lo=float(np.exp(b - 1.96 * se)), ivw_or_hi=float(np.exp(b + 1.96 * se)),
                   ivw_p=float(2 * (1 - stats.norm.cdf(abs(b / se)))),
                   q=q, q_p=qp)
        loo = []
        for i in range(n):
            m_ = np.ones(n, bool)
            m_[i] = False
            if m_.sum() >= 1:
                b_loo, se_loo, *_ = ivw(bx[m_], by[m_], sx[m_], sy[m_])
                loo.append(b_loo)
        rec['loo_max_shift'] = float(np.max(np.abs(np.array(loo) - b))) if loo else np.nan
    if n >= 3:
        wm_b, wm_se, wm_p = weighted_median(bx, by, sx, sy)
        rec.update(wm_beta=wm_b, wm_or=float(np.exp(wm_b)), wm_se=wm_se, wm_p=wm_p)
        eg_b, eg_se, eg_p, ic, ic_se, ic_p = egger(bx, by, sx, sy)
        rec.update(egger_beta=eg_b, egger_se=eg_se, egger_p=eg_p,
                   egger_intercept=ic, egger_intercept_se=ic_se, egger_intercept_p=ic_p)
    res.append(rec)

res = pd.DataFrame(res)
bris = pd.read_csv(ROOT / 'IBS/gmrepo/results_bristol_genus/biomarkers.csv')
sev = pd.read_csv(ROOT / 'IBS/gmrepo/results_clinical_severity/biomarkers.csv')
res = res.merge(bris[['taxon', 'spearman', 'q']].rename(
    columns={'taxon': 'cell_name', 'spearman': 'obs_rho_bristol', 'q': 'obs_q_bristol'}),
    on='cell_name', how='left')
res = res.merge(sev[['taxon', 'clr_effect', 'q']].rename(
    columns={'taxon': 'cell_name', 'clr_effect': 'obs_effect_severity', 'q': 'obs_q_severity'}),
    on='cell_name', how='left')


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.clip(qv, 0, 1)
    q[ok] = qq
    return q


res['ivw_fdr'] = bh(res['ivw_p'])
res = res.sort_values('ivw_p')
res.to_csv(OUT / 'mr_results.csv', index=False)
pd.set_option('display.width', 250)
cols_show = ['cell_name', 'taxon', 'n_snp', 'ivw_or', 'ivw_p', 'ivw_fdr', 'wm_p',
             'egger_p', 'egger_intercept_p', 'q_p', 'obs_rho_bristol', 'obs_q_bristol']
print(res[cols_show].to_string(index=False))
print('\nsaved:', OUT / 'mr_results.csv')
