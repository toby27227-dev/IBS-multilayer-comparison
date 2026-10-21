"""完全嵌套敏感性（fully nested）：门控亦在重抽样内重算。

v2.4 方法学审查落地：原 nested_sensitivity.py 仅将 BH q<0.05 预筛选下沉，
π_uni_cell 与 KS 门值固定（已正名为"预筛选嵌套敏感性 pre-screening nested"）。
本脚本实现两级严格嵌套，采用两种外层设计：

  A（主，fully_nested.csv）          外层 = 受试者级全量 bootstrap（有放回，
     保持样本量与单变量功效），内层 = 原 π 协议（外层重抽样内 50% 受试者
     子抽样 × B_in=100）；此时 π 门重算的估计精度与原管线一致，
     频率下降只反映门/特征的真实不稳定性。
  B（保守，fully_nested_conservative_halved.csv）外层 = 50% 受试者子抽样
     （模拟"半数研究"），内层再取外层 50%（复合半样本，功效最低版本）。

每次重抽样内四关全重算：① BH q<0.05 单变量筛选（方向一致）② π_uni_cell
内层 bootstrap 重算 ③ KS 重算（Cell 子集 vs Jacobs 全量 X，仅用 X）
④ 预注册名可测性（与重抽样无关）。
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, t as t_dist

warnings.filterwarnings('ignore')

ROOT = Path.cwd()
CE = ROOT / 'IBS/cellomics'
NW = ROOT / 'IBS/jacobs2023'
OUT = ROOT / 'IBS/jacobs_validation'
B_OUT = 100
B_IN = 100


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def spear(Xm, ym):
    n = len(ym)
    Xr = rankdata(Xm, axis=0)
    yr = rankdata(ym)
    Xc = Xr - Xr.mean(0)
    yc = yr - yr.mean()
    rho = np.clip((Xc.T @ yc) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()),
                  -0.999999, 0.999999)
    tstat = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return rho, 2 * t_dist.sf(np.abs(tstat), n - 2)


nmr = pd.read_csv(CE / 'nmr.csv', index_col=0)
sm = pd.read_csv(CE / 'sample_map.csv', index_col=0)
tr = nmr.join(sm[['bristol', 'subject']]).dropna()
FEATS = list(nmr.columns)
X = np.log1p(tr[FEATS].values.astype(float))
y = tr['bristol'].values.astype(float)
g = tr['subject'].values.astype(str)
print(f'Cell 2020: n={len(tr)} | subjects={len(set(g))} | features={len(FEATS)}')

sel = json.load(open(OUT / 'frozen_selector.json', encoding='utf-8'))
FMAP = sel['feature_map']
IN_TARGET = np.array([f in FMAP for f in FEATS])

met = pd.read_csv(NW / 'jacobs2023_metabolites.csv', index_col=0)
ZJ = {}
for f in FEATS:
    if f in FMAP:
        xj = np.log1p(met[FMAP[f]].values.astype(float))
        ZJ[f] = (xj - xj.mean()) / xj.std()

subjects = np.unique(g)
idx_by_subject = {s: np.where(g == s)[0] for s in subjects}


def run_design(mode):
    rng = np.random.default_rng(42)
    cnt_screen = np.zeros(len(FEATS))
    cnt_pi = np.zeros(len(FEATS))
    cnt_ks = np.zeros(len(FEATS))
    cnt_full = np.zeros(len(FEATS))
    for b in range(B_OUT):
        if mode == 'bootstrap':
            pick = rng.choice(subjects, size=len(subjects), replace=True)
            pool = np.unique(pick)
        else:
            pick = rng.choice(subjects, size=int(0.5 * len(subjects)),
                              replace=False)
            pool = pick
        idx = np.concatenate([idx_by_subject[s] for s in pick])
        Xb, yb = X[idx], y[idx]
        rho_o, p_o = spear(Xb, yb)
        alive = bh_fdr(p_o) < 0.05
        sign_o = np.sign(rho_o)
        cnt_screen += alive

        counts = np.zeros(len(FEATS))
        for _ in range(B_IN):
            pick2 = rng.choice(pool, size=int(0.5 * len(pool)), replace=False)
            idx2 = np.concatenate([idx_by_subject[s] for s in pick2])
            rho_i, p_i = spear(X[idx2], y[idx2])
            counts += (p_i < 0.05) & (np.sign(rho_i) == sign_o)
        pi_b = counts / B_IN
        cnt_pi += (pi_b >= 0.8)

        ks_b = np.full(len(FEATS), np.nan)
        for j, f in enumerate(FEATS):
            if f in FMAP:
                xc = Xb[:, j]
                zc = (xc - xc.mean()) / xc.std()
                ks_b[j] = ks_2samp(zc, ZJ[f]).statistic
        ks_pass = ks_b < 0.15
        cnt_ks += ks_pass
        cnt_full += alive & IN_TARGET & (pi_b >= 0.8) & ks_pass
    return (cnt_screen / B_OUT, cnt_pi / B_OUT, cnt_ks / B_OUT,
            cnt_full / B_OUT)


scr = pd.read_csv(OUT / 'nested_sensitivity.csv')
fr_a = run_design('bootstrap')
print('\nsaved bootstrap design')
fr_b = run_design('halved')
print('saved halved design')

rows = []
for f in FEATS:
    d = sel['features'].get(f, {})
    j = FEATS.index(f)
    rows.append({
        'feature': f,
        'full_pi_uni_cell': d.get('pi_uni_cell', np.nan),
        'full_ks': d.get('ks', np.nan),
        'in_target': bool(f in FMAP),
        'prescreen_survival_fixed_gates': float(
            scr.loc[scr['feature'] == f, 'screen_survival'].iloc[0]),
        'screen_freq_bs': round(float(fr_a[0][j]), 3),
        'pi_gate_freq_bs': round(float(fr_a[1][j]), 3),
        'ks_gate_freq_bs': round(float(fr_a[2][j]), 3),
        'fully_nested_freq_bs': round(float(fr_a[3][j]), 3),
        'fully_nested_freq_halved': round(float(fr_b[3][j]), 3),
    })
res = pd.DataFrame(rows).sort_values('fully_nested_freq_bs', ascending=False)
res.to_csv(OUT / 'fully_nested.csv', index=False)
res[['feature', 'fully_nested_freq_bs', 'fully_nested_freq_halved']].to_csv(
    OUT / 'fully_nested_conservative_halved.csv', index=False)
print('\n' + res.to_string(index=False))

for tag, freqs in [('外层全量bootstrap', fr_a[3]), ('复合半样本(保守)', fr_b[3])]:
    core = [FEATS[j] for j in np.where(freqs >= 0.8)[0]]
    print(f'\n{tag} 核心 (freq>=0.8): {core if core else "无"}')
    print('  排序前5: ' + ' > '.join(
        f'{FEATS[j]}={freqs[j]:.2f}' for j in np.argsort(freqs)[::-1][:5]))
print(f'\nsaved: fully_nested.csv + fully_nested_conservative_halved.csv -> {OUT}')
