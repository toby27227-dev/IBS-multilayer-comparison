"""对 DMM 三个亚型做生物学解释与临床关联探索。

1. Driver taxa：CLR 上的 Kruskal-Wallis + BH-FDR，并用 log-ratio 效应量排序
2. 两两亚型差异：Mann-Whitney U（log-ratio），每对比较内 BH-FDR
3. 临床表型：亚型 × 项目/国家/性别（卡方）、年龄（Kruskal-Wallis），统一 BH-FDR
4. 趋势代理：各亚型到"健康对照中心"的 CLR 距离
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import IBS


def main():
    abund_all = pd.read_csv(HERE / 'abundance_genus.csv', index_col=0)
    meta_all = pd.read_csv(HERE / 'gmrepo_metadata.csv', index_col=0)
    labels = pd.read_csv(HERE / 'dmm_labels.csv', index_col=0).iloc[:, 0]
    labels.index = labels.index.astype(str)

    ibs_ids = labels.index.intersection(abund_all.index).intersection(meta_all.index)
    labels = labels.loc[ibs_ids]
    subtypes = sorted(labels.unique())

    # 与主流程一致的 log-ratio 变换
    vals = abund_all.values
    pseudo = IBS.resolve_pseudo(vals, IBS.PSEUDOCOUNT)
    clr = pd.DataFrame(IBS.clr_transform(vals, pseudo, IBS.CLR_METHOD),
                       index=abund_all.index, columns=abund_all.columns)
    clr_ibs = clr.loc[ibs_ids]
    rel_ibs = abund_all.loc[ibs_ids]

    # ---------- 1) 全局差异丰度（log-ratio）----------
    rows = []
    for g in abund_all.columns:
        groups = [clr_ibs.loc[labels == s, g].values for s in subtypes]
        groups = [x for x in groups if len(x) > 1]
        H, p = kruskal(*groups)
        means = {s: clr_ibs.loc[labels == s, g].mean() for s in subtypes}
        relab = {s: rel_ibs.loc[labels == s, g].mean() for s in subtypes}
        row = {'genus': g, 'kw_H': H, 'kw_p': p,
               **{f'clr_{s}': means[s] for s in subtypes},
               **{f'relab_{s}': relab[s] for s in subtypes}}
        for s in subtypes:
            row[f'enrich_{s}'] = means[s] - np.mean([means[o] for o in subtypes if o != s])
        rows.append(row)
    drv = pd.DataFrame(rows)
    drv['kw_q'] = IBS.bh_fdr(drv['kw_p'].values)

    # ---------- 2) 两两亚型差异（每对 BH 校正）----------
    print('=== 两两亚型差异（CLR, Mann-Whitney U, BH-FDR）===')
    for a, b in combinations(subtypes, 2):
        pvals, diffs = [], []
        for g in abund_all.columns:
            xa = clr_ibs.loc[labels == a, g].values
            xb = clr_ibs.loc[labels == b, g].values
            try:
                _, p = mannwhitneyu(xa, xb, alternative='two-sided')
            except ValueError:
                p = 1.0
            pvals.append(p)
            diffs.append(xa.mean() - xb.mean())
        q = IBS.bh_fdr(np.array(pvals))
        drv[f'{a}_vs_{b}_p'] = pvals
        drv[f'{a}_vs_{b}_q'] = q
        drv[f'{a}_vs_{b}_clrdiff'] = diffs

        order = np.argsort(-np.abs(diffs))
        print(f'\n[{a} vs {b}]  显著(q<0.05): {int((q < 0.05).sum())}/{len(q)}')
        shown = 0
        for i in order:
            if q[i] < 0.05:
                g = abund_all.columns[i]
                fc = np.exp(diffs[i])
                print(f"  {g[:40]:40s} ΔCLR={diffs[i]:+.2f} fold={fc:.2f} q={q[i]:.3g}")
                shown += 1
                if shown >= 8:
                    break

    drv.sort_values('kw_q').to_csv(HERE / 'differential_taxa.csv', index=False)

    print('\n=== 各亚型 Top 驱动属（按 CLR 富集，全局 BH-FDR）===')
    for s in subtypes:
        top = drv.sort_values(f'enrich_{s}', ascending=False).head(6)
        print(f'\n[{s}]  n={int((labels == s).sum())}')
        for _, r in top.iterrows():
            print(f"  {r['genus'][:40]:40s} enrich={r[f'enrich_{s}']:+.2f} "
                  f"relab%={r[f'relab_{s}']:.2f} q={r['kw_q']:.3g}")

    print('\n=== 经典 enterotype 标记属 ===')
    for g in ['Bacteroides', 'Prevotella', 'Paraprevotella', 'Ruminococcus',
              'Blautia', 'Roseburia', 'Faecalibacterium']:
        row = drv[drv['genus'] == g]
        if row.empty:
            print(f'  {g}: 不在矩阵中')
            continue
        r = row.iloc[0]
        vals = ', '.join(f"{s}:{r[f'enrich_{s}']:+.2f}({r[f'relab_{s}']:.1f}%)"
                         for s in subtypes)
        print(f'  {g:16s} q={r["kw_q"]:.3g} | {vals}')

    # 热图
    top_genera = set()
    for s in subtypes:
        top_genera.update(drv.sort_values(f'enrich_{s}', ascending=False).head(6)['genus'])
    hm = drv.set_index('genus').loc[sorted(top_genera), [f'enrich_{s}' for s in subtypes]]
    fig, ax = plt.subplots(figsize=(6.5, max(6, 0.32 * len(hm))))
    vmax = np.abs(hm.values).max()
    im = ax.imshow(hm.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(subtypes)))
    ax.set_xticklabels(subtypes)
    ax.set_yticks(range(len(hm)))
    ax.set_yticklabels([q[:38] for q in hm.index], fontsize=8)
    ax.set_title('Driver genera: CLR enrichment by subtype')
    fig.colorbar(im, ax=ax, shrink=0.7, label='CLR enrichment')
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'driver_taxa_heatmap.png', dpi=150)
    plt.close(fig)

    # ---------- 3) 临床/人口学关联（BH 校正）----------
    print('\n=== 亚型 × 可得临床/人口学变量（BH-FDR）===')
    tests = []
    for col in ['project_id', 'country', 'sex']:
        if col not in meta_all.columns:
            continue
        sub = meta_all.loc[ibs_ids, col]
        tab = pd.crosstab(labels, sub)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        stat, p, _, _ = chi2_contingency(tab)
        tests.append({'variable': col, 'test': 'chi2', 'stat': stat, 'p': p})
        print(f'\n[{col}] 卡方 p={p:.3g}')
        print(tab)

    age = pd.to_numeric(meta_all.loc[ibs_ids, 'host_age'], errors='coerce')
    groups = [age[labels == s].dropna().values for s in subtypes]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) > 1:
        H, p = kruskal(*groups)
        tests.append({'variable': 'host_age', 'test': 'kruskal', 'stat': H, 'p': p})
        print(f'\n[host_age] H={H:.3f}, p={p:.3g}, n={int(age.notna().sum())}/{len(age)}')
        print('  各亚型年龄均值:', {s: round(float(age[labels == s].mean()), 1)
                                    for s in subtypes})

    if tests:
        q = IBS.bh_fdr([t['p'] for t in tests])
        print('\n临床关联汇总（BH 校正后）:')
        for t, qi in zip(tests, q):
            print(f"  {t['variable']:<14s} {t['test']:<7s} p={t['p']:.3g}  q={qi:.3g}")
        pd.DataFrame([{**t, 'q': qi} for t, qi in zip(tests, q)]).to_csv(
            HERE / 'clinical_association.csv', index=False)

    # ---------- 4) 健康趋势代理 ----------
    health_ids = meta_all.index[meta_all['health_state'] == 'Health']
    if len(health_ids) > 0:
        centroid = clr.loc[health_ids].mean(axis=0)
        dist = pd.Series(np.linalg.norm(clr.loc[ibs_ids].values - centroid.values, axis=1),
                         index=ibs_ids)
        print('\n=== 各亚型到健康对照中心的平均 CLR 距离（越小越接近健康）===')
        for s in subtypes:
            print(f'  {s}: {dist[labels == s].mean():.3f} (n={int((labels == s).sum())})')


if __name__ == '__main__':
    main()
