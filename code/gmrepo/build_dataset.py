"""从 GMrepo 原始下载文件构建 IBS 属/种相对丰度矩阵与元数据。

输入（raw/）：
  - sample_to_run_info.txt.gz                 所有 run 的元数据
  - species_abundance.txt.gz                  长表：run × taxon 相对丰度（含 genus/species 两个 rank）
  - superkingdom2descendents.txt.gz           ncbi_taxon_id -> scientific_name / rank
  - all_projects_contain_samples_of_D043183.tsv  含 IBS 样本的项目

输出（本目录）：
  - abundance_species.csv   样本 × 物种（相对丰度 %）
  - abundance_genus.csv     样本 × 属
  - gmrepo_metadata.csv            样本级临床/人群学元数据
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
IBS_MESH = 'D043183'          # Irritable Bowel Syndrome
HEALTHY_MESH = 'D006262'      # Health


def load_ibs_projects():
    p = pd.read_csv(RAW / 'all_projects_contain_samples_of_D043183.tsv', sep='\t')
    return p['project_id'].astype(str).tolist()


def load_metadata(ibs_projects):
    info = pd.read_csv(RAW / 'sample_to_run_info.txt.gz', sep='\t', dtype=str, low_memory=False)
    sub = info[info['project_id'].isin(ibs_projects)].copy()
    sub = sub[sub['disease'].isin([IBS_MESH, HEALTHY_MESH])]
    sub['health_state'] = sub['disease'].map({IBS_MESH: 'IBS', HEALTHY_MESH: 'Health'})
    sub = sub.drop_duplicates(subset='run_id')
    keep = ['run_id', 'project_id', 'sample_id', 'disease', 'phenotype', 'health_state',
            'host_age', 'sex', 'BMI', 'country', 'experiment_type', 'nr_reads_sequenced']
    meta = sub[[c for c in keep if c in sub.columns]].copy()
    meta['host_age'] = pd.to_numeric(meta['host_age'], errors='coerce')
    meta['BMI'] = pd.to_numeric(meta['BMI'], errors='coerce')
    return meta.set_index('run_id')


def load_taxonomy():
    tx = pd.read_csv(RAW / 'superkingdom2descendents.txt.gz', sep='\t', dtype=str)
    tx = tx[['ncbi_taxon_id', 'scientific_name', 'node_rank']].drop_duplicates('ncbi_taxon_id')
    return tx.set_index('ncbi_taxon_id')


def load_abundance(run_ids):
    """分块读取 348MB 长表，只保留目标 run，返回 (species_long, genus_long)。"""
    usecols = ['ncbi_taxon_id', 'taxon_rank_level', 'relative_abundance', 'accession_id']
    species, genus = [], []
    reader = pd.read_csv(
        RAW / 'species_abundance.txt.gz', sep='\t', usecols=usecols,
        dtype={'ncbi_taxon_id': str, 'taxon_rank_level': str,
               'accession_id': str, 'relative_abundance': float},
        chunksize=1_000_000, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk['accession_id'].isin(run_ids)]
        if chunk.empty:
            continue
        for rank, bucket in (('species', species), ('genus', genus)):
            part = chunk[chunk['taxon_rank_level'] == rank]
            if not part.empty:
                bucket.append(part[['accession_id', 'ncbi_taxon_id', 'relative_abundance']])
    sp = pd.concat(species, ignore_index=True) if species else pd.DataFrame()
    ge = pd.concat(genus, ignore_index=True) if genus else pd.DataFrame()
    return sp, ge


def make_matrix(long_df, taxonomy, rank):
    tx = taxonomy[taxonomy['node_rank'] == rank][['scientific_name']]
    df = long_df.merge(tx, left_on='ncbi_taxon_id', right_index=True, how='left')
    df['scientific_name'] = df['scientific_name'].fillna(
        'unclassified' if False else df['ncbi_taxon_id']).astype(str)
    # 同名物种合并（如不同 taxon id 同名），避免 pivot 报错
    df = df.groupby(['accession_id', 'scientific_name'], as_index=False)['relative_abundance'].sum()
    mat = df.pivot(index='accession_id', columns='scientific_name',
                   values='relative_abundance').fillna(0.0)
    mat.index.name = 'run_id'
    return mat


def main():
    ibs_projects = load_ibs_projects()
    print('IBS projects:', len(ibs_projects), ibs_projects)

    meta = load_metadata(ibs_projects)
    print('runs with IBS/Health label:', len(meta))
    print(meta['health_state'].value_counts().to_string())

    taxonomy = load_taxonomy()
    print('taxonomy entries:', len(taxonomy))

    sp_long, ge_long = load_abundance(set(meta.index))
    print('filtered abundance rows | species:', len(sp_long), '| genus:', len(ge_long))

    sp_mat = make_matrix(sp_long, taxonomy, 'species')
    ge_mat = make_matrix(ge_long, taxonomy, 'genus')

    # 只保留有丰度数据的样本，并按元数据顺序对齐
    common = meta.index.intersection(sp_mat.index)
    meta = meta.loc[common]
    sp_mat = sp_mat.loc[common]
    ge_mat = ge_mat.loc[common]
    print('final samples:', len(common))
    print('  species matrix:', sp_mat.shape, '| genus matrix:', ge_mat.shape)

    # 相对丰度：行和检查（GMrepo 以百分比给出）
    print('species row-sum median:', round(float(sp_mat.sum(axis=1).median()), 3))
    print('genus   row-sum median:', round(float(ge_mat.sum(axis=1).median()), 3))

    # 丢弃检出率极低（<1%）的列，减小文件
    sp_mat = sp_mat.loc[:, (sp_mat > 0).mean() >= 0.01]
    ge_mat = ge_mat.loc[:, (ge_mat > 0).mean() >= 0.01]
    print('after prevalence>=1% | species:', sp_mat.shape, '| genus:', ge_mat.shape)

    sp_mat.to_csv(HERE / 'abundance_species.csv')
    ge_mat.to_csv(HERE / 'abundance_genus.csv')
    meta.to_csv(HERE / 'gmrepo_metadata.csv')
    print('\nsaved:')
    for f in ['abundance_species.csv', 'abundance_genus.csv', 'gmrepo_metadata.csv']:
        print('  ', f, (HERE / f).stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
