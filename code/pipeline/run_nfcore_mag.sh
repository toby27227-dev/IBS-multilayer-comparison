#!/usr/bin/env bash
# 用 nf-core/mag 对 fetchngs 产出的 fastq 做统一 QC + 组装 + 物种/功能谱。
# 用法: bash run_nfcore_mag.sh data/fetchngs/results/samplesheet/samplesheet.csv [outdir]
# 依赖: nextflow, docker/singularity, 参考数据库
set -euo pipefail

INPUT="${1:?需要 nf-core/mag samplesheet.csv}"
OUTDIR="${2:-results/mag}"

# 如有人类宿主污染，建议提供 --host_genome 的 bowtie2 索引
nextflow run nf-core/mag \
  -r 3.2.1 \
  -profile docker \
  --input "$INPUT" \
  --outdir "$OUTDIR" \
  --run_metaphlan true \
  --metaphlan_db "$METAPHLAN_DB" \
  --skip_quast false \
  -resume

# 关键输出（供下游有监督分析）：
#   $OUTDIR/taxonomy/metaphlan/merged_abundance_table_species.tsv
#   $OUTDIR/taxonomy/metaphlan/merged_abundance_table_genus.tsv
#   $OUTDIR/functional_annotation/... (eggNOG/KEGG)
echo "物种谱: $OUTDIR/taxonomy/metaphlan/"
