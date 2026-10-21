#!/usr/bin/env bash
# 16S/18S amplicon 统一 Profiling（用于 PRJNA812699 等扩增子队列）。
# 用法: bash run_nfcore_ampliseq.sh samplesheet.csv [outdir]
# 依赖: nextflow, docker/singularity, SILVA/GTDB 参考库
set -euo pipefail

INPUT="${1:?需要 nf-core/ampliseq samplesheet.csv}"
OUTDIR="${2:-results/ampliseq}"

nextflow run nf-core/ampliseq \
  -r 2.9.0 \
  -profile docker \
  --input "$INPUT" \
  --outdir "$OUTDIR" \
  --dada_ref_taxonomy silva=138 \
  --skip_qiime \
  -resume

# 关键输出（供下游有监督分析）：
#   $OUTDIR/dada2/ASV_table/ASV_table.tsv
#   $OUTDIR/dada2/ASV_tax_silva.tsv
#   $OUTDIR/qiime2/rel_abundance_tables/rel_table-ASV.tsv (若未 skip qiime)
echo "ASV 表: $OUTDIR/dada2/ASV_table/ASV_table.tsv"
