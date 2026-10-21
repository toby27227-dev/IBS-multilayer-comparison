#!/usr/bin/env bash
# 用 nf-core/fetchngs 从 ENA/SRA 批量下载原始 fastq。
# 用法: bash fetchngs.sh PRJEB37924 PRJEB34103
# 依赖: nextflow, docker/singularity, 网络
set -euo pipefail

OUTDIR="${OUTDIR:-data/fetchngs}"
mkdir -p "$OUTDIR"

# 把 BioProject accession 写成 fetchngs 接受的 ids 文件
: > "$OUTDIR/ids.csv"
for acc in "$@"; do
  echo "$acc" >> "$OUTDIR/ids.csv"
done

nextflow run nf-core/fetchngs \
  -r 1.12.0 \
  -profile docker \
  --input "$OUTDIR/ids.csv" \
  --outdir "$OUTDIR/results" \
  -resume

echo "FASTQ 输出: $OUTDIR/results/fastq"
echo "样本表:     $OUTDIR/results/samplesheet/samplesheet.csv"
