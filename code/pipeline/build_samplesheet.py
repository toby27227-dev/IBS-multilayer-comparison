"""从 ENA 生成 nf-core/mag samplesheet（本地 fastq 路径）。

用法:
  python build_samplesheet.py --accession PRJEB37924 --fastq-dir data/PRJEB37924 \
      --out samplesheet.csv

会查询 ENA read_run 的 fastq_ftp，按 <run>_1.fastq.gz / <run>_2.fastq.gz 命名，
输出列: sample,group,fastq_1,fastq_2
"""
import argparse
import csv
import io
import urllib.request
from pathlib import Path

ENA = 'https://www.ebi.ac.uk/ena/portal/api/filereport'


def fetch(accession):
    url = (f'{ENA}?accession={accession}&result=read_run'
           '&fields=run_accession,fastq_ftp,library_layout&format=tsv')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return list(csv.DictReader(io.StringIO(
        urllib.request.urlopen(req, timeout=60).read().decode()), delimiter='\t'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--accession', required=True)
    ap.add_argument('--fastq-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    fq = Path(args.fastq_dir)
    rows = []
    for r in fetch(args.accession):
        run = r['run_accession']
        ftp = (r.get('fastq_ftp') or '').split(';')
        f1 = fq / f'{run}_1.fastq.gz'
        f2 = fq / f'{run}_2.fastq.gz'
        if not f1.exists() and ftp and ftp[0]:
            f1 = fq / Path(ftp[0]).name.replace('.fastq.gz', '_1.fastq.gz')
        if len(ftp) > 1 and ftp[1]:
            f2 = fq / Path(ftp[1]).name.replace('.fastq.gz', '_2.fastq.gz')
        else:
            f2 = None
        rows.append({'sample': run, 'group': args.accession,
                     'fastq_1': str(f1), 'fastq_2': str(f2) if f2 else ''})

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample', 'group', 'fastq_1', 'fastq_2'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
