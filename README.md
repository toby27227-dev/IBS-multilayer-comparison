# IBS multi-omics layer comparison — analysis code

Code accompanying the manuscript:

> A layer-wise multi-omics comparison identifies the NMR metabolome as the layer with the highest point-estimate OOF R² for stool consistency regression (Bristol) in IBS: subject-grouped cross-validation, label-blind transductive stability selection, and held-out zero-shot validation

This repository contains analysis code only. Raw and processed data are **not** included. Derived summary tables are provided separately as Supplementary Data S1; processed multi-omics matrices are available from Mendeley Data (see below).

## Data sources

- Cell 2020 cohort: ENA `PRJEB37924`; processed multi-omics matrices: Mendeley Data `10.17632/29n2z5r5ph.3`
- Jacobs 2023 external cohort: `PRJNA812699`
- Taxonomy and functional abundance: GMrepo
- MR: MiBioGen (Kurilshikov et al., 2021) and IBS GWAS `GCST90016564` (Eijsbouts et al., 2021)

## Structure

```
code/
  IBS.py                     main exploratory analysis
  pipeline/                  feature construction and supervised models
  cellomics/                 Cell 2020 multi-omics analyses
  gmrepo/                    GMrepo taxonomy analyses and domain adaptation
  jacobs2023/                Jacobs 2023 data preparation
  jacobs_validation/         zero-shot validation and stability selection
  mr/                        two-sample Mendelian randomization
  revision_p0/               P0 revision analyses (classification uncertainty,
                             external baselines, sample flow)
```

## Environment

Tested with Python 3.13.13:

```
numpy==2.4.6
scipy==1.17.1
pandas==3.0.3
scikit-learn==1.8.0
matplotlib==3.10.9
torch==2.13.0  # optional; CPU build used, only gmrepo/dann.py
```

Install with:

```
python -m pip install -r requirements.txt
```

A full environment snapshot is provided in `requirements-lock.txt`; the direct pins above are sufficient for the core analyses.

## Notes

- Scripts resolve paths relative to the current working directory (`Path.cwd()`). Run them from the project root that contains `IBS/`, and place the downloaded data in the corresponding `IBS/...` directories before running.
- The code is provided for review and reproducibility. No license is granted for redistribution.
- The manuscript is under review; please do not cite without permission.
