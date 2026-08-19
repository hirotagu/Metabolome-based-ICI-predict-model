# ICI metabolomics validation analyses

This repository contains a consolidated public implementation of the Python
analyses used to compare repeated nested cross-validation with intentionally
non-nested validation in the ICI metabolomics study. It preserves the recorded
analysis settings while removing duplicated scenario-specific script copies.
The workflow starts from the 17 analysis-ready matrices released as
`Supplementary Data 1.xlsx`.

The code is intended for research reproducibility and method inspection. It
does not provide a clinically deployable prediction model or a prespecified
clinical decision threshold.

## Contents

| File | Purpose |
|---|---|
| `prepare_inputs.py` | Validates and converts the 17 sheets in Supplementary Data 1 into the canonical input workbooks used by the analysis. |
| `ICI_predict.py` | Primary repeated nested CV and the KNN, kmax30, outer3, regularization-grid, and SD sensitivity configurations. |
| `ICI_analysis_common.py` | Shared model-fitting, validation, metrics, bootstrap, receipt, and provenance functions. This module is imported by the other scripts and is not run directly. |
| `Compare_method.py` | Fully non-nested analysis and comparison with the primary nested-CV estimates. |
| `Compare_nest.py` | Selective-nesting diagnostic for the six formal pre-ICI cohorts (Datasets 1-5 and 7). |
| `Compare_sensitivity.py` | Scenario-matched nested versus non-nested sensitivity comparisons. |
| `Holdout.py` | Prespecified seed-42 holdout and 25 repeated stratified holdouts. Dataset 3 Post 2 automatically uses the required higher elastic-net iteration ceiling. |
| `Metrics_Calibration.py` | Discrimination, Brier score, and calibration analyses from saved predictions. |

Publication-only figure assembly and manual manuscript table formatting are not
part of this repository. The submitted figures, tables, and source-data files
are provided with the article.

## Software environment

The analyses were run with Python 3.11.14 and the package versions pinned in
`requirements.txt`:

- numpy 2.3.3
- pandas 2.3.3
- scikit-learn 1.7.2
- scipy 1.16.2
- openpyxl 3.1.5

Example environment setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the public inputs

Place `Supplementary Data 1.xlsx` beside the scripts, then run:

```bash
python prepare_inputs.py --input "Supplementary Data 1.xlsx"
```

The script creates `raw/` with 17 one-sheet workbooks and `input_qc.csv`. It
checks sheet order, IDs, binary outcomes, numeric feature values, cohort counts,
feature counts, and output hashes. Existing inputs are never overwritten.

| Source sheet | Canonical analysis key |
|---|---|
| Sheet1 | dataset1_pre |
| Sheet2 | dataset2_pre |
| Sheet3 | dataset3_pre |
| Sheet4 | dataset4_pre |
| Sheet5 | dataset5_pre |
| Sheet6 | dataset6_pre |
| Sheet7 | dataset7_pre |
| Sheet8 | dataset1_post1 |
| Sheet9 | dataset2_post1 |
| Sheet10 | dataset3_post1 |
| Sheet11 | dataset4_post1 |
| Sheet12 | dataset5_post1 |
| Sheet13 | dataset6_post1 |
| Sheet14 | dataset2_post2 |
| Sheet15 | dataset3_post2 |
| Sheet16 | dataset4_post2 |
| Sheet17 | dataset5_post2 |

## Run the analyses

Run the following commands from the repository root. Each script writes to a
separate subdirectory under `results_revision/` and refuses to overwrite a
completed run.

```bash
# Primary repeated nested CV: 17 matrices
python ICI_predict.py --scenario primary

# Primary nested versus fully non-nested comparison
python Compare_method.py

# Selective-nesting diagnostic: Datasets 1-5 and 7, pre-ICI
python Compare_nest.py

# Publicly reproducible sensitivity scenarios
python ICI_predict.py --scenario knn
python ICI_predict.py --scenario kmax30
python ICI_predict.py --scenario outer3
python ICI_predict.py --scenario reggrid
python Compare_sensitivity.py

# Repeated holdout and calibration
python Holdout.py
python Metrics_Calibration.py
```

These analyses are computationally intensive. Do not rename or edit the Python
files between dependent runs: downstream scripts verify producer and artifact
SHA256 hashes. If code or settings are changed, start again in a clean output
directory rather than mixing outputs from different versions.

## Analysis scenarios

`ICI_predict.py` contains one modeling implementation. The command-line
scenario selects only the prespecified targets and settings:

| Scenario | Analysis |
|---|---|
| `primary` | Primary repeated nested CV on all 17 matrices |
| `knn` | KNN-imputation sensitivity analysis for Dataset 2 Pre |
| `kmax30` | Maximum top-k increased from 10 to 30 on all 17 matrices |
| `outer3` | Three-fold outer-CV sensitivity analysis on all 17 matrices |
| `reggrid` | Expanded regularization-grid sensitivity analysis on Datasets 1-5 and 7 Pre |
| `sd` | Stable disease recoded as responder/non-responder for Dataset 1 Pre and Post 1 |

### Stable-disease sensitivity analysis

The two alternative stable-disease recodings used by the `sd` scenario are not
included in Supplementary Data 1. The scenario code is retained for transparent
method reporting, but this branch cannot be reproduced from the publicly
provided workbook alone. Consequently, the public default in
`Compare_sensitivity.py` compares `knn`, `kmax30`, `outer3`, and `reggrid`; it
does not silently claim to reproduce the unavailable SD branch.

## Reproducibility safeguards

- Random seed: 42.
- Imputation, scaling, feature ranking, feature selection, and tuning are fitted
  within training partitions.
- Primary discrimination is calculated from held-out, patient-level averaged
  out-of-fold predictions.
- Sample counts, settings, software versions, warnings, output hashes, and
  producer hashes are recorded in run receipts.
- Bootstrap intervals are conditional on the saved patient-level averaged OOF
  predictions and do not include uncertainty from repartitioning, feature
  selection, tuning, or model refitting.
- The repeated holdout splits overlap and are used descriptively rather than as
  independent replicates.

## Generated files

The principal run directories are:

```text
results_revision/All_data_primary
results_revision/Compare_method_v1
results_revision/Compare_nest_v1
results_revision/All_data_KNN
results_revision/All_data_Kmax30
results_revision/All_data_outer3
results_revision/All_data_RegGrid
results_revision/Compare_sensitivity_v1
results_revision/Holdout_v1
results_revision/Metrics_Calibration_v1
```

Generated input and result directories are excluded by `.gitignore`.
