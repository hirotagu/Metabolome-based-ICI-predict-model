# ICI metabolomics validation analyses

Research code for comparing nested, intentionally non-nested, selective-nesting,
and holdout validation in the ICI metabolomics study.

The public input data comprise **12 analysis-ready matrices from Datasets 3–7**,
provided in Supplementary Data 1. The workflow can use all 12 matrices or any
nonempty subset of them. Dataset numbers and time-point labels follow the
manuscript.

## Data availability

- **Supplementary Data 1 contains 12 matrices for Datasets 3–7.** The table below
  identifies every public matrix and its required input name.
- **Individual-level data for Datasets 1 and 2 are not publicly deposited** and
  are not included in Supplementary Data 1 or this repository. Enquiries about
  access should be directed to the corresponding author identified in the
  manuscript. Any provision requires the applicable data-owner permissions and
  access conditions.
- Supplementary Tables S1–S9 report dataset-specific methods and aggregate
  results for the study, including Datasets 1 and 2.
- This repository and its release archives provide code and documentation.
  They do not contain individual-level matrices or patient-level prediction files.

The public source studies are:

| Datasets | Source |
|---|---|
| 3–5 | Li et al., https://doi.org/10.1038/s41467-019-12361-9 |
| 6 | Darragh et al., https://doi.org/10.1038/s43018-022-00450-6 |
| 7 | Costantini et al., https://doi.org/10.1186/s13046-025-03378-8 |

These correspond to manuscript references 13–15. Use the analysis-ready
matrices in Supplementary Data 1 for the workflow below. Source-study
workbooks require study-specific reconstruction and are not direct inputs to
`prepare_inputs.py`.

## Files

| File | Purpose |
|---|---|
| `prepare_inputs.py` | Validate supplied matrices and create one input workbook per matrix. |
| `dataset_selection.py` | Dataset selection and input-coverage records. |
| `ICI_predict.py` | Nested CV and primary/sensitivity scenario settings. |
| `ICI_analysis_common.py` | Shared fitting, metrics, artifact validation, and provenance. |
| `Compare_method.py` | Intentionally non-nested comparison with nested CV. |
| `Compare_nest.py` | Selective nesting for available formal pre-ICI datasets. |
| `Compare_sensitivity.py` | Scenario-matched nested/non-nested comparisons. |
| `Holdout.py` | Seed-42 holdout and 25 repeated stratified holdouts. |
| `Metrics_Calibration.py` | Saved-prediction metrics and bin-wise Wilson intervals. |
| `Revision2_reporting.py` | Dataset-omission, LC/MS-only, paired-repeat and KNN summaries; no fitting. |
| `Revision2_calibration.py` | Wilson-bin tables from saved OOF predictions; no fitting. |
| `Revision2_figures.py` | Calibration figures from validated aggregate Wilson-bin tables. |
| `tests/` | Input-subset and numerical reporting tests. |

## Environment

Use Python 3.11 with the scientific package versions specified in
`requirements.txt`. The manuscript analyses used Python 3.11.14 and 3.11.16.
Validation details are recorded in `REVISION2_VALIDATION.md`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select this environment in Positron. Run the commands below from the repository
folder. Script settings are near the top of each file. Model runs are
computationally intensive.

## Prepare the 12 public input matrices

`prepare_inputs.py` identifies each dataset from its worksheet name. For the
public workbook, **use the canonical worksheet names in the rightmost column
below before running the converter**. Dataset identity is not inferred from
sheet position or from the workbook README.

| Supplementary Data 1 sheet | Dataset | Time point | Canonical worksheet name |
|---|---|---|---|
| Sheet1 | 3 | Pre | `dataset3_pre` |
| Sheet2 | 4 | Pre | `dataset4_pre` |
| Sheet3 | 5 | Pre | `dataset5_pre` |
| Sheet4 | 6 | Pre | `dataset6_pre` |
| Sheet5 | 7 | Pre | `dataset7_pre` |
| Sheet6 | 3 | Post1 | `dataset3_post1` |
| Sheet7 | 4 | Post1 | `dataset4_post1` |
| Sheet8 | 5 | Post1 | `dataset5_post1` |
| Sheet9 | 6 | Post1 | `dataset6_post1` |
| Sheet10 | 3 | Post2 | `dataset3_post2` |
| Sheet11 | 4 | Post2 | `dataset4_post2` |
| Sheet12 | 5 | Post2 | `dataset5_post2` |

1. Make a working copy of Supplementary Data 1.
2. Rename its data-sheet tabs using the canonical names above. If they already
   have those names, no renaming is needed. Update the sheet-name column in the
   workbook README to match. Keep IDs, labels and feature values unchanged.
3. Save the copy beside the scripts as `Supplementary Data 1_canonical.xlsx`.
4. Run the converter with a new or empty `raw/` output directory:

```bash
python prepare_inputs.py --input "Supplementary Data 1_canonical.xlsx"
```

Do not pass a workbook whose data tabs are still named Sheet1–Sheet12 directly
to the converter. Complete the worksheet-name preparation above first.

The converter checks IDs, labels, numeric values and the expected sample,
class and feature counts. It writes one workbook per supplied matrix to
`raw/`, named using the canonical key, such as `dataset3_pre.xlsx`. It also
writes `input_qc.csv` and `input_coverage.csv`, including file hashes.
Existing output content is not overwritten.

**Any nonempty subset of the 12 public matrices may be used.** To run a
subset, retain only the desired data sheets in the working copy, with
their canonical names unchanged. A workbook README sheet is optional.
Unknown names, duplicate mappings, malformed present inputs and explicitly
requested missing matrices cause an error. Missing inputs are recorded as
`not_provided`.

Alternatively, already prepared workbooks named using the canonical keys in
the table can be placed in `raw/` directly. Each must contain a single `Sheet1`
with `id`, `category` (or the supported `Responder` column), and numeric
features. Keep unrelated workbooks outside `raw/`.

## Run the analyses

```bash
python ICI_predict.py --scenario primary
python Compare_method.py
python Compare_nest.py

python ICI_predict.py --scenario knn
python ICI_predict.py --scenario kmax30
python ICI_predict.py --scenario outer3
python ICI_predict.py --scenario reggrid
python Compare_sensitivity.py

python Holdout.py
python Metrics_Calibration.py
```

Each script uses the intersection of the supplied inputs and the datasets
eligible for its analysis. Among the public inputs, the formal primary pre-ICI
cohorts and the KNN/expanded-grid sensitivity cohorts are **Datasets 3, 4, 5
and 7 Pre**. Dataset 6 is evaluated separately as a descriptive analysis.

To request particular matrices explicitly:

```bash
python ICI_predict.py --scenario knn --datasets dataset3_pre dataset4_pre
```

Downstream scripts expose `DATASETS_TO_RUN` and run-root settings. Selected
inputs must have their required completed upstream outputs. A missing result
for a selected input is an error, rather than an instruction to omit that input.

The public data support analyses of the supplied Datasets 3–7 matrices.
They do not reproduce the individual-level analyses of Datasets 1 and 2 or
summaries that require every study cohort. Six-cohort nominal tests are marked
`not_run` when any required cohort is absent. Each analysis records its input
coverage so that public-data results can be distinguished from the full-study
results reported in Supplementary Tables S1–S9.

## Aggregate reporting from completed runs

`Revision2_reporting.py` reads saved analysis artifacts and exports aggregate
tables, coverage, and input/output hashes. It does not select features, tune
parameters, fit models, or export patient-level predictions in its tables.
`Revision2_calibration.py` creates bin-wise Wilson-interval tables from saved
OOF predictions. These scripts require the completed outputs of the analyses
to be summarized.

For outputs generated with the default run locations:

```bash
python Revision2_reporting.py --knn-root results_revision/All_data_KNN --sensitivity-root results_revision/Compare_sensitivity_v1
python Revision2_calibration.py --out results_revision/round2_calibration
```

Adjust input roots for your own run directories; see
`python Revision2_reporting.py --help` and
`python Revision2_calibration.py --help`. If KNN results are unavailable, omit
`--knn-root` and `--sensitivity-root` to generate the primary-analysis summaries
with KNN marked `not_requested`.

To render calibration figures from aggregate bins:

```bash
python -m pip install -r requirements-figures.txt
python Revision2_figures.py --input-dir results_revision/round2_calibration --out results_revision/round2_figures
```

Set `--input-dir` to the output directory from `Revision2_calibration.py`.
The renderer checks bin-table hashes and coverage before plotting, uses
red/black curves with vertical bin-wise Wilson error bars, and labels partial
cohort coverage. The default font is Arial; use `--font "DejaVu Sans"` if an
alternative is needed. The actual font is recorded in the output.

## Interpretation and provenance

Primary nCV fits data-dependent preprocessing and selection within training
partitions, apart from the documented initial all-missing feature removal.
Intentionally non-nested and selective-nesting comparisons use the global
ranking/tuning or eligibility decisions described in Methods. They are not
fully nested pipelines.

Bootstrap intervals condition on saved patient-level averaged OOF predictions.
Wilson intervals condition on saved predictions and bin assignments; they are
bin-wise and not simultaneous confidence bands. Neither quantifies full
model-development uncertainty. Repeat-specific CV results and repeated-holdout
variation are descriptive. Patient-averaged OOF AUC and the mean of
repeat-specific AUCs are different quantities. KNN results assess preprocessing
sensitivity, rather than the effect of imputation alone.

Keep completed results with their producer scripts and receipts. The reporting
scripts check producer identities and artifact hashes; outputs must retain the
provenance of the code that generated them. Source provenance is documented in
`REVISION2_SOURCE_AUDIT.md` and `REVISION2_SOURCE_PROVENANCE.json`.

The MIT code license does not grant data-sharing permissions. Clinical use
requires independent validation.

## Validation

```bash
python -m unittest discover -s tests -v
```

See `REVISION2_VALIDATION.md` and `REVISION2_CHANGES.md` for validation records
and implementation details.
