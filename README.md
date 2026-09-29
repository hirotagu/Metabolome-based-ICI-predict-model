# ICI metabolomics validation analyses

Research code comparing nested, intentionally non-nested, selective-nesting,
and holdout validation in the ICI metabolomics study.

This revision adds revision-2 reporting and accepts any nonempty subset of
known study matrices. Dataset numbers retain their original meaning.
File presence does not establish permission to share data.

## Data availability

This repository and its release archives distribute code and documentation,
not individual-level data or patient-level prediction files.

- Individual-level data for **Datasets 1 and 2 are not included in the public
  Supplementary Data 1 or in this repository**, at any time point. Enquiries
  about access should be directed to the corresponding author identified in
  the manuscript. Any provision requires the applicable data-owner permissions
  and access conditions; the code license does not authorize data sharing.
- The revised public Supplementary Data 1 contains **12 analysis-ready matrices
  for Datasets 3–7**. Supplementary Tables S1–S9 retain the dataset-specific
  methods and aggregate results for the complete study, including Datasets 1
  and 2. Withholding their individual-level data does not remove them from the
  reported analyses or change the reported results.
- The original public sources are Li et al. (Datasets 3–5;
  https://doi.org/10.1038/s41467-019-12361-9), Darragh et al. (Dataset 6;
  https://doi.org/10.1038/s43018-022-00450-6), and Costantini et al. (Dataset 7;
  https://doi.org/10.1186/s13046-025-03378-8), corresponding to manuscript
  references 13–15. Original source workbooks and analysis-ready matrices are
  different inputs; see the preparation instructions below.

Using the public subset reproduces only the analyses for the supplied matrices.
The available formal primary pre-ICI cohorts are Datasets 3, 4, 5 and 7;
Dataset 6 remains a separate descriptive analysis. The full six-cohort and
17-matrix results require the corresponding complete, authorized inputs.

## Files

| File | Purpose |
|---|---|
| `prepare_inputs.py` | Validate supplied sheets and create canonical input workbooks. |
| `dataset_selection.py` | Study mapping, subset selection, and coverage records. |
| `ICI_predict.py` | Nested CV and primary/sensitivity scenario settings. |
| `ICI_analysis_common.py` | Shared fitting, metrics, artifact validation, and provenance. |
| `Compare_method.py` | Intentionally non-nested comparison with nested CV. |
| `Compare_nest.py` | Selective nesting for available formal pre-ICI datasets. |
| `Compare_sensitivity.py` | Scenario-matched nested/non-nested comparisons. |
| `Holdout.py` | Seed-42 holdout and 25 repeated stratified holdouts. |
| `Metrics_Calibration.py` | Saved-prediction metrics and bin-wise Wilson intervals. |
| `Revision2_reporting.py` | Dataset-omission, LC/MS-only, paired-repeat and KNN summaries; no fitting. |
| `Revision2_calibration.py` | Wilson-bin source tables from archived or current saved OOF predictions; no fitting. |
| `Revision2_figures.py` | Calibration figures from validated aggregate Wilson-bin tables, including partial-coverage labels. |
| `tests/` | Subset-input and numerical reporting regression tests. |

## Environment

Use Python 3.11 with `requirements.txt` for the original analysis environment.
The main analyses and existing Dataset 2 KNN results used Python 3.11.14;
the additional revision-2 KNN runs used Python 3.11.16 with the same scientific
package versions. See `REVISION2_VALIDATION.md` for the update's validation
environment; that validation is not a claim to have refitted every model.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select this environment in Positron. Script settings are near the top of each
file. Model runs are computationally intensive.

## Prepare the available input matrices

`prepare_inputs.py` identifies a matrix by its sheet name. It accepts the
original 17-sheet numbering or canonical keys such as `dataset3_pre`.
It does not infer a new mapping from sheet order or from the workbook README.

### Revised public workbook with 12 consecutively numbered sheets

In the revised public workbook, `Sheet1` identifies Dataset 3 Pre. In the
original 17-sheet layout, `Sheet1` identifies Dataset 1 Pre. **Do not pass the
renumbered public workbook directly to the converter.** Its counts are checked,
and that ambiguous naming is rejected rather than silently reassigning datasets.

Make a working copy of the public workbook. After checking its README, rename
the 12 data-sheet tabs using the mapping below and update the sheet-name column
in its README to match. Leave the patient rows, labels and feature values
unchanged. Save the copy as `Supplementary Data 1_canonical.xlsx`.

| Revised public sheet | Canonical sheet name |
|---|---|
| Sheet1 | dataset3_pre |
| Sheet2 | dataset4_pre |
| Sheet3 | dataset5_pre |
| Sheet4 | dataset6_pre |
| Sheet5 | dataset7_pre |
| Sheet6 | dataset3_post1 |
| Sheet7 | dataset4_post1 |
| Sheet8 | dataset5_post1 |
| Sheet9 | dataset6_post1 |
| Sheet10 | dataset3_post2 |
| Sheet11 | dataset4_post2 |
| Sheet12 | dataset5_post2 |

This mapping applies only to the revised 12-matrix public layout described
above. The original 17-sheet layout must keep its original mapping.

From a clean analysis directory, with no previous private inputs in `raw/`, run:

```bash
python prepare_inputs.py --input "Supplementary Data 1_canonical.xlsx"
```

The original public workbook is retained separately. Existing input/output
directories are not overwritten by the converter.

### Original sheet numbering or already canonical sheets

A workbook may contain **any nonempty subset** of the original sheets below,
in any order, or use their canonical keys as sheet names. A `README` sheet is
optional. This table describes the original input schema, not which data are
publicly distributed. Datasets 1 and 2 must be supplied only with the required
permissions. Do not renumber retained sheets, and do not supply both names for
the same matrix.

```bash
python prepare_inputs.py --input "Supplementary Data 1.xlsx"
```

| Original sheet | Canonical key |
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

For example, only Sheet3, Sheet4, Sheet5, and Sheet7 is valid and remains
Datasets 3, 4, 5, and 7 Pre. The converter validates IDs, labels, numeric
values, original sample/class/feature counts, and hashes. It writes supplied
matrices to `raw/`, plus `input_qc.csv` and `input_coverage.csv`.
Existing input directories are not overwritten.

Already prepared canonical `datasetN_pre.xlsx`, `datasetN_post1.xlsx`, or
`datasetN_post2.xlsx` files can instead be placed in `raw/` directly.
Each has `Sheet1` with `id`, `category` (or the supported `Responder`
column), and features. Only the 17 known keys are accepted; keep unrelated
workbooks outside `raw/`.

Absent inputs are `not_provided`. Unknown names, duplicate mappings,
malformed present inputs, and explicitly requested missing datasets are errors.
This is a study-reproduction workflow, not a generic new-cohort interface.

These inputs are **analysis-ready study matrices**, not the original source
study workbooks. The uploaded `data3-5作成.py` performs clinical/metabolite joins,
outcome construction and cohort exclusions; `data6作成.py` combines source
mwTab measurements. Those reconstruction steps are not performed by
`prepare_inputs.py`. Their outputs also use different sheet/file names and
must be mapped to the documented canonical analysis-ready format before use.

## Run on available inputs

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

Defaults intersect supplied inputs with the intended scenario scope.
KNN and expanded-grid sensitivity target Datasets 1–5 and 7 Pre.
Explicit requests, for example, remain strict:

```bash
python ICI_predict.py --scenario knn --datasets dataset3_pre dataset4_pre
```

Downstream scripts expose `DATASETS_TO_RUN` and run-root settings.
Selected inputs must have their required completed upstream outputs;
missing results are not silently treated as unavailable data.

The optional `sd` scenario needs the separately authorized stable-disease
recoding sheets, which were not included in the published input workbook.

Each analysis records `input_coverage.csv`. Subset results do not reproduce
the full six-dataset or 17-matrix analysis. Six-dataset nominal tests remain
`not_run` when any formal pre-ICI dataset is missing.

## Revision-2 reporting from completed results

`Revision2_reporting.py` reads saved artifacts without feature selection,
tuning, or model fitting. Use `python Revision2_reporting.py --help` for
input-root options and the explicit completed-KNN-extension adapter.
It exports aggregate tables, coverage, and input/output hashes.
Patient-level predictions used for validation are not exported in these tables.

For newly generated outputs:

```bash
python Revision2_reporting.py --knn-root results_revision/All_data_KNN --sensitivity-root results_revision/Compare_sensitivity_v1
python Revision2_calibration.py
```

For the completed original results (edit paths to your local folders):

```bash
python Revision2_reporting.py --main-root ../results_revision_v2/All_data_primary --compare-root ../results_revision_v2/Compare_method_v1 --knn-extension-root ../results_revision_round2_knn --out ../round2_reporting
python Revision2_calibration.py --main-root ../results_revision_v2/All_data_primary --compare-root ../results_revision_v2/Compare_method_v1 --out ../round2_calibration
```

The extension adapter is explicitly selected; it retains the original Dataset
2 reused-result identity and the five new-result identities. `Revision2_calibration.py`
recreates the original 10 quantile bins and exact positive counts from saved
OOF predictions. These two reporting commands do not require raw matrices or
rerunning the modified model producers. If KNN results are absent, omit
`--knn-root`, `--sensitivity-root`, and `--knn-extension-root` to generate
main-only reports with KNN marked `not_requested`.

To render the revised calibration figure from aggregate bins:

```bash
python -m pip install -r requirements-figures.txt
python Revision2_figures.py --input-dir ../round2_calibration --out ../round2_figures
```

The renderer validates the bin-table hashes and coverage before plotting.
It uses the accepted red/black calibration style and vertical bin-wise Wilson
error bars. A partial set of cohorts is labelled as such in the figure.
The default font is Arial; if it is unavailable, select an installed font
explicitly, for example `--font "DejaVu Sans"`. The actual font is recorded.
Other manuscript figures and the older full-result publication workbook
assembly are outside this focused reporting update.

It distinguishes patient-averaged OOF AUC from repeat-specific AUC, paired
repeat differences from independent samples, subset summaries from the full
six-dataset results, omission ranges from confidence intervals, and KNN
preprocessing sensitivity from an isolated imputation-only comparison.

## Interpretation and provenance

Primary nCV fits data-dependent preprocessing and selection within training
partitions, apart from the documented initial all-missing feature removal.
Intentionally non-nested and selective-nesting comparisons use the global
ranking/tuning or eligibility decisions described in Methods; these are not
fully nested pipelines.

Bootstrap intervals condition on saved patient-level averaged OOF predictions.
Wilson intervals condition on saved predictions and bin assignments; they are
bin-wise and not simultaneous confidence bands. Neither quantifies full
model-development uncertainty. Across-dataset p-values and repeated-holdout
variation are interpreted descriptively.

Keep completed results with original producer scripts and receipts.
Strict producer/artifact hash checks remain. Do not relabel old results as
outputs of an edited producer. The reporting adapter retains archived
producer identities without equating them to the current code.

No patient matrices, prediction tables, or result archives are distributed
in this repository or its release archives.
The MIT code license does not grant data-sharing permissions.
Clinical use requires independent validation.

## Validation

```bash
python -m unittest discover -s tests -v
```

See `REVISION2_VALIDATION.md` and `REVISION2_CHANGES.md`.

The author's uploaded `revision_script.zip` has also been compared with the
public implementation. See `REVISION2_SOURCE_AUDIT.md` for the source mapping,
preserved settings, older publication-code scope and archived-output
compatibility; exact uploaded-source hashes are recorded in
`REVISION2_SOURCE_PROVENANCE.json`.
