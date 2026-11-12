# ICI metabolomics validation analyses

Research code for comparing nested cross-validation, intentionally non-nested cross-validation, selective nesting and holdout evaluation in metabolomics-based prediction of immune checkpoint inhibitor outcomes.

This repository accompanies **Comparison of nested and non-nested cross-validation in metabolomics-based prediction of immune checkpoint inhibitor outcomes**.

## Repository contents

| File or folder | Purpose |
|---|---|
| `prepare_inputs.py` | Validate analysis-ready matrices and create per-matrix input workbooks. |
| `dataset_selection.py` | Select datasets and record input coverage. |
| `ICI_predict.py` | Fit nested-CV models under the main and sensitivity settings. |
| `ICI_analysis_common.py` | Shared fitting, metrics and result-validation functions. |
| `Compare_method.py` | Fit the non-nested workflow and compare it with nested CV. |
| `Compare_nest.py` | Evaluate selective-nesting workflows. |
| `Compare_sensitivity.py` | Fit scenario-matched non-nested comparators and summarize sensitivity results. |
| `Holdout.py` | Run single and repeated stratified holdout evaluations. |
| `Metrics_Calibration.py` | Calculate discrimination and calibration summaries from saved predictions; no model fitting. |
| `Revision2_reporting.py` | Summarize completed analyses, including dataset-omission and KNN comparisons. |
| `Revision2_calibration.py`, `Revision2_figures.py` | Produce calibration tables and figures from saved results. |
| `prepare_revision3_inputs.py`, `run_revision3_6_6.py` | Prepare inputs and run input-scale sensitivity analyses for Datasets 3–5. |
| `revision3_engine/`, `revision3_inputs/` | Fitting code and input-validation information for the input-scale analysis, including the Dataset 5 Post2 missingness map. |
| `plot_revision3_scale_sensitivity.py`, `figure_data/` | Recreate Supplementary Figure S7 from included aggregate comparison values. |
| `requirements.txt`, `requirements-figures.txt` | Python dependencies. |
| `tests/` | Automated input, reporting and workflow checks. |

## Data availability

**Supplementary Data 1 is supplied with the manuscript, not through this repository.** It contains 12 analysis-ready matrices for Datasets 3–7. Individual-level data for Datasets 1 and 2 are available from the corresponding author upon reasonable request.

The source studies are [Li et al.](https://doi.org/10.1038/s41467-019-12361-9) (Datasets 3–5), [Darragh et al.](https://doi.org/10.1038/s43018-022-00450-6) (Dataset 6) and [Costantini et al.](https://doi.org/10.1186/s13046-025-03378-8) (Dataset 7). Use the analysis-ready supplementary matrices for this workflow; source-study workbooks require separate reconstruction.

The repository includes aggregate Figure S7 values and a Dataset 5 Post2 missingness map containing public-source IDs and feature names. It does not include participant-level intensity matrices, outcomes or predictions. Public inputs support the Datasets 3–7 analyses; summaries requiring Datasets 1 and 2 cannot be reproduced from public inputs alone.

## Setup

Use Python 3.11 and the package versions pinned in `requirements.txt`. From the repository folder:

```bash
python3 -m pip install -r requirements.txt
```

Select the same Python environment in Positron, if used. Run the commands below from the repository folder. Model fitting can take substantial time.

## Prepare analysis inputs

If per-matrix inputs are already available, place copies in `raw/` with canonical filenames such as `dataset3_pre.xlsx`. Each workbook must contain one `Sheet1`, with `id`, `category` (or `Responder`), followed by numeric features.

Otherwise, make a local working copy of Supplementary Data 1. Rename its data-sheet tabs using this mapping, keeping IDs, labels and feature values unchanged:

| Supplementary Data 1 sheet | Canonical worksheet name |
|---|---|
| Sheet1 | `dataset3_pre` |
| Sheet2 | `dataset4_pre` |
| Sheet3 | `dataset5_pre` |
| Sheet4 | `dataset6_pre` |
| Sheet5 | `dataset7_pre` |
| Sheet6 | `dataset3_post1` |
| Sheet7 | `dataset4_post1` |
| Sheet8 | `dataset5_post1` |
| Sheet9 | `dataset6_post1` |
| Sheet10 | `dataset3_post2` |
| Sheet11 | `dataset4_post2` |
| Sheet12 | `dataset5_post2` |

Save the working copy beside the scripts as `Supplementary Data 1_canonical.xlsx`. Update any worksheet references in the copy's README sheet to match. Keep the original supplement unchanged, then run:

```bash
python3 prepare_inputs.py --input "Supplementary Data 1_canonical.xlsx"
```

Use a new or empty `raw/` directory. The converter checks IDs, labels, values and matrix dimensions, and records input coverage. **Do not pass the public Sheet1–Sheet12 workbook directly without renaming the tabs.** Any nonempty subset of the public matrices can be supplied for the main analyses.

## Run the analyses

Fit the main nested-CV models, their non-nested and selective-nesting comparisons, and holdout evaluations:

```bash
python3 ICI_predict.py --scenario primary
python3 Compare_method.py
python3 Compare_nest.py
python3 Holdout.py
```

Alternative settings cover KNN preprocessing, a larger feature-count limit, three outer folds and an expanded regularization grid:

```bash
python3 ICI_predict.py --scenario knn
python3 ICI_predict.py --scenario kmax30
python3 ICI_predict.py --scenario outer3
python3 ICI_predict.py --scenario reggrid
python3 Compare_sensitivity.py
```

Results are written under `results_revision/`. Each workflow uses eligible datasets from the supplied inputs; Dataset 6 is handled as a separate small-sample analysis. `ICI_predict.py` accepts a subset, for example `--datasets dataset3_pre dataset4_pre`. Downstream scripts expose dataset and result-directory settings near the top; selected datasets require completed upstream outputs.

Calculate metrics from saved predictions without refitting models:

```bash
python3 Metrics_Calibration.py
```

For additional aggregate tables and calibration figures, use `Revision2_reporting.py`, `Revision2_calibration.py` and `Revision2_figures.py`; their `--help` options describe the required saved-result paths.

## Input-scale sensitivity analysis

This workflow evaluates Datasets 3–5 using the supplied log10-transformed values directly. It requires all nine pre/post matrices in `raw/`. Input preparation restores documented source-missing entries in Dataset 5 Post2 to missing values and writes separate inputs to `raw_revision3/`.

Prepare and check the inputs without fitting models:

```bash
python3 prepare_revision3_inputs.py
python3 run_revision3_6_6.py --check
```

Run the nested/non-nested, holdout, selective-nesting and alternative-setting analyses:

```bash
python3 run_revision3_6_6.py
```

Results are written under `result_revision3/`, including `Revision3_results.xlsx`. The input-scale results are reported separately from the main analyses in Supplementary Table S10.

## Recreate the comparison figure

Supplementary Figure S7 can be generated from `figure_data/FigureS7_data.csv` without participant-level inputs or model fitting:

```bash
python3 plot_revision3_scale_sensitivity.py
```

PDF, PNG, CSV and a rendering record are written to `result_revision3/`. Arial is preferred; an available fallback is reported and recorded. Use `--font` to select another installed font.

## Validation and provenance

```bash
python3 -m unittest discover -s tests -v
```

See [validation records](REVISION2_VALIDATION.md), [source provenance](REVISION2_SOURCE_PROVENANCE.json), [input-scale engine provenance](revision3_engine/PROVENANCE.json) and the [change log](REVISION3_CHANGES.md) for implementation and verification details. Non-nested and selective-nesting workflows intentionally retain the global selection steps described in the manuscript Methods.

The [MIT license](LICENSE) applies to code. Source data remain subject to their original terms and attribution.
