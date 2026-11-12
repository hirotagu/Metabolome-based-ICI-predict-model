# ICI metabolomics validation analyses — manuscript

Code supporting **Comparison of nested and non-nested cross-validation in metabolomics-based prediction of immune checkpoint inhibitor outcomes**.

The original main analyses are retained. Revision 3 adds an input-scale sensitivity analysis for Datasets 3–5 and Supplementary Figure S7; these do not replace the main results.

## Data and access

**Supplementary Data 1 is supplied with the manuscript, separately from this code repository.** It contains 12 analysis-ready matrices for Datasets 3–7. This code update does not replace or modify the submitted supplementary workbook.

| Dataset | Canonical worksheet names | Source |
|---|---|---|
| 3 | `dataset3_pre`, `dataset3_post1`, `dataset3_post2` | [Li et al.](https://doi.org/10.1038/s41467-019-12361-9) |
| 4 | `dataset4_pre`, `dataset4_post1`, `dataset4_post2` | [Li et al.](https://doi.org/10.1038/s41467-019-12361-9) |
| 5 | `dataset5_pre`, `dataset5_post1`, `dataset5_post2` | [Li et al.](https://doi.org/10.1038/s41467-019-12361-9) |
| 6 | `dataset6_pre`, `dataset6_post1` | [Darragh et al.](https://doi.org/10.1038/s43018-022-00450-6) |
| 7 | `dataset7_pre` | [Costantini et al.](https://doi.org/10.1186/s13046-025-03378-8) |

Individual-level data for Datasets 1 and 2 are available from the corresponding author upon reasonable request; they are not included in this repository or Supplementary Data 1. Full-study summaries are reported in Supplementary Tables S1–S10. Public-only runs cannot reproduce analyses requiring Datasets 1 and 2; the original six-cohort test is not performed on a smaller subset.

This repository contains code, documentation, aggregate Figure S7 values and a Dataset 5 Post2 missingness map (public-source IDs and feature names). It contains no participant-level outcomes, predictions or intensity matrices. Supplementary Data 1 is not included in the repository or its code release archives. The MIT license applies to code; source data remain subject to their original terms and attribution.

## Environment

The manuscript analyses used Python 3.11.14 or 3.11.16. The completed revision-3 scale-aware analyses used **Python 3.11.16**, NumPy 2.3.3, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.16.2, openpyxl 3.1.5 and Matplotlib 3.10.6. Package versions are pinned in `requirements.txt`.

Select that Python environment in Positron. Check it with `python3 --version` and `python3 -m pip show numpy pandas scikit-learn scipy openpyxl matplotlib`. For a new Python 3.11 environment, install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Run commands from the repository folder. No personal Mac path or private archive is required. Figure S7 prefers Arial; if unavailable, it reports and records a DejaVu Sans fallback. Use `--font` to request another installed font.

## Prepare inputs using the existing supplementary workbook

The existing input procedure is retained. If canonical workbooks are already available, place copies in `raw/` using names such as `dataset3_pre.xlsx`. Each must contain one `Sheet1` with `id`, `category` (or `Responder`), then numeric features. Keep unrelated workbooks outside `raw/`.

Otherwise, obtain Supplementary Data 1 from the manuscript supplementary files. Make a **local working copy** and rename its data-sheet tabs according to the existing public mapping below. This working copy is only an analysis input; do not replace or resubmit the original supplement. Keep IDs, labels and feature values unchanged. If the copy contains a README sheet with worksheet references, update those references in the copy only.

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

Save the working copy beside the scripts as `Supplementary Data 1_canonical.xlsx`, then use a new or empty `raw/` destination:

```bash
python3 prepare_inputs.py --input "Supplementary Data 1_canonical.xlsx"
```

Do not pass the public Sheet1–Sheet12 workbook directly to the converter before completing the tab-name step. Source-study workbooks are also not direct converter inputs. Existing outputs are not overwritten. Any nonempty subset of the 12 public matrices may be supplied for the original analyses; the complete scale-aware run below requires all nine Dataset 3–5 matrices.

## Quick checks and Figure S7 — no model fitting

With the original canonical inputs in `raw/`:

```bash
python3 prepare_revision3_inputs.py --check
python3 prepare_revision3_inputs.py
python3 run_revision3_6_6.py --check
python3 plot_revision3_scale_sensitivity.py --check
```

These commands prepare and validate inputs without fitting models. The sensitivity inputs are written to a new `raw_revision3/` folder; original inputs are not overwritten.

Recreate Figure S7 from included aggregate values without participant-level inputs or model fitting:

```bash
python3 plot_revision3_scale_sensitivity.py
```

This writes PDF, PNG, CSV and a rendering receipt to `result_revision3/`. The default source, `figure_data/FigureS7_data.csv`, contains 33 comparison rows (66 plotted points) for Datasets 3–5 only.

## Original main analyses

The first converter writes the original canonical inputs to `raw/`, with QC and coverage records. The main-analysis programs from the previous public release are retained:

```bash
python3 ICI_predict.py --scenario primary
python3 Compare_method.py
python3 Compare_nest.py
python3 Holdout.py
python3 Metrics_Calibration.py
```

These commands fit models and can take substantial time. They operate on available public inputs; Dataset 6 is handled separately as a small-sample analysis. Existing sensitivity analyses remain available:

```bash
python3 ICI_predict.py --scenario knn
python3 ICI_predict.py --scenario kmax30
python3 ICI_predict.py --scenario outer3
python3 ICI_predict.py --scenario reggrid
python3 Compare_sensitivity.py
```

`ICI_predict.py` accepts an explicit subset, for example `--datasets dataset3_pre dataset4_pre`. Downstream scripts expose `DATASETS_TO_RUN` and run-root settings. Selected datasets require completed upstream results. Historical `Revision2_*` filenames are retained because those reporting utilities remain applicable.

## Input-scale sensitivity — Datasets 3–5

The supplied measurements in Datasets 3–5 are already log10-transformed. The main pipeline applies its original additional log1p transformation. The scale-aware analysis retains log10 values, including finite zero and negative values. Missing intensities are imputed within training partitions as the observed minimum minus log10(2); the signed Dataset 4 Kyn/Trp feature uses the training-partition median.

`prepare_revision3_inputs.py` reads `raw/` and writes the nine affected matrices to `raw_revision3/`. In Dataset 5 Post2, it restores 3,275 source-missing entries encoded as zero during assembly. The included public-source ID/feature map distinguishes these cells from observed zeros. IDs, outcomes, feature order and other observed values are retained. Original `raw/` inputs are unchanged. The main analysis treated zeros as missing; restoration is needed to preserve genuine zeros in the scale-aware analysis.

After preparation and checks, run the complete scale-aware analyses:

```bash
python3 run_revision3_6_6.py
```

This computationally intensive command evaluates only Datasets 3–5: pre/post nested and non-nested CV, repeated holdout, pre-ICI selective nesting, and matching k≤30, outer-3-fold and expanded-C-grid settings. It does not rerun or reuse participant-level results for Datasets 1, 2, 6 or 7. Outputs are under `result_revision3/`; use `--help` for options.

The scientific engine is in `revision3_engine/`. Fitting functions, grids and seeds are retained from the completed local revision-3 analyses; the original input-scale handling is retained in the separate main-analysis code. `revision3_engine/PROVENANCE.json` records original hashes and configuration/orchestration changes. New runs create their own result receipts.

## Scripts and manuscript outputs

| Scripts / source | Scope | Manuscript output |
|---|---|---|
| `ICI_predict.py`, `Compare_method.py` | Original nested/non-nested CV | Table 2; corresponding main results |
| `Metrics_Calibration.py`, `Revision2_reporting.py`, `Revision2_calibration.py` | Saved-prediction metrics and summaries | Brier/calibration summaries; Tables S2–S4 |
| `Holdout.py` | Original repeated holdout | Figure 4; Table S7 |
| `Compare_nest.py` | Original selective nesting | Figure 5; Table S8 |
| `ICI_predict.py` sensitivity scenarios, `Compare_sensitivity.py` | Original alternative settings | Table S9 |
| `prepare_revision3_inputs.py`, `run_revision3_6_6.py` | Scale-aware Datasets 3–5 | Table S10A–E: discrimination, Brier, holdout, selective nesting, additional settings |
| `plot_revision3_scale_sensitivity.py`, `figure_data/` | Saved original/scale-aware aggregates | Figure S7 |

This maps numerical analyses to manuscript outputs, not automatic reproduction of every publication layout. Public inputs cannot reproduce full-study summaries requiring Datasets 1 and 2. For saved main-analysis reporting, use `--help` with `Revision2_reporting.py`, `Revision2_calibration.py` and `Revision2_figures.py`.

## Interpretation and verification

Non-nested and selective-nesting workflows intentionally retain the global selection steps described in Methods. Bootstrap intervals condition on saved OOF predictions; repeated-CV and holdout summaries are descriptive.

The public-package update checks the archived canonical analysis inputs, preservation of scientific functions, tests and saved-data figure regeneration without a full model rerun. The exact submitted Supplementary Data 1 file was not available for a separate end-to-end check; the existing worksheet mapping is retained from the public README. The manuscript scale-aware results came from the completed local Python 3.11.16 run. Packaging checks do not establish an independent reproduction of every fitted result. Historical validation records remain in `REVISION2_*.md`.

```bash
python3 -m unittest discover -s tests -v
```

The intended release is `v3.0.0` (Manuscript revision 3). Once published and archived, cite the version-specific Zenodo DOI shown on that release. No DOI has been assigned in this prepared package.
