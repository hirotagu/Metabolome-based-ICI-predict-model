# Uploaded revision-script comparison — 2026-09-27

The author supplied `revision_script.zip`, containing 18 Python files.
Archive SHA-256:
`2c1128b91d33c3afe1e9b3ba4eb68afc525ee01da0517001d064267c4de4f3fe`.
This comparison concerns the uploaded snapshot; it does not assert access to
the author's live Mac directory. Per-file hashes are recorded in
`REVISION2_SOURCE_PROVENANCE.json`.

## Findings

No unexplained numerical model change was found between the supplied producer
code and the consolidated public implementation. All 30 numerical/data-rule
settings matched for each of the six scenario configurations (180 checks).
Of 55 primary-script definitions, 46 are exactly AST-identical; the nested-CV
routine is identical after excluding additional receipt metadata. Other
differences concern scenario selection, input discovery, orchestration and
provenance. Of 38 common-module definitions, 37 are exactly AST-identical and
the remaining difference is a filename in an error message.

The source checks complement the saved-result numerical checks in
`REVISION2_VALIDATION.md`; they are not a new model-refitting experiment.
Across the baseline result archive, 171 saved receipts have producer hashes
matching 12 of the uploaded script identities. Two receipts under the old
discarded-output folder have different historical identities and were not
used to validate the selected completed runs.

## Source-to-public mapping

| Uploaded source | Public entry point / treatment |
|---|---|
| `ICI_predict_v3.py` | `ICI_predict.py --scenario primary` |
| `ICI_predict_v3_KNN.py` | `ICI_predict.py --scenario knn`; uploaded scope is D2 only, six-cohort expansion uses the separately accepted KNN extension |
| `ICI_predict_v3_kmax.py` | `ICI_predict.py --scenario kmax30` |
| `ICI_predict_v3_out3.py` | `ICI_predict.py --scenario outer3` |
| `ICI_predict_v3_RedGrid.py` | `ICI_predict.py --scenario reggrid`; nonconvergence eligibility setting retained |
| `ICI_predict_v3_SD.py` | `ICI_predict.py --scenario sd`; separately supplied recoding sheets required |
| `ICI_revision_common.py` | `ICI_analysis_common.py` |
| `Compare_method.py` | Same public name, with input availability and provenance additions |
| `Compare_nest.py` | Same public name; public primary scope is D1–5 and D7, as explained below |
| `Compare_sensitivity.py` | Same public name; consolidated producer activation, six-cohort KNN and available-input selection |
| `Holdout.py`, `Holdout_d3-2.py` | Unified `Holdout.py`; D3 Post2 retains the 80000 iteration ceiling, other datasets 20000 |
| `Metrics_Calibration.py` | Same public name for new runs; archived OOF reporting uses `Revision2_calibration.py` |
| `Publication_outputs.py`, `publication_figures.py`, `publication_tables.py` | Older full-result publication assembly; not copied into the subset workflow. Revision-2 aggregate additions and calibration rendering are exposed separately |
| `data3-5作成.py`, `data6作成.py` | Original-source reconstruction, outside the analysis-ready input contract; not silently applied by `prepare_inputs.py` |

## Differences that matter

- The uploaded KNN code targets only D2 Pre. Its inclusion is not evidence
  that the uploaded folder contains the later six-cohort extension.
  `KNN_Revision_R4_8_v1_0_2.zip` remains the source for that extension.
- The uploaded publication code has no Wilson error bars, requires all
  expected datasets, and uses earlier figure/table numbering. The accepted
  `Publication_figure_updates_original_style.zip` is the reference for the
  revised calibration figure. The subset renderer does not call the older
  full-17 publication assembler.
- Uploaded selective nesting includes D6 Pre. The pre-existing public
  baseline already restricts this analysis to the six formal primary
  cohorts. That difference has been retained; the current workflow does not
  claim to reproduce the original seven-row diagnostic output.
- Uploaded sensitivity defaults include SD recodings. Public defaults use
  ordinary supplied matrices; SD remains explicitly selectable when its
  separate inputs are available.
- Original holdout output uses a separate D3 Post2 retry folder. The public
  producer consolidates this setting and records the effective iteration
  ceiling. Old outputs must not be relabelled as new public-producer outputs.
  The explicit saved-result reporting route preserves original producer
  identities and available artifact checks.
- The two source-data builders' expected N, class counts and feature counts
  match current converter QC for all 11 matrices they construct. This is
  a static expectation check, not proof of row-level reconstruction identity.
  Their outcome construction, clinical joins and exclusions occur before
  the analysis-ready input stage. Their output sheets are named `Analysis`,
  and some filenames differ from the public canonical names.
- Source-data reconstruction and redistribution permissions were not
  revalidated. The supplied D3–5 source workbook headers were inspected,
  but no reconstruction or model refitting was performed in this comparison.

## Resulting update

The consolidated model code and scenario definitions were retained. The
documentation now distinguishes original source data from analysis-ready
matrices and records the uploaded source comparison. Calibration plotting is
available from aggregate Wilson-bin outputs for either all six formal cohorts
or an explicitly labelled subset. The public bundle contains code and
documentation only; no clinical matrices or patient predictions are included.
