# Revision 2 validation — 2026-09-27

## Result

39 unit/regression tests passed (35 input/reporting tests and four plotting
validation tests). The updated reporting code also processed the
actual archived 17-matrix baseline and the completed six-dataset KNN extension
without model fitting. Independent subset checks passed with D1/D2 absent.

## Patch packaging correction

The initial patch used a locally reconstructed baseline that inadvertently
added one trailing blank line to each retrieved source file. This caused
`git apply --check` to fail at the end-of-file hunks in `.gitignore`,
`ICI_predict.py`, and `README.md` on the real upstream checkout.

The corrected patch is generated from the exact GitHub bytes at commit
`a8c252ec4b64bbc4409c8f17ccd097e9fe009f65`. All 12 original Git blob IDs were
verified before generating it. Applying the patch with `--index` to that
pristine baseline reproduces all 25 delivered files byte for byte. Numerical
code is unchanged; source trailing blank lines were normalized. The prior
patch check against the reconstructed baseline was insufficient and is
superseded by this exact-blob check.

## Numerical parity

Compared with the accepted revision-2 reporting output, absolute tolerance
`1e-12`:

| Output | Checked |
|---|---|
| Main individual values | 6 rows × 6 numeric fields |
| Dataset-omission summaries | 7 rows × 5 numeric fields |
| Paired Main/KNN repeats | 30 rows × 8 numeric fields |
| KNN summaries | 6 rows × 7 numeric fields |

The original transfer files used older supplementary-table numbers.
The current outputs use descriptive names to avoid implying those old numbers
remain the manuscript's final numbering.

All 120 raw-probability calibration bins across six datasets and two methods
were compared with the accepted publication-figure code. Bin N and event counts
were exactly equal; point estimates and Wilson bounds differed by at most
`2.22e-16`.

## Uploaded originals and figure rendering

All 18 uploaded Python files were compared with the public implementation and
accepted revision-2 sources. No unexplained numerical-model change was found.
The original-folder source mapping and scope differences are recorded in
`REVISION2_SOURCE_AUDIT.md`.

The new aggregate-only renderer was checked with the actual six-cohort
calibration tables and a derived four-cohort subset. Full coverage uses a 2×3
layout; the four-cohort plot uses 2×2 and a visible subset label. PNGs were
visually inspected for legibility and clipping; full-cohort PDF and SVG were
also generated. Rendering used Matplotlib 3.10.8, Pillow 12.3.0, 150 dpi and
explicitly selected DejaVu Sans because Arial is unavailable in this runtime.
The script defaults to Arial and 600 dpi, and requires an explicit installed
font choice if Arial is unavailable. This is a style/structure check, not a
claim of pixel-identical reproduction with a different font or resolution.

Plot tests reject changed artifact hashes, invalid counts, duplicate bins and
inconsistent coverage, and verify full/subset layout. They are skipped when
the optional Matplotlib dependency is not installed.

## Partial-input and failure checks

- Noncontiguous original sheets and canonical aliases preserve Dataset IDs.
- Unknown, duplicate, empty, malformed and overwritten inputs are rejected.
- Explicitly requested missing datasets fail.
- No-applicable/zero-scheduled analysis does not claim completed model runs.
- A 12-matrix saved-output subset excluding D1/D2 produces a four-of-six main
  Pre report with explicit missing-cohort labels.
- The same subset produces 80 calibration bins with correct N/event totals.
- Removing only a selected comparison result fails before report output.
- Stale receipt linkage, changed hashed artifacts, invalid probabilities,
  duplicate repeats and mismatched patients/labels are rejected.
- Six-cohort nominal tests are not run on incomplete subsets.
- Reporting exports contain aggregate data, not patient IDs or predictions.

## Scope

Source review found numerical model routines unchanged; changes concern
selection/orchestration, coverage/provenance and reporting. Shared
`ICI_analysis_common.py` is unchanged from the public baseline.

Checks ran with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
scikit-learn 1.8.0 and openpyxl 3.1.5 in the available environment.
`requirements.txt` retains the original scientific-library versions and the
recommended analysis environment remains Python 3.11.14/3.11.16 as recorded.

Models were not refitted, and raw-input provenance/data-sharing
permissions were not re-audited. Original nested producers lack output-hash
manifests: reporting checks available comparison/extension hashes, saved
prediction consistency and receipt links and records that limitation.
The uploaded `revision_script.zip` has now been compared: 18 files, 180 matching
scenario-setting checks, and 171 saved receipts matching 12 uploaded producer
identities. See `REVISION2_SOURCE_AUDIT.md` for intentional scope and layout
differences. This verifies the uploaded snapshot, not the live Mac directory.
GitHub upload/Release remains a separate author-side action.
