# Manuscript revision 3 public package

Prepared from public repository commit `b414fcb503af2963bc965a4aaaedd98563812c6e`.

## Changes

- Retains all original main-analysis and revision-2 reporting source files unchanged.
- Adds `prepare_revision3_inputs.py` and the public-source Dataset 5 Post2 missingness map.
- Adds `run_revision3_6_6.py` and the scientific engine from the completed local revision-3 analyses, isolated in `revision3_engine/`.
- Adds `plot_revision3_scale_sensitivity.py` and 33 aggregate comparison rows for Figure S7.
- Updates README, dependency versions and generated-file exclusions. Existing revision-2 documentation remains as a historical record.
- Retains the existing input instructions for the manuscript's Supplementary Data 1. No supplementary workbook is created, modified or supplied by this code package. Datasets 1 and 2 remain available upon reasonable request.

## Verification performed without refitting models

- All 48 unit tests pass in the packaging runtime.
- Canonical analysis inputs from the existing analysis archive were used for input checks. The exact submitted Supplementary Data 1 was not available for a separate end-to-end check; its worksheet-name mapping is retained from the current public README.
- The unchanged original XLSX converter introduces 18 floating-point serialization differences, at most 8.88e-16 (two ULP). The input manifest records exact fingerprints of the original and verified converter outputs; no tolerance-based acceptance is used.
- Sensitivity-input preparation and runner `--check` pass for all nine archived Dataset 3–5 canonical matrices.
- Dataset 5 Post2 restores exactly 3,275 source-missing entries and preserves 24 observed zeros; original main inputs remain unchanged.
- Preparing from the original raw workbooks reproduces the nine actual local revision-3 input files byte-for-byte.
- Scientific function/class definitions in the packaged engine match the executed local sources. Configuration/orchestration differences are recorded in `revision3_engine/PROVENANCE.json`.
- The new summary writer was checked against saved aggregate results. Figure S7 was regenerated and visually checked; the original drawing function is retained.
- No private Dataset 1/2 inputs, participant-level predictions or private analysis archives are included in the code folder.

The validation runtime used Python 3.12.14; no models were fitted in that environment. Manuscript revision-3 results came from the completed local Python 3.11.16 run, whose scientific-package versions are pinned in `requirements.txt`. These packaging checks do not constitute an independent full numerical replication.
