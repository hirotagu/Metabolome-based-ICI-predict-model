# Revision 2 public-code update — 2026-09-27

Based on upstream commit `a8c252ec4b64bbc4409c8f17ccd097e9fe009f65`.

- Input workbooks may contain any nonempty subset of original Sheet1–Sheet17 or canonical dataset-key sheets, with optional README. Dataset numbering, counts, labels and feature QC are preserved.
- Default analyses operate on available inputs within each scenario's intended scope. Explicit missing selections, malformed present files and missing upstream results are errors.
- KNN preprocessing targets the available subset of Datasets 1–5 and 7 Pre.
- Each run records input coverage and the dataset-selection helper identity.
- Revision2_reporting.py adds saved-result omission/LCMS summaries, paired CV repeats, KNN comparisons, missingness and explicit coverage. Native and historical completed-KNN-extension layouts are explicitly distinguished.
- Metrics_Calibration.py adds exact bin events and bin-wise Wilson 95% bounds. Revision2_calibration.py provides a read-only saved-OOF route for historical results as well.
- Existing six-dataset test calculations are retained with descriptive nominal interpretation and are skipped when the six intended cohorts are not all present.
- Original producer/artifact checks remain. No data files or completed-result archives are included.
- Model preprocessing, ranking, grids, seeds, fitting and CV design have not been redesigned.

The uploaded `revision_script.zip` (18 Python files) has now been compared
with this update and the public baseline. Numerical model routines were
preserved; source-specific scope and saved-result layout differences are
documented in `REVISION2_SOURCE_AUDIT.md`. The upload's KNN and publication
code precede the six-cohort KNN and Wilson-interval additions, so the
previously accepted revision-2 packages remain the sources for those changes.
`Revision2_figures.py` renders the validated Wilson-bin tables without
requiring all six cohorts. No GitHub push or Release was performed.
