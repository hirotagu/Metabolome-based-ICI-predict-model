#!/usr/bin/env python3
"""Compare fully nested CV with intentionally non-nested CV.

The completed primary nested-CV predictions are read and verified; they are not
recomputed.  The non-nested analysis deliberately uses the same repeated CV
for global feature/hyperparameter selection and OOF evaluation, thereby
quantifying selection optimism.  Holdout and plotting are intentionally kept
in separate scripts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import ICI_predict as base
import ICI_analysis_common as common
import dataset_selection as selection


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Point this to the completed main run (the folder containing run_manifest.csv).
MAIN_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_primary"

OUTPUT_ROOT = SCRIPT_DIR / "results_revision"
RUN_TAG = "Compare_method_v1"

# "ALL" runs available canonical matrices, preserving original dataset IDs.
# Explicitly requested missing datasets are errors.
DATASETS_TO_RUN: Union[str, List[str]] = "ALL"

MAX_K = base.PRIMARY_MAX_K
BOOTSTRAP_N = base.BOOTSTRAP_N
RANDOM_STATE = base.RANDOM_STATE

STRICT_BASE_SCRIPT_HASH = True
CONTINUE_ON_DATASET_ERROR = True
ALLOW_OVERWRITE = False

# Prespecified comparisons across independent Pre cohorts 1-5 and 7.
# Nominal p-values are descriptive for these six small, heterogeneous cohorts.
FORMAL_PRE_KEYS: Tuple[str, ...] = (
    "dataset1_pre",
    "dataset2_pre",
    "dataset3_pre",
    "dataset4_pre",
    "dataset5_pre",
    "dataset7_pre",
)
CANONICAL_KEYS: Tuple[str, ...] = (
    "dataset1_pre",
    "dataset1_post1",
    "dataset2_pre",
    "dataset2_post1",
    "dataset2_post2",
    "dataset3_pre",
    "dataset3_post1",
    "dataset3_post2",
    "dataset4_pre",
    "dataset4_post1",
    "dataset4_post2",
    "dataset5_pre",
    "dataset5_post1",
    "dataset5_post2",
    "dataset6_pre",
    "dataset6_post1",
    "dataset7_pre",
)


def _run_root() -> Path:
    return OUTPUT_ROOT / RUN_TAG if RUN_TAG else OUTPUT_ROOT


def _selected_keys(specs: Dict[str, base.DatasetSpec]) -> List[str]:
    return selection.select_available(
        DATASETS_TO_RUN, specs, default_keys=CANONICAL_KEYS
    )


def _fold_parameter_row(
    artifact: common.MainArtifacts,
    record: base.SplitRecord,
    global_spec: common.GlobalSpec,
    model_features: Sequence[str],
) -> Dict[str, Any]:
    y_train = artifact.bundle.y[record.train_idx]
    y_test = artifact.bundle.y[record.test_idx]
    return {
        "dataset": artifact.spec.key,
        "method": "fully_non_nested",
        "outer_repeat": record.repeat,
        "outer_fold": record.fold,
        "outer_splits_used": artifact.outer_plan.n_splits,
        "outer_repeats": artifact.outer_plan.n_repeats,
        "train_n": len(record.train_idx),
        "train_positive": int(np.sum(y_train == 1)),
        "train_negative": int(np.sum(y_train == 0)),
        "test_n": len(record.test_idx),
        "test_positive": int(np.sum(y_test == 1)),
        "test_negative": int(np.sum(y_test == 0)),
        **common.class_weights(y_train),
        "global_en_C": global_spec.en_C,
        "global_en_l1_ratio": global_spec.en_l1_ratio,
        "global_k_selected": global_spec.k_selected,
        "global_l2_C": global_spec.l2_C,
        "n_global_selected_features": len(global_spec.selected_features),
        "n_model_features": len(model_features),
    }


def run_non_nested(
    artifact: common.MainArtifacts,
    output_dir: Path,
) -> Dict[str, Any]:
    bundle = artifact.bundle
    label = f"{artifact.spec.key} | fully_non_nested"
    global_spec = common.tune_global_spec(
        bundle,
        selection_plan=artifact.outer_plan,
        context=label,
        max_k=MAX_K,
        imputation="half_minimum",
    )

    prediction_rows: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    fold_count_rows: List[Dict[str, Any]] = []
    preprocessing_frames: List[pd.DataFrame] = []
    warning_rows = list(global_spec.warning_rows)
    log_lines = list(global_spec.log_lines)
    rank_lookup = {
        feature: rank
        for rank, feature in enumerate(global_spec.ranking, start=1)
    }

    ids = bundle.ids.astype(str).tolist()
    for outer_iter, record in enumerate(artifact.outer_plan.records, start=1):
        probability, model_features, coefficients, prep_audit = common.predict_fixed_spec(
            bundle,
            record.train_idx,
            record.test_idx,
            selected_features=global_spec.selected_features,
            l2_C=global_spec.l2_C,
            context=(
                f"{label} repeat={record.repeat} fold={record.fold}"
            ),
            imputation="half_minimum",
            warning_rows=warning_rows,
            log_lines=log_lines,
        )
        for local_index, original_index in enumerate(record.test_idx):
            prediction_rows.append(
                {
                    "dataset": artifact.spec.key,
                    "method": "fully_non_nested",
                    base.ID_COL: ids[original_index],
                    "repeat": record.repeat,
                    "outer_fold": record.fold,
                    "y_true": int(bundle.y[original_index]),
                    "p": float(probability[local_index]),
                }
            )
        parameter_rows.append(
            _fold_parameter_row(artifact, record, global_spec, model_features)
        )
        y_train = bundle.y[record.train_idx]
        y_test = bundle.y[record.test_idx]
        for subset_name, values in (("train", y_train), ("test", y_test)):
            fold_count_rows.append(
                {
                    "dataset": artifact.spec.key,
                    "method": "fully_non_nested",
                    "outer_repeat": record.repeat,
                    "outer_fold": record.fold,
                    "subset": subset_name,
                    **base.class_counts(values),
                }
            )
        model_feature_set = set(model_features)
        selected_set = set(global_spec.selected_features)
        for feature in bundle.feature_names:
            feature_rows.append(
                {
                    "dataset": artifact.spec.key,
                    "method": "fully_non_nested",
                    "outer_repeat": record.repeat,
                    "outer_fold": record.fold,
                    "feature": feature,
                    "global_rank": rank_lookup.get(feature, np.nan),
                    "selected_global_topk": feature in selected_set,
                    "model_used": feature in model_feature_set,
                    "outer_coefficient": coefficients.get(feature, np.nan),
                    "global_k_selected": global_spec.k_selected,
                    "n_model_features": len(model_features),
                }
            )
        prep_audit.insert(0, "outer_fold", record.fold)
        prep_audit.insert(0, "outer_repeat", record.repeat)
        prep_audit.insert(0, "dataset", artifact.spec.key)
        preprocessing_frames.append(prep_audit)
        print(
            f"[{label}] outer {outer_iter}/{len(artifact.outer_plan.records)} complete"
        )

    predictions = pd.DataFrame(prediction_rows)
    expected_rows = len(bundle.y) * artifact.outer_plan.n_repeats
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"{label}: expected {expected_rows} prediction rows, found {len(predictions)}"
        )
    oof = common.averaged_oof(
        predictions,
        bundle.ids.astype(str).tolist(),
        expected_repeats=artifact.outer_plan.n_repeats,
    )
    repeat_metrics = common.repeat_metric_table(
        predictions, artifact.spec.key, "fully_non_nested"
    )
    nonnested_metrics = common.metric_values(
        oof["y_true"].to_numpy(dtype=int), oof["p_oof"].to_numpy(dtype=float)
    )

    nested_oof = artifact.oof.set_index(base.ID_COL).loc[oof[base.ID_COL]].reset_index()
    if not np.array_equal(
        nested_oof["y_true"].to_numpy(dtype=int), oof["y_true"].to_numpy(dtype=int)
    ):
        raise ValueError(f"Nested/non-nested labels are misaligned for {artifact.spec.key}")
    nested_metrics = common.metric_values(
        nested_oof["y_true"].to_numpy(dtype=int),
        nested_oof["p_oof"].to_numpy(dtype=float),
    )
    paired_bootstrap = common.paired_stratified_auc_bootstrap(
        oof["y_true"].to_numpy(dtype=int),
        nested_oof["p_oof"].to_numpy(dtype=float),
        oof["p_oof"].to_numpy(dtype=float),
        n_boot=BOOTSTRAP_N,
        seed=RANDOM_STATE,
    )
    comparison = {
        "dataset": artifact.spec.key,
        "dataset_id": artifact.spec.dataset_id,
        "timepoint": artifact.spec.timepoint,
        "analysis_role": common.dataset_role(artifact.spec),
        "inference_group": common.inference_group(artifact.spec),
        "n_samples": len(bundle.y),
        "n_positive": int(np.sum(bundle.y == 1)),
        "n_negative": int(np.sum(bundle.y == 0)),
        "nested_roc_auc": nested_metrics["roc_auc"],
        "nested_average_precision": nested_metrics["average_precision"],
        "nested_brier": nested_metrics["brier"],
        "nonnested_roc_auc": nonnested_metrics["roc_auc"],
        "nonnested_average_precision": nonnested_metrics["average_precision"],
        "nonnested_brier": nonnested_metrics["brier"],
        "delta_auc_nonnested_minus_nested": (
            nonnested_metrics["roc_auc"] - nested_metrics["roc_auc"]
        ),
        "delta_ap_nonnested_minus_nested": (
            nonnested_metrics["average_precision"]
            - nested_metrics["average_precision"]
        ),
        "delta_brier_nonnested_minus_nested": (
            nonnested_metrics["brier"] - nested_metrics["brier"]
        ),
        "nested_auc_ci95_lo_descriptive": paired_bootstrap["auc_a_ci95_lo"],
        "nested_auc_ci95_hi_descriptive": paired_bootstrap["auc_a_ci95_hi"],
        "nonnested_auc_ci95_lo_descriptive": paired_bootstrap["auc_b_ci95_lo"],
        "nonnested_auc_ci95_hi_descriptive": paired_bootstrap["auc_b_ci95_hi"],
        "delta_auc_ci95_lo_descriptive": paired_bootstrap["delta_ci95_lo"],
        "delta_auc_ci95_hi_descriptive": paired_bootstrap["delta_ci95_hi"],
        "auc_ci_method": (
            "paired stratified patient bootstrap of fixed averaged OOF predictions; "
            "descriptive, not CV-variance corrected"
        ),
        "global_en_C": global_spec.en_C,
        "global_en_l1_ratio": global_spec.en_l1_ratio,
        "global_k_selected": global_spec.k_selected,
        "global_l2_C": global_spec.l2_C,
        "outer_splits_used": artifact.outer_plan.n_splits,
        "outer_repeats": artifact.outer_plan.n_repeats,
    }

    parameters = pd.DataFrame(parameter_rows)
    features = pd.DataFrame(feature_rows)
    folds = pd.DataFrame(fold_count_rows)
    preprocessing = pd.concat(preprocessing_frames, ignore_index=True)
    warnings_table = pd.DataFrame(
        warning_rows,
        columns=["dataset", "stage", "warning_type", "message"],
    )

    common.write_csv(predictions, output_dir / "outer_predictions_long.csv")
    common.write_csv(predictions, output_dir / "nonnested_outer_predictions_long.csv")
    common.write_csv(oof, output_dir / "oof_predictions.csv")
    common.write_csv(oof, output_dir / "nonnested_oof_predictions.csv")
    common.write_csv(repeat_metrics, output_dir / "repeat_metrics.csv")
    common.write_csv(pd.DataFrame([comparison]), output_dir / "metrics_summary.csv")
    common.write_csv(parameters, output_dir / "outer_parameters.csv")
    common.write_csv(features, output_dir / "outer_features_long.csv")
    common.write_csv(folds, output_dir / "fold_counts.csv")
    common.write_csv(preprocessing, output_dir / "preprocessing_by_outer.csv")
    ranking_table = global_spec.ranking_table.copy()
    ranking_table.insert(0, "dataset", artifact.spec.key)
    ranking_table["selected_global_topk"] = ranking_table["rank"] <= global_spec.k_selected
    common.write_csv(ranking_table, output_dir / "global_ranking.csv")
    common.write_csv(
        global_spec.k_curve.assign(dataset=artifact.spec.key),
        output_dir / "global_k_curve.csv",
    )
    common.write_csv(warnings_table, output_dir / "warnings.csv")
    (output_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    common.write_json(
        output_dir / "global_model_spec.json",
        {
            "dataset": artifact.spec.key,
            "method": "fully_non_nested",
            "statement": (
                "Diagnostic optimistic estimate: selection/tuning and OOF evaluation "
                "reuse the same repeated CV"
            ),
            "global_en_C": global_spec.en_C,
            "global_en_l1_ratio": global_spec.en_l1_ratio,
            "global_k_selected": global_spec.k_selected,
            "global_l2_C": global_spec.l2_C,
            "global_selected_features": global_spec.selected_features,
            "class_weight": "balanced",
            "deployment_threshold": None,
        },
    )
    common.write_json(
        output_dir / "settings_used.json",
        {
            "dataset_selection_sha256": common.sha256_file(Path(selection.__file__).resolve()),
            "main_run_dir": str(artifact.run_dir),
            "main_run_receipt_sha256": common.sha256_file(
                artifact.run_dir / "run_receipt.json"
            ),
            "input_path": str(artifact.spec.path),
            "input_sha256": bundle.input_sha256,
            "selection_and_evaluation_cv": {
                "splits": artifact.outer_plan.n_splits,
                "repeats": artifact.outer_plan.n_repeats,
                "random_state": RANDOM_STATE,
            },
            "max_k": MAX_K,
            "bootstrap_n": BOOTSTRAP_N,
            "missing_rate_threshold": base.MISSING_RATE_THRESHOLD,
            "class_weight": "balanced",
            "threshold_metrics": "not computed",
        },
    )
    output_hashes = common.output_sha256_manifest(
        output_dir,
        (
            "outer_predictions_long.csv",
            "oof_predictions.csv",
            "repeat_metrics.csv",
            "metrics_summary.csv",
            "outer_parameters.csv",
            "outer_features_long.csv",
            "fold_counts.csv",
            "preprocessing_by_outer.csv",
            "global_ranking.csv",
            "global_k_curve.csv",
            "global_model_spec.json",
            "settings_used.json",
            "warnings.csv",
            "run_log.txt",
        ),
    )
    common.write_json(
        output_dir / "run_receipt.json",
        {
            "status": "completed",
            "completed_utc": common.utc_now(),
            "dataset": artifact.spec.key,
            "method": "fully_non_nested",
            "input_path": str(artifact.spec.path),
            "input_sha256": bundle.input_sha256,
            "main_source_dir": str(artifact.run_dir),
            "sample_counts": base.class_counts(bundle.y),
            "metrics": comparison,
            "warning_count": len(warning_rows),
            "output_sha256": output_hashes,
            "dataset_selection_sha256": common.sha256_file(Path(selection.__file__).resolve()),
            **common.provenance_payload(Path(__file__)),
        },
    )
    return comparison


def build_formal_inference(comparison: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in comparison.columns:
        comparison = pd.DataFrame(columns=["dataset", "delta_auc_nonnested_minus_nested"])
    available = comparison.set_index("dataset")
    missing = [key for key in FORMAL_PRE_KEYS if key not in available.index]
    if missing:
        return pd.DataFrame(
            [
                {
                    "test": "not_run",
                    "inferential_role": "descriptive_nominal_primary",
                    "status": "missing_prespecified_cohorts",
                    "required_datasets": "|".join(FORMAL_PRE_KEYS),
                    "missing_datasets": "|".join(missing),
                    "multiplicity": (
                        "not applicable; one prespecified primary comparison was planned"
                    ),
                }
            ]
        )
    ordered = available.loc[list(FORMAL_PRE_KEYS)]
    delta = ordered["delta_auc_nonnested_minus_nested"].to_numpy(dtype=float)
    signed_rank = common.exact_signed_rank_sign_permutation(delta)
    sign = common.exact_sign_test(delta)
    shared = {
        "status": "completed",
        "effect": "AUROC_fully_non_nested_minus_fully_nested",
        "unit": "independent_pre_cohort",
        "datasets": "|".join(FORMAL_PRE_KEYS),
        "n_cohorts": len(delta),
        "median_delta_auc": float(np.median(delta)),
        "min_delta_auc": float(np.min(delta)),
        "max_delta_auc": float(np.max(delta)),
        "n_positive": int(np.sum(delta > 0)),
        "n_negative": int(np.sum(delta < 0)),
        "n_zero": int(np.sum(delta == 0)),
        "multiplicity": (
            "nominal unadjusted p-values; prespecified primary comparison with sign-test sensitivity"
        ),
        "inference_scope": "descriptive nominal comparisons of the six included heterogeneous Pre cohorts",
        "p_value_interpretation": (
            "descriptive nominal p-values; not evidence for a common effect or "
            "generalization to a wider population of datasets"
        ),
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "test": "exact_two_sided_signed_rank_by_sign_enumeration",
                "inferential_role": "descriptive_nominal_primary",
                "p_value": signed_rank["p_value"],
                "n_nonzero_used": signed_rank["n_nonzero"],
                "test_detail": (
                    "average ranks for tied absolute differences; all sign assignments enumerated"
                ),
            },
            {
                **shared,
                "test": "exact_two_sided_sign_test",
                "inferential_role": "descriptive_nominal_sensitivity",
                "p_value": sign["p_value"],
                "n_nonzero_used": sign["n_nonzero"],
                "test_detail": "zeros excluded; exact binomial probability under p=0.5",
            },
        ]
    )


def main() -> None:
    if int(MAX_K) != int(base.PRIMARY_MAX_K):
        raise ValueError("Compare_method MAX_K must equal ICI_predict.PRIMARY_MAX_K")
    if int(RANDOM_STATE) != int(base.RANDOM_STATE):
        raise ValueError("Compare_method RANDOM_STATE must equal ICI_predict.RANDOM_STATE")
    if int(BOOTSTRAP_N) < 1:
        raise ValueError("Compare_method BOOTSTRAP_N must be >= 1")
    started_time = time.time()
    started_utc = common.utc_now()
    specs = base.discover_datasets()
    keys = _selected_keys(specs)
    if not keys:
        raise RuntimeError("DATASETS_TO_RUN did not match any canonical input")
    run_root = common.prepare_run_root(_run_root(), ALLOW_OVERWRITE)
    common.write_csv(
        pd.DataFrame(selection.coverage_rows(specs, keys, expected=CANONICAL_KEYS)),
        run_root / "input_coverage.csv",
    )

    comparison_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for key in keys:
        print("=" * 88)
        print(f"[START] Compare_method | {key}")
        try:
            main_dir = common.resolve_main_run_dirs(
                MAIN_RUN_ROOT, specs, [key]
            )[key]
            artifact = common.validate_main_artifacts(
                specs[key], main_dir, strict_script_hash=STRICT_BASE_SCRIPT_HASH
            )
            with common.staged_dataset_dir(run_root, key) as stage:
                comparison = run_non_nested(artifact, stage)
            comparison_rows.append(comparison)
            manifest_rows.append(
                {
                    "dataset": key,
                    "status": "completed",
                    "analysis_role": common.dataset_role(specs[key]),
                    "output_dir": str((run_root / "datasets" / key).resolve()),
                    "message": "",
                    "nested_roc_auc": comparison["nested_roc_auc"],
                    "nonnested_roc_auc": comparison["nonnested_roc_auc"],
                    "delta_auc": comparison["delta_auc_nonnested_minus_nested"],
                }
            )
            print(f"[DONE] Compare_method | {key}")
        except Exception as exc:
            manifest_rows.append(
                {
                    "dataset": key,
                    "status": "failed",
                    "analysis_role": common.dataset_role(specs[key]),
                    "output_dir": "",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[FAILED] Compare_method | {key}: {type(exc).__name__}: {exc}")
            if not CONTINUE_ON_DATASET_ERROR:
                raise

    comparison_table = pd.DataFrame(comparison_rows)
    manifest = pd.DataFrame(manifest_rows)
    common.write_csv(manifest, run_root / "run_manifest.csv")
    common.write_csv(comparison_table, run_root / "comparison_all_datasets.csv")
    inference = build_formal_inference(comparison_table)
    common.write_csv(inference, run_root / "pre_primary_inference.csv")

    common.finalize_run_receipt(
        run_root,
        Path(__file__),
        started_utc,
        started_time,
        manifest,
        {
            "analysis": "fully_nested_vs_fully_non_nested",
            "main_run_root": str(Path(MAIN_RUN_ROOT).resolve()),
            "datasets_to_run": DATASETS_TO_RUN,
            "selected_datasets": keys,
            "input_coverage_file": "input_coverage.csv",
            "dataset_selection_sha256": common.sha256_file(Path(selection.__file__).resolve()),
            "formal_pre_keys": list(FORMAL_PRE_KEYS),
            "nominal_primary_comparison": "exact two-sided signed-rank by sign enumeration",
            "nominal_sensitivity_comparison": "exact two-sided sign test",
            "p_value_interpretation": "descriptive nominal across six small heterogeneous Pre cohorts; not a common-effect or population-generalization claim",
            "multiplicity": "nominal unadjusted p-values; one prespecified descriptive primary comparison",
            "dataset_level_tests": "not performed",
            "post_tests": "not performed",
            "dataset6_test": "not performed; descriptive supplement",
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_interpretation": (
                "descriptive fixed-prediction patient bootstrap; not CV-variance corrected"
            ),
            "max_k": MAX_K,
            "random_state": RANDOM_STATE,
            "continue_on_dataset_error": CONTINUE_ON_DATASET_ERROR,
        },
    )
    completed = int((manifest["status"] == "completed").sum())
    failed = int((manifest["status"] == "failed").sum())
    print("=" * 88)
    print(f"Compare_method completed: {completed}; failed: {failed}")
    print(f"Output: {run_root}")
    if failed:
        raise RuntimeError(
            f"Compare_method finished with {failed} failed dataset(s); inspect run_manifest.csv"
        )


if __name__ == "__main__":
    main()
