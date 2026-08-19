#!/usr/bin/env python3
"""Prespecified and repeated stratified holdout analysis.

For every split, the complete modeling sequence (preprocessing, elastic-net tuning,
feature ranking, k/C selection, and L2 fitting) is learned from the training
subset only. Split 1 is the prespecified seed-42 70/30 holdout; all 25 splits
are retained to describe split dependence. The overlapping splits are not
treated as independent replicates and no comparison p-values are calculated.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import ICI_predict as base
import ICI_analysis_common as common


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "results_revision"
RUN_TAG = "Holdout_v1"

DATASETS_TO_RUN: Union[str, List[str]] = "ALL"

TEST_SIZE = 0.30
HOLDOUT_SPLITS = 25
RANDOM_STATE = 42
MAX_K = base.PRIMARY_MAX_K
FIXED_SPLIT_BOOTSTRAP_N = base.BOOTSTRAP_N

# Dataset 3 Post 2 required a higher elastic-net iteration ceiling to reach
# convergence in the original analysis.  The model, grids, folds, and stopping
# tolerance are otherwise unchanged.  Keeping this as a dataset-specific
# setting removes the need for a separate retry script.
DEFAULT_EN_MAX_ITER_LIMIT = int(base.EN_MAX_ITER_LIMIT)
DATASET_EN_MAX_ITER_LIMIT_OVERRIDES: Dict[str, int] = {
    "dataset3_post2": 80000,
}

CONTINUE_ON_DATASET_ERROR = True
ALLOW_OVERWRITE = False

CANONICAL_KEYS: Tuple[str, ...] = (
    "dataset1_pre", "dataset1_post1",
    "dataset2_pre", "dataset2_post1", "dataset2_post2",
    "dataset3_pre", "dataset3_post1", "dataset3_post2",
    "dataset4_pre", "dataset4_post1", "dataset4_post2",
    "dataset5_pre", "dataset5_post1", "dataset5_post2",
    "dataset6_pre", "dataset6_post1", "dataset7_pre",
)


def _run_root() -> Path:
    return OUTPUT_ROOT / RUN_TAG if RUN_TAG else OUTPUT_ROOT


def _en_max_iter_limit(dataset_key: str) -> int:
    return int(
        DATASET_EN_MAX_ITER_LIMIT_OVERRIDES.get(
            dataset_key,
            DEFAULT_EN_MAX_ITER_LIMIT,
        )
    )


def _selected_keys(specs: Dict[str, base.DatasetSpec]) -> List[str]:
    if isinstance(DATASETS_TO_RUN, str) and DATASETS_TO_RUN.upper() in {"ALL", "ALL_PRIMARY"}:
        missing = sorted(set(CANONICAL_KEYS).difference(specs))
        extra = sorted(set(specs).difference(CANONICAL_KEYS))
        if missing or extra:
            raise ValueError(f"Canonical 17-matrix set mismatch; missing={missing}; extra={extra}")
        return list(CANONICAL_KEYS)
    selected = base.select_targets(DATASETS_TO_RUN, list(specs))
    if not isinstance(DATASETS_TO_RUN, str):
        requested = [str(value).strip().lower() for value in DATASETS_TO_RUN]
        if len(requested) != len(set(requested)):
            raise ValueError("DATASETS_TO_RUN contains duplicate keys")
        missing = sorted(set(requested).difference(selected))
        if missing:
            raise ValueError(f"Unknown requested dataset keys: {missing}")
    return selected


def _holdout_splits(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros((len(y), 1), dtype=float)
    repeated = StratifiedShuffleSplit(
        n_splits=HOLDOUT_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    splits = [
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in repeated.split(dummy, y)
    ]
    fixed = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    fixed_train, fixed_test = next(fixed.split(dummy, y))
    if not np.array_equal(splits[0][0], fixed_train) or not np.array_equal(
        splits[0][1], fixed_test
    ):
        raise RuntimeError("Repeated holdout split 1 does not equal the prespecified seed-42 split")
    return splits


def _summarize_metric(frame: pd.DataFrame, metric: str) -> Dict[str, float]:
    values = frame[metric].to_numpy(dtype=float)
    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    return {
        f"{metric}_median": float(np.median(values)),
        f"{metric}_q1": q1,
        f"{metric}_q3": q3,
        f"{metric}_iqr": q3 - q1,
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        f"{metric}_min": float(np.min(values)),
        f"{metric}_max": float(np.max(values)),
    }


def run_dataset(
    spec: base.DatasetSpec,
    output_dir: Path,
    en_max_iter_limit: int,
) -> Dict[str, Any]:
    if int(base.EN_MAX_ITER_LIMIT) != int(en_max_iter_limit):
        raise RuntimeError(
            f"Elastic-net iteration setting was not applied for {spec.key}: "
            f"expected {en_max_iter_limit}, found {base.EN_MAX_ITER_LIMIT}"
        )
    bundle = base.load_dataset(spec, spec.primary_sheet)
    splits = _holdout_splits(bundle.y)
    ids = bundle.ids.astype(str).tolist()

    prediction_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, Any]] = []
    log_lines: List[str] = []

    for split_number, (train_idx, test_idx) in enumerate(splits, start=1):
        role = (
            "prespecified_fixed_seed42"
            if split_number == 1
            else "repeated_split_sensitivity"
        )
        context = f"{spec.key} | holdout split={split_number}"
        fit = common.tune_train_predict(
            bundle,
            train_idx,
            test_idx,
            context=context,
            outer_iter=split_number,
            max_k=MAX_K,
            imputation="half_minimum",
        )
        warning_rows.extend(fit.warning_rows)
        log_lines.extend(fit.log_lines)
        y_train = bundle.y[train_idx]
        y_test = bundle.y[test_idx]
        weights = common.class_weights(y_train)

        for local_index, original_index in enumerate(test_idx):
            prediction_rows.append(
                {
                    "dataset": spec.key,
                    "method": "holdout",
                    "split": split_number,
                    "split_role": role,
                    base.ID_COL: ids[original_index],
                    "y_true": int(bundle.y[original_index]),
                    "p": float(fit.probability[local_index]),
                    "train_positive": int(np.sum(y_train == 1)),
                    "train_negative": int(np.sum(y_train == 0)),
                    **weights,
                }
            )
        train_set = set(train_idx.tolist())
        for original_index in range(len(bundle.y)):
            assignment_rows.append(
                {
                    "dataset": spec.key,
                    "split": split_number,
                    "split_role": role,
                    base.ID_COL: ids[original_index],
                    "y_true": int(bundle.y[original_index]),
                    "subset": "train" if original_index in train_set else "test",
                }
            )

        split_metrics = common.metric_values(y_test, fit.probability)
        metric_rows.append(
            {
                "dataset": spec.key,
                "split": split_number,
                "split_role": role,
                **base.class_counts(y_test),
                **split_metrics,
            }
        )
        parameter_rows.append(
            {
                "dataset": spec.key,
                "split": split_number,
                "split_role": role,
                "test_size_requested": TEST_SIZE,
                "train_n": len(train_idx),
                "train_positive": int(np.sum(y_train == 1)),
                "train_negative": int(np.sum(y_train == 0)),
                "test_n": len(test_idx),
                "test_positive": int(np.sum(y_test == 1)),
                "test_negative": int(np.sum(y_test == 0)),
                "en_max_iter_limit": int(en_max_iter_limit),
                **weights,
                "inner_splits_used": fit.inner_plan.n_splits,
                "inner_repeats": fit.inner_plan.n_repeats,
                "en_C": fit.en_C,
                "en_l1_ratio": fit.en_l1_ratio,
                "k_selected": fit.k_selected,
                "l2_C": fit.l2_C,
                "n_ranked_features": len(fit.ranking),
                "n_selected_features": len(fit.selected_features),
                "n_model_features": len(fit.model_features),
            }
        )
        ranking_lookup = fit.ranking_table.set_index("feature")
        selected_set = set(fit.selected_features)
        model_set = set(fit.model_features)
        for feature in bundle.feature_names:
            in_ranking = feature in ranking_lookup.index
            feature_rows.append(
                {
                    "dataset": spec.key,
                    "split": split_number,
                    "split_role": role,
                    "feature": feature,
                    "rank": (
                        int(ranking_lookup.loc[feature, "rank"])
                        if in_ranking
                        else np.nan
                    ),
                    "eligible_freq_inner": (
                        float(ranking_lookup.loc[feature, "eligible_freq"])
                        if in_ranking
                        else 0.0
                    ),
                    "nonzero_freq_inner": (
                        float(ranking_lookup.loc[feature, "nonzero_freq"])
                        if in_ranking
                        else 0.0
                    ),
                    "mean_abs_coef_inner": (
                        float(ranking_lookup.loc[feature, "mean_abs_coef"])
                        if in_ranking
                        else 0.0
                    ),
                    "selected_topk": feature in selected_set,
                    "model_used": feature in model_set,
                    "model_coefficient": fit.coefficients.get(feature, np.nan),
                    "k_selected": fit.k_selected,
                }
            )
        print(f"[{spec.key}] holdout split {split_number}/{len(splits)} complete")

    predictions = pd.DataFrame(prediction_rows)
    assignments = pd.DataFrame(assignment_rows)
    parameters = pd.DataFrame(parameter_rows)
    features = pd.DataFrame(feature_rows)
    split_metrics = pd.DataFrame(metric_rows)
    warnings_table = pd.DataFrame(
        warning_rows,
        columns=["dataset", "stage", "warning_type", "message"],
    )

    if predictions.duplicated(["split", base.ID_COL]).any():
        raise RuntimeError(f"Duplicate holdout test predictions for {spec.key}")
    expected_test_rows = int(sum(len(test) for _, test in splits))
    if len(predictions) != expected_test_rows:
        raise RuntimeError(
            f"Expected {expected_test_rows} holdout predictions, found {len(predictions)}"
        )
    if assignments.duplicated(["split", base.ID_COL]).any() or len(assignments) != (
        len(bundle.y) * HOLDOUT_SPLITS
    ):
        raise RuntimeError(f"Incomplete holdout split assignments for {spec.key}")

    fixed_predictions = predictions.loc[predictions["split"] == 1].copy()
    fixed_metrics = split_metrics.loc[split_metrics["split"] == 1].iloc[0].to_dict()
    fixed_minority_n = int(
        min(
            np.sum(fixed_predictions["y_true"].to_numpy(dtype=int) == 0),
            np.sum(fixed_predictions["y_true"].to_numpy(dtype=int) == 1),
        )
    )
    if fixed_minority_n < 2:
        fixed_ci_low, fixed_ci_high = np.nan, np.nan
        fixed_ci_method = (
            "not estimated: fixed test set has fewer than two observations in one class"
        )
        warning_rows.append(
            {
                "dataset": spec.key,
                "stage": "fixed-holdout-bootstrap",
                "warning_type": "minority_class_too_small",
                "message": (
                    f"minority_n={fixed_minority_n}; AUROC CI suppressed because a "
                    "stratified bootstrap would repeatedly reuse a single patient"
                ),
            }
        )
    else:
        fixed_ci_low, fixed_ci_high = base.bootstrap_auc_ci(
            fixed_predictions["y_true"].to_numpy(dtype=int),
            fixed_predictions["p"].to_numpy(dtype=float),
            n_boot=FIXED_SPLIT_BOOTSTRAP_N,
            seed=RANDOM_STATE,
        )
        fixed_ci_method = (
            "stratified patient bootstrap of the fixed test set; descriptive; "
            "model-selection uncertainty not included"
        )
    fixed_metrics.update(
        {
            "roc_auc_ci95_lo_descriptive": float(fixed_ci_low),
            "roc_auc_ci95_hi_descriptive": float(fixed_ci_high),
            "ci_method": fixed_ci_method,
        }
    )

    summary: Dict[str, Any] = {
        "dataset": spec.key,
        "dataset_id": spec.dataset_id,
        "timepoint": spec.timepoint,
        "analysis_role": common.dataset_role(spec),
        "n_samples": len(bundle.y),
        "n_positive": int(np.sum(bundle.y == 1)),
        "n_negative": int(np.sum(bundle.y == 0)),
        "holdout_splits": HOLDOUT_SPLITS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "en_max_iter_limit": int(en_max_iter_limit),
        "fixed_split": 1,
        "fixed_roc_auc": float(fixed_metrics["roc_auc"]),
        "fixed_roc_auc_ci95_lo_descriptive": float(fixed_ci_low),
        "fixed_roc_auc_ci95_hi_descriptive": float(fixed_ci_high),
        "fixed_average_precision": float(fixed_metrics["average_precision"]),
        "fixed_brier": float(fixed_metrics["brier"]),
        "repeated_split_interpretation": (
            "descriptive split dependence; overlapping splits are not independent replicates"
        ),
    }
    for metric in ("roc_auc", "average_precision", "brier"):
        summary.update(_summarize_metric(split_metrics, metric))

    warnings_table = pd.DataFrame(
        warning_rows,
        columns=["dataset", "stage", "warning_type", "message"],
    )

    common.write_csv(predictions, output_dir / "test_predictions_long.csv")
    common.write_csv(fixed_predictions, output_dir / "fixed_split_test_predictions.csv")
    common.write_csv(assignments, output_dir / "split_assignments.csv")
    common.write_csv(parameters, output_dir / "split_parameters.csv")
    common.write_csv(features, output_dir / "split_features_long.csv")
    common.write_csv(split_metrics, output_dir / "split_metrics.csv")
    common.write_csv(pd.DataFrame([fixed_metrics]), output_dir / "fixed_split_metrics.csv")
    common.write_csv(pd.DataFrame([summary]), output_dir / "metrics_summary.csv")
    common.write_csv(bundle.feature_manifest, output_dir / "feature_manifest.csv")
    common.write_csv(warnings_table, output_dir / "warnings.csv")
    (output_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    common.write_json(
        output_dir / "settings_used.json",
        {
            "input_path": str(spec.path),
            "input_sha256": bundle.input_sha256,
            "test_size": TEST_SIZE,
            "holdout_splits": HOLDOUT_SPLITS,
            "random_state": RANDOM_STATE,
            "fixed_split_definition": (
                "split 1 from StratifiedShuffleSplit(n_splits=25, test_size=0.30, "
                "random_state=42); verified against n_splits=1"
            ),
            "max_k": MAX_K,
            "fixed_split_bootstrap_n": FIXED_SPLIT_BOOTSTRAP_N,
            "run_tag": RUN_TAG,
            "en_max_iter_limit": int(en_max_iter_limit),
            "en_max_iter_source": (
                "dataset-specific override"
                if spec.key in DATASET_EN_MAX_ITER_LIMIT_OVERRIDES
                else "default"
            ),
            "model_sequence": [
                "training-only elastic-net tuning",
                "training-only resampling feature ranking",
                "training-only top-k/L2-C 1-SE selection",
                "training-only preprocessing and L2 fit",
                "test prediction",
            ],
            "class_weight": "balanced",
            "missing_rate_threshold": base.MISSING_RATE_THRESHOLD,
            "threshold_metrics": "not computed",
            "representative_repeat": "not selected",
            "inferential_tests": "not performed",
        },
    )
    output_hashes = common.output_sha256_manifest(
        output_dir,
        (
            "test_predictions_long.csv",
            "fixed_split_test_predictions.csv",
            "split_assignments.csv",
            "split_parameters.csv",
            "split_features_long.csv",
            "split_metrics.csv",
            "fixed_split_metrics.csv",
            "metrics_summary.csv",
            "feature_manifest.csv",
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
            "dataset": spec.key,
            "input_path": str(spec.path),
            "input_sha256": bundle.input_sha256,
            "sample_counts": base.class_counts(bundle.y),
            "summary": summary,
            "effective_settings": {
                "run_tag": RUN_TAG,
                "en_max_iter_limit": int(en_max_iter_limit),
                "en_max_iter_source": (
                    "dataset-specific override"
                    if spec.key in DATASET_EN_MAX_ITER_LIMIT_OVERRIDES
                    else "default"
                ),
            },
            "warning_count": len(warning_rows),
            "output_sha256": output_hashes,
            **common.provenance_payload(Path(__file__)),
        },
    )
    return summary


def main() -> None:
    if not np.isclose(float(TEST_SIZE), 0.30) or int(HOLDOUT_SPLITS) != 25:
        raise ValueError("Holdout protocol is fixed at 25 stratified 70/30 splits")
    if int(RANDOM_STATE) != 42:
        raise ValueError("Holdout protocol requires RANDOM_STATE=42")
    if int(MAX_K) != int(base.PRIMARY_MAX_K):
        raise ValueError("Holdout MAX_K must equal ICI_predict.PRIMARY_MAX_K")
    if int(FIXED_SPLIT_BOOTSTRAP_N) < 1:
        raise ValueError("Holdout FIXED_SPLIT_BOOTSTRAP_N must be >= 1")
    if int(DEFAULT_EN_MAX_ITER_LIMIT) != 20000:
        raise ValueError(
            "Unexpected default elastic-net iteration ceiling; expected 20000"
        )
    invalid_limits = {
        key: value
        for key, value in DATASET_EN_MAX_ITER_LIMIT_OVERRIDES.items()
        if key not in CANONICAL_KEYS or int(value) < DEFAULT_EN_MAX_ITER_LIMIT
    }
    if invalid_limits:
        raise ValueError(f"Invalid dataset-specific EN iteration settings: {invalid_limits}")
    started_time = time.time()
    started_utc = common.utc_now()
    specs = base.discover_datasets()
    keys = _selected_keys(specs)
    if not keys:
        raise RuntimeError("DATASETS_TO_RUN did not match any canonical input")
    run_root = common.prepare_run_root(_run_root(), ALLOW_OVERWRITE)

    summary_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for key in keys:
        print("=" * 88)
        en_max_iter_limit = _en_max_iter_limit(key)
        print(
            f"[START] Holdout | {key} | "
            f"EN max_iter limit={en_max_iter_limit}"
        )
        original_en_max_iter_limit = int(base.EN_MAX_ITER_LIMIT)
        base.EN_MAX_ITER_LIMIT = int(en_max_iter_limit)
        try:
            with common.staged_dataset_dir(run_root, key) as stage:
                summary = run_dataset(
                    specs[key],
                    stage,
                    en_max_iter_limit=en_max_iter_limit,
                )
            summary_rows.append(summary)
            manifest_rows.append(
                {
                    "dataset": key,
                    "status": "completed",
                    "analysis_role": common.dataset_role(specs[key]),
                    "output_dir": str((run_root / "datasets" / key).resolve()),
                    "message": "",
                    "en_max_iter_limit": en_max_iter_limit,
                    "fixed_roc_auc": summary["fixed_roc_auc"],
                    "median_roc_auc": summary["roc_auc_median"],
                }
            )
            print(f"[DONE] Holdout | {key}")
        except Exception as exc:
            manifest_rows.append(
                {
                    "dataset": key,
                    "status": "failed",
                    "analysis_role": common.dataset_role(specs[key]),
                    "output_dir": "",
                    "message": f"{type(exc).__name__}: {exc}",
                    "en_max_iter_limit": en_max_iter_limit,
                }
            )
            print(f"[FAILED] Holdout | {key}: {type(exc).__name__}: {exc}")
            if not CONTINUE_ON_DATASET_ERROR:
                raise
        finally:
            base.EN_MAX_ITER_LIMIT = original_en_max_iter_limit

    summary_table = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)
    common.write_csv(manifest, run_root / "run_manifest.csv")
    common.write_csv(summary_table, run_root / "holdout_summary_all_datasets.csv")
    common.finalize_run_receipt(
        run_root,
        Path(__file__),
        started_utc,
        started_time,
        manifest,
        {
            "analysis": "prespecified_and_repeated_stratified_holdout",
            "datasets_to_run": DATASETS_TO_RUN,
            "test_size": TEST_SIZE,
            "holdout_splits": HOLDOUT_SPLITS,
            "random_state": RANDOM_STATE,
            "fixed_split": 1,
            "max_k": MAX_K,
            "fixed_split_bootstrap_n": FIXED_SPLIT_BOOTSTRAP_N,
            "run_tag": RUN_TAG,
            "default_en_max_iter_limit": DEFAULT_EN_MAX_ITER_LIMIT,
            "dataset_en_max_iter_overrides": DATASET_EN_MAX_ITER_LIMIT_OVERRIDES,
            "representative_repeat": "not selected",
            "inferential_tests": "not performed",
            "continue_on_dataset_error": CONTINUE_ON_DATASET_ERROR,
        },
    )
    completed = int((manifest["status"] == "completed").sum())
    failed = int((manifest["status"] == "failed").sum())
    print("=" * 88)
    print(f"Holdout completed: {completed}; failed: {failed}")
    print(f"Output: {run_root}")
    if failed:
        raise RuntimeError(
            f"Holdout finished with {failed} failed dataset(s); inspect run_manifest.csv"
        )


if __name__ == "__main__":
    main()
