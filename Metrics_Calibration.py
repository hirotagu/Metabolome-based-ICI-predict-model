#!/usr/bin/env python3
"""Evaluate discrimination, Brier score, and calibration from saved predictions.

No model is refitted. Raw probabilities are the primary analysis. A
fold/split-specific prior adjustment is added as a Supplement sensitivity
analysis because the upstream logistic models use ``class_weight='balanced'``.
The adjustment is applied to every held-out prediction before repeated OOF
probabilities are averaged.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, roc_curve

import ICI_predict as base
import ICI_analysis_common as common


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_primary"
COMPARE_METHOD_RUN_ROOT = SCRIPT_DIR / "results_revision" / "Compare_method_v1"
HOLDOUT_DEFAULT_RUN_ROOT = SCRIPT_DIR / "results_revision" / "Holdout_v1"
HOLDOUT_DEFAULT_PRODUCER = SCRIPT_DIR / "Holdout.py"

HOLDOUT_EXTENDED_ITER_KEY = "dataset3_post2"
HOLDOUT_DEFAULT_EN_MAX_ITER_LIMIT = 20000
HOLDOUT_EXTENDED_EN_MAX_ITER_LIMIT = 80000

OUTPUT_ROOT = SCRIPT_DIR / "results_revision"
RUN_TAG = "Metrics_Calibration_v1"

DATASETS_TO_RUN: Union[str, List[str]] = "ALL"

CALIBRATION_BINS = 10
RUN_BOOTSTRAP_CI = True
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
MIN_BOOTSTRAP_SUCCESS = max(100, BOOTSTRAP_N // 10)

STRICT_BASE_SCRIPT_HASH = True
CONTINUE_ON_DATASET_ERROR = True
ALLOW_OVERWRITE = False

METHOD_ORDER: Tuple[str, ...] = ("fully_nested", "fully_non_nested", "holdout")
PROBABILITY_VARIANTS: Tuple[str, ...] = ("raw", "prior_adjusted")
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

def _holdout_source(dataset_key: str) -> Tuple[Path, Path]:
    return HOLDOUT_DEFAULT_RUN_ROOT, HOLDOUT_DEFAULT_PRODUCER

def _selected_keys(specs: Mapping[str, base.DatasetSpec]) -> List[str]:
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


def _resolve_completed_dataset_dir(run_root: Path, dataset_key: str) -> Path:
    root = Path(run_root).expanduser().resolve()
    manifest_path = root / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = common.read_csv(manifest_path)
    required = {"dataset", "status", "output_dir"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Run manifest lacks columns: {sorted(missing)}")
    row = manifest.loc[manifest["dataset"].astype(str) == dataset_key]
    if len(row) != 1 or row.iloc[0]["status"] != "completed":
        raise FileNotFoundError(f"No unique completed result for {dataset_key} in {root}")
    manifest_dir = Path(str(row.iloc[0]["output_dir"])).expanduser()
    fallback = root / "datasets" / dataset_key
    directory = next((path.resolve() for path in (manifest_dir, fallback) if path.exists()), None)
    if directory is None:
        raise FileNotFoundError(f"Dataset output directory is missing for {dataset_key} in {root}")
    return directory


def _validate_downstream_receipt(
    directory: Path,
    dataset_key: str,
    input_sha256: str,
    producer_script: Path,
    required_outputs: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    receipt_path = directory / "run_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"Missing downstream receipt: {receipt_path}")
    receipt = common.read_json(receipt_path)
    if receipt.get("status") != "completed" or receipt.get("dataset") != dataset_key:
        raise ValueError(f"Invalid downstream receipt for {dataset_key}: {directory}")
    if receipt.get("input_sha256") != input_sha256:
        raise ValueError(f"Input SHA mismatch for {dataset_key}: {directory}")
    if STRICT_BASE_SCRIPT_HASH:
        current_hash = common.sha256_file(Path(base.__file__).resolve())
        if receipt.get("base_script_sha256") != current_hash:
            raise ValueError(f"Base-script SHA mismatch for {dataset_key}: {directory}")
    producer = Path(producer_script).resolve()
    if not producer.is_file() or receipt.get("script_sha256") != common.sha256_file(producer):
        raise ValueError(f"Producer-script SHA mismatch for {dataset_key}: {producer}")
    if receipt.get("common_module_sha256") != common.sha256_file(Path(common.__file__).resolve()):
        raise ValueError(f"Common-module SHA mismatch for {dataset_key}: {directory}")
    common.validate_output_sha256_manifest(directory, receipt, required_outputs)
    settings = common.read_json(directory / "settings_used.json")
    return receipt, settings


def _strict_probabilities(probability: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite probabilities in {context}")
    if not np.logical_and(values > 0.0, values < 1.0).all():
        raise ValueError(
            f"Boundary/out-of-range probabilities in {context}; calibration logits "
            "are not silently clipped"
        )
    return values


def _prior_adjust(
    probability: np.ndarray,
    train_positive: np.ndarray,
    train_negative: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    p = _strict_probabilities(probability, "prior adjustment")
    n1 = np.asarray(train_positive, dtype=float)
    n0 = np.asarray(train_negative, dtype=float)
    if not (len(p) == len(n1) == len(n0)) or (n1 <= 0).any() or (n0 <= 0).any():
        raise ValueError("Invalid fold class counts for prior adjustment")
    logit = np.log(p / (1.0 - p))
    shift = np.log(n1 / n0)
    adjusted = expit(logit + shift)
    if not np.isfinite(adjusted).all() or not np.logical_and(adjusted > 0, adjusted < 1).all():
        raise RuntimeError("Prior adjustment produced invalid probabilities")
    return adjusted, shift


def _validate_balanced_weights(frame: pd.DataFrame) -> None:
    if not {"class_weight_negative", "class_weight_positive"}.issubset(frame.columns):
        return
    expected_negative = frame["train_n"] / (2.0 * frame["train_negative"])
    expected_positive = frame["train_n"] / (2.0 * frame["train_positive"])
    if not np.allclose(frame["class_weight_negative"], expected_negative, atol=1e-12):
        raise ValueError("Saved negative-class weights do not match class_weight='balanced'")
    if not np.allclose(frame["class_weight_positive"], expected_positive, atol=1e-12):
        raise ValueError("Saved positive-class weights do not match class_weight='balanced'")


def _prepare_cv_predictions(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    dataset_key: str,
    method: str,
) -> pd.DataFrame:
    required_prediction = {base.ID_COL, "repeat", "outer_fold", "y_true", "p"}
    required_parameter = {
        "outer_repeat",
        "outer_fold",
        "train_n",
        "train_positive",
        "train_negative",
        "test_n",
        "test_positive",
        "test_negative",
    }
    if missing := required_prediction.difference(predictions.columns):
        raise ValueError(f"{method} predictions lack columns: {sorted(missing)}")
    if missing := required_parameter.difference(parameters.columns):
        raise ValueError(f"{method} parameters lack columns: {sorted(missing)}")
    if "dataset" in predictions and not (
        predictions["dataset"].astype(str) == dataset_key
    ).all():
        raise ValueError(f"{method} prediction dataset mismatch")
    if "method" in predictions and not (
        predictions["method"].astype(str) == method
    ).all():
        raise ValueError(f"{method} prediction method mismatch")
    raw_labels = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(float)
    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0.0, 1.0]).all():
        raise ValueError(f"{method} labels must be exactly 0 or 1")
    if parameters.duplicated(["outer_repeat", "outer_fold"]).any():
        raise ValueError(f"Duplicate {method} outer parameter keys")
    params = parameters.copy()
    _validate_balanced_weights(params)
    params = params.rename(
        columns={"outer_repeat": "repeat", "train_n": "training_n"}
    )
    merge_columns = [
        "repeat",
        "outer_fold",
        "training_n",
        "train_positive",
        "train_negative",
    ]
    frame = predictions.merge(
        params[merge_columns],
        on=["repeat", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    if frame[["training_n", "train_positive", "train_negative"]].isna().any().any():
        raise ValueError(f"Unmatched {method} prediction-to-parameter rows")
    if not np.array_equal(
        frame["training_n"].to_numpy(dtype=int),
        (
            frame["train_positive"].to_numpy(dtype=int)
            + frame["train_negative"].to_numpy(dtype=int)
        ),
    ):
        raise ValueError(f"Invalid {method} training class counts")
    fold_counts = (
        frame.groupby(["repeat", "outer_fold"], sort=True)
        .agg(
            observed_test_n=("y_true", "size"),
            observed_test_positive=("y_true", "sum"),
        )
        .reset_index()
        .merge(
            parameters.rename(columns={"outer_repeat": "repeat"})[
                ["repeat", "outer_fold", "train_n", "test_n", "test_positive", "test_negative"]
            ],
            on=["repeat", "outer_fold"],
            validate="one_to_one",
        )
    )
    if not (
        (fold_counts["observed_test_n"] == fold_counts["test_n"])
        & (fold_counts["observed_test_positive"] == fold_counts["test_positive"])
        & (
            fold_counts["observed_test_n"] - fold_counts["observed_test_positive"]
            == fold_counts["test_negative"]
        )
        & (
            fold_counts["train_n"] + fold_counts["test_n"]
            == frame[base.ID_COL].astype(str).nunique()
        )
    ).all():
        raise ValueError(f"{method} fold counts do not reproduce")
    frame["p_raw"] = frame["p"].astype(float)
    frame["p_prior_adjusted"], frame["prior_log_odds_shift"] = _prior_adjust(
        frame["p_raw"].to_numpy(float),
        frame["train_positive"].to_numpy(float),
        frame["train_negative"].to_numpy(float),
    )
    frame["dataset"] = dataset_key
    frame["method"] = method
    frame[base.ID_COL] = frame[base.ID_COL].astype(str)
    if frame.duplicated([base.ID_COL, "repeat"]).any():
        raise ValueError(f"Duplicate {method} ID/repeat predictions")
    return frame


def _prepare_holdout_predictions(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    dataset_key: str,
) -> pd.DataFrame:
    required_prediction = {base.ID_COL, "split", "split_role", "y_true", "p"}
    required_parameter = {
        "split",
        "train_n",
        "train_positive",
        "train_negative",
        "test_n",
        "test_positive",
        "test_negative",
    }
    if missing := required_prediction.difference(predictions.columns):
        raise ValueError(f"Holdout predictions lack columns: {sorted(missing)}")
    if missing := required_parameter.difference(parameters.columns):
        raise ValueError(f"Holdout parameters lack columns: {sorted(missing)}")
    if "dataset" in predictions and not (
        predictions["dataset"].astype(str) == dataset_key
    ).all():
        raise ValueError("Holdout prediction dataset mismatch")
    if "method" in predictions and not (
        predictions["method"].astype(str) == "holdout"
    ).all():
        raise ValueError("Holdout prediction method mismatch")
    raw_labels = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(float)
    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0.0, 1.0]).all():
        raise ValueError("Holdout labels must be exactly 0 or 1")
    split_roles = predictions[["split", "split_role"]].drop_duplicates()
    if split_roles.duplicated("split").any() or not (
        split_roles.set_index("split").loc[1, "split_role"]
        == "prespecified_fixed_seed42"
    ):
        raise ValueError("Holdout split 1 is not the prespecified fixed seed-42 split")
    if parameters.duplicated(["split"]).any():
        raise ValueError("Duplicate holdout parameter split")
    _validate_balanced_weights(parameters)
    params = parameters.rename(columns={"train_n": "training_n"})
    frame = predictions.merge(
        params[
            ["split", "training_n", "train_positive", "train_negative"]
        ],
        on="split",
        how="left",
        validate="many_to_one",
        suffixes=("", "_parameter"),
    )
    # Prediction files also retain train counts. Verify them if present.
    for column in ("train_positive", "train_negative"):
        parameter_column = f"{column}_parameter"
        if parameter_column in frame.columns:
            if not np.array_equal(
                frame[column].to_numpy(dtype=int),
                frame[parameter_column].to_numpy(dtype=int),
            ):
                raise ValueError(f"Holdout {column} differs between prediction and parameter files")
            frame[column] = frame[parameter_column]
            frame = frame.drop(columns=[parameter_column])
    if frame[["training_n", "train_positive", "train_negative"]].isna().any().any():
        raise ValueError("Unmatched holdout prediction-to-parameter rows")
    if not np.array_equal(
        frame["training_n"].to_numpy(dtype=int),
        (
            frame["train_positive"].to_numpy(dtype=int)
            + frame["train_negative"].to_numpy(dtype=int)
        ),
    ):
        raise ValueError("Invalid holdout training class counts")
    observed = (
        frame.groupby("split", sort=True)
        .agg(observed_test_n=("y_true", "size"), observed_test_positive=("y_true", "sum"))
        .reset_index()
        .merge(
            parameters[["split", "train_n", "test_n", "test_positive", "test_negative"]],
            on="split",
            validate="one_to_one",
        )
    )
    if not (
        (observed["observed_test_n"] == observed["test_n"])
        & (observed["observed_test_positive"] == observed["test_positive"])
        & (
            observed["observed_test_n"] - observed["observed_test_positive"]
            == observed["test_negative"]
        )
    ).all():
        raise ValueError("Holdout test counts do not reproduce")
    frame["p_raw"] = frame["p"].astype(float)
    frame["p_prior_adjusted"], frame["prior_log_odds_shift"] = _prior_adjust(
        frame["p_raw"].to_numpy(float),
        frame["train_positive"].to_numpy(float),
        frame["train_negative"].to_numpy(float),
    )
    frame["dataset"] = dataset_key
    frame["method"] = "holdout"
    frame[base.ID_COL] = frame[base.ID_COL].astype(str)
    if frame.duplicated([base.ID_COL, "split"]).any():
        raise ValueError("Duplicate holdout ID/split predictions")
    return frame


def _logit_probability(probability: np.ndarray) -> np.ndarray:
    p = _strict_probabilities(probability, "calibration fit")
    return np.log(p / (1.0 - p))


def _fit_calibration_intercept(y: np.ndarray, p: np.ndarray) -> Tuple[float, str]:
    labels = np.asarray(y, dtype=int)
    if set(np.unique(labels).tolist()) != {0, 1}:
        return np.nan, "both_classes_required"
    z = _logit_probability(p)

    def score(intercept: float) -> float:
        return float(np.sum(expit(intercept + z) - labels))

    try:
        value = float(brentq(score, -50.0, 50.0, maxiter=200))
    except Exception as exc:
        return np.nan, f"fit_failed:{type(exc).__name__}:{exc}"
    return value, "ok"


def _fit_calibration_slope(
    y: np.ndarray, p: np.ndarray
) -> Tuple[float, float, str]:
    labels = np.asarray(y, dtype=int)
    if set(np.unique(labels).tolist()) != {0, 1}:
        return np.nan, np.nan, "both_classes_required"
    z = _logit_probability(p)
    design = np.column_stack([np.ones(len(z)), z])
    if np.ptp(z) <= 1e-12 or np.linalg.matrix_rank(design, tol=1e-12) < 2:
        return np.nan, np.nan, "nonidentifiable_constant_logit"
    negative_z = z[labels == 0]
    positive_z = z[labels == 1]
    if (
        np.max(negative_z) <= np.min(positive_z)
        or np.max(positive_z) <= np.min(negative_z)
    ):
        return np.nan, np.nan, "nonidentifiable_complete_or_quasi_separation"

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - labels * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = design @ beta
        return design.T @ (expit(eta) - labels)

    intercept0, _ = _fit_calibration_intercept(labels, p)
    initial = np.array([0.0 if not np.isfinite(intercept0) else intercept0, 1.0])
    try:
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(-25.0, 25.0), (-25.0, 25.0)],
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
    except Exception as exc:
        return np.nan, np.nan, f"fit_failed:{type(exc).__name__}:{exc}"
    beta = np.asarray(result.x, dtype=float)
    if not result.success or not np.isfinite(beta).all():
        return np.nan, np.nan, f"fit_failed:{result.message}"
    if np.any(np.isclose(np.abs(beta), 25.0, atol=1e-5)):
        return np.nan, np.nan, "fit_failed:parameter_bound_reached"
    return float(beta[0]), float(beta[1]), "ok"


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    raw_labels = np.asarray(y)
    numeric_labels = raw_labels.astype(float)
    if not np.isfinite(numeric_labels).all() or not np.isin(
        numeric_labels, [0.0, 1.0]
    ).all():
        raise ValueError("Calibration labels must be exactly 0 or 1")
    labels = numeric_labels.astype(int)
    probability = _strict_probabilities(p, "calibration metrics")
    discrimination = common.metric_values(labels, probability)
    intercept, intercept_status = _fit_calibration_intercept(labels, probability)
    free_intercept, slope, slope_status = _fit_calibration_slope(labels, probability)
    statuses = [intercept_status, slope_status]
    return {
        **discrimination,
        "event_rate": float(np.mean(labels)),
        "mean_predicted_probability": float(np.mean(probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "recalibration_intercept_free": free_intercept,
        "calibration_intercept_status": intercept_status,
        "calibration_slope_status": slope_status,
        "calibration_fit_ok": all(value == "ok" for value in statuses),
    }


def bootstrap_cis(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    labels = np.asarray(y, dtype=int)
    probability = np.asarray(p, dtype=float)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Both classes are required for calibration bootstrap")
    rng = np.random.default_rng(int(seed))
    metric_names = (
        "roc_auc",
        "average_precision",
        "brier",
        "event_rate",
        "mean_predicted_probability",
        "calibration_intercept",
        "calibration_slope",
    )
    values: Dict[str, List[float]] = {name: [] for name in metric_names}
    both_class_draws = 0
    for _ in range(int(n_boot)):
        sample = rng.integers(0, len(labels), size=len(labels))
        sampled_y = labels[sample]
        sampled_p = probability[sample]
        values["brier"].append(float(np.mean((sampled_y - sampled_p) ** 2)))
        values["event_rate"].append(float(np.mean(sampled_y)))
        values["mean_predicted_probability"].append(float(np.mean(sampled_p)))
        if set(np.unique(sampled_y).tolist()) != {0, 1}:
            continue
        both_class_draws += 1
        result = calibration_metrics(sampled_y, sampled_p)
        for name in (
            "roc_auc",
            "average_precision",
            "calibration_intercept",
            "calibration_slope",
        ):
            value = float(result[name])
            if np.isfinite(value):
                values[name].append(value)
    output: Dict[str, Any] = {
        "bootstrap_target": int(n_boot),
        "bootstrap_draws_with_both_classes": int(both_class_draws),
        "bootstrap_method": (
            "ordinary n-out-of-n patient bootstrap of fixed predictions; "
            "class-dependent metrics omit single-class draws; descriptive; "
            "model-fitting and CV-selection uncertainty not included"
        ),
    }
    for name in metric_names:
        array = np.asarray(values[name], dtype=float)
        output[f"{name}_bootstrap_success"] = int(len(array))
        output[f"{name}_ci95_lo"] = (
            float(np.quantile(array, 0.025)) if len(array) else np.nan
        )
        output[f"{name}_ci95_hi"] = (
            float(np.quantile(array, 0.975)) if len(array) else np.nan
        )
    return output


def _calibration_coordinates(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    unique_n = int(frame["p"].nunique())
    bins = max(1, min(int(n_bins), len(frame), unique_n))
    try:
        frame["bin"] = pd.qcut(frame["p"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        frame["bin"] = 0
    result = (
        frame.groupby("bin", dropna=False, sort=True)
        .agg(
            n=("y_true", "size"),
            predicted_mean=("p", "mean"),
            observed_rate=("y_true", "mean"),
            predicted_min=("p", "min"),
            predicted_max=("p", "max"),
        )
        .reset_index()
    )
    result["bin"] = np.arange(1, len(result) + 1, dtype=int)
    return result


def _primary_unit(
    method: str,
    unit_type: str,
    unit_id: int,
) -> bool:
    if method in {"fully_nested", "fully_non_nested"}:
        return unit_type == "averaged_oof"
    return method == "holdout" and unit_type == "holdout_split" and unit_id == 1


def _unit_frames_from_cv(
    frame: pd.DataFrame,
    bundle: base.DataBundle,
    method: str,
) -> List[Tuple[str, int, pd.DataFrame]]:
    units: List[Tuple[str, int, pd.DataFrame]] = []
    for repeat, subset in frame.groupby("repeat", sort=True):
        if subset[base.ID_COL].duplicated().any() or len(subset) != len(bundle.y):
            raise ValueError(f"{method} repeat {repeat} is not one prediction per input ID")
        ordered = subset.set_index(base.ID_COL).loc[bundle.ids.astype(str)].reset_index()
        if not np.array_equal(ordered["y_true"].to_numpy(dtype=int), bundle.y):
            raise ValueError(f"{method} repeat {repeat} labels differ from the input data")
        units.append(("repeat", int(repeat), ordered))
    aggregate = (
        frame.groupby(base.ID_COL, sort=False)
        .agg(
            y_true=("y_true", "first"),
            p_raw=("p_raw", "mean"),
            p_prior_adjusted=("p_prior_adjusted", "mean"),
            contributing_predictions=("p_raw", "size"),
        )
        .reset_index()
    )
    aggregate = aggregate.set_index(base.ID_COL).loc[bundle.ids.astype(str)].reset_index()
    if not np.array_equal(aggregate["y_true"].to_numpy(dtype=int), bundle.y):
        raise ValueError(f"{method} averaged OOF labels differ from the input data")
    expected_repeats = int(frame["repeat"].nunique())
    if not (aggregate["contributing_predictions"] == expected_repeats).all():
        raise ValueError(f"Incomplete {method} adjusted-before-averaging OOF predictions")
    units.append(("averaged_oof", 0, aggregate))
    return units


def _unit_frames_from_holdout(
    frame: pd.DataFrame,
    bundle: base.DataBundle,
) -> List[Tuple[str, int, pd.DataFrame]]:
    units: List[Tuple[str, int, pd.DataFrame]] = []
    label_lookup = dict(zip(bundle.ids.astype(str), bundle.y.astype(int)))
    for split, subset in frame.groupby("split", sort=True):
        if subset[base.ID_COL].duplicated().any():
            raise ValueError(f"Holdout split {split} has duplicate test IDs")
        observed_ids = subset[base.ID_COL].astype(str)
        if not observed_ids.isin(label_lookup).all():
            raise ValueError(f"Holdout split {split} contains unknown IDs")
        expected_labels = observed_ids.map(label_lookup).to_numpy(dtype=int)
        if not np.array_equal(subset["y_true"].to_numpy(dtype=int), expected_labels):
            raise ValueError(f"Holdout split {split} labels differ from the input data")
        units.append(("holdout_split", int(split), subset.copy()))
    return units


def _source_group(spec: base.DatasetSpec) -> str:
    if spec.dataset_id == "dataset6":
        return "dataset6_supplement"
    if spec.timepoint == "pre" and spec.dataset_id in {
        "dataset1",
        "dataset2",
        "dataset3",
        "dataset4",
        "dataset5",
        "dataset7",
    }:
        return "pre_primary_6_cohorts"
    return "post_descriptive"


def _load_sources(
    artifact: common.MainArtifacts,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    compare_dir = _resolve_completed_dataset_dir(
        COMPARE_METHOD_RUN_ROOT, artifact.spec.key
    )
    holdout_root, holdout_producer = _holdout_source(artifact.spec.key)
    holdout_dir = _resolve_completed_dataset_dir(
        holdout_root, artifact.spec.key
    )
    _, compare_settings = _validate_downstream_receipt(
        compare_dir,
        artifact.spec.key,
        artifact.bundle.input_sha256,
        SCRIPT_DIR / "Compare_method.py",
        (
            "outer_predictions_long.csv",
            "oof_predictions.csv",
            "outer_parameters.csv",
            "settings_used.json",
        ),
    )
    _, holdout_settings = _validate_downstream_receipt(
        holdout_dir,
        artifact.spec.key,
        artifact.bundle.input_sha256,
        holdout_producer,
        (
            "test_predictions_long.csv",
            "split_assignments.csv",
            "split_parameters.csv",
            "settings_used.json",
        ),
    )
    compare_cv = compare_settings.get("selection_and_evaluation_cv", {})
    if (
        compare_settings.get("class_weight") != "balanced"
        or int(compare_cv.get("splits", -1)) != artifact.outer_plan.n_splits
        or int(compare_cv.get("repeats", -1)) != artifact.outer_plan.n_repeats
        or int(compare_cv.get("random_state", -1)) != int(base.RANDOM_STATE)
        or int(compare_settings.get("max_k", -1)) != int(base.PRIMARY_MAX_K)
        or compare_settings.get("main_run_receipt_sha256")
        != common.sha256_file(artifact.run_dir / "run_receipt.json")
    ):
        raise ValueError(f"Compare_method settings mismatch for {artifact.spec.key}")
    if (
        holdout_settings.get("class_weight") != "balanced"
        or int(holdout_settings.get("holdout_splits", -1)) != 25
        or not np.isclose(float(holdout_settings.get("test_size", np.nan)), 0.30)
        or int(holdout_settings.get("random_state", -1)) != 42
        or int(holdout_settings.get("max_k", -1)) != int(base.PRIMARY_MAX_K)
    ):
        raise ValueError(f"Holdout settings mismatch for {artifact.spec.key}")
    expected_en_max_iter_limit = (
        HOLDOUT_EXTENDED_EN_MAX_ITER_LIMIT
        if artifact.spec.key == HOLDOUT_EXTENDED_ITER_KEY
        else HOLDOUT_DEFAULT_EN_MAX_ITER_LIMIT
    )
    if (
        int(holdout_settings.get("en_max_iter_limit", -1))
        != expected_en_max_iter_limit
    ):
        raise ValueError(
            f"Holdout en_max_iter_limit mismatch for {artifact.spec.key}: "
            f"expected {expected_en_max_iter_limit}"
        )
    compare_predictions = common.read_csv(
        compare_dir / "outer_predictions_long.csv", id_columns=[base.ID_COL]
    )
    compare_parameters = common.read_csv(compare_dir / "outer_parameters.csv")
    holdout_predictions = common.read_csv(
        holdout_dir / "test_predictions_long.csv", id_columns=[base.ID_COL]
    )
    holdout_parameters = common.read_csv(holdout_dir / "split_parameters.csv")

    nested = _prepare_cv_predictions(
        artifact.predictions_long,
        artifact.parameters,
        artifact.spec.key,
        "fully_nested",
    )
    nonnested = _prepare_cv_predictions(
        compare_predictions,
        compare_parameters,
        artifact.spec.key,
        "fully_non_nested",
    )
    holdout = _prepare_holdout_predictions(
        holdout_predictions,
        holdout_parameters,
        artifact.spec.key,
    )
    return nested, nonnested, holdout, {
        "main": str(artifact.run_dir),
        "compare_method": str(compare_dir),
        "holdout": str(holdout_dir),
    }


def analyze_dataset(
    artifact: common.MainArtifacts,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nested, nonnested, holdout, source_dirs = _load_sources(artifact)
    source_frames = {
        "fully_nested": nested,
        "fully_non_nested": nonnested,
        "holdout": holdout,
    }
    prediction_export = pd.concat(source_frames.values(), ignore_index=True, sort=False)

    metric_rows: List[Dict[str, Any]] = []
    roc_frames: List[pd.DataFrame] = []
    pr_frames: List[pd.DataFrame] = []
    calibration_frames: List[pd.DataFrame] = []
    qc_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, Any]] = []

    for method in METHOD_ORDER:
        frame = source_frames[method]
        units = (
            _unit_frames_from_cv(frame, artifact.bundle, method)
            if method != "holdout"
            else _unit_frames_from_holdout(frame, artifact.bundle)
        )
        for unit_type, unit_id, unit in units:
            primary = _primary_unit(method, unit_type, unit_id)
            for variant in PROBABILITY_VARIANTS:
                p_column = "p_raw" if variant == "raw" else "p_prior_adjusted"
                labels = unit["y_true"].to_numpy(dtype=int)
                probability = unit[p_column].to_numpy(dtype=float)
                point = calibration_metrics(labels, probability)
                row: Dict[str, Any] = {
                    "dataset": artifact.spec.key,
                    "dataset_id": artifact.spec.dataset_id,
                    "timepoint": artifact.spec.timepoint,
                    "analysis_role": common.dataset_role(artifact.spec),
                    "source_group": _source_group(artifact.spec),
                    "method": method,
                    "unit_type": unit_type,
                    "unit_id": unit_id,
                    "is_primary_unit": primary,
                    "probability_variant": variant,
                    "probability_analysis_role": (
                        "primary_raw" if variant == "raw" else "supplement_prior_sensitivity"
                    ),
                    "n": len(unit),
                    "n_positive": int(np.sum(labels == 1)),
                    "n_negative": int(np.sum(labels == 0)),
                    **point,
                }
                if RUN_BOOTSTRAP_CI and primary:
                    boot = bootstrap_cis(
                        labels,
                        probability,
                        n_boot=BOOTSTRAP_N,
                        seed=(
                            BOOTSTRAP_SEED
                            + artifact.spec.dataset_number * 100
                            + METHOD_ORDER.index(method) * 10
                        ),
                    )
                    row.update(boot)
                    for metric_name in (
                        "roc_auc",
                        "average_precision",
                        "brier",
                        "event_rate",
                        "mean_predicted_probability",
                        "calibration_intercept",
                        "calibration_slope",
                    ):
                        success = int(boot[f"{metric_name}_bootstrap_success"])
                        if success < MIN_BOOTSTRAP_SUCCESS:
                            warning_rows.append(
                                {
                                    "dataset": artifact.spec.key,
                                    "method": method,
                                    "unit_type": unit_type,
                                    "unit_id": unit_id,
                                    "probability_variant": variant,
                                    "issue": f"low_{metric_name}_bootstrap_success",
                                    "detail": (
                                        f"successful={success}; required={MIN_BOOTSTRAP_SUCCESS}; "
                                        f"target={BOOTSTRAP_N}"
                                    ),
                                }
                            )
                metric_rows.append(row)
                qc_rows.append(
                    {
                        "dataset": artifact.spec.key,
                        "method": method,
                        "unit_type": unit_type,
                        "unit_id": unit_id,
                        "probability_variant": variant,
                        "n": len(unit),
                        "unique_ids": int(unit[base.ID_COL].astype(str).nunique()),
                        "duplicate_id_count": int(unit[base.ID_COL].astype(str).duplicated().sum()),
                        "probability_min": float(np.min(probability)),
                        "probability_max": float(np.max(probability)),
                        "finite_probability": bool(np.isfinite(probability).all()),
                        "both_classes": set(np.unique(labels).tolist()) == {0, 1},
                        "calibration_fit_ok": point["calibration_fit_ok"],
                    }
                )
                if primary:
                    fpr, tpr, thresholds = roc_curve(labels, probability)
                    roc = pd.DataFrame(
                        {"fpr": fpr, "tpr": tpr, "threshold": thresholds}
                    )
                    precision, recall, pr_thresholds = precision_recall_curve(
                        labels, probability
                    )
                    pr = pd.DataFrame(
                        {
                            "precision": precision,
                            "recall": recall,
                            "threshold": np.append(pr_thresholds, np.nan),
                        }
                    )
                    calibration = _calibration_coordinates(
                        labels, probability, CALIBRATION_BINS
                    )
                    for coordinates in (roc, pr, calibration):
                        coordinates.insert(0, "probability_variant", variant)
                        coordinates.insert(0, "method", method)
                        coordinates.insert(0, "dataset", artifact.spec.key)
                    roc_frames.append(roc)
                    pr_frames.append(pr)
                    calibration_frames.append(calibration)

    metrics = pd.DataFrame(metric_rows)
    primary_metrics = metrics.loc[metrics["is_primary_unit"]].copy()
    if len(primary_metrics) != len(METHOD_ORDER) * len(PROBABILITY_VARIANTS):
        raise RuntimeError(
            f"Expected {len(METHOD_ORDER) * len(PROBABILITY_VARIANTS)} primary metric rows "
            f"for {artifact.spec.key}; found {len(primary_metrics)}"
        )
    qc = pd.DataFrame(qc_rows)
    warnings_table = pd.DataFrame(
        warning_rows,
        columns=[
            "dataset",
            "method",
            "unit_type",
            "unit_id",
            "probability_variant",
            "issue",
            "detail",
        ],
    )

    common.write_csv(prediction_export, output_dir / "predictions_with_prior_adjustment.csv")
    common.write_csv(metrics, output_dir / "calibration_metrics_all_units.csv")
    common.write_csv(primary_metrics, output_dir / "calibration_primary_metrics.csv")
    common.write_csv(qc, output_dir / "prediction_qc.csv")
    common.write_csv(pd.concat(roc_frames, ignore_index=True), output_dir / "roc_coordinates.csv")
    common.write_csv(pd.concat(pr_frames, ignore_index=True), output_dir / "pr_coordinates.csv")
    common.write_csv(
        pd.concat(calibration_frames, ignore_index=True),
        output_dir / "calibration_coordinates.csv",
    )
    common.write_csv(warnings_table, output_dir / "warnings.csv")
    common.write_json(
        output_dir / "settings_used.json",
        {
            "source_directories": source_dirs,
            "input_path": str(artifact.spec.path),
            "input_sha256": artifact.bundle.input_sha256,
            "methods": list(METHOD_ORDER),
            "probability_variants": list(PROBABILITY_VARIANTS),
            "raw_probability_role": "primary",
            "prior_adjusted_probability_role": "Supplement sensitivity",
            "prior_adjustment_formula": (
                "logit(p_adjusted) = logit(p_raw) + log(train_positive/train_negative)"
            ),
            "adjustment_order": (
                "adjust each held-out fold/split prediction before averaging repeated OOF predictions"
            ),
            "prior_adjustment_scope": (
                "corrects the balanced-class prior/intercept only; does not correct slope, "
                "overfitting, regularization, or feature-selection effects"
            ),
            "calibration_intercept_definition": "CITL; calibration slope fixed at 1",
            "calibration_slope_definition": (
                "logistic recalibration slope with a freely estimated intercept"
            ),
            "calibration_bins": CALIBRATION_BINS,
            "bootstrap_enabled": RUN_BOOTSTRAP_CI,
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_interpretation": (
                "descriptive ordinary patient bootstrap of fixed predictions; "
                "not CV-variance corrected"
            ),
            "formal_tests": "not performed",
            "threshold_metrics": "not computed",
        },
    )
    common.write_json(
        output_dir / "run_receipt.json",
        {
            "status": "completed",
            "completed_utc": common.utc_now(),
            "dataset": artifact.spec.key,
            "input_path": str(artifact.spec.path),
            "input_sha256": artifact.bundle.input_sha256,
            "source_directories": source_dirs,
            "sample_counts": base.class_counts(artifact.bundle.y),
            "primary_metrics": primary_metrics.to_dict(orient="records"),
            "warning_count": len(warning_rows),
            **common.provenance_payload(Path(__file__)),
        },
    )
    return metric_rows, primary_metrics.to_dict(orient="records")


def _group_summary(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    if primary_metrics.empty:
        return pd.DataFrame()
    metric_names = (
        "roc_auc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "event_rate",
        "mean_predicted_probability",
    )
    rows: List[Dict[str, Any]] = []
    for keys, subset in primary_metrics.groupby(
        ["source_group", "method", "probability_variant"], sort=True
    ):
        source_group, method, variant = keys
        for metric in metric_names:
            values = subset[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            q1 = float(np.quantile(values, 0.25))
            q3 = float(np.quantile(values, 0.75))
            rows.append(
                {
                    "source_group": source_group,
                    "method": method,
                    "probability_variant": variant,
                    "metric": metric,
                    "n_datasets": int(len(values)),
                    "median": float(np.median(values)),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "inferential_test": "not performed",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if RUN_BOOTSTRAP_CI and int(BOOTSTRAP_N) < 1:
        raise ValueError("Metrics_Calibration BOOTSTRAP_N must be >= 1")
    started_time = time.time()
    started_utc = common.utc_now()
    specs = base.discover_datasets()
    keys = _selected_keys(specs)
    if not keys:
        raise RuntimeError("DATASETS_TO_RUN did not match any canonical input")
    run_root = common.prepare_run_root(_run_root(), ALLOW_OVERWRITE)

    all_metrics: List[Dict[str, Any]] = []
    all_primary: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for key in keys:
        print("=" * 88)
        print(f"[START] Metrics_Calibration | {key}")
        try:
            main_dir = common.resolve_main_run_dirs(MAIN_RUN_ROOT, specs, [key])[key]
            artifact = common.validate_main_artifacts(
                specs[key], main_dir, strict_script_hash=STRICT_BASE_SCRIPT_HASH
            )
            with common.staged_dataset_dir(run_root, key) as stage:
                metric_rows, primary_rows = analyze_dataset(artifact, stage)
            all_metrics.extend(metric_rows)
            all_primary.extend(primary_rows)
            manifest_rows.append(
                {
                    "dataset": key,
                    "status": "completed",
                    "analysis_role": common.dataset_role(specs[key]),
                    "output_dir": str((run_root / "datasets" / key).resolve()),
                    "message": "",
                }
            )
            print(f"[DONE] Metrics_Calibration | {key}")
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
            print(
                f"[FAILED] Metrics_Calibration | {key}: {type(exc).__name__}: {exc}"
            )
            if not CONTINUE_ON_DATASET_ERROR:
                raise

    manifest = pd.DataFrame(manifest_rows)
    metrics_table = pd.DataFrame(all_metrics)
    primary_table = pd.DataFrame(all_primary)
    group_summary = _group_summary(primary_table)
    common.write_csv(manifest, run_root / "run_manifest.csv")
    common.write_csv(metrics_table, run_root / "calibration_metrics_all_datasets.csv")
    common.write_csv(primary_table, run_root / "calibration_primary_metrics_all_datasets.csv")
    common.write_csv(group_summary, run_root / "calibration_group_summary.csv")
    common.finalize_run_receipt(
        run_root,
        Path(__file__),
        started_utc,
        started_time,
        manifest,
        {
            "analysis": "saved_prediction_calibration_and_prior_sensitivity",
            "main_run_root": str(Path(MAIN_RUN_ROOT).resolve()),
            "compare_method_run_root": str(Path(COMPARE_METHOD_RUN_ROOT).resolve()),
            "holdout_default_run_root": str(
                Path(HOLDOUT_DEFAULT_RUN_ROOT).resolve()
            ),
            "holdout_default_producer": str(
                Path(HOLDOUT_DEFAULT_PRODUCER).resolve()
            ),
            "holdout_extended_iter_key": HOLDOUT_EXTENDED_ITER_KEY,
            "holdout_default_en_max_iter_limit": HOLDOUT_DEFAULT_EN_MAX_ITER_LIMIT,
            "holdout_extended_en_max_iter_limit": HOLDOUT_EXTENDED_EN_MAX_ITER_LIMIT,
            "datasets_to_run": DATASETS_TO_RUN,
            "methods": list(METHOD_ORDER),
            "probability_variants": list(PROBABILITY_VARIANTS),
            "raw_probability_role": "primary",
            "prior_adjusted_probability_role": "Supplement sensitivity",
            "bootstrap_enabled": RUN_BOOTSTRAP_CI,
            "bootstrap_n": BOOTSTRAP_N,
            "formal_tests": "not performed",
            "continue_on_dataset_error": CONTINUE_ON_DATASET_ERROR,
        },
    )
    completed = int((manifest["status"] == "completed").sum())
    failed = int((manifest["status"] == "failed").sum())
    print("=" * 88)
    print(f"Metrics_Calibration completed: {completed}; failed: {failed}")
    print(f"Output: {run_root}")
    if failed:
        raise RuntimeError(
            f"Metrics_Calibration finished with {failed} failed dataset(s); inspect run_manifest.csv"
        )


if __name__ == "__main__":
    main()
