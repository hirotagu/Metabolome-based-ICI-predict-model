#!/usr/bin/env python3
"""Shared, auditable utilities for the public ICI analyses.

This module deliberately reuses the preprocessing and model-fitting engine in
``ICI_predict.py``. It is not an analysis entry point; place it beside the
analysis scripts and ``ICI_predict.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import openpyxl
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest, rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    import ICI_predict as base
except ImportError as exc:  # pragma: no cover - user-facing import guard
    raise ImportError(
        "ICI_predict.py must be in the same directory as the analysis scripts"
    ) from exc


REQUIRED_MAIN_FILES = (
    "run_receipt.json",
    "settings_used.json",
    "outer_predictions_long.csv",
    "oof_predictions.csv",
    "repeat_metrics.csv",
    "metrics_summary.csv",
    "outer_parameters.csv",
    "outer_features_long.csv",
    "fold_counts.csv",
    "feature_manifest.csv",
)


@dataclass
class MainArtifacts:
    spec: base.DatasetSpec
    bundle: base.DataBundle
    run_dir: Path
    receipt: Dict[str, Any]
    settings: Dict[str, Any]
    predictions_long: pd.DataFrame
    oof: pd.DataFrame
    repeat_metrics: pd.DataFrame
    metrics: pd.DataFrame
    parameters: pd.DataFrame
    outer_features: pd.DataFrame
    fold_counts: pd.DataFrame
    feature_manifest: pd.DataFrame
    outer_plan: base.CVPlan


@dataclass
class TunedFit:
    probability: np.ndarray
    en_C: float
    en_l1_ratio: float
    ranking: List[str]
    ranking_table: pd.DataFrame
    k_selected: int
    l2_C: float
    selected_features: List[str]
    model_features: List[str]
    coefficients: Dict[str, float]
    inner_plan: base.CVPlan
    k_curve: pd.DataFrame
    preprocessor_audit: pd.DataFrame
    log_lines: List[str]
    warning_rows: List[Dict[str, Any]]


@dataclass
class GlobalSpec:
    en_C: float
    en_l1_ratio: float
    ranking: List[str]
    ranking_table: pd.DataFrame
    k_selected: int
    l2_C: float
    selected_features: List[str]
    k_curve: pd.DataFrame
    selection_plan: base.CVPlan
    log_lines: List[str]
    warning_rows: List[Dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_sha256_manifest(
    directory: Path,
    filenames: Sequence[str],
) -> Dict[str, str]:
    """Hash a declared set of completed outputs for downstream validation."""
    root = Path(directory).resolve()
    result: Dict[str, str] = {}
    for filename in filenames:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Cannot hash missing output: {path}")
        result[str(filename)] = sha256_file(path)
    return result


def validate_output_sha256_manifest(
    directory: Path,
    receipt: Mapping[str, Any],
    required_filenames: Sequence[str],
) -> None:
    """Require producer-declared hashes and verify every consumed artifact."""
    recorded = receipt.get("output_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Receipt lacks output_sha256 manifest: {directory}")
    root = Path(directory).resolve()
    for filename in required_filenames:
        expected = recorded.get(str(filename))
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"Receipt lacks SHA256 for {filename}: {directory}")
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing hashed output: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Output SHA256 mismatch for {path}")


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    clean_payload = _json_sanitize(payload)
    Path(path).write_text(
        json.dumps(
            clean_payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_sanitize(item) for item in value.tolist()]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path, id_columns: Sequence[str] = ()) -> pd.DataFrame:
    dtype = {column: "string" for column in id_columns}
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=dtype)
    for column in id_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def class_weights(y: np.ndarray) -> Dict[str, float]:
    raw = np.asarray(y)
    if not np.isfinite(raw.astype(float)).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("Class labels must be exactly 0 or 1")
    counts = base.class_counts(raw.astype(int))
    if counts["positive"] <= 0 or counts["negative"] <= 0:
        raise ValueError("Both classes are required to calculate balanced class weights")
    n = float(counts["n"])
    return {
        "class_weight_negative": n / (2.0 * counts["negative"]),
        "class_weight_positive": n / (2.0 * counts["positive"]),
    }


def metric_values(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    raw_labels = np.asarray(y)
    if not np.isfinite(raw_labels.astype(float)).all() or not np.isin(
        raw_labels, [0, 1]
    ).all():
        raise ValueError("Metric labels must be exactly 0 or 1")
    labels = raw_labels.astype(int)
    p = np.asarray(probability, dtype=float)
    if len(labels) != len(p) or len(labels) == 0:
        raise ValueError("Metric inputs must be non-empty and have equal length")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Both classes are required for AUROC/AP/Brier")
    if not np.isfinite(p).all() or not np.logical_and(p >= 0, p <= 1).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return {
        "roc_auc": float(roc_auc_score(labels, p)),
        "average_precision": float(average_precision_score(labels, p)),
        "brier": float(brier_score_loss(labels, p)),
    }


def averaged_oof(
    predictions_long: pd.DataFrame,
    ids_in_order: Sequence[str],
    expected_repeats: int,
    id_col: str = base.ID_COL,
    probability_col: str = "p",
) -> pd.DataFrame:
    required = {id_col, "repeat", "y_true", probability_col}
    missing = required.difference(predictions_long.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    frame = predictions_long.copy()
    frame[id_col] = frame[id_col].astype(str)
    raw_labels = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0.0, 1.0]).all():
        raise ValueError("Prediction labels must be exactly 0 or 1")
    frame["y_true"] = raw_labels.astype(int)
    per_id_repeat = frame.groupby([id_col, "repeat"], sort=False).size()
    if not (per_id_repeat == 1).all():
        raise ValueError("Each ID must be predicted exactly once within each repeat")
    if frame[probability_col].isna().any() or not frame[probability_col].between(0, 1).all():
        raise ValueError("Invalid prediction probability")
    label_nunique = frame.groupby(id_col, sort=False)["y_true"].nunique(dropna=False)
    if not (label_nunique == 1).all():
        raise ValueError("An ID has inconsistent labels across predictions")
    oof = (
        frame.groupby(id_col, sort=False)
        .agg(
            y_true=("y_true", "first"),
            p_oof=(probability_col, "mean"),
            p_oof_sd=(probability_col, "std"),
            oof_count=(probability_col, "size"),
        )
        .reset_index()
    )
    ordered = [str(value) for value in ids_in_order]
    if set(oof[id_col]) != set(ordered) or len(oof) != len(ordered):
        raise ValueError("OOF IDs do not match the input dataset")
    oof = oof.set_index(id_col).loc[ordered].reset_index()
    if not (oof["oof_count"] == int(expected_repeats)).all():
        raise ValueError("Incomplete repeated OOF predictions")
    return oof


def repeat_metric_table(
    predictions_long: pd.DataFrame,
    dataset: str,
    method: str,
    probability_col: str = "p",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for repeat, subset in predictions_long.groupby("repeat", sort=True):
        values = metric_values(
            subset["y_true"].to_numpy(dtype=int),
            subset[probability_col].to_numpy(dtype=float),
        )
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "repeat": int(repeat),
                **base.class_counts(subset["y_true"].to_numpy(dtype=int)),
                **values,
            }
        )
    return pd.DataFrame(rows)


def dataset_role(spec: base.DatasetSpec) -> str:
    if spec.dataset_id == "dataset6":
        return "supplement_small_sample"
    if spec.dataset_id == "dataset7":
        return "nmr_separate"
    return "main_lcms"


def inference_group(spec: base.DatasetSpec) -> str:
    if spec.timepoint == "pre" and spec.dataset_id in {
        "dataset1",
        "dataset2",
        "dataset3",
        "dataset4",
        "dataset5",
        "dataset7",
    }:
        return "pre_primary_independent_cohorts"
    if spec.dataset_id == "dataset6":
        return "dataset6_supplement"
    if spec.timepoint.startswith("post"):
        return "post_descriptive"
    return "other_descriptive"


def prepare_run_root(path: Path, allow_overwrite: bool) -> Path:
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        if not allow_overwrite:
            raise FileExistsError(
                f"Run output already contains files: {root}. Change RUN_TAG or set "
                "ALLOW_OVERWRITE=True intentionally."
            )
        backup = root.with_name(f"{root.name}.backup_{timestamp_token()}")
        counter = 1
        while backup.exists():
            backup = root.with_name(f"{root.name}.backup_{timestamp_token()}_{counter}")
            counter += 1
        root.rename(backup)
    root.mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(exist_ok=True)
    (root / "failed").mkdir(exist_ok=True)
    (root / "_staging").mkdir(exist_ok=True)
    return root


@contextmanager
def staged_dataset_dir(run_root: Path, dataset_key: str) -> Iterator[Path]:
    root = Path(run_root)
    token = uuid.uuid4().hex[:10]
    stage = root / "_staging" / f"{dataset_key}_{token}"
    final = root / "datasets" / dataset_key
    stage.mkdir(parents=True, exist_ok=False)
    try:
        yield stage
        if final.exists():
            raise FileExistsError(f"Dataset output already exists unexpectedly: {final}")
        stage.rename(final)
    except Exception as exc:
        failure = root / "failed" / f"{dataset_key}_{timestamp_token()}_{token}"
        try:
            write_json(
                stage / "failure.json",
                {
                    "status": "failed",
                    "dataset": dataset_key,
                    "failed_utc": utc_now(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            stage.rename(failure)
        except Exception:
            pass
        raise


def _same_sequence(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return json.dumps(list(left), default=_json_default) == json.dumps(
        list(right), default=_json_default
    )


def _expected_main_dir(main_run_root: Path, spec: base.DatasetSpec) -> Path:
    if spec.dataset_id == "dataset6":
        return (
            Path(main_run_root)
            / "supplement"
            / "dataset6_small_sample"
            / "primary"
            / spec.key
        )
    return Path(main_run_root) / "main" / "primary" / spec.key


def resolve_main_run_dirs(
    main_run_root: Path,
    specs: Mapping[str, base.DatasetSpec],
    required_keys: Sequence[str],
) -> Dict[str, Path]:
    root = Path(main_run_root).expanduser().resolve()
    manifest_path = root / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Current-v3 run_manifest.csv was not found: {manifest_path}. "
            "Wait for the main run to complete and point MAIN_RUN_ROOT to that run."
        )
    manifest = read_csv(manifest_path)
    required_columns = {"dataset", "scenario", "output_dir"}
    if missing := required_columns.difference(manifest.columns):
        raise ValueError(f"Main run manifest lacks columns: {sorted(missing)}")
    primary = manifest.loc[manifest["scenario"].astype(str) == "primary"].copy()
    if "status" in manifest.columns:
        primary = primary.loc[primary["status"].astype(str) == "completed"].copy()
    if primary["dataset"].duplicated().any():
        duplicates = primary.loc[primary["dataset"].duplicated(False), "dataset"].tolist()
        raise ValueError(f"Duplicate primary rows in run manifest: {duplicates}")

    resolved: Dict[str, Path] = {}
    for key in required_keys:
        if key not in specs:
            raise KeyError(f"Input dataset was not discovered: {key}")
        rows = primary.loc[primary["dataset"].astype(str) == key]
        if len(rows) != 1:
            raise FileNotFoundError(
                f"Expected one completed primary manifest row for {key}; found {len(rows)}"
            )
        manifest_dir = Path(str(rows.iloc[0]["output_dir"])).expanduser()
        candidates = [manifest_dir, _expected_main_dir(root, specs[key])]
        run_dir = next((path.resolve() for path in candidates if path.exists()), None)
        if run_dir is None:
            raise FileNotFoundError(
                f"Result directory for {key} does not exist. Tried: "
                + ", ".join(str(path) for path in candidates)
            )
        resolved[key] = run_dir
    return resolved


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _assert_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=atol, rtol=1e-10):
        raise ValueError(f"{label} mismatch: actual={actual}, expected={expected}")


def exact_integer(value: Any, label: str) -> int:
    """Parse an integer-like saved value without silently truncating corruption."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer; found {value!r}") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be a finite integer; found {value!r}")
    return int(number)


def validate_main_artifacts(
    spec: base.DatasetSpec,
    run_dir: Path,
    strict_script_hash: bool = True,
) -> MainArtifacts:
    directory = Path(run_dir).resolve()
    for filename in REQUIRED_MAIN_FILES:
        if not (directory / filename).exists():
            raise FileNotFoundError(f"Missing current-v3 output: {directory / filename}")

    bundle = base.load_dataset(spec, spec.primary_sheet)
    receipt = read_json(directory / "run_receipt.json")
    settings = read_json(directory / "settings_used.json")
    if receipt.get("status") != "completed":
        raise ValueError(f"Main receipt is not completed for {spec.key}")
    if receipt.get("dataset") != spec.key or receipt.get("scenario") != "primary":
        raise ValueError(f"Main receipt dataset/scenario mismatch for {spec.key}")
    if receipt.get("input_sha256") != bundle.input_sha256:
        raise ValueError(f"Input SHA256 differs from the completed main run for {spec.key}")
    if strict_script_hash:
        current_hash = sha256_file(Path(base.__file__).resolve())
        if receipt.get("script_sha256") != current_hash:
            raise ValueError(
                f"ICI_predict.py SHA256 differs from the main run for {spec.key}"
            )
    saved_versions = receipt.get("package_versions", {})
    current_sklearn = sklearn.__version__
    if saved_versions.get("scikit_learn") != current_sklearn:
        raise ValueError(
            f"scikit-learn version differs from the main run for {spec.key}: "
            f"saved={saved_versions.get('scikit_learn')}, current={current_sklearn}"
        )
    if settings.get("class_weight") != "balanced":
        raise ValueError(f"Main class_weight is not balanced for {spec.key}")
    if int(settings.get("random_state", -1)) != int(base.RANDOM_STATE):
        raise ValueError(f"Main random_state differs from current v3 for {spec.key}")
    if int(settings.get("outer_repeats", -1)) != int(base.OUTER_REPEATS):
        raise ValueError(f"Main outer repeats differ from current v3 for {spec.key}")
    if int(settings.get("inner_repeats", -1)) != int(base.INNER_REPEATS):
        raise ValueError(f"Main inner repeats differ from current v3 for {spec.key}")
    scenario = settings.get("scenario", {})
    if not isinstance(scenario, Mapping) or (
        scenario.get("name") != "primary"
        or scenario.get("source") != "primary"
        or scenario.get("sheet_name") != spec.primary_sheet
        or scenario.get("imputation") != "half_minimum"
        or int(scenario.get("max_k", -1)) != int(base.PRIMARY_MAX_K)
        or int(scenario.get("outer_splits_target", -1))
        != int(base.OUTER_SPLITS_TARGET)
    ):
        raise ValueError(f"Main primary-scenario settings mismatch for {spec.key}")
    expected_outer_splits = min(
        int(base.OUTER_SPLITS_TARGET),
        int(np.sum(bundle.y == 0)),
        int(np.sum(bundle.y == 1)),
    )
    if (
        int(settings.get("outer_splits_used", -1)) != expected_outer_splits
        or int(settings.get("inner_splits_target", -1))
        != int(base.INNER_SPLITS_TARGET)
    ):
        raise ValueError(f"Main CV split settings mismatch for {spec.key}")
    _assert_close(
        float(settings.get("missing_rate_threshold", np.nan)),
        float(base.MISSING_RATE_THRESHOLD),
        f"{spec.key} missing-rate threshold",
    )
    for setting_name, current in (
        ("en_C_grid", base.EN_C_GRID),
        ("en_l1_ratio_grid", base.EN_L1_RATIO_GRID),
        ("l2_C_grid", base.L2_C_GRID),
    ):
        if not _same_sequence(settings.get(setting_name, []), current):
            raise ValueError(f"{spec.key}: {setting_name} differs from current v3")

    predictions = read_csv(
        directory / "outer_predictions_long.csv", id_columns=[base.ID_COL]
    )
    oof = read_csv(directory / "oof_predictions.csv", id_columns=[base.ID_COL])
    repeat_metrics = read_csv(directory / "repeat_metrics.csv")
    metrics = read_csv(directory / "metrics_summary.csv")
    parameters = read_csv(directory / "outer_parameters.csv")
    outer_features = read_csv(directory / "outer_features_long.csv")
    fold_counts = read_csv(directory / "fold_counts.csv")
    feature_manifest = read_csv(directory / "feature_manifest.csv")

    _require_columns(
        predictions,
        ["dataset", "scenario", base.ID_COL, "repeat", "outer_fold", "y_true", "p"],
        f"{spec.key} outer predictions",
    )
    _require_columns(oof, [base.ID_COL, "y_true", "p_oof", "p_oof_sd", "oof_count"], f"{spec.key} OOF")
    _require_columns(
        parameters,
        [
            "dataset",
            "scenario",
            "outer_repeat",
            "outer_fold",
            "train_n",
            "train_positive",
            "train_negative",
            "test_n",
            "test_positive",
            "test_negative",
            "en_C",
            "en_l1_ratio",
            "k_selected",
            "l2_C",
        ],
        f"{spec.key} outer parameters",
    )
    _require_columns(
        outer_features,
        [
            "dataset",
            "scenario",
            "outer_repeat",
            "outer_fold",
            "feature",
            "rank",
            "eligible",
            "selected_topk",
            "model_used",
            "k_selected",
        ],
        f"{spec.key} outer features",
    )
    if not (predictions["dataset"].astype(str) == spec.key).all() or not (
        predictions["scenario"].astype(str) == "primary"
    ).all():
        raise ValueError(f"Prediction dataset/scenario mismatch for {spec.key}")
    if predictions.duplicated([base.ID_COL, "repeat"]).any():
        raise ValueError(f"Duplicate ID/repeat predictions for {spec.key}")
    if predictions["p"].isna().any() or not predictions["p"].between(0, 1).all():
        raise ValueError(f"Invalid probabilities in main predictions for {spec.key}")

    log_lines: List[str] = []
    outer_plan = base.make_cv_plan(
        bundle.y,
        target_splits=base.OUTER_SPLITS_TARGET,
        n_repeats=base.OUTER_REPEATS,
        context=f"{spec.key} main-validation outer",
        log_lines=log_lines,
        random_state=base.RANDOM_STATE,
    )
    ids = bundle.ids.astype(str).tolist()
    for record in outer_plan.records:
        subset = predictions.loc[
            (predictions["repeat"].astype(int) == record.repeat)
            & (predictions["outer_fold"].astype(int) == record.fold)
        ]
        expected_ids = {ids[index] for index in record.test_idx}
        if set(subset[base.ID_COL].astype(str)) != expected_ids or len(subset) != len(expected_ids):
            raise ValueError(
                f"Main fold assignment differs from reconstructed v3 split: "
                f"{spec.key} repeat={record.repeat}, fold={record.fold}"
            )
        expected_labels = {
            ids[index]: int(bundle.y[index]) for index in record.test_idx
        }
        observed_labels = dict(
            zip(subset[base.ID_COL].astype(str), subset["y_true"].astype(int))
        )
        if observed_labels != expected_labels:
            raise ValueError(
                f"Main prediction labels differ from input labels: {spec.key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )
    if parameters.duplicated(["outer_repeat", "outer_fold"]).any() or len(parameters) != len(
        outer_plan.records
    ):
        raise ValueError(f"Main parameter rows do not match outer fits for {spec.key}")
    expected_fold_keys = {(record.repeat, record.fold) for record in outer_plan.records}
    observed_fold_keys = set(
        zip(
            parameters["outer_repeat"].astype(int),
            parameters["outer_fold"].astype(int),
        )
    )
    if observed_fold_keys != expected_fold_keys:
        raise ValueError(f"Main parameter fold keys differ from the outer plan for {spec.key}")
    for outer_iter, record in enumerate(outer_plan.records, start=1):
        row = parameters.loc[
            (parameters["outer_repeat"].astype(int) == record.repeat)
            & (parameters["outer_fold"].astype(int) == record.fold)
        ].iloc[0]
        train_counts = base.class_counts(bundle.y[record.train_idx])
        test_counts = base.class_counts(bundle.y[record.test_idx])
        expected_counts = {
            "train_n": train_counts["n"],
            "train_positive": train_counts["positive"],
            "train_negative": train_counts["negative"],
            "test_n": test_counts["n"],
            "test_positive": test_counts["positive"],
            "test_negative": test_counts["negative"],
        }
        if any(
            exact_integer(row[column], f"{spec.key} {column}") != int(value)
            for column, value in expected_counts.items()
        ):
            raise ValueError(
                f"Main parameter class counts differ from reconstructed fold for "
                f"{spec.key} repeat={record.repeat}, fold={record.fold}"
            )
        if (
            str(row["dataset"]) != spec.key
            or str(row["scenario"]) != "primary"
            or exact_integer(row["outer_iter"], f"{spec.key} outer_iter") != outer_iter
            or exact_integer(
                row["outer_splits_used"], f"{spec.key} outer_splits_used"
            )
            != outer_plan.n_splits
            or exact_integer(row["outer_repeats"], f"{spec.key} outer_repeats")
            != outer_plan.n_repeats
            or float(row["en_C"]) not in {float(value) for value in base.EN_C_GRID}
            or float(row["en_l1_ratio"])
            not in {float(value) for value in base.EN_L1_RATIO_GRID}
            or float(row["l2_C"]) not in {float(value) for value in base.L2_C_GRID}
            or not 1
            <= exact_integer(row["k_selected"], f"{spec.key} k_selected")
            <= int(base.PRIMARY_MAX_K)
            or exact_integer(
                row["n_model_features"], f"{spec.key} n_model_features"
            )
            > exact_integer(row["k_selected"], f"{spec.key} k_selected")
        ):
            raise ValueError(
                f"Main parameter metadata/hyperparameters are invalid for {spec.key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )

    if outer_features.duplicated(["outer_repeat", "outer_fold", "feature"]).any():
        raise ValueError(f"Duplicate main outer-feature rows for {spec.key}")
    if not (outer_features["dataset"].astype(str) == spec.key).all() or not (
        outer_features["scenario"].astype(str) == "primary"
    ).all():
        raise ValueError(f"Main outer-feature dataset/scenario mismatch for {spec.key}")
    feature_universe = set(bundle.feature_names)
    observed_feature_fold_keys = set(
        zip(
            outer_features["outer_repeat"].astype(int),
            outer_features["outer_fold"].astype(int),
        )
    )
    if (
        observed_feature_fold_keys != expected_fold_keys
        or len(outer_features) != len(outer_plan.records) * len(feature_universe)
    ):
        raise ValueError(f"Main outer-feature fold/row universe mismatch for {spec.key}")
    parsed_flags: Dict[str, pd.Series] = {}
    for flag_column in ("eligible", "selected_topk", "model_used"):
        normalized = outer_features[flag_column].astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError(f"Invalid boolean values in main {flag_column} for {spec.key}")
        parsed_flags[flag_column] = normalized.eq("true")
    for record in outer_plan.records:
        fold_features = outer_features.loc[
            (outer_features["outer_repeat"].astype(int) == record.repeat)
            & (outer_features["outer_fold"].astype(int) == record.fold)
        ].copy()
        if set(fold_features["feature"].astype(str)) != feature_universe or len(
            fold_features
        ) != len(feature_universe):
            raise ValueError(
                f"Main outer-feature universe mismatch for {spec.key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )
        ranked = fold_features.loc[fold_features["rank"].notna()].sort_values("rank")
        ranks = ranked["rank"].to_numpy(dtype=float)
        if not np.array_equal(ranks, np.arange(1, len(ranked) + 1, dtype=float)):
            raise ValueError(
                f"Main feature ranks are not unique contiguous integers for {spec.key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )
        fold_index = fold_features.index
        eligible = parsed_flags["eligible"].loc[fold_index]
        selected = parsed_flags["selected_topk"].loc[fold_index]
        model_used = parsed_flags["model_used"].loc[fold_index]
        parameter_row = parameters.loc[
            (parameters["outer_repeat"].astype(int) == record.repeat)
            & (parameters["outer_fold"].astype(int) == record.fold)
        ].iloc[0]
        k_selected = exact_integer(
            parameter_row["k_selected"], f"{spec.key} k_selected"
        )
        expected_selected = fold_features["rank"].notna() & (
            fold_features["rank"].astype(float) <= k_selected
        )
        if (
            not (eligible.loc[ranked.index]).all()
            or not selected.equals(expected_selected)
            or not model_used.equals(selected)
        ):
            raise ValueError(
                f"Main selected/model-used features do not match k for {spec.key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )

    recomputed_oof = averaged_oof(
        predictions,
        bundle.ids.astype(str).tolist(),
        expected_repeats=base.OUTER_REPEATS,
    )
    if not np.array_equal(recomputed_oof["y_true"].to_numpy(dtype=int), bundle.y):
        raise ValueError(f"Main outer-prediction labels are misaligned for {spec.key}")
    expected_ids = bundle.ids.astype(str).tolist()
    if (
        oof[base.ID_COL].astype(str).duplicated().any()
        or len(oof) != len(expected_ids)
        or set(oof[base.ID_COL].astype(str)) != set(expected_ids)
    ):
        raise ValueError(f"Saved main OOF IDs do not exactly match input IDs for {spec.key}")
    saved = oof.set_index(base.ID_COL).loc[expected_ids].reset_index()
    saved_labels = pd.to_numeric(saved["y_true"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(saved_labels).all()
        or not np.isin(saved_labels, [0.0, 1.0]).all()
        or not np.array_equal(saved_labels.astype(int), bundle.y)
    ):
        raise ValueError(f"Saved main OOF labels are misaligned for {spec.key}")
    if not np.allclose(saved["p_oof"], recomputed_oof["p_oof"], atol=1e-12, rtol=1e-10):
        raise ValueError(f"Saved main OOF probabilities do not reproduce for {spec.key}")
    if not np.allclose(
        saved["p_oof_sd"],
        recomputed_oof["p_oof_sd"],
        atol=1e-12,
        rtol=1e-10,
        equal_nan=True,
    ) or not np.array_equal(
        saved["oof_count"].to_numpy(dtype=int),
        recomputed_oof["oof_count"].to_numpy(dtype=int),
    ):
        raise ValueError(f"Saved main OOF dispersion/counts do not reproduce for {spec.key}")
    recomputed_metrics = metric_values(bundle.y, recomputed_oof["p_oof"].to_numpy(float))
    if len(metrics) != 1:
        raise ValueError(f"Expected one metrics row for {spec.key}")
    metric_row = metrics.iloc[0]
    for saved_name, computed_name in (
        ("roc_auc_oof", "roc_auc"),
        ("average_precision_oof", "average_precision"),
        ("brier_oof", "brier"),
    ):
        _assert_close(
            float(metric_row[saved_name]),
            recomputed_metrics[computed_name],
            f"{spec.key} {saved_name}",
        )

    return MainArtifacts(
        spec=spec,
        bundle=bundle,
        run_dir=directory,
        receipt=receipt,
        settings=settings,
        predictions_long=predictions,
        oof=saved,
        repeat_metrics=repeat_metrics,
        metrics=metrics,
        parameters=parameters,
        outer_features=outer_features,
        fold_counts=fold_counts,
        feature_manifest=feature_manifest,
        outer_plan=outer_plan,
    )


def _filter_ranking_to_training_eligibility(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    feature_rules: Mapping[str, base.FeatureRule],
    ranking: Sequence[str],
    imputation: str = "half_minimum",
) -> Tuple[List[str], base.MetaboPreprocessor]:
    eligibility = base._make_preprocessor(
        feature_names, feature_rules, imputation
    ).fit(X_train[list(feature_names)])
    eligible = set(eligibility.kept_features_)
    filtered = [feature for feature in ranking if feature in eligible]
    if not filtered:
        raise ValueError("No ranked feature is eligible in the training data")
    return filtered, eligibility


def tune_train_predict(
    bundle: base.DataBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    context: str,
    outer_iter: int,
    max_k: int = base.PRIMARY_MAX_K,
    imputation: str = "half_minimum",
) -> TunedFit:
    X_train = bundle.X.iloc[np.asarray(train_idx, dtype=int)]
    X_test = bundle.X.iloc[np.asarray(test_idx, dtype=int)]
    y_train = bundle.y[np.asarray(train_idx, dtype=int)]
    log_lines: List[str] = []
    warning_rows: List[Dict[str, Any]] = []
    inner_plan = base.make_cv_plan(
        y_train,
        target_splits=base.INNER_SPLITS_TARGET,
        n_repeats=base.INNER_REPEATS,
        context=f"{context} inner",
        log_lines=log_lines,
        random_state=base.RANDOM_STATE,
    )
    en_C, en_l1, _ = base.tune_elasticnet(
        X_train,
        y_train,
        bundle.feature_names,
        inner_plan,
        dataset_name=context,
        outer_iter=outer_iter,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=imputation,
        warning_rows=warning_rows,
    )
    ranking, ranking_table = base.stability_select_and_rank(
        X_train,
        y_train,
        bundle.feature_names,
        inner_plan,
        en_C=en_C,
        en_l1=en_l1,
        dataset_name=context,
        outer_iter=outer_iter,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=imputation,
        warning_rows=warning_rows,
    )
    ranking, _ = _filter_ranking_to_training_eligibility(
        X_train,
        bundle.feature_names,
        bundle.feature_rules,
        ranking,
        imputation,
    )
    ranking_table = ranking_table.loc[ranking_table["feature"].isin(ranking)].copy()
    ranking_table = ranking_table.sort_values("rank").reset_index(drop=True)
    ranking_table["rank"] = np.arange(1, len(ranking_table) + 1, dtype=int)
    k_selected, l2_C, k_curve = base.select_k_by_1se(
        X_train,
        y_train,
        ranking,
        inner_plan,
        dataset_name=context,
        outer_iter=outer_iter,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=imputation,
        max_k=max_k,
        warning_rows=warning_rows,
    )
    selected = ranking[:k_selected]
    preprocessor = base._make_preprocessor(
        selected, bundle.feature_rules, imputation
    ).fit(X_train[selected])
    train_matrix = preprocessor.transform(X_train[selected])
    test_matrix = preprocessor.transform(X_test[selected])
    model = base.fit_logreg_with_refit(
        train_matrix,
        y_train,
        kind="l2",
        C=l2_C,
        l1_ratio=None,
        dataset_name=context,
        stage="L2-train-fit",
        log_lines=log_lines,
        warning_rows=warning_rows,
    )
    probability = model.predict_proba(test_matrix)[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite test probabilities: {context}")
    coefficients = {
        feature: float(value)
        for feature, value in zip(preprocessor.kept_features_, model.coef_.ravel())
    }
    return TunedFit(
        probability=probability,
        en_C=float(en_C),
        en_l1_ratio=float(en_l1),
        ranking=ranking,
        ranking_table=ranking_table,
        k_selected=int(k_selected),
        l2_C=float(l2_C),
        selected_features=selected,
        model_features=preprocessor.kept_features_,
        coefficients=coefficients,
        inner_plan=inner_plan,
        k_curve=k_curve,
        preprocessor_audit=preprocessor.audit_table(),
        log_lines=log_lines,
        warning_rows=warning_rows,
    )


def tune_global_spec(
    bundle: base.DataBundle,
    selection_plan: base.CVPlan,
    context: str,
    max_k: int = base.PRIMARY_MAX_K,
    imputation: str = "half_minimum",
) -> GlobalSpec:
    log_lines: List[str] = []
    warning_rows: List[Dict[str, Any]] = []
    en_C, en_l1, _ = base.tune_elasticnet(
        bundle.X,
        bundle.y,
        bundle.feature_names,
        selection_plan,
        dataset_name=context,
        outer_iter=0,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=imputation,
        warning_rows=warning_rows,
    )
    ranking, ranking_table = base.stability_select_and_rank(
        bundle.X,
        bundle.y,
        bundle.feature_names,
        selection_plan,
        en_C=en_C,
        en_l1=en_l1,
        dataset_name=context,
        outer_iter=0,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=imputation,
        warning_rows=warning_rows,
    )
    ranking, _ = _filter_ranking_to_training_eligibility(
        bundle.X,
        bundle.feature_names,
        bundle.feature_rules,
        ranking,
        imputation,
    )
    ranking_table = ranking_table.loc[ranking_table["feature"].isin(ranking)].copy()
    ranking_table = ranking_table.sort_values("rank").reset_index(drop=True)
    ranking_table["rank"] = np.arange(1, len(ranking_table) + 1, dtype=int)
    k_selected, l2_C, k_curve = base.select_k_by_1se(
        bundle.X,
        bundle.y,
        ranking,
        selection_plan,
        dataset_name=context,
        outer_iter=0,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=imputation,
        max_k=max_k,
        warning_rows=warning_rows,
    )
    return GlobalSpec(
        en_C=float(en_C),
        en_l1_ratio=float(en_l1),
        ranking=ranking,
        ranking_table=ranking_table,
        k_selected=int(k_selected),
        l2_C=float(l2_C),
        selected_features=ranking[: int(k_selected)],
        k_curve=k_curve,
        selection_plan=selection_plan,
        log_lines=log_lines,
        warning_rows=warning_rows,
    )


def predict_fixed_spec(
    bundle: base.DataBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    selected_features: Sequence[str],
    l2_C: float,
    context: str,
    imputation: str = "half_minimum",
    warning_rows: Optional[List[Dict[str, Any]]] = None,
    log_lines: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, float], pd.DataFrame]:
    logs = log_lines if log_lines is not None else []
    warnings_out = warning_rows if warning_rows is not None else []
    selected = list(selected_features)
    X_train = bundle.X.iloc[np.asarray(train_idx, dtype=int)]
    X_test = bundle.X.iloc[np.asarray(test_idx, dtype=int)]
    y_train = bundle.y[np.asarray(train_idx, dtype=int)]
    preprocessor = base._make_preprocessor(
        selected, bundle.feature_rules, imputation
    ).fit(X_train[selected])
    train_matrix = preprocessor.transform(X_train[selected])
    test_matrix = preprocessor.transform(X_test[selected])
    model = base.fit_logreg_with_refit(
        train_matrix,
        y_train,
        kind="l2",
        C=float(l2_C),
        l1_ratio=None,
        dataset_name=context,
        stage="L2-fixed-spec-fit",
        log_lines=logs,
        warning_rows=warnings_out,
    )
    probability = model.predict_proba(test_matrix)[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite fixed-spec probabilities: {context}")
    coefficients = {
        feature: float(value)
        for feature, value in zip(preprocessor.kept_features_, model.coef_.ravel())
    }
    return probability, preprocessor.kept_features_, coefficients, preprocessor.audit_table()


def paired_stratified_auc_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    labels = np.asarray(y, dtype=int)
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    if not (len(labels) == len(a) == len(b)):
        raise ValueError("Paired bootstrap inputs have unequal length")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Both classes are required for paired AUROC bootstrap")
    rng = np.random.default_rng(int(seed))
    auc_a = np.empty(int(n_boot), dtype=float)
    auc_b = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        sample = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        auc_a[index] = roc_auc_score(labels[sample], a[sample])
        auc_b[index] = roc_auc_score(labels[sample], b[sample])
    delta = auc_b - auc_a
    return {
        "auc_a_ci95_lo": float(np.quantile(auc_a, 0.025)),
        "auc_a_ci95_hi": float(np.quantile(auc_a, 0.975)),
        "auc_b_ci95_lo": float(np.quantile(auc_b, 0.025)),
        "auc_b_ci95_hi": float(np.quantile(auc_b, 0.975)),
        "delta_ci95_lo": float(np.quantile(delta, 0.025)),
        "delta_ci95_hi": float(np.quantile(delta, 0.975)),
    }


def exact_signed_rank_sign_permutation(differences: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Exact signed-rank differences must be finite")
    comparison_tolerance = np.finfo(float).eps * 100
    nonzero = values[values != 0.0]
    zero_n = int(len(values) - len(nonzero))
    if len(nonzero) == 0:
        return {
            "p_value": 1.0,
            "n_total": int(len(values)),
            "n_nonzero": 0,
            "n_zero": zero_n,
            "t_plus": 0.0,
            "rank_sum": 0.0,
            "enumerations": 1,
        }
    ranks = rankdata(np.abs(nonzero), method="average")
    observed = float(ranks[nonzero > 0].sum())
    rank_sum = float(ranks.sum())
    observed_distance = abs(observed - rank_sum / 2.0)
    extreme = 0
    total = 0
    for signs in product((0, 1), repeat=len(nonzero)):
        t_plus = float(ranks[np.asarray(signs, dtype=bool)].sum())
        if abs(t_plus - rank_sum / 2.0) + comparison_tolerance >= observed_distance:
            extreme += 1
        total += 1
    return {
        "p_value": float(extreme / total),
        "n_total": int(len(values)),
        "n_nonzero": int(len(nonzero)),
        "n_zero": zero_n,
        "t_plus": observed,
        "rank_sum": rank_sum,
        "enumerations": int(total),
    }


def exact_sign_test(differences: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    positive = int(np.sum(values > 0.0))
    negative = int(np.sum(values < 0.0))
    zero = int(len(values) - positive - negative)
    n = positive + negative
    p_value = 1.0 if n == 0 else float(binomtest(positive, n, 0.5, alternative="two-sided").pvalue)
    return {
        "p_value": p_value,
        "n_total": int(len(values)),
        "n_nonzero": int(n),
        "n_positive": positive,
        "n_negative": negative,
        "n_zero": zero,
    }


def provenance_payload(script_path: Path) -> Dict[str, Any]:
    common_path = Path(__file__).resolve()
    return {
        "script_path": str(Path(script_path).resolve()),
        "script_sha256": sha256_file(Path(script_path).resolve()),
        "common_module_path": str(common_path),
        "common_module_sha256": sha256_file(common_path),
        "base_script_path": str(Path(base.__file__).resolve()),
        "base_script_sha256": sha256_file(Path(base.__file__).resolve()),
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "openpyxl": openpyxl.__version__,
        },
    }


def finalize_run_receipt(
    run_root: Path,
    script_path: Path,
    started_utc: str,
    started_time: float,
    manifest: pd.DataFrame,
    settings: Mapping[str, Any],
) -> None:
    completed = int((manifest["status"] == "completed").sum()) if not manifest.empty else 0
    failed = int((manifest["status"] == "failed").sum()) if not manifest.empty else 0
    root = Path(run_root)
    write_json(root / "settings_used.json", dict(settings))
    top_level_files = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "run_receipt.json"
    )
    output_hashes = output_sha256_manifest(root, top_level_files)
    write_json(
        root / "run_receipt.json",
        {
            "status": (
                "failed" if completed == 0 else "completed_with_failures" if failed else "completed"
            ),
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "elapsed_seconds": float(time.time() - started_time),
            "n_completed": completed,
            "n_failed": failed,
            "output_root": str(Path(run_root).resolve()),
            "output_sha256": output_hashes,
            **provenance_payload(script_path),
        },
    )
