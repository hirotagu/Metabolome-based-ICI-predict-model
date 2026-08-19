#!/usr/bin/env python3
"""Scenario-matched nested versus intentionally non-nested sensitivity analysis.

This script consumes completed nested-CV sensitivity runs and computes the
corresponding fully non-nested estimate under exactly the same input sheet,
imputation method, hyperparameter grids, maximum k, and repeated outer-CV
plan.  Existing nested predictions are validated and reused; they are not
recomputed.

The script is intentionally separate from Compare_method.py so the completed
primary provenance chain remains unchanged.  It performs no new hypothesis
tests and produces no figures.  Plotting scripts should consume the compact
top-level CSV/XLSX outputs generated here.

IMPORTANT: Complete every requested nested sensitivity scenario with
``ICI_predict.py`` before running this script. The single producer file must
remain byte-identical to the version recorded in each ``run_receipt.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import sklearn

import ICI_predict as base
import ICI_analysis_common as common


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PRIMARY_COMPARE_RUN_ROOT = SCRIPT_DIR / "results_revision" / "Compare_method_v1"

SD_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_SD"
KNN_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_KNN"
KMAX30_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_Kmax30"
OUTER3_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_outer3"
REGGRID_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_RegGrid"

# All nested scenarios are produced by the same public entry point. Its SHA256
# must match the value recorded by every scenario-specific run receipt.
SENSITIVITY_PRODUCER_SCRIPT = SCRIPT_DIR / "ICI_predict.py"

OUTPUT_ROOT = SCRIPT_DIR / "results_revision"
RUN_TAG = "Compare_sensitivity_v1"

# The public Supplementary Data do not contain the two SD recodings, so the
# default public run excludes them. Add "sd_as_nr" and/or "sd_as_r" only when
# those non-public analysis sheets are available. "ALL" still means all six
# scenarios and remains available for an internal exact-reproduction run.
SCENARIOS_TO_RUN: Union[str, List[str]] = ["knn", "kmax30", "outer3", "reggrid"]

# "ALL" uses each scenario's prespecified target set.  A dataset list can be
# used for a technical smoke test; it does not expand any scenario's scope.
DATASETS_TO_RUN: Union[str, List[str]] = "ALL"

STRICT_PRODUCER_SCRIPT_HASH = True
STRICT_PRIMARY_REFERENCE_HASH = True
CONTINUE_ON_UNIT_ERROR = True
RESUME_COMPLETED_UNITS = True
ALLOW_OVERWRITE = False
WRITE_EXCEL = True

# No new p-values or bootstrap confidence intervals are generated here.
PERFORM_HYPOTHESIS_TESTS = False
COMPUTE_BOOTSTRAP_INTERVALS = False


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

FORMAL_PRE_KEYS: Tuple[str, ...] = (
    "dataset1_pre",
    "dataset2_pre",
    "dataset3_pre",
    "dataset4_pre",
    "dataset5_pre",
    "dataset7_pre",
)

SD_KEYS: Tuple[str, ...] = ("dataset1_pre", "dataset1_post1")
KNN_KEYS: Tuple[str, ...] = ("dataset2_pre",)

# Fourteen Primary matrices had at least one outer fit selecting k=10.  Kmax30
# was nevertheless run on all 17; the remaining three are labelled surplus.
KMAX_PRIMARY_TRIGGER_KEYS: Tuple[str, ...] = (
    "dataset1_pre",
    "dataset1_post1",
    "dataset2_pre",
    "dataset2_post1",
    "dataset2_post2",
    "dataset3_pre",
    "dataset3_post1",
    "dataset3_post2",
    "dataset4_pre",
    "dataset4_post2",
    "dataset5_pre",
    "dataset5_post1",
    "dataset5_post2",
    "dataset7_pre",
)

REQUIRED_SENSITIVITY_FILES: Tuple[str, ...] = (
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
    "warnings.csv",
)

UNIT_OUTPUT_FILES: Tuple[str, ...] = (
    "outer_predictions_long.csv",
    "oof_predictions.csv",
    "paired_oof_predictions.csv",
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
    "source_artifact_hashes.csv",
    "warnings.csv",
    "run_log.txt",
)


@dataclass(frozen=True)
class ScenarioConfig:
    family: str
    name: str
    invocation_scenario: str
    run_root: Path
    producer_script: Path
    expected_keys: Tuple[str, ...]
    sheet_name: str
    imputation: str
    max_k: int
    outer_splits_target: int
    source_label: str
    same_analysis_population_as_primary: bool


@dataclass
class SensitivityArtifacts:
    config: ScenarioConfig
    engine: ModuleType
    spec: Any
    bundle: Any
    run_dir: Path
    root_manifest_path: Path
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
    source_warnings: pd.DataFrame
    outer_plan: Any
    source_hashes: pd.DataFrame


@dataclass
class ScenarioGlobalSpec:
    en_C: float
    en_l1_ratio: float
    ranking: List[str]
    ranking_table: pd.DataFrame
    k_selected: int
    l2_C: float
    selected_features: List[str]
    k_curve: pd.DataFrame
    log_lines: List[str]
    warning_rows: List[Dict[str, Any]]


def _all_configs() -> Dict[str, ScenarioConfig]:
    return {
        "sd_as_nr": ScenarioConfig(
            family="sd",
            name="sd_as_nr",
            invocation_scenario="sd",
            run_root=SD_RUN_ROOT,
            producer_script=SENSITIVITY_PRODUCER_SCRIPT,
            expected_keys=SD_KEYS,
            sheet_name="Sheet2",
            imputation="half_minimum",
            max_k=10,
            outer_splits_target=5,
            source_label="sd_sensitivity",
            same_analysis_population_as_primary=False,
        ),
        "sd_as_r": ScenarioConfig(
            family="sd",
            name="sd_as_r",
            invocation_scenario="sd",
            run_root=SD_RUN_ROOT,
            producer_script=SENSITIVITY_PRODUCER_SCRIPT,
            expected_keys=SD_KEYS,
            sheet_name="Sheet3",
            imputation="half_minimum",
            max_k=10,
            outer_splits_target=5,
            source_label="sd_sensitivity",
            same_analysis_population_as_primary=False,
        ),
        "knn": ScenarioConfig(
            family="knn",
            name="knn",
            invocation_scenario="knn",
            run_root=KNN_RUN_ROOT,
            producer_script=SENSITIVITY_PRODUCER_SCRIPT,
            expected_keys=KNN_KEYS,
            sheet_name="Sheet1",
            imputation="knn",
            max_k=10,
            outer_splits_target=5,
            source_label="imputation_sensitivity",
            same_analysis_population_as_primary=True,
        ),
        "kmax30": ScenarioConfig(
            family="kmax",
            name="kmax30",
            invocation_scenario="kmax30",
            run_root=KMAX30_RUN_ROOT,
            producer_script=SENSITIVITY_PRODUCER_SCRIPT,
            expected_keys=CANONICAL_KEYS,
            sheet_name="Sheet1",
            imputation="half_minimum",
            max_k=30,
            outer_splits_target=5,
            source_label="kmax_sensitivity",
            same_analysis_population_as_primary=True,
        ),
        "outer3": ScenarioConfig(
            family="outer_folds",
            name="outer3",
            invocation_scenario="outer3",
            run_root=OUTER3_RUN_ROOT,
            producer_script=SENSITIVITY_PRODUCER_SCRIPT,
            expected_keys=CANONICAL_KEYS,
            sheet_name="Sheet1",
            imputation="half_minimum",
            max_k=10,
            outer_splits_target=3,
            source_label="fold_sensitivity",
            same_analysis_population_as_primary=True,
        ),
        "reggrid": ScenarioConfig(
            family="regularization_grid",
            name="reggrid",
            invocation_scenario="reggrid",
            run_root=REGGRID_RUN_ROOT,
            producer_script=SENSITIVITY_PRODUCER_SCRIPT,
            expected_keys=FORMAL_PRE_KEYS,
            sheet_name="Sheet1",
            imputation="half_minimum",
            max_k=10,
            outer_splits_target=5,
            source_label="regularization_grid_sensitivity",
            same_analysis_population_as_primary=True,
        ),
    }


def _run_root() -> Path:
    return OUTPUT_ROOT / RUN_TAG if RUN_TAG else OUTPUT_ROOT


def _selected_configs() -> List[ScenarioConfig]:
    configs = _all_configs()
    if isinstance(SCENARIOS_TO_RUN, str):
        if SCENARIOS_TO_RUN.upper() == "ALL":
            names = list(configs)
        else:
            names = [SCENARIOS_TO_RUN.strip().lower()]
    else:
        names = [str(value).strip().lower() for value in SCENARIOS_TO_RUN]
    if len(names) != len(set(names)):
        raise ValueError("SCENARIOS_TO_RUN contains duplicate scenario names")
    missing = sorted(set(names).difference(configs))
    if missing:
        raise ValueError(f"Unknown sensitivity scenarios: {missing}")
    return [configs[name] for name in names]


def _selected_keys(config: ScenarioConfig) -> List[str]:
    allowed = list(config.expected_keys)
    if isinstance(DATASETS_TO_RUN, str):
        if DATASETS_TO_RUN.upper() == "ALL":
            return allowed
        requested = [DATASETS_TO_RUN.strip().lower()]
    else:
        requested = [str(value).strip().lower() for value in DATASETS_TO_RUN]
    if len(requested) != len(set(requested)):
        raise ValueError("DATASETS_TO_RUN contains duplicate dataset keys")
    unknown = sorted(set(requested).difference(CANONICAL_KEYS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}")
    return [key for key in allowed if key in set(requested)]


def _same_sequence(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return json.dumps(list(left), default=str) == json.dumps(list(right), default=str)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{label} mismatch: actual={actual}, expected={expected}")


def _load_engine(path: Path) -> ModuleType:
    producer = Path(path).expanduser().resolve()
    if not producer.is_file():
        raise FileNotFoundError(
            f"Sensitivity producer script was not found: {producer}. "
            "Place the exact ICI_predict.py used for the nested runs beside "
            "Compare_sensitivity.py."
        )
    module = base
    if producer != Path(module.__file__).resolve():
        raise ImportError(
            "The configured producer must be the imported ICI_predict.py: "
            f"configured={producer}, imported={Path(module.__file__).resolve()}"
        )
    required = (
        "discover_datasets",
        "load_dataset",
        "make_cv_plan",
        "tune_elasticnet",
        "stability_select_and_rank",
        "select_k_by_1se",
        "_make_preprocessor",
        "fit_logreg_with_refit",
        "class_counts",
        "EN_C_GRID",
        "EN_L1_RATIO_GRID",
        "L2_C_GRID",
    )
    absent = [name for name in required if not hasattr(module, name)]
    if absent:
        raise ImportError(f"ICI_predict.py lacks required analysis objects: {absent}")
    return module


def _activate_engine(config: ScenarioConfig) -> Tuple[ModuleType, Dict[str, Any]]:
    """Reset the unified producer to the exact profile for one scenario."""
    engine = _load_engine(config.producer_script)
    if not hasattr(engine, "configure_scenario") or not hasattr(
        engine, "get_public_scenario_profile"
    ):
        raise ImportError(
            "ICI_predict.py must expose configure_scenario() and "
            "get_public_scenario_profile()."
        )
    profile = engine.configure_scenario(config.invocation_scenario)
    if engine.ACTIVE_SCENARIO != config.invocation_scenario:
        raise RuntimeError(
            f"Failed to activate ICI_predict scenario {config.invocation_scenario!r}"
        )
    if str(profile.get("run_tag")) != Path(config.run_root).name:
        raise ValueError(
            f"Run-root/profile mismatch for {config.name}: "
            f"{Path(config.run_root).name!r} vs {profile.get('run_tag')!r}"
        )
    profile_targets = profile.get("datasets_to_run")
    if isinstance(profile_targets, str) and profile_targets.upper() == "ALL":
        expected_profile_keys = set(CANONICAL_KEYS)
    else:
        expected_profile_keys = {str(value) for value in profile_targets or []}
    if expected_profile_keys != set(config.expected_keys):
        raise ValueError(
            f"Dataset/profile mismatch for {config.name}: expected "
            f"{list(config.expected_keys)}, profile has {sorted(expected_profile_keys)}"
        )
    return engine, profile


def _validate_primary_reference() -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(PRIMARY_COMPARE_RUN_ROOT).expanduser().resolve()
    required = (
        "run_receipt.json",
        "comparison_all_datasets.csv",
        "pre_primary_inference.csv",
        "run_manifest.csv",
        "settings_used.json",
    )
    for filename in required:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Missing Primary Compare artifact: {root / filename}")
    receipt = common.read_json(root / "run_receipt.json")
    if receipt.get("status") != "completed" or int(receipt.get("n_failed", -1)) != 0:
        raise ValueError("Primary Compare receipt is not a completed zero-failure run")
    if STRICT_PRIMARY_REFERENCE_HASH:
        common.validate_output_sha256_manifest(
            root,
            receipt,
            (
                "comparison_all_datasets.csv",
                "pre_primary_inference.csv",
                "run_manifest.csv",
                "settings_used.json",
            ),
        )
        producer = SCRIPT_DIR / "Compare_method.py"
        if not producer.is_file() or common.sha256_file(producer) != receipt.get("script_sha256"):
            raise ValueError("Compare_method.py SHA256 differs from the Primary Compare run")
        if common.sha256_file(Path(common.__file__).resolve()) != receipt.get(
            "common_module_sha256"
        ):
            raise ValueError("ICI_analysis_common.py SHA256 differs from Primary Compare")
        if common.sha256_file(Path(base.__file__).resolve()) != receipt.get(
            "base_script_sha256"
        ):
            raise ValueError("ICI_predict.py SHA256 differs from Primary Compare")
    comparison = common.read_csv(root / "comparison_all_datasets.csv")
    required_columns = (
        "dataset",
        "n_samples",
        "n_positive",
        "n_negative",
        "nested_roc_auc",
        "nested_average_precision",
        "nested_brier",
        "nonnested_roc_auc",
        "nonnested_average_precision",
        "nonnested_brier",
        "delta_auc_nonnested_minus_nested",
        "delta_ap_nonnested_minus_nested",
        "delta_brier_nonnested_minus_nested",
    )
    _require_columns(comparison, required_columns, "Primary comparison table")
    if comparison["dataset"].duplicated().any() or set(comparison["dataset"]) != set(
        CANONICAL_KEYS
    ):
        raise ValueError("Primary comparison table is not the canonical unique 17-matrix set")
    comparison = comparison.set_index("dataset").loc[list(CANONICAL_KEYS)].reset_index()
    hash_rows = []
    for filename in required:
        path = root / filename
        hash_rows.append(
            {
                "source_kind": "primary_compare",
                "scenario": "primary",
                "dataset": "ALL",
                "artifact": filename,
                "path": str(path),
                "sha256": common.sha256_file(path),
            }
        )
    return comparison, pd.DataFrame(hash_rows)


def _resolve_sensitivity_dir(
    config: ScenarioConfig,
    dataset_key: str,
) -> Tuple[Path, Path]:
    root = Path(config.run_root).expanduser().resolve()
    manifest_path = root / "run_manifest.csv"
    if not manifest_path.is_file():
        if config.name == "reggrid":
            raise FileNotFoundError(
                f"RegGrid run is not available: {manifest_path}. Complete All_data_RegGrid first."
            )
        raise FileNotFoundError(f"Sensitivity run manifest was not found: {manifest_path}")
    manifest = common.read_csv(manifest_path)
    _require_columns(
        manifest,
        ("dataset", "scenario", "invocation_scenario", "output_dir"),
        str(manifest_path),
    )
    rows = manifest.loc[
        (manifest["dataset"].astype(str) == dataset_key)
        & (manifest["scenario"].astype(str) == config.name)
    ]
    if "status" in manifest.columns:
        rows = rows.loc[rows["status"].astype(str) == "completed"]
    if len(rows) != 1:
        raise FileNotFoundError(
            f"Expected one completed {config.name}/{dataset_key} manifest row; found {len(rows)}"
        )
    if str(rows.iloc[0]["invocation_scenario"]) != config.invocation_scenario:
        raise ValueError(
            f"Invocation-scenario mismatch for {config.name}/{dataset_key}: "
            f"found {rows.iloc[0]['invocation_scenario']!r}, "
            f"expected {config.invocation_scenario!r}"
        )
    expected = root / "supplement" / "sensitivity" / config.name / dataset_key
    recorded = Path(str(rows.iloc[0]["output_dir"])).expanduser()
    candidates = (expected, recorded)
    run_dir = next((path.resolve() for path in candidates if path.is_dir()), None)
    if run_dir is None:
        raise FileNotFoundError(
            f"Sensitivity result directory does not exist for {config.name}/{dataset_key}. "
            f"Tried: {expected}, {recorded}"
        )
    return run_dir, manifest_path


def _source_hash_table(
    config: ScenarioConfig,
    dataset_key: str,
    run_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    paths: List[Tuple[str, Path]] = [("root_manifest", manifest_path)]
    paths.append(("producer_script", Path(config.producer_script).resolve()))
    paths.extend((filename, run_dir / filename) for filename in REQUIRED_SENSITIVITY_FILES)
    rows = []
    for artifact, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot hash missing sensitivity source: {path}")
        rows.append(
            {
                "source_kind": "nested_sensitivity",
                "scenario": config.name,
                "dataset": dataset_key,
                "artifact": artifact,
                "path": str(path.resolve()),
                "sha256": common.sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def _inventory_fingerprint(table: pd.DataFrame) -> str:
    required = ("source_kind", "scenario", "dataset", "artifact", "path", "sha256")
    _require_columns(table, required, "Source inventory")
    records = (
        table.loc[:, list(required)]
        .astype(str)
        .sort_values(list(required), kind="stable")
        .to_dict(orient="records")
    )
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preflight_source_inventory(
    configs: Sequence[ScenarioConfig],
    units: Sequence[Tuple[ScenarioConfig, str]],
) -> pd.DataFrame:
    """Fail before output creation if any planned source is absent or ambiguous."""
    frames: List[pd.DataFrame] = []
    for config in configs:
        producer = Path(config.producer_script).expanduser().resolve()
        if not producer.is_file():
            if config.name == "reggrid":
                raise FileNotFoundError(
                    f"RegGrid producer is not available: {producer}. "
                    "Run the reggrid scenario with ICI_predict.py before running "
                    "Compare_sensitivity."
                )
            raise FileNotFoundError(f"Sensitivity producer script was not found: {producer}")
        root = Path(config.run_root).expanduser().resolve()
        manifest_path = root / "run_manifest.csv"
        if not manifest_path.is_file():
            if config.name == "reggrid":
                raise FileNotFoundError(
                    f"RegGrid results are not available: {manifest_path}. "
                    "Complete the six Pre RegGrid runs before running Compare_sensitivity."
                )
            raise FileNotFoundError(f"Sensitivity manifest was not found: {manifest_path}")
        manifest = common.read_csv(manifest_path)
        _require_columns(
            manifest,
            ("dataset", "scenario", "invocation_scenario", "output_dir"),
            str(manifest_path),
        )
        scenario_rows = manifest.loc[manifest["scenario"].astype(str) == config.name].copy()
        if "status" in scenario_rows.columns:
            scenario_rows = scenario_rows.loc[
                scenario_rows["status"].astype(str) == "completed"
            ].copy()
        if scenario_rows["dataset"].duplicated().any() or set(
            scenario_rows["dataset"].astype(str)
        ) != set(config.expected_keys):
            raise ValueError(
                f"{config.name} manifest dataset set mismatch; expected "
                f"{list(config.expected_keys)}, found "
                f"{scenario_rows['dataset'].astype(str).tolist()}"
            )
        if not (
            scenario_rows["invocation_scenario"].astype(str)
            == config.invocation_scenario
        ).all():
            raise ValueError(
                f"{config.name} manifest invocation_scenario must be "
                f"{config.invocation_scenario!r}"
            )
    for config, dataset_key in units:
        run_dir, manifest_path = _resolve_sensitivity_dir(config, dataset_key)
        receipt_path = run_dir / "run_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Missing sensitivity receipt: {receipt_path}")
        receipt = common.read_json(receipt_path)
        if (
            receipt.get("status") != "completed"
            or receipt.get("scenario") != config.name
            or receipt.get("invocation_scenario") != config.invocation_scenario
            or receipt.get("dataset") != dataset_key
        ):
            raise ValueError(f"Incomplete or mismatched source receipt: {config.name}/{dataset_key}")
        settings = common.read_json(run_dir / "settings_used.json")
        scenario_settings = settings.get("scenario")
        if not isinstance(scenario_settings, Mapping) or scenario_settings.get(
            "name"
        ) != config.name:
            raise ValueError(
                f"Source settings scenario mismatch: {config.name}/{dataset_key}"
            )
        if settings.get("invocation_scenario") != config.invocation_scenario:
            raise ValueError(
                f"Source settings invocation mismatch: {config.name}/{dataset_key}"
            )
        expected_profile = base.get_public_scenario_profile(
            config.invocation_scenario
        )
        if receipt.get("invocation_scenario_metadata") != expected_profile:
            raise ValueError(
                f"Source receipt profile mismatch: {config.name}/{dataset_key}"
            )
        if settings.get("invocation_scenario_metadata") != expected_profile:
            raise ValueError(
                f"Source settings profile mismatch: {config.name}/{dataset_key}"
            )
        producer_sha = common.sha256_file(Path(config.producer_script).resolve())
        if STRICT_PRODUCER_SCRIPT_HASH and receipt.get("script_sha256") != producer_sha:
            raise ValueError(
                f"Producer-script SHA mismatch for {config.name}/{dataset_key}: "
                f"{config.producer_script}"
            )
        frames.append(_source_hash_table(config, dataset_key, run_dir, manifest_path))
    inventory = pd.concat(frames, ignore_index=True)
    duplicates = inventory.groupby("path", sort=False)["sha256"].nunique()
    if (duplicates > 1).any():
        bad = duplicates.loc[duplicates > 1].index.tolist()
        raise ValueError(f"A source path has inconsistent SHA values: {bad}")
    return inventory.drop_duplicates(["path", "sha256"]).reset_index(drop=True)


def _assert_source_inventory_unchanged(
    observed: pd.DataFrame,
    locked_inventory: pd.DataFrame,
) -> None:
    locked = dict(
        zip(locked_inventory["path"].astype(str), locked_inventory["sha256"].astype(str))
    )
    for row in observed.itertuples(index=False):
        expected = locked.get(str(row.path))
        if expected is None or expected != str(row.sha256):
            raise ValueError(f"Sensitivity source changed after preflight: {row.path}")


def _validate_sensitivity_artifacts(
    config: ScenarioConfig,
    dataset_key: str,
) -> SensitivityArtifacts:
    run_dir, manifest_path = _resolve_sensitivity_dir(config, dataset_key)
    for filename in REQUIRED_SENSITIVITY_FILES:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(f"Missing sensitivity artifact: {run_dir / filename}")

    engine, expected_invocation_profile = _activate_engine(config)
    producer_sha = common.sha256_file(Path(config.producer_script).resolve())
    receipt = common.read_json(run_dir / "run_receipt.json")
    settings = common.read_json(run_dir / "settings_used.json")
    if receipt.get("status") != "completed":
        raise ValueError(f"Sensitivity receipt is not completed: {config.name}/{dataset_key}")
    if (
        receipt.get("dataset") != dataset_key
        or receipt.get("scenario") != config.name
        or receipt.get("invocation_scenario") != config.invocation_scenario
    ):
        raise ValueError(f"Sensitivity receipt dataset/scenario mismatch: {config.name}/{dataset_key}")
    if receipt.get("invocation_scenario_metadata") != expected_invocation_profile:
        raise ValueError(
            f"Sensitivity receipt scenario profile mismatch: "
            f"{config.name}/{dataset_key}"
        )
    if STRICT_PRODUCER_SCRIPT_HASH and receipt.get("script_sha256") != producer_sha:
        raise ValueError(
            f"Producer-script SHA mismatch for {config.name}/{dataset_key}: "
            f"{config.producer_script}"
        )
    saved_versions = receipt.get("package_versions", {})
    if saved_versions.get("scikit_learn") != sklearn.__version__:
        raise ValueError(
            f"scikit-learn version mismatch for {config.name}/{dataset_key}: "
            f"saved={saved_versions.get('scikit_learn')}, current={sklearn.__version__}"
        )

    specs = engine.discover_datasets()
    if dataset_key not in specs:
        raise KeyError(f"Producer did not discover required input dataset: {dataset_key}")
    spec = specs[dataset_key]
    bundle = engine.load_dataset(spec, config.sheet_name)
    if receipt.get("input_sha256") != bundle.input_sha256:
        raise ValueError(f"Input SHA mismatch for {config.name}/{dataset_key}")
    if receipt.get("input_sheet") != config.sheet_name:
        raise ValueError(f"Input sheet mismatch for {config.name}/{dataset_key}")

    scenario = settings.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"Missing scenario settings for {config.name}/{dataset_key}")
    expected_scenario = {
        "name": config.name,
        "sheet_name": config.sheet_name,
        "imputation": config.imputation,
        "max_k": config.max_k,
        "outer_splits_target": config.outer_splits_target,
        "source": config.source_label,
    }
    for key, expected in expected_scenario.items():
        actual = scenario.get(key)
        if isinstance(expected, int):
            try:
                actual = common.exact_integer(actual, f"scenario.{key}")
            except ValueError as exc:
                raise ValueError(f"{config.name}/{dataset_key}: {exc}") from exc
        if actual != expected:
            raise ValueError(
                f"Scenario setting mismatch for {config.name}/{dataset_key}: "
                f"{key}={actual!r}, expected={expected!r}"
            )
    if settings.get("invocation_scenario") != config.invocation_scenario:
        raise ValueError(
            f"Invocation scenario mismatch for {config.name}/{dataset_key}: "
            f"found {settings.get('invocation_scenario')!r}, "
            f"expected {config.invocation_scenario!r}"
        )
    if settings.get("invocation_scenario_metadata") != expected_invocation_profile:
        raise ValueError(
            f"Invocation scenario metadata differs from ICI_predict.py for "
            f"{config.name}/{dataset_key}"
        )
    scalar_expectations = {
        "random_state": int(engine.RANDOM_STATE),
        "outer_repeats": int(engine.OUTER_REPEATS),
        "inner_splits_target": int(engine.INNER_SPLITS_TARGET),
        "inner_repeats": int(engine.INNER_REPEATS),
        "knn_neighbors": int(engine.KNN_NEIGHBORS),
    }
    for key, expected in scalar_expectations.items():
        if common.exact_integer(settings.get(key), key) != expected:
            raise ValueError(f"{key} mismatch for {config.name}/{dataset_key}")
    if settings.get("class_weight") != "balanced":
        raise ValueError(f"class_weight mismatch for {config.name}/{dataset_key}")
    _assert_close(
        float(settings.get("missing_rate_threshold", np.nan)),
        float(engine.MISSING_RATE_THRESHOLD),
        f"{config.name}/{dataset_key} missing_rate_threshold",
    )
    for setting_name, current in (
        ("en_C_grid", engine.EN_C_GRID),
        ("en_l1_ratio_grid", engine.EN_L1_RATIO_GRID),
        ("l2_C_grid", engine.L2_C_GRID),
    ):
        if not _same_sequence(settings.get(setting_name, []), current):
            raise ValueError(f"{setting_name} differs from producer for {config.name}/{dataset_key}")

    plan_logs: List[str] = []
    outer_plan = engine.make_cv_plan(
        bundle.y,
        target_splits=config.outer_splits_target,
        n_repeats=int(engine.OUTER_REPEATS),
        context=f"{dataset_key} {config.name} source-validation outer",
        log_lines=plan_logs,
        random_state=int(engine.RANDOM_STATE),
    )
    if common.exact_integer(settings.get("outer_splits_used"), "outer_splits_used") != int(
        outer_plan.n_splits
    ):
        raise ValueError(f"outer_splits_used mismatch for {config.name}/{dataset_key}")
    if int(receipt.get("outer_splits_used", -1)) != int(outer_plan.n_splits):
        raise ValueError(f"Receipt outer folds mismatch for {config.name}/{dataset_key}")

    predictions = common.read_csv(
        run_dir / "outer_predictions_long.csv", id_columns=[engine.ID_COL]
    )
    saved_oof = common.read_csv(run_dir / "oof_predictions.csv", id_columns=[engine.ID_COL])
    repeat_metrics = common.read_csv(run_dir / "repeat_metrics.csv")
    metrics = common.read_csv(run_dir / "metrics_summary.csv")
    parameters = common.read_csv(run_dir / "outer_parameters.csv")
    outer_features = common.read_csv(run_dir / "outer_features_long.csv")
    fold_counts = common.read_csv(run_dir / "fold_counts.csv")
    feature_manifest = common.read_csv(run_dir / "feature_manifest.csv")
    source_warnings = common.read_csv(run_dir / "warnings.csv")

    _require_columns(
        predictions,
        ("dataset", "scenario", engine.ID_COL, "repeat", "outer_fold", "y_true", "p"),
        f"{config.name}/{dataset_key} predictions",
    )
    if not (predictions["dataset"].astype(str) == dataset_key).all() or not (
        predictions["scenario"].astype(str) == config.name
    ).all():
        raise ValueError(f"Prediction dataset/scenario mismatch for {config.name}/{dataset_key}")
    if predictions.duplicated([engine.ID_COL, "repeat"]).any():
        raise ValueError(f"Duplicate ID/repeat predictions for {config.name}/{dataset_key}")
    expected_prediction_rows = len(bundle.y) * int(outer_plan.n_repeats)
    if len(predictions) != expected_prediction_rows:
        raise ValueError(
            f"Prediction row count mismatch for {config.name}/{dataset_key}: "
            f"{len(predictions)} vs {expected_prediction_rows}"
        )
    probabilities = pd.to_numeric(predictions["p"], errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.logical_and(
        probabilities >= 0, probabilities <= 1
    ).all():
        raise ValueError(f"Invalid source probabilities for {config.name}/{dataset_key}")

    ids = bundle.ids.astype(str).tolist()
    for record in outer_plan.records:
        subset = predictions.loc[
            (predictions["repeat"].astype(int) == int(record.repeat))
            & (predictions["outer_fold"].astype(int) == int(record.fold))
        ]
        expected_ids = {ids[index] for index in record.test_idx}
        if len(subset) != len(expected_ids) or set(subset[engine.ID_COL].astype(str)) != expected_ids:
            raise ValueError(
                f"Fold membership mismatch for {config.name}/{dataset_key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )
        expected_labels = {ids[index]: int(bundle.y[index]) for index in record.test_idx}
        observed_labels = dict(
            zip(subset[engine.ID_COL].astype(str), subset["y_true"].astype(int))
        )
        if observed_labels != expected_labels:
            raise ValueError(
                f"Fold label mismatch for {config.name}/{dataset_key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )

    _require_columns(
        parameters,
        (
            "dataset",
            "scenario",
            "outer_iter",
            "outer_repeat",
            "outer_fold",
            "outer_splits_used",
            "outer_repeats",
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
            "n_model_features",
        ),
        f"{config.name}/{dataset_key} parameters",
    )
    if parameters.duplicated(["outer_repeat", "outer_fold"]).any() or len(parameters) != len(
        outer_plan.records
    ):
        raise ValueError(f"Parameter rows do not match outer folds for {config.name}/{dataset_key}")
    expected_fold_keys = {(int(r.repeat), int(r.fold)) for r in outer_plan.records}
    actual_fold_keys = set(
        zip(parameters["outer_repeat"].astype(int), parameters["outer_fold"].astype(int))
    )
    if actual_fold_keys != expected_fold_keys:
        raise ValueError(f"Parameter fold keys mismatch for {config.name}/{dataset_key}")
    en_grid = {float(value) for value in engine.EN_C_GRID}
    l1_grid = {float(value) for value in engine.EN_L1_RATIO_GRID}
    l2_grid = {float(value) for value in engine.L2_C_GRID}
    for outer_iter, record in enumerate(outer_plan.records, start=1):
        row = parameters.loc[
            (parameters["outer_repeat"].astype(int) == int(record.repeat))
            & (parameters["outer_fold"].astype(int) == int(record.fold))
        ].iloc[0]
        train_counts = engine.class_counts(bundle.y[record.train_idx])
        test_counts = engine.class_counts(bundle.y[record.test_idx])
        expected_counts = {
            "train_n": train_counts["n"],
            "train_positive": train_counts["positive"],
            "train_negative": train_counts["negative"],
            "test_n": test_counts["n"],
            "test_positive": test_counts["positive"],
            "test_negative": test_counts["negative"],
        }
        for column, expected in expected_counts.items():
            if common.exact_integer(row[column], column) != int(expected):
                raise ValueError(
                    f"Fold class count mismatch for {config.name}/{dataset_key} "
                    f"repeat={record.repeat}, fold={record.fold}"
                )
        selected_k = common.exact_integer(row["k_selected"], "k_selected")
        if (
            str(row["dataset"]) != dataset_key
            or str(row["scenario"]) != config.name
            or common.exact_integer(row["outer_iter"], "outer_iter") != outer_iter
            or float(row["en_C"]) not in en_grid
            or float(row["en_l1_ratio"]) not in l1_grid
            or float(row["l2_C"]) not in l2_grid
            or not 1 <= selected_k <= config.max_k
            or common.exact_integer(row["n_model_features"], "n_model_features") > selected_k
        ):
            raise ValueError(
                f"Invalid saved hyperparameter metadata for {config.name}/{dataset_key} "
                f"repeat={record.repeat}, fold={record.fold}"
            )

    recomputed_oof = common.averaged_oof(
        predictions,
        ids,
        expected_repeats=int(outer_plan.n_repeats),
        id_col=engine.ID_COL,
    )
    expected_ids = bundle.ids.astype(str).tolist()
    if saved_oof[engine.ID_COL].astype(str).duplicated().any() or set(
        saved_oof[engine.ID_COL].astype(str)
    ) != set(expected_ids):
        raise ValueError(f"Saved OOF IDs mismatch for {config.name}/{dataset_key}")
    saved_oof = saved_oof.set_index(engine.ID_COL).loc[expected_ids].reset_index()
    if not np.array_equal(saved_oof["y_true"].to_numpy(int), bundle.y):
        raise ValueError(f"Saved OOF labels mismatch for {config.name}/{dataset_key}")
    if not np.allclose(
        saved_oof["p_oof"].to_numpy(float),
        recomputed_oof["p_oof"].to_numpy(float),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(f"Saved OOF probabilities do not reproduce for {config.name}/{dataset_key}")
    nested_metrics = common.metric_values(
        bundle.y, recomputed_oof["p_oof"].to_numpy(float)
    )
    if len(metrics) != 1:
        raise ValueError(f"Expected one nested metrics row for {config.name}/{dataset_key}")
    for saved_name, computed_name in (
        ("roc_auc_oof", "roc_auc"),
        ("average_precision_oof", "average_precision"),
        ("brier_oof", "brier"),
    ):
        _assert_close(
            float(metrics.iloc[0][saved_name]),
            nested_metrics[computed_name],
            f"{config.name}/{dataset_key} {saved_name}",
        )

    source_hashes = _source_hash_table(config, dataset_key, run_dir, manifest_path)
    return SensitivityArtifacts(
        config=config,
        engine=engine,
        spec=spec,
        bundle=bundle,
        run_dir=run_dir,
        root_manifest_path=manifest_path,
        receipt=receipt,
        settings=settings,
        predictions_long=predictions,
        oof=saved_oof,
        repeat_metrics=repeat_metrics,
        metrics=metrics,
        parameters=parameters,
        outer_features=outer_features,
        fold_counts=fold_counts,
        feature_manifest=feature_manifest,
        source_warnings=source_warnings,
        outer_plan=outer_plan,
        source_hashes=source_hashes,
    )


def _filter_ranking_to_eligibility(
    engine: ModuleType,
    bundle: Any,
    ranking: Sequence[str],
    imputation: str,
) -> List[str]:
    eligibility = engine._make_preprocessor(
        bundle.feature_names, bundle.feature_rules, imputation
    ).fit(bundle.X[list(bundle.feature_names)])
    eligible = set(eligibility.kept_features_)
    filtered = [feature for feature in ranking if feature in eligible]
    if not filtered:
        raise ValueError("No globally ranked feature is eligible in the full dataset")
    return filtered


def _tune_global_spec(artifact: SensitivityArtifacts) -> ScenarioGlobalSpec:
    engine = artifact.engine
    bundle = artifact.bundle
    config = artifact.config
    context = f"{artifact.spec.key} | {config.name} | fully_non_nested"
    log_lines: List[str] = []
    warning_rows: List[Dict[str, Any]] = []
    en_C, en_l1, _ = engine.tune_elasticnet(
        bundle.X,
        bundle.y,
        bundle.feature_names,
        artifact.outer_plan,
        dataset_name=context,
        outer_iter=0,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=config.imputation,
        warning_rows=warning_rows,
    )
    ranking, ranking_table = engine.stability_select_and_rank(
        bundle.X,
        bundle.y,
        bundle.feature_names,
        artifact.outer_plan,
        en_C=en_C,
        en_l1=en_l1,
        dataset_name=context,
        outer_iter=0,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=config.imputation,
        warning_rows=warning_rows,
    )
    ranking = _filter_ranking_to_eligibility(
        engine, bundle, ranking, config.imputation
    )
    ranking_table = ranking_table.loc[ranking_table["feature"].isin(ranking)].copy()
    ranking_table = ranking_table.sort_values("rank").reset_index(drop=True)
    ranking_table["rank"] = np.arange(1, len(ranking_table) + 1, dtype=int)
    k_selected, l2_C, k_curve = engine.select_k_by_1se(
        bundle.X,
        bundle.y,
        ranking,
        artifact.outer_plan,
        dataset_name=context,
        outer_iter=0,
        log_lines=log_lines,
        feature_rules=bundle.feature_rules,
        imputation=config.imputation,
        max_k=config.max_k,
        warning_rows=warning_rows,
    )
    return ScenarioGlobalSpec(
        en_C=float(en_C),
        en_l1_ratio=float(en_l1),
        ranking=ranking,
        ranking_table=ranking_table,
        k_selected=int(k_selected),
        l2_C=float(l2_C),
        selected_features=ranking[: int(k_selected)],
        k_curve=k_curve,
        log_lines=log_lines,
        warning_rows=warning_rows,
    )


def _predict_fixed_spec(
    artifact: SensitivityArtifacts,
    global_spec: ScenarioGlobalSpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    context: str,
    warning_rows: List[Dict[str, Any]],
    log_lines: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, float], pd.DataFrame]:
    engine = artifact.engine
    bundle = artifact.bundle
    selected = list(global_spec.selected_features)
    X_train = bundle.X.iloc[np.asarray(train_idx, dtype=int)]
    X_test = bundle.X.iloc[np.asarray(test_idx, dtype=int)]
    y_train = bundle.y[np.asarray(train_idx, dtype=int)]
    preprocessor = engine._make_preprocessor(
        selected, bundle.feature_rules, artifact.config.imputation
    ).fit(X_train[selected])
    train_matrix = preprocessor.transform(X_train[selected])
    test_matrix = preprocessor.transform(X_test[selected])
    model = engine.fit_logreg_with_refit(
        train_matrix,
        y_train,
        kind="l2",
        C=float(global_spec.l2_C),
        l1_ratio=None,
        dataset_name=context,
        stage="L2-fixed-spec-fit",
        log_lines=log_lines,
        warning_rows=warning_rows,
    )
    probability = model.predict_proba(test_matrix)[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite fixed-spec probabilities: {context}")
    coefficients = {
        feature: float(value)
        for feature, value in zip(preprocessor.kept_features_, model.coef_.ravel())
    }
    return probability, preprocessor.kept_features_, coefficients, preprocessor.audit_table()


def _boundary_counts(parameters: pd.DataFrame, engine: ModuleType) -> Dict[str, Any]:
    result: Dict[str, Any] = {"nested_outer_fit_count": int(len(parameters))}
    for column, grid, prefix in (
        ("en_C", engine.EN_C_GRID, "nested_en_C"),
        ("en_l1_ratio", engine.EN_L1_RATIO_GRID, "nested_en_l1_ratio"),
        ("l2_C", engine.L2_C_GRID, "nested_l2_C"),
    ):
        values = pd.to_numeric(parameters[column], errors="coerce").to_numpy(float)
        lower = float(min(grid))
        upper = float(max(grid))
        result[f"{prefix}_grid_min"] = lower
        result[f"{prefix}_grid_max"] = upper
        result[f"{prefix}_at_min_count"] = int(np.sum(np.isclose(values, lower)))
        result[f"{prefix}_at_max_count"] = int(np.sum(np.isclose(values, upper)))
    return result


def _primary_row(primary_reference: pd.DataFrame, dataset_key: str) -> pd.Series:
    rows = primary_reference.loc[primary_reference["dataset"].astype(str) == dataset_key]
    if len(rows) != 1:
        raise ValueError(f"Primary reference row is not unique for {dataset_key}")
    return rows.iloc[0]


def _run_matching_non_nested(
    artifact: SensitivityArtifacts,
    primary_reference: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    config = artifact.config
    engine = artifact.engine
    bundle = artifact.bundle
    dataset_key = artifact.spec.key
    label = f"{dataset_key} | {config.name} | fully_non_nested"
    global_spec = _tune_global_spec(artifact)
    warning_rows = list(global_spec.warning_rows)
    log_lines = list(global_spec.log_lines)

    prediction_rows: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    fold_count_rows: List[Dict[str, Any]] = []
    preprocessing_frames: List[pd.DataFrame] = []
    rank_lookup = {feature: rank for rank, feature in enumerate(global_spec.ranking, start=1)}
    ids = bundle.ids.astype(str).tolist()

    for outer_iter, record in enumerate(artifact.outer_plan.records, start=1):
        probability, model_features, coefficients, prep_audit = _predict_fixed_spec(
            artifact,
            global_spec,
            record.train_idx,
            record.test_idx,
            context=f"{label} repeat={record.repeat} fold={record.fold}",
            warning_rows=warning_rows,
            log_lines=log_lines,
        )
        for local_index, original_index in enumerate(record.test_idx):
            prediction_rows.append(
                {
                    "dataset": dataset_key,
                    "scenario": config.name,
                    "method": "fully_non_nested",
                    engine.ID_COL: ids[original_index],
                    "repeat": int(record.repeat),
                    "outer_fold": int(record.fold),
                    "y_true": int(bundle.y[original_index]),
                    "p": float(probability[local_index]),
                }
            )
        y_train = bundle.y[record.train_idx]
        y_test = bundle.y[record.test_idx]
        parameter_rows.append(
            {
                "dataset": dataset_key,
                "scenario": config.name,
                "method": "fully_non_nested",
                "outer_iter": outer_iter,
                "outer_repeat": int(record.repeat),
                "outer_fold": int(record.fold),
                "outer_splits_used": int(artifact.outer_plan.n_splits),
                "outer_repeats": int(artifact.outer_plan.n_repeats),
                "train_n": int(len(record.train_idx)),
                "train_positive": int(np.sum(y_train == 1)),
                "train_negative": int(np.sum(y_train == 0)),
                "test_n": int(len(record.test_idx)),
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
        )
        for subset_name, values in (("train", y_train), ("test", y_test)):
            fold_count_rows.append(
                {
                    "dataset": dataset_key,
                    "scenario": config.name,
                    "method": "fully_non_nested",
                    "outer_repeat": int(record.repeat),
                    "outer_fold": int(record.fold),
                    "subset": subset_name,
                    **engine.class_counts(values),
                }
            )
        model_feature_set = set(model_features)
        selected_set = set(global_spec.selected_features)
        for feature in bundle.feature_names:
            feature_rows.append(
                {
                    "dataset": dataset_key,
                    "scenario": config.name,
                    "method": "fully_non_nested",
                    "outer_repeat": int(record.repeat),
                    "outer_fold": int(record.fold),
                    "feature": feature,
                    "global_rank": rank_lookup.get(feature, np.nan),
                    "selected_global_topk": feature in selected_set,
                    "model_used": feature in model_feature_set,
                    "outer_coefficient": coefficients.get(feature, np.nan),
                    "global_k_selected": global_spec.k_selected,
                    "n_model_features": len(model_features),
                }
            )
        prep_audit.insert(0, "outer_fold", int(record.fold))
        prep_audit.insert(0, "outer_repeat", int(record.repeat))
        prep_audit.insert(0, "scenario", config.name)
        prep_audit.insert(0, "dataset", dataset_key)
        preprocessing_frames.append(prep_audit)
        print(
            f"[{label}] outer {outer_iter}/{len(artifact.outer_plan.records)} complete"
        )

    predictions = pd.DataFrame(prediction_rows)
    expected_rows = len(bundle.y) * int(artifact.outer_plan.n_repeats)
    if len(predictions) != expected_rows:
        raise RuntimeError(f"{label}: expected {expected_rows} predictions, found {len(predictions)}")
    oof = common.averaged_oof(
        predictions,
        ids,
        expected_repeats=int(artifact.outer_plan.n_repeats),
        id_col=engine.ID_COL,
    )
    repeat_metrics = common.repeat_metric_table(
        predictions, dataset_key, "fully_non_nested"
    )
    nonnested_metrics = common.metric_values(
        oof["y_true"].to_numpy(int), oof["p_oof"].to_numpy(float)
    )
    nested_oof = artifact.oof.set_index(engine.ID_COL).loc[
        oof[engine.ID_COL].astype(str)
    ].reset_index()
    if not np.array_equal(
        nested_oof["y_true"].to_numpy(int), oof["y_true"].to_numpy(int)
    ):
        raise ValueError(f"Nested/non-nested labels are misaligned for {config.name}/{dataset_key}")
    nested_metrics = common.metric_values(
        nested_oof["y_true"].to_numpy(int), nested_oof["p_oof"].to_numpy(float)
    )
    primary = _primary_row(primary_reference, dataset_key)
    same_population = bool(config.same_analysis_population_as_primary)
    if same_population:
        expected_counts = (
            int(primary["n_samples"]),
            int(primary["n_positive"]),
            int(primary["n_negative"]),
        )
        observed_counts = (
            int(len(bundle.y)),
            int(np.sum(bundle.y == 1)),
            int(np.sum(bundle.y == 0)),
        )
        if observed_counts != expected_counts:
            raise ValueError(
                f"Population count mismatch relative to Primary for {config.name}/{dataset_key}: "
                f"{observed_counts} vs {expected_counts}"
            )

    delta_auc = nonnested_metrics["roc_auc"] - nested_metrics["roc_auc"]
    delta_ap = nonnested_metrics["average_precision"] - nested_metrics["average_precision"]
    delta_brier = nonnested_metrics["brier"] - nested_metrics["brier"]
    primary_delta_auc = float(primary["delta_auc_nonnested_minus_nested"])
    primary_delta_ap = float(primary["delta_ap_nonnested_minus_nested"])
    primary_delta_brier = float(primary["delta_brier_nonnested_minus_nested"])
    comparison: Dict[str, Any] = {
        "family": config.family,
        "scenario": config.name,
        "invocation_scenario": config.invocation_scenario,
        "dataset": dataset_key,
        "dataset_id": artifact.spec.dataset_id,
        "timepoint": artifact.spec.timepoint,
        "analysis_role": common.dataset_role(artifact.spec),
        "inference_group": common.inference_group(artifact.spec),
        "same_analysis_population_as_primary": same_population,
        "primary_comparison_is_paired_population": same_population,
        "n_samples": int(len(bundle.y)),
        "n_positive": int(np.sum(bundle.y == 1)),
        "n_negative": int(np.sum(bundle.y == 0)),
        "nested_roc_auc": nested_metrics["roc_auc"],
        "nested_average_precision": nested_metrics["average_precision"],
        "nested_brier": nested_metrics["brier"],
        "nonnested_roc_auc": nonnested_metrics["roc_auc"],
        "nonnested_average_precision": nonnested_metrics["average_precision"],
        "nonnested_brier": nonnested_metrics["brier"],
        "delta_auc_nonnested_minus_nested": delta_auc,
        "delta_ap_nonnested_minus_nested": delta_ap,
        "delta_brier_nonnested_minus_nested": delta_brier,
        "primary_nested_roc_auc": float(primary["nested_roc_auc"]),
        "primary_nonnested_roc_auc": float(primary["nonnested_roc_auc"]),
        "primary_delta_auc_nonnested_minus_nested": primary_delta_auc,
        "change_nested_auc_vs_primary": nested_metrics["roc_auc"]
        - float(primary["nested_roc_auc"]),
        "change_nonnested_auc_vs_primary": nonnested_metrics["roc_auc"]
        - float(primary["nonnested_roc_auc"]),
        "change_delta_auc_vs_primary": delta_auc - primary_delta_auc,
        "primary_nested_average_precision": float(primary["nested_average_precision"]),
        "primary_nonnested_average_precision": float(primary["nonnested_average_precision"]),
        "primary_delta_ap_nonnested_minus_nested": primary_delta_ap,
        "change_delta_ap_vs_primary": delta_ap - primary_delta_ap,
        "primary_nested_brier": float(primary["nested_brier"]),
        "primary_nonnested_brier": float(primary["nonnested_brier"]),
        "primary_delta_brier_nonnested_minus_nested": primary_delta_brier,
        "change_delta_brier_vs_primary": delta_brier - primary_delta_brier,
        "global_en_C": global_spec.en_C,
        "global_en_l1_ratio": global_spec.en_l1_ratio,
        "global_k_selected": global_spec.k_selected,
        "global_l2_C": global_spec.l2_C,
        "global_en_C_at_grid_min": bool(
            np.isclose(global_spec.en_C, min(engine.EN_C_GRID))
        ),
        "global_en_C_at_grid_max": bool(
            np.isclose(global_spec.en_C, max(engine.EN_C_GRID))
        ),
        "global_l2_C_at_grid_min": bool(
            np.isclose(global_spec.l2_C, min(engine.L2_C_GRID))
        ),
        "global_l2_C_at_grid_max": bool(
            np.isclose(global_spec.l2_C, max(engine.L2_C_GRID))
        ),
        "outer_splits_used": int(artifact.outer_plan.n_splits),
        "outer_repeats": int(artifact.outer_plan.n_repeats),
        "imputation": config.imputation,
        "input_sheet": config.sheet_name,
        "max_k": config.max_k,
        "kmax_triggered_by_primary": (
            dataset_key in KMAX_PRIMARY_TRIGGER_KEYS if config.name == "kmax30" else np.nan
        ),
        "kmax_exploratory_surplus": (
            dataset_key not in KMAX_PRIMARY_TRIGGER_KEYS if config.name == "kmax30" else np.nan
        ),
        **_boundary_counts(artifact.parameters, engine),
    }

    paired_oof = pd.DataFrame(
        {
            engine.ID_COL: oof[engine.ID_COL].astype(str),
            "y_true": oof["y_true"].astype(int),
            "p_nested": nested_oof["p_oof"].to_numpy(float),
            "p_non_nested": oof["p_oof"].to_numpy(float),
        }
    )
    parameters = pd.DataFrame(parameter_rows)
    features = pd.DataFrame(feature_rows)
    folds = pd.DataFrame(fold_count_rows)
    preprocessing = pd.concat(preprocessing_frames, ignore_index=True)
    generated_warnings = pd.DataFrame(
        warning_rows, columns=["dataset", "stage", "warning_type", "message"]
    )
    generated_warnings.insert(0, "origin", "matching_non_nested")
    source_warnings = artifact.source_warnings.copy()
    if source_warnings.empty:
        source_warnings = pd.DataFrame(columns=["dataset", "stage", "warning_type", "message"])
    for column in ("dataset", "stage", "warning_type", "message"):
        if column not in source_warnings.columns:
            source_warnings[column] = ""
    source_warnings = source_warnings[["dataset", "stage", "warning_type", "message"]]
    source_warnings.insert(0, "origin", "nested_source")
    warnings_table = pd.concat([source_warnings, generated_warnings], ignore_index=True)

    common.write_csv(predictions, output_dir / "outer_predictions_long.csv")
    common.write_csv(oof, output_dir / "oof_predictions.csv")
    common.write_csv(paired_oof, output_dir / "paired_oof_predictions.csv")
    common.write_csv(repeat_metrics, output_dir / "repeat_metrics.csv")
    common.write_csv(pd.DataFrame([comparison]), output_dir / "metrics_summary.csv")
    common.write_csv(parameters, output_dir / "outer_parameters.csv")
    common.write_csv(features, output_dir / "outer_features_long.csv")
    common.write_csv(folds, output_dir / "fold_counts.csv")
    common.write_csv(preprocessing, output_dir / "preprocessing_by_outer.csv")
    ranking_table = global_spec.ranking_table.copy()
    ranking_table.insert(0, "scenario", config.name)
    ranking_table.insert(0, "dataset", dataset_key)
    ranking_table["selected_global_topk"] = ranking_table["rank"] <= global_spec.k_selected
    common.write_csv(ranking_table, output_dir / "global_ranking.csv")
    k_curve = global_spec.k_curve.copy()
    k_curve.insert(0, "scenario", config.name)
    k_curve.insert(0, "dataset", dataset_key)
    common.write_csv(k_curve, output_dir / "global_k_curve.csv")
    common.write_csv(warnings_table, output_dir / "warnings.csv")
    common.write_csv(artifact.source_hashes, output_dir / "source_artifact_hashes.csv")
    common.write_json(
        output_dir / "global_model_spec.json",
        {
            "dataset": dataset_key,
            "scenario": config.name,
            "method": "fully_non_nested",
            "statement": (
                "Diagnostic optimistic estimate: global selection/tuning and OOF "
                "evaluation reuse the same scenario-matched repeated CV"
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
            "family": config.family,
            "scenario": config.name,
            "invocation_scenario": config.invocation_scenario,
            "nested_source_dir": str(artifact.run_dir),
            "nested_source_receipt_sha256": common.sha256_file(
                artifact.run_dir / "run_receipt.json"
            ),
            "nested_producer_script": str(Path(config.producer_script).resolve()),
            "nested_producer_script_sha256": common.sha256_file(
                Path(config.producer_script).resolve()
            ),
            "nested_invocation_scenario_metadata": artifact.settings.get(
                "invocation_scenario_metadata"
            ),
            "input_path": str(artifact.spec.path),
            "input_sha256": bundle.input_sha256,
            "input_sheet": config.sheet_name,
            "imputation": config.imputation,
            "selection_and_evaluation_cv": {
                "splits": int(artifact.outer_plan.n_splits),
                "repeats": int(artifact.outer_plan.n_repeats),
                "random_state": int(engine.RANDOM_STATE),
            },
            "max_k": config.max_k,
            "en_C_grid": list(engine.EN_C_GRID),
            "en_l1_ratio_grid": list(engine.EN_L1_RATIO_GRID),
            "l2_C_grid": list(engine.L2_C_GRID),
            "missing_rate_threshold": float(engine.MISSING_RATE_THRESHOLD),
            "class_weight": "balanced",
            "same_analysis_population_as_primary": same_population,
            "hypothesis_tests": "not performed",
            "bootstrap_intervals": "not computed",
            "threshold_metrics": "not computed; no prespecified threshold",
        },
    )
    log_lines.append(
        f"[DONE] scenario={config.name}, dataset={dataset_key}, output={output_dir}"
    )
    (output_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    output_hashes = common.output_sha256_manifest(output_dir, UNIT_OUTPUT_FILES)
    common.write_json(
        output_dir / "run_receipt.json",
        {
            "status": "completed",
            "completed_utc": common.utc_now(),
            "family": config.family,
            "scenario": config.name,
            "invocation_scenario": config.invocation_scenario,
            "dataset": dataset_key,
            "method": "fully_non_nested_sensitivity_compare",
            "input_path": str(artifact.spec.path),
            "input_sha256": bundle.input_sha256,
            "nested_source_dir": str(artifact.run_dir),
            "nested_source_receipt_sha256": common.sha256_file(
                artifact.run_dir / "run_receipt.json"
            ),
            "nested_producer_script": str(Path(config.producer_script).resolve()),
            "nested_producer_script_sha256": common.sha256_file(
                Path(config.producer_script).resolve()
            ),
            "sample_counts": engine.class_counts(bundle.y),
            "metrics": comparison,
            "warning_count": int(len(warnings_table)),
            "output_sha256": output_hashes,
            **common.provenance_payload(Path(__file__)),
        },
    )
    return comparison


def _unit_dir(run_root: Path, config: ScenarioConfig, dataset_key: str) -> Path:
    return run_root / "datasets" / config.name / dataset_key


@contextmanager
def _staged_unit_dir(
    run_root: Path,
    config: ScenarioConfig,
    dataset_key: str,
) -> Iterator[Path]:
    token = uuid.uuid4().hex[:10]
    stage = run_root / "_staging" / f"{config.name}__{dataset_key}__{token}"
    final = _unit_dir(run_root, config, dataset_key)
    stage.mkdir(parents=True, exist_ok=False)
    try:
        yield stage
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"Unit output already exists unexpectedly: {final}")
        stage.rename(final)
    except Exception as exc:
        failure_parent = run_root / "failed" / config.name
        failure_parent.mkdir(parents=True, exist_ok=True)
        failure = failure_parent / f"{dataset_key}_{common.timestamp_token()}_{token}"
        try:
            common.write_json(
                stage / "failure.json",
                {
                    "status": "failed",
                    "scenario": config.name,
                    "dataset": dataset_key,
                    "failed_utc": common.utc_now(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            stage.rename(failure)
        except Exception:
            pass
        raise


def _run_plan(
    configs: Sequence[ScenarioConfig],
    locked_source_inventory: pd.DataFrame,
) -> Dict[str, Any]:
    scenario_plan = []
    for config in configs:
        scenario_plan.append(
            {
                "family": config.family,
                "name": config.name,
                "invocation_scenario": config.invocation_scenario,
                "run_root": str(Path(config.run_root).resolve()),
                "producer_script": str(Path(config.producer_script).resolve()),
                "expected_keys": list(config.expected_keys),
                "selected_keys": _selected_keys(config),
                "sheet_name": config.sheet_name,
                "imputation": config.imputation,
                "max_k": config.max_k,
                "outer_splits_target": config.outer_splits_target,
                "source_label": config.source_label,
            }
        )
    return {
        "script_sha256": common.sha256_file(Path(__file__).resolve()),
        "primary_compare_run_root": str(Path(PRIMARY_COMPARE_RUN_ROOT).resolve()),
        "locked_source_inventory_sha256": _inventory_fingerprint(locked_source_inventory),
        "scenarios": scenario_plan,
        "perform_hypothesis_tests": PERFORM_HYPOTHESIS_TESTS,
        "compute_bootstrap_intervals": COMPUTE_BOOTSTRAP_INTERVALS,
    }


def _initialize_run_root(
    configs: Sequence[ScenarioConfig],
    locked_source_inventory: pd.DataFrame,
) -> Path:
    root = _run_root().expanduser().resolve()
    plan = _run_plan(configs, locked_source_inventory)
    if root.exists() and any(root.iterdir()):
        if ALLOW_OVERWRITE:
            root = common.prepare_run_root(root, allow_overwrite=True)
            common.write_json(root / "run_plan.json", {"created_utc": common.utc_now(), "plan": plan})
            return root
        if not RESUME_COMPLETED_UNITS:
            raise FileExistsError(f"Run output already contains files: {root}")
        plan_path = root / "run_plan.json"
        if not plan_path.is_file():
            raise FileExistsError(
                f"Cannot safely resume because run_plan.json is missing: {root}"
            )
        saved = common.read_json(plan_path).get("plan")
        if saved != plan:
            raise ValueError(
                "Existing run plan differs from the current script/settings; use a new RUN_TAG"
            )
        completed_receipt = root / "run_receipt.json"
        if completed_receipt.is_file():
            status = common.read_json(completed_receipt).get("status")
            if status == "completed":
                raise FileExistsError(
                    f"Compare sensitivity run is already completed: {root}. Use a new RUN_TAG."
                )
    else:
        root.mkdir(parents=True, exist_ok=True)
        common.write_json(root / "run_plan.json", {"created_utc": common.utc_now(), "plan": plan})
    for name in ("datasets", "failed", "_staging"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _validate_recorded_source_hashes(path: Path) -> pd.DataFrame:
    table = common.read_csv(path)
    _require_columns(table, ("path", "sha256"), str(path))
    for row in table.itertuples(index=False):
        source = Path(str(row.path)).expanduser()
        if not source.is_file() or common.sha256_file(source) != str(row.sha256):
            raise ValueError(f"Source artifact changed since completed unit: {source}")
    return table


def _load_completed_unit(
    run_root: Path,
    config: ScenarioConfig,
    dataset_key: str,
) -> Optional[Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    directory = _unit_dir(run_root, config, dataset_key)
    if not directory.exists():
        return None
    receipt_path = directory / "run_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Existing unit lacks run_receipt.json: {directory}")
    receipt = common.read_json(receipt_path)
    if (
        receipt.get("status") != "completed"
        or receipt.get("scenario") != config.name
        or receipt.get("invocation_scenario") != config.invocation_scenario
        or receipt.get("dataset") != dataset_key
    ):
        raise ValueError(f"Existing unit receipt mismatch: {directory}")
    if receipt.get("script_sha256") != common.sha256_file(Path(__file__).resolve()):
        raise ValueError(f"Existing unit was produced by a different Compare_sensitivity.py: {directory}")
    producer = Path(config.producer_script).resolve()
    if receipt.get("nested_producer_script_sha256") != common.sha256_file(producer):
        raise ValueError(f"Nested producer changed since completed unit: {producer}")
    common.validate_output_sha256_manifest(directory, receipt, UNIT_OUTPUT_FILES)
    source_hashes = _validate_recorded_source_hashes(
        directory / "source_artifact_hashes.csv"
    )
    metrics = common.read_csv(directory / "metrics_summary.csv")
    if len(metrics) != 1:
        raise ValueError(f"Completed unit metrics row is not unique: {directory}")
    warnings_table = common.read_csv(directory / "warnings.csv")
    return metrics.iloc[0].to_dict(), source_hashes, warnings_table


def _scenario_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "scenario",
        "inference_group",
        "n_matrices",
        "median_delta_auc",
        "min_delta_auc",
        "max_delta_auc",
        "n_delta_positive",
        "n_delta_negative",
        "n_delta_zero",
        "median_change_delta_auc_vs_primary",
        "min_change_delta_auc_vs_primary",
        "max_change_delta_auc_vs_primary",
        "hypothesis_test",
        "interpretation",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (family, scenario, group), subset in comparison.groupby(
        ["family", "scenario", "inference_group"], sort=False
    ):
        delta = subset["delta_auc_nonnested_minus_nested"].to_numpy(float)
        change = subset["change_delta_auc_vs_primary"].to_numpy(float)
        rows.append(
            {
                "family": family,
                "scenario": scenario,
                "inference_group": group,
                "n_matrices": int(len(subset)),
                "median_delta_auc": float(np.median(delta)),
                "min_delta_auc": float(np.min(delta)),
                "max_delta_auc": float(np.max(delta)),
                "n_delta_positive": int(np.sum(delta > 0)),
                "n_delta_negative": int(np.sum(delta < 0)),
                "n_delta_zero": int(np.sum(delta == 0)),
                "median_change_delta_auc_vs_primary": float(np.median(change)),
                "min_change_delta_auc_vs_primary": float(np.min(change)),
                "max_change_delta_auc_vs_primary": float(np.max(change)),
                "hypothesis_test": "not performed",
                "interpretation": (
                    "descriptive independent Pre-cohort summary"
                    if group == "pre_primary_independent_cohorts"
                    else "descriptive only"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_excel(
    path: Path,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    primary_reference: pd.DataFrame,
    manifest: pd.DataFrame,
    warnings_table: pd.DataFrame,
    source_hashes: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheets = {
            "AllComparisons": comparison,
            "ScenarioSummary": summary,
            "PrimaryReference": primary_reference,
            "Manifest": manifest,
            "Warnings": warnings_table,
            "SourceHashes": source_hashes,
        }
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max((len(value) for value in values), default=0) + 2, 10), 42)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def _write_top_outputs(
    run_root: Path,
    comparison_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    primary_reference: pd.DataFrame,
    warning_frames: Sequence[pd.DataFrame],
    source_hash_frames: Sequence[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = pd.DataFrame(comparison_rows)
    manifest = pd.DataFrame(manifest_rows)
    summary = _scenario_summary(comparison)
    warnings_table = (
        pd.concat(warning_frames, ignore_index=True)
        if warning_frames
        else pd.DataFrame(columns=["origin", "dataset", "stage", "warning_type", "message"])
    )
    source_hashes = (
        pd.concat(source_hash_frames, ignore_index=True)
        .drop_duplicates(["path", "sha256"])
        .reset_index(drop=True)
        if source_hash_frames
        else pd.DataFrame(
            columns=["source_kind", "scenario", "dataset", "artifact", "path", "sha256"]
        )
    )
    common.write_csv(comparison, run_root / "comparison_all_scenarios.csv")
    common.write_csv(summary, run_root / "scenario_summary.csv")
    common.write_csv(primary_reference, run_root / "primary_delta_reference.csv")
    common.write_csv(manifest, run_root / "run_manifest.csv")
    common.write_csv(warnings_table, run_root / "warnings_all.csv")
    common.write_csv(source_hashes, run_root / "source_artifact_hashes.csv")
    if WRITE_EXCEL:
        _write_excel(
            run_root / "Compare_sensitivity_results.xlsx",
            comparison,
            summary,
            primary_reference,
            manifest,
            warnings_table,
            source_hashes,
        )
    return comparison, manifest, warnings_table, source_hashes


def main() -> None:
    if PERFORM_HYPOTHESIS_TESTS:
        raise ValueError("Sensitivity hypothesis tests are intentionally disabled")
    if COMPUTE_BOOTSTRAP_INTERVALS:
        raise ValueError("Sensitivity bootstrap intervals are intentionally disabled")
    if not RUN_TAG:
        raise ValueError("RUN_TAG must be non-empty for an auditable sensitivity comparison")
    configs = _selected_configs()
    units = [(config, key) for config in configs for key in _selected_keys(config)]
    if not units:
        raise RuntimeError("No scenario/dataset units were selected")

    started_time = time.time()
    started_utc = common.utc_now()
    primary_reference, primary_hashes = _validate_primary_reference()
    sensitivity_inventory = _preflight_source_inventory(configs, units)
    locked_source_inventory = pd.concat(
        [primary_hashes, sensitivity_inventory], ignore_index=True
    ).drop_duplicates(["path", "sha256"])
    run_root = _initialize_run_root(configs, locked_source_inventory)
    comparison_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    warning_frames: List[pd.DataFrame] = []
    source_hash_frames: List[pd.DataFrame] = [locked_source_inventory]

    for config, dataset_key in units:
        print("=" * 88)
        print(f"[START] Compare_sensitivity | {config.name} | {dataset_key}")
        try:
            cached = (
                _load_completed_unit(run_root, config, dataset_key)
                if RESUME_COMPLETED_UNITS
                else None
            )
            if cached is not None:
                comparison, hashes, warnings_table = cached
                action = "resumed_validated"
                print(f"[RESUME] validated completed unit | {config.name} | {dataset_key}")
            else:
                artifact = _validate_sensitivity_artifacts(config, dataset_key)
                _assert_source_inventory_unchanged(
                    artifact.source_hashes, locked_source_inventory
                )
                with _staged_unit_dir(run_root, config, dataset_key) as stage:
                    comparison = _run_matching_non_nested(
                        artifact, primary_reference, stage
                    )
                hashes = artifact.source_hashes
                warnings_table = common.read_csv(
                    _unit_dir(run_root, config, dataset_key) / "warnings.csv"
                )
                action = "calculated"
                print(f"[DONE] Compare_sensitivity | {config.name} | {dataset_key}")
            comparison_rows.append(dict(comparison))
            source_hash_frames.append(hashes)
            if not warnings_table.empty:
                warning_copy = warnings_table.copy()
                warning_copy.insert(0, "scenario", config.name)
                warning_copy.insert(1, "source_dataset", dataset_key)
                warning_frames.append(warning_copy)
            manifest_rows.append(
                {
                    "family": config.family,
                    "scenario": config.name,
                    "invocation_scenario": config.invocation_scenario,
                    "dataset": dataset_key,
                    "status": "completed",
                    "execution_action": action,
                    "output_dir": str(_unit_dir(run_root, config, dataset_key)),
                    "message": "",
                    "nested_roc_auc": comparison["nested_roc_auc"],
                    "nonnested_roc_auc": comparison["nonnested_roc_auc"],
                    "delta_auc": comparison["delta_auc_nonnested_minus_nested"],
                    "change_delta_auc_vs_primary": comparison[
                        "change_delta_auc_vs_primary"
                    ],
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "family": config.family,
                    "scenario": config.name,
                    "invocation_scenario": config.invocation_scenario,
                    "dataset": dataset_key,
                    "status": "failed",
                    "execution_action": "failed",
                    "output_dir": "",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[FAILED] Compare_sensitivity | {config.name} | {dataset_key}: "
                f"{type(exc).__name__}: {exc}"
            )
            _write_top_outputs(
                run_root,
                comparison_rows,
                manifest_rows,
                primary_reference,
                warning_frames,
                source_hash_frames,
            )
            if not CONTINUE_ON_UNIT_ERROR:
                raise
        else:
            _write_top_outputs(
                run_root,
                comparison_rows,
                manifest_rows,
                primary_reference,
                warning_frames,
                source_hash_frames,
            )

    comparison, manifest, _, _ = _write_top_outputs(
        run_root,
        comparison_rows,
        manifest_rows,
        primary_reference,
        warning_frames,
        source_hash_frames,
    )
    common.finalize_run_receipt(
        run_root,
        Path(__file__),
        started_utc,
        started_time,
        manifest,
        {
            "analysis": "scenario_matched_nested_vs_fully_non_nested_sensitivity",
            "primary_compare_run_root": str(Path(PRIMARY_COMPARE_RUN_ROOT).resolve()),
            "scenario_names": [config.name for config in configs],
            "selected_units": [f"{config.name}/{key}" for config, key in units],
            "n_expected_units": len(units),
            "hypothesis_tests": "not performed; Primary Pre inference remains the sole formal test",
            "bootstrap_intervals": "not computed",
            "post_analysis": "descriptive only; repeated time points are not independent cohorts",
            "dataset6_analysis": "descriptive exploratory supplement only",
            "sd_primary_comparison": (
                "not paired because SD inclusion changes sample size and label definition"
            ),
            "kmax30_scope": (
                "all 17 completed; 14 Primary ceiling-triggered and 3 labelled exploratory surplus"
            ),
            "resume_completed_units": RESUME_COMPLETED_UNITS,
            "continue_on_unit_error": CONTINUE_ON_UNIT_ERROR,
            "write_excel": WRITE_EXCEL,
        },
    )
    completed = int((manifest["status"] == "completed").sum())
    failed = int((manifest["status"] == "failed").sum())
    print("=" * 88)
    print(f"Compare_sensitivity completed: {completed}; failed: {failed}")
    print(f"Output: {run_root}")
    if len(comparison) != completed:
        raise RuntimeError("Top comparison row count does not match completed units")
    if failed:
        raise RuntimeError(
            f"Compare_sensitivity finished with {failed} failed unit(s); inspect run_manifest.csv"
        )


if __name__ == "__main__":
    main()
