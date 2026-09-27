#!/usr/bin/env python3
"""Diagnose where selection optimism enters the CV pipeline.

Four methods are reported on the identical primary-analysis outer folds:

1. fully_nested (read from the completed main run)
2. ranking_nested_kC_global (diagnostic hybrid; intentionally leaky)
3. ranking_global_kC_nested (diagnostic hybrid; intentionally leaky)
4. fully_non_nested (read from Compare_method.py)

The two hybrids are not valid performance estimators. They isolate whether
optimism is introduced mainly by global ranking or by global k/C selection.
No p-values or figures are produced here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import ICI_predict as base
import ICI_analysis_common as common
import dataset_selection as selection


# =============================================================================
# USER SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_primary"
COMPARE_METHOD_RUN_ROOT = SCRIPT_DIR / "results_revision" / "Compare_method_v1"

OUTPUT_ROOT = SCRIPT_DIR / "results_revision"
RUN_TAG = "Compare_nest_v1"

PRE_KEYS: Tuple[str, ...] = tuple(
    f"dataset{number}_pre" for number in (1, 2, 3, 4, 5, 7)
)
DATASETS_TO_RUN: Union[str, List[str]] = "ALL"

MAX_K = base.PRIMARY_MAX_K
STRICT_BASE_SCRIPT_HASH = True
CONTINUE_ON_DATASET_ERROR = True
ALLOW_OVERWRITE = False


METHOD_ORDER: Tuple[str, ...] = (
    "fully_nested",
    "ranking_nested_kC_global",
    "ranking_global_kC_nested",
    "fully_non_nested",
)


def _run_root() -> Path:
    return OUTPUT_ROOT / RUN_TAG if RUN_TAG else OUTPUT_ROOT


def _selected_keys(specs: Mapping[str, base.DatasetSpec]) -> List[str]:
    selected = selection.select_available(
        DATASETS_TO_RUN, specs, default_keys=PRE_KEYS
    )
    invalid = [key for key in selected if key not in PRE_KEYS]
    if invalid:
        raise ValueError(f"Compare_nest supports only Datasets 1-5 and 7 Pre: {invalid}")
    return selected


def _load_compare_method(
    artifact: common.MainArtifacts,
) -> Dict[str, Any]:
    root = Path(COMPARE_METHOD_RUN_ROOT).expanduser().resolve()
    manifest_path = root / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Compare_method run_manifest.csv is missing: {manifest_path}"
        )
    manifest = common.read_csv(manifest_path)
    required_manifest = {"dataset", "status", "output_dir"}
    if missing := required_manifest.difference(manifest.columns):
        raise ValueError(f"Compare_method manifest lacks columns: {sorted(missing)}")
    row = manifest.loc[manifest["dataset"].astype(str) == artifact.spec.key]
    if len(row) != 1 or row.iloc[0]["status"] != "completed":
        raise FileNotFoundError(
            f"No unique completed Compare_method result for {artifact.spec.key}"
        )
    manifest_dir = Path(str(row.iloc[0]["output_dir"])).expanduser()
    fallback = root / "datasets" / artifact.spec.key
    directory = next(
        (path.resolve() for path in (manifest_dir, fallback) if path.exists()), None
    )
    if directory is None:
        raise FileNotFoundError(f"Compare_method dataset directory is missing for {artifact.spec.key}")
    required = (
        "run_receipt.json",
        "outer_predictions_long.csv",
        "oof_predictions.csv",
        "outer_parameters.csv",
        "outer_features_long.csv",
        "global_ranking.csv",
        "global_model_spec.json",
    )
    for filename in required:
        if not (directory / filename).exists():
            raise FileNotFoundError(f"Missing Compare_method output: {directory / filename}")
    receipt = common.read_json(directory / "run_receipt.json")
    if receipt.get("status") != "completed" or receipt.get("dataset") != artifact.spec.key:
        raise ValueError(f"Invalid Compare_method receipt for {artifact.spec.key}")
    if receipt.get("input_sha256") != artifact.bundle.input_sha256:
        raise ValueError(f"Compare_method input SHA mismatch for {artifact.spec.key}")
    if STRICT_BASE_SCRIPT_HASH:
        current_base_hash = common.sha256_file(Path(base.__file__).resolve())
        if receipt.get("base_script_sha256") != current_base_hash:
            raise ValueError(f"Compare_method base-script SHA mismatch for {artifact.spec.key}")
    producer_path = SCRIPT_DIR / "Compare_method.py"
    if not producer_path.is_file() or receipt.get("script_sha256") != common.sha256_file(
        producer_path
    ):
        raise ValueError(f"Compare_method producer-script SHA mismatch for {artifact.spec.key}")
    current_common_hash = common.sha256_file(Path(common.__file__).resolve())
    if receipt.get("common_module_sha256") != current_common_hash:
        raise ValueError(f"Compare_method common-module SHA mismatch for {artifact.spec.key}")
    common.validate_output_sha256_manifest(
        directory,
        receipt,
        (
            "outer_predictions_long.csv",
            "oof_predictions.csv",
            "outer_parameters.csv",
            "outer_features_long.csv",
            "global_ranking.csv",
            "global_model_spec.json",
            "settings_used.json",
        ),
    )
    compare_settings = common.read_json(directory / "settings_used.json")
    cv_settings = compare_settings.get("selection_and_evaluation_cv", {})
    expected_main_receipt_hash = common.sha256_file(artifact.run_dir / "run_receipt.json")
    if compare_settings.get("main_run_receipt_sha256") != expected_main_receipt_hash:
        raise ValueError(f"Compare_method is not linked to the validated main run for {artifact.spec.key}")
    if (
        compare_settings.get("class_weight") != "balanced"
        or int(cv_settings.get("splits", -1)) != artifact.outer_plan.n_splits
        or int(cv_settings.get("repeats", -1)) != artifact.outer_plan.n_repeats
        or int(cv_settings.get("random_state", -1)) != int(base.RANDOM_STATE)
        or int(compare_settings.get("max_k", -1)) != int(MAX_K)
        or not np.isclose(
            float(compare_settings.get("missing_rate_threshold", np.nan)),
            float(base.MISSING_RATE_THRESHOLD),
        )
    ):
        raise ValueError(f"Compare_method settings mismatch for {artifact.spec.key}")

    predictions = common.read_csv(
        directory / "outer_predictions_long.csv", id_columns=[base.ID_COL]
    )
    oof = common.read_csv(directory / "oof_predictions.csv", id_columns=[base.ID_COL])
    parameters = common.read_csv(directory / "outer_parameters.csv")
    outer_features = common.read_csv(directory / "outer_features_long.csv")
    ranking = common.read_csv(directory / "global_ranking.csv")
    model_spec = common.read_json(directory / "global_model_spec.json")
    required_prediction = {
        base.ID_COL,
        "repeat",
        "outer_fold",
        "y_true",
        "p",
    }
    if missing := required_prediction.difference(predictions.columns):
        raise ValueError(f"Compare_method predictions lack columns: {sorted(missing)}")
    if predictions.duplicated([base.ID_COL, "repeat"]).any():
        raise ValueError(f"Duplicate Compare_method ID/repeat rows for {artifact.spec.key}")
    ids = artifact.bundle.ids.astype(str).tolist()
    for record in artifact.outer_plan.records:
        subset = predictions.loc[
            (predictions["repeat"].astype(int) == record.repeat)
            & (predictions["outer_fold"].astype(int) == record.fold)
        ]
        expected_ids = {ids[index] for index in record.test_idx}
        if set(subset[base.ID_COL].astype(str)) != expected_ids:
            raise ValueError(
                f"Compare_method outer fold does not match main: {artifact.spec.key}, "
                f"repeat={record.repeat}, fold={record.fold}"
            )
        expected_labels = {
            ids[index]: int(artifact.bundle.y[index]) for index in record.test_idx
        }
        observed_labels = dict(
            zip(subset[base.ID_COL].astype(str), subset["y_true"].astype(int))
        )
        if observed_labels != expected_labels:
            raise ValueError(
                f"Compare_method labels differ from input labels for {artifact.spec.key}"
            )
    recomputed_oof = common.averaged_oof(
        predictions, ids, expected_repeats=artifact.outer_plan.n_repeats
    )
    saved_oof = oof.set_index(base.ID_COL).loc[ids].reset_index()
    if not np.allclose(saved_oof["p_oof"], recomputed_oof["p_oof"], atol=1e-12):
        raise ValueError(f"Compare_method OOF does not reproduce for {artifact.spec.key}")
    ranking = ranking.sort_values("rank").reset_index(drop=True)
    ranking_features = ranking["feature"].astype(str).tolist()
    if len(ranking_features) != len(set(ranking_features)) or not ranking_features:
        raise ValueError(f"Invalid global ranking for {artifact.spec.key}")
    selected = [str(value) for value in model_spec.get("global_selected_features", [])]
    global_k = int(model_spec.get("global_k_selected", -1))
    if global_k < 1 or selected != ranking_features[:global_k]:
        raise ValueError(f"Global k/feature list mismatch for {artifact.spec.key}")
    return {
        "run_dir": directory,
        "receipt": receipt,
        "predictions": predictions,
        "oof": oof,
        "parameters": parameters,
        "outer_features": outer_features,
        "ranking_table": ranking,
        "ranking": ranking_features,
        "model_spec": model_spec,
        "global_k": global_k,
        "global_l2_C": float(model_spec["global_l2_C"]),
        "global_en_C": float(model_spec["global_en_C"]),
        "global_en_l1_ratio": float(model_spec["global_en_l1_ratio"]),
    }


def _fold_nested_ranking(
    artifact: common.MainArtifacts,
    record: base.SplitRecord,
) -> List[str]:
    table = artifact.outer_features.loc[
        (artifact.outer_features["outer_repeat"].astype(int) == record.repeat)
        & (artifact.outer_features["outer_fold"].astype(int) == record.fold)
        & artifact.outer_features["rank"].notna()
    ].copy()
    if table.empty:
        raise ValueError(
            f"Missing nested ranking: {artifact.spec.key}, repeat={record.repeat}, fold={record.fold}"
        )
    table = table.sort_values("rank")
    ranking = table["feature"].astype(str).tolist()
    if len(ranking) != len(set(ranking)):
        raise ValueError("Nested fold ranking contains duplicate features")
    return ranking


def _main_parameter_row(
    artifact: common.MainArtifacts,
    record: base.SplitRecord,
) -> pd.Series:
    table = artifact.parameters.loc[
        (artifact.parameters["outer_repeat"].astype(int) == record.repeat)
        & (artifact.parameters["outer_fold"].astype(int) == record.fold)
    ]
    if len(table) != 1:
        raise ValueError("Expected one main parameter row per outer fold")
    return table.iloc[0]


def _feature_rows(
    artifact: common.MainArtifacts,
    method: str,
    record: base.SplitRecord,
    ranking: Sequence[str],
    selected: Sequence[str],
    model_features: Sequence[str],
    coefficients: Mapping[str, float],
    k_selected: int,
) -> List[Dict[str, Any]]:
    rank_lookup = {feature: rank for rank, feature in enumerate(ranking, start=1)}
    selected_set = set(selected)
    model_set = set(model_features)
    return [
        {
            "dataset": artifact.spec.key,
            "method": method,
            "outer_repeat": record.repeat,
            "outer_fold": record.fold,
            "feature": feature,
            "rank": rank_lookup.get(feature, np.nan),
            "selected_topk": feature in selected_set,
            "model_used": feature in model_set,
            "outer_coefficient": coefficients.get(feature, np.nan),
            "k_selected": k_selected,
            "n_model_features": len(model_features),
        }
        for feature in artifact.bundle.feature_names
    ]


def compute_hybrids(
    artifact: common.MainArtifacts,
    compare: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    bundle = artifact.bundle
    ids = bundle.ids.astype(str).tolist()
    prediction_rows: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    k_curve_frames: List[pd.DataFrame] = []
    warning_rows: List[Dict[str, Any]] = []
    log_lines: List[str] = []

    for outer_iter, record in enumerate(artifact.outer_plan.records, start=1):
        X_train = bundle.X.iloc[record.train_idx]
        y_train = bundle.y[record.train_idx]
        y_test = bundle.y[record.test_idx]
        weights = common.class_weights(y_train)
        main_params = _main_parameter_row(artifact, record)

        # Hybrid A: ranking is nested in each outer fold; k and C are selected
        # globally by the deliberately non-nested analysis.
        method_a = "ranking_nested_kC_global"
        ranking_a = _fold_nested_ranking(artifact, record)
        effective_global_k = min(compare["global_k"], len(ranking_a))
        selected_a = ranking_a[:effective_global_k]
        p_a, model_a, coefficients_a, _ = common.predict_fixed_spec(
            bundle,
            record.train_idx,
            record.test_idx,
            selected_features=selected_a,
            l2_C=compare["global_l2_C"],
            context=(
                f"{artifact.spec.key} | {method_a} repeat={record.repeat} fold={record.fold}"
            ),
            warning_rows=warning_rows,
            log_lines=log_lines,
        )
        for local_index, original_index in enumerate(record.test_idx):
            prediction_rows.append(
                {
                    "dataset": artifact.spec.key,
                    "method": method_a,
                    base.ID_COL: ids[original_index],
                    "repeat": record.repeat,
                    "outer_fold": record.fold,
                    "y_true": int(bundle.y[original_index]),
                    "p": float(p_a[local_index]),
                }
            )
        parameter_rows.append(
            {
                "dataset": artifact.spec.key,
                "method": method_a,
                "outer_repeat": record.repeat,
                "outer_fold": record.fold,
                "train_n": len(record.train_idx),
                "train_positive": int(np.sum(y_train == 1)),
                "train_negative": int(np.sum(y_train == 0)),
                "test_n": len(record.test_idx),
                "test_positive": int(np.sum(y_test == 1)),
                "test_negative": int(np.sum(y_test == 0)),
                **weights,
                "ranking_source": "nested_outer_training",
                "k_C_source": "global_full_dataset_leaky",
                "en_C_for_ranking": float(main_params["en_C"]),
                "en_l1_ratio_for_ranking": float(main_params["en_l1_ratio"]),
                "k_selected_requested": compare["global_k"],
                "k_selected_effective": effective_global_k,
                "l2_C": compare["global_l2_C"],
                "n_ranked_features": len(ranking_a),
                "n_model_features": len(model_a),
            }
        )
        feature_rows.extend(
            _feature_rows(
                artifact,
                method_a,
                record,
                ranking_a,
                selected_a,
                model_a,
                coefficients_a,
                effective_global_k,
            )
        )

        # Hybrid B: the ranking is selected globally, while k and C are tuned
        # solely within each outer-training set.
        method_b = "ranking_global_kC_nested"
        eligibility = base._make_preprocessor(
            bundle.feature_names, bundle.feature_rules, "half_minimum"
        ).fit(X_train[bundle.feature_names])
        eligible = set(eligibility.kept_features_)
        ranking_b = [feature for feature in compare["ranking"] if feature in eligible]
        if not ranking_b:
            raise ValueError("No global-ranked feature is eligible in an outer-training fold")
        inner_plan = base.make_cv_plan(
            y_train,
            target_splits=base.INNER_SPLITS_TARGET,
            n_repeats=base.INNER_REPEATS,
            context=(
                f"{artifact.spec.key} | {method_b} repeat={record.repeat} "
                f"fold={record.fold} inner"
            ),
            log_lines=log_lines,
            random_state=base.RANDOM_STATE,
        )
        k_b, C_b, k_curve = base.select_k_by_1se(
            X_train,
            y_train,
            ranking_b,
            inner_plan,
            dataset_name=f"{artifact.spec.key} | {method_b}",
            outer_iter=outer_iter,
            log_lines=log_lines,
            feature_rules=bundle.feature_rules,
            imputation="half_minimum",
            max_k=MAX_K,
            warning_rows=warning_rows,
        )
        selected_b = ranking_b[:k_b]
        p_b, model_b, coefficients_b, _ = common.predict_fixed_spec(
            bundle,
            record.train_idx,
            record.test_idx,
            selected_features=selected_b,
            l2_C=C_b,
            context=(
                f"{artifact.spec.key} | {method_b} repeat={record.repeat} fold={record.fold}"
            ),
            warning_rows=warning_rows,
            log_lines=log_lines,
        )
        for local_index, original_index in enumerate(record.test_idx):
            prediction_rows.append(
                {
                    "dataset": artifact.spec.key,
                    "method": method_b,
                    base.ID_COL: ids[original_index],
                    "repeat": record.repeat,
                    "outer_fold": record.fold,
                    "y_true": int(bundle.y[original_index]),
                    "p": float(p_b[local_index]),
                }
            )
        parameter_rows.append(
            {
                "dataset": artifact.spec.key,
                "method": method_b,
                "outer_repeat": record.repeat,
                "outer_fold": record.fold,
                "train_n": len(record.train_idx),
                "train_positive": int(np.sum(y_train == 1)),
                "train_negative": int(np.sum(y_train == 0)),
                "test_n": len(record.test_idx),
                "test_positive": int(np.sum(y_test == 1)),
                "test_negative": int(np.sum(y_test == 0)),
                **weights,
                "ranking_source": "global_full_dataset_leaky",
                "k_C_source": "nested_outer_training",
                "en_C_for_ranking": compare["global_en_C"],
                "en_l1_ratio_for_ranking": compare["global_en_l1_ratio"],
                "inner_splits_used": inner_plan.n_splits,
                "inner_repeats": inner_plan.n_repeats,
                "k_selected_requested": k_b,
                "k_selected_effective": k_b,
                "l2_C": C_b,
                "n_ranked_features": len(ranking_b),
                "n_model_features": len(model_b),
            }
        )
        feature_rows.extend(
            _feature_rows(
                artifact,
                method_b,
                record,
                ranking_b,
                selected_b,
                model_b,
                coefficients_b,
                k_b,
            )
        )
        k_curve.insert(0, "dataset", artifact.spec.key)
        k_curve.insert(1, "method", method_b)
        k_curve["outer_repeat"] = record.repeat
        k_curve["outer_fold"] = record.fold
        k_curve_frames.append(k_curve)
        print(
            f"[{artifact.spec.key}] Compare_nest outer {outer_iter}/"
            f"{len(artifact.outer_plan.records)} complete"
        )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(parameter_rows),
        pd.DataFrame(feature_rows),
        pd.concat(k_curve_frames, ignore_index=True),
        pd.DataFrame(
            warning_rows,
            columns=["dataset", "stage", "warning_type", "message"],
        ),
        "\n".join(log_lines),
    )


def run_dataset(
    artifact: common.MainArtifacts,
    compare: Dict[str, Any],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    (
        hybrid_predictions,
        hybrid_parameters,
        hybrid_features,
        k_curves,
        warnings_table,
        log_text,
    ) = (
        compute_hybrids(artifact, compare)
    )

    nested_predictions = artifact.predictions_long.copy()
    nested_predictions["method"] = "fully_nested"
    nested_predictions = nested_predictions[
        ["dataset", "method", base.ID_COL, "repeat", "outer_fold", "y_true", "p"]
    ]
    nonnested_predictions = compare["predictions"].copy()
    nonnested_predictions["method"] = "fully_non_nested"
    nonnested_predictions = nonnested_predictions[
        ["dataset", "method", base.ID_COL, "repeat", "outer_fold", "y_true", "p"]
    ]
    all_predictions = pd.concat(
        [nested_predictions, hybrid_predictions, nonnested_predictions],
        ignore_index=True,
    )

    expected_per_method = len(artifact.bundle.y) * artifact.outer_plan.n_repeats
    metric_rows: List[Dict[str, Any]] = []
    oof_frames: List[pd.DataFrame] = []
    repeat_frames: List[pd.DataFrame] = []
    for method in METHOD_ORDER:
        subset = all_predictions.loc[all_predictions["method"] == method].copy()
        if len(subset) != expected_per_method:
            raise RuntimeError(
                f"{artifact.spec.key} {method}: expected {expected_per_method} rows, found {len(subset)}"
            )
        oof = common.averaged_oof(
            subset,
            artifact.bundle.ids.astype(str).tolist(),
            expected_repeats=artifact.outer_plan.n_repeats,
        )
        oof.insert(0, "method", method)
        oof.insert(0, "dataset", artifact.spec.key)
        oof_frames.append(oof)
        repeat_frames.append(common.repeat_metric_table(subset, artifact.spec.key, method))
        values = common.metric_values(
            oof["y_true"].to_numpy(dtype=int), oof["p_oof"].to_numpy(dtype=float)
        )
        metric_rows.append(
            {
                "dataset": artifact.spec.key,
                "dataset_id": artifact.spec.dataset_id,
                "timepoint": artifact.spec.timepoint,
                "analysis_role": common.dataset_role(artifact.spec),
                "method": method,
                "valid_performance_estimator": method == "fully_nested",
                "diagnostic_leakage": method != "fully_nested",
                "n_samples": len(artifact.bundle.y),
                "n_positive": int(np.sum(artifact.bundle.y == 1)),
                "n_negative": int(np.sum(artifact.bundle.y == 0)),
                **values,
                "delta_auc_vs_fully_nested": (
                    values["roc_auc"]
                    - common.metric_values(
                        artifact.oof["y_true"].to_numpy(dtype=int),
                        artifact.oof["p_oof"].to_numpy(dtype=float),
                    )["roc_auc"]
                ),
            }
        )

    nested_parameters = artifact.parameters.copy()
    nested_parameters["method"] = "fully_nested"
    nonnested_parameters = compare["parameters"].copy()
    nonnested_parameters["method"] = "fully_non_nested"
    nested_parameters["k_effective"] = nested_parameters["k_selected"]
    nested_parameters["l2_C_effective"] = nested_parameters["l2_C"]
    nested_parameters["ranking_source"] = "nested_outer_training"
    nested_parameters["k_C_source"] = "nested_outer_training"
    hybrid_parameters["k_effective"] = hybrid_parameters["k_selected_effective"]
    hybrid_parameters["l2_C_effective"] = hybrid_parameters["l2_C"]
    nonnested_parameters["k_effective"] = nonnested_parameters["global_k_selected"]
    nonnested_parameters["l2_C_effective"] = nonnested_parameters["global_l2_C"]
    nonnested_parameters["ranking_source"] = "global_full_dataset_leaky"
    nonnested_parameters["k_C_source"] = "global_full_dataset_leaky"
    all_parameters = pd.concat(
        [nested_parameters, hybrid_parameters, nonnested_parameters],
        ignore_index=True,
        sort=False,
    )
    nested_features = artifact.outer_features.copy()
    nested_features["method"] = "fully_nested"
    nonnested_features = compare["outer_features"].copy()
    nonnested_features["method"] = "fully_non_nested"
    nested_features["rank_effective"] = nested_features["rank"]
    nested_features["selected_effective"] = nested_features["selected_topk"]
    nested_features["ranking_source"] = "nested_outer_training"
    hybrid_features["rank_effective"] = hybrid_features["rank"]
    hybrid_features["selected_effective"] = hybrid_features["selected_topk"]
    hybrid_features["ranking_source"] = np.where(
        hybrid_features["method"].eq("ranking_nested_kC_global"),
        "nested_outer_training",
        "global_full_dataset_leaky",
    )
    nonnested_features["rank_effective"] = nonnested_features["global_rank"]
    nonnested_features["selected_effective"] = nonnested_features[
        "selected_global_topk"
    ]
    nonnested_features["ranking_source"] = "global_full_dataset_leaky"
    all_features = pd.concat(
        [nested_features, hybrid_features, nonnested_features],
        ignore_index=True,
        sort=False,
    )
    metrics = pd.DataFrame(metric_rows)

    common.write_csv(all_predictions, output_dir / "outer_predictions_long.csv")
    common.write_csv(pd.concat(oof_frames, ignore_index=True), output_dir / "oof_predictions.csv")
    common.write_csv(
        pd.concat(repeat_frames, ignore_index=True), output_dir / "repeat_metrics.csv"
    )
    common.write_csv(metrics, output_dir / "metrics_summary.csv")
    common.write_csv(all_parameters, output_dir / "outer_parameters.csv")
    common.write_csv(all_features, output_dir / "outer_features_long.csv")
    common.write_csv(k_curves, output_dir / "nested_kC_k_curves.csv")
    common.write_csv(warnings_table, output_dir / "warnings.csv")
    (output_dir / "run_log.txt").write_text(log_text + "\n", encoding="utf-8")
    common.write_json(
        output_dir / "settings_used.json",
        {
            "dataset_selection_sha256": common.sha256_file(Path(selection.__file__).resolve()),
            "main_source_dir": str(artifact.run_dir),
            "compare_method_source_dir": str(compare["run_dir"]),
            "input_path": str(artifact.spec.path),
            "input_sha256": artifact.bundle.input_sha256,
            "outer_folds": (
                "identical reconstructed primary-analysis outer folds for all methods"
            ),
            "methods": {
                "fully_nested": "ranking and k/C selected inside outer training",
                "ranking_nested_kC_global": (
                    "diagnostic hybrid; fold-local ranking, globally selected k/C"
                ),
                "ranking_global_kC_nested": (
                    "diagnostic hybrid; global ranking, fold-local k/C selection"
                ),
                "fully_non_nested": (
                    "diagnostic optimistic estimate; global ranking and k/C"
                ),
            },
                "formal_tests": "not performed",
                "hybrid_interpretation": (
                    "conditional diagnostic only; global k/C was selected jointly with the "
                    "global ranking, so method differences are not pure causal component effects"
                ),
            "max_k": MAX_K,
            "class_weight": "balanced",
            "missing_rate_threshold": base.MISSING_RATE_THRESHOLD,
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
            "nested_kC_k_curves.csv",
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
            "analysis_role": common.dataset_role(artifact.spec),
            "input_path": str(artifact.spec.path),
            "input_sha256": artifact.bundle.input_sha256,
            "main_source_dir": str(artifact.run_dir),
            "compare_method_source_dir": str(compare["run_dir"]),
            "sample_counts": base.class_counts(artifact.bundle.y),
            "metrics": metric_rows,
            "output_sha256": output_hashes,
            "dataset_selection_sha256": common.sha256_file(Path(selection.__file__).resolve()),
            **common.provenance_payload(Path(__file__)),
        },
    )
    return metric_rows


def main() -> None:
    if int(MAX_K) != int(base.PRIMARY_MAX_K):
        raise ValueError("Compare_nest MAX_K must equal ICI_predict.PRIMARY_MAX_K")
    started_time = time.time()
    started_utc = common.utc_now()
    specs = base.discover_datasets()
    keys = _selected_keys(specs)
    if not keys:
        raise RuntimeError("No Dataset 1-7 Pre inputs were selected")
    run_root = common.prepare_run_root(_run_root(), ALLOW_OVERWRITE)
    common.write_csv(
        pd.DataFrame(selection.coverage_rows(specs, keys, expected=PRE_KEYS)),
        run_root / "input_coverage.csv",
    )

    metrics_all: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for key in keys:
        print("=" * 88)
        print(f"[START] Compare_nest | {key}")
        try:
            main_dir = common.resolve_main_run_dirs(MAIN_RUN_ROOT, specs, [key])[key]
            artifact = common.validate_main_artifacts(
                specs[key], main_dir, strict_script_hash=STRICT_BASE_SCRIPT_HASH
            )
            compare = _load_compare_method(artifact)
            with common.staged_dataset_dir(run_root, key) as stage:
                metric_rows = run_dataset(artifact, compare, stage)
            metrics_all.extend(metric_rows)
            manifest_rows.append(
                {
                    "dataset": key,
                    "status": "completed",
                    "analysis_role": common.dataset_role(specs[key]),
                    "output_dir": str((run_root / "datasets" / key).resolve()),
                    "message": "",
                }
            )
            print(f"[DONE] Compare_nest | {key}")
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
            print(f"[FAILED] Compare_nest | {key}: {type(exc).__name__}: {exc}")
            if not CONTINUE_ON_DATASET_ERROR:
                raise

    manifest = pd.DataFrame(manifest_rows)
    metrics_table = pd.DataFrame(metrics_all)
    common.write_csv(manifest, run_root / "run_manifest.csv")
    common.write_csv(metrics_table, run_root / "nesting_comparison_all_pre.csv")
    common.finalize_run_receipt(
        run_root,
        Path(__file__),
        started_utc,
        started_time,
        manifest,
        {
            "analysis": "four_way_nesting_diagnostic",
            "main_run_root": str(Path(MAIN_RUN_ROOT).resolve()),
            "compare_method_run_root": str(Path(COMPARE_METHOD_RUN_ROOT).resolve()),
            "datasets_to_run": DATASETS_TO_RUN,
            "selected_datasets": keys,
            "input_coverage_file": "input_coverage.csv",
            "dataset_selection_sha256": common.sha256_file(Path(selection.__file__).resolve()),
            "methods": list(METHOD_ORDER),
            "hybrids_are_valid_performance_estimators": False,
            "formal_tests": "not performed",
            "max_k": MAX_K,
            "continue_on_dataset_error": CONTINUE_ON_DATASET_ERROR,
        },
    )
    completed = int((manifest["status"] == "completed").sum())
    failed = int((manifest["status"] == "failed").sum())
    print("=" * 88)
    print(f"Compare_nest completed: {completed}; failed: {failed}")
    print(f"Output: {run_root}")
    if failed:
        raise RuntimeError(
            f"Compare_nest finished with {failed} failed dataset(s); inspect run_manifest.csv"
        )


if __name__ == "__main__":
    main()
