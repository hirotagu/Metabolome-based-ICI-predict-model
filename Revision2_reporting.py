#!/usr/bin/env python3
"""Read-only revision-2 summaries of completed, explicitly selected run roots.

Edit USER SETTINGS for Positron, or use --help. No raw matrix is required and no
model is fitted. The main AUC is calculated from averaged patient OOF predictions;
repeat AUCs are paired descriptive results, never independent study replicates.
Only aggregate tables are exported (no patient identifiers or predictions).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Optional

import numpy as np
import pandas as pd
import openpyxl

# USER SETTINGS: roots must identify one completed run, not a results parent.
SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_primary"
COMPARE_RUN_ROOT = SCRIPT_DIR / "results_revision" / "Compare_method_v1"
KNN_RUN_ROOT = None  # e.g. SCRIPT_DIR / "results_revision" / "All_data_KNN"
KNN_EXTENSION_ROOT = None  # Historical KNN_Revision_R4_8 results; alternative to the two roots above.
SENSITIVITY_RUN_ROOT = None  # e.g. SCRIPT_DIR / "results_revision" / "Compare_sensitivity_v1"
OUTPUT_DIR = SCRIPT_DIR / "results_revision" / "Revision2_reporting"
EXPECTED_REPEATS = 5

PRE = tuple(f"dataset{i}_pre" for i in (1, 2, 3, 4, 5, 7))
CANONICAL = tuple(f"dataset{i}_{t}" for i in range(1, 8)
                  for t in (("pre",) if i == 7 else ("pre", "post1") if i in (1, 6)
                            else ("pre", "post1", "post2")))
NOTES = [
    "Main ROC-AUC uses patient-level predictions averaged across CV repeats; it is not mean repeat ROC-AUC.",
    "Delta = non-nested ROC-AUC minus nested ROC-AUC for the same dataset and analysis population.",
    "Dataset-omission summaries and repeat distributions are descriptive, not confidence intervals.",
    "CV repeats reuse patients and global non-nested selections; repeats are not independent replicates.",
    "KNN preprocessing changes imputation and preprocessing order; this is not an isolated imputation-method effect.",
    "Dataset 6 and post-treatment matrices are excluded from the main six-cohort omission summary.",
    "Full coverage means all six intended cohorts are present; it does not establish numerical reproduction of the manuscript.",
    "Availability describes supplied saved outputs, not permission to share underlying data.",
    "Validation checks saved predictions, metrics and source links; no model fitting or raw-data audit is performed.",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal(a, b, label):
    try:
        valid = math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
            float(a), float(b), abs_tol=1e-12, rel_tol=0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Value mismatch: {label}: {a!r} versus {b!r}")


def number(value, name, low=None, high=None, integer=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value: {name}: {value!r}") from exc
    if not math.isfinite(result) or (low is not None and result < low) or (
            high is not None and result > high) or (integer and result != int(result)):
        raise ValueError(f"Invalid numeric value: {name}: {value!r}")
    return int(result) if integer else result


def numeric(frame, columns, low=None, high=None, integer=False):
    for col in columns:
        if col not in frame:
            raise ValueError(f"Missing column {col}")
        frame[col] = [number(v, col, low, high, integer) for v in frame[col]]


def require(frame, columns, context):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing columns in {context}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Empty table is not an absent dataset: {context}")


def unique(frame, cols, context):
    if frame.duplicated(cols).any():
        raise ValueError(f"Duplicate {cols}: {context}")


def auc(y, probability):
    """Mann-Whitney AUC with average ranks for tied predictions."""
    y = np.asarray(y, dtype=int)
    ranks = pd.Series(np.asarray(probability, dtype=float)).rank(method="average").to_numpy()
    positive = int(y.sum())
    negative = len(y) - positive
    if positive <= 0 or negative <= 0:
        raise ValueError("Both outcome classes are required for ROC-AUC")
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


class Audit:
    def __init__(self):
        self.files = {}

    def path(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Declared artifact is missing: {path}")
        digest = sha256(path)
        if str(path) in self.files and self.files[str(path)] != digest:
            raise ValueError(f"Artifact changed while reporting: {path}")
        self.files[str(path)] = digest
        return path

    def json(self, path):
        value = json.loads(self.path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return value

    def csv(self, path):
        # dtype=str preserves patient IDs, including leading zeros.
        return pd.read_csv(self.path(path), dtype=str, keep_default_na=False)

    def verify_declared(self, directory, receipt, filenames, required):
        hashes = receipt.get("output_sha256")
        if hashes is None and not required:
            return  # Original nested producer predates output hash manifests.
        if not isinstance(hashes, dict):
            raise ValueError(f"Missing producer output hashes: {directory}")
        for filename in filenames:
            path = self.path(directory / filename)
            if hashes.get(filename) != self.files[str(path)]:
                raise ValueError(f"Producer hash mismatch or missing hash: {path}")

    def assert_unchanged(self):
        for path, expected in self.files.items():
            if sha256(path) != expected:
                raise ValueError(f"Input changed during reporting: {path}")


def inventory(root, scenario, kind, audit):
    root = Path(root).expanduser().resolve()
    frame = audit.csv(root / "run_manifest.csv")
    require(frame, ["dataset", "output_dir"], root)
    if "scenario" in frame:
        unique(frame, ["dataset", "scenario"], root)
        frame = frame.loc[frame.scenario == scenario]
    elif kind != "compare":
        raise ValueError(f"Scenario column missing from {root}")
    else:
        unique(frame, ["dataset"], root)
    result = {}
    for row in frame.to_dict("records"):
        key = row["dataset"]
        if key not in CANONICAL:
            raise ValueError(f"Unknown dataset key {key!r} in {root}")
        if row.get("status", "completed") != "completed":
            raise ValueError(f"Manifest contains incomplete or failed {key}: {root}")
        if kind == "compare":
            directory = root / "datasets" / key
        elif kind == "sensitivity":
            directory = root / "datasets" / scenario / key
        elif scenario == "primary":
            section = "supplement/dataset6_small_sample" if key.startswith("dataset6_") else "main"
            directory = root / section / scenario / key
        else:
            directory = root / "supplement/sensitivity" / scenario / key
        # Recorded absolute paths may refer to the original machine. The fixed
        # layout plus source-receipt hashes below makes relocation explicit/safe.
        if not directory.is_dir():
            raise FileNotFoundError(f"Manifest-declared output directory is missing: {directory}")
        result[key] = directory
    return result


@dataclass
class SavedUnit:
    directory: Path
    receipt: dict
    settings: dict
    metrics: dict
    oof: pd.DataFrame
    predictions: pd.DataFrame
    repeats: pd.DataFrame
    id_col: str


def load_unit(directory, key, scenario, nested, audit, expected_repeats):
    receipt = audit.json(directory / "run_receipt.json")
    settings = audit.json(directory / "settings_used.json")
    if receipt.get("status") != "completed" or receipt.get("dataset") != key:
        raise ValueError(f"Incomplete or mismatched receipt: {directory}")
    if nested and receipt.get("scenario") != scenario:
        raise ValueError(f"Wrong nested scenario: {directory}")
    if not nested and receipt.get("method") != (
            "fully_non_nested" if scenario == "primary" else "fully_non_nested_sensitivity_compare"):
        raise ValueError(f"Wrong comparison method: {directory}")
    if not nested and scenario != "primary" and receipt.get("scenario") != scenario:
        raise ValueError(f"Wrong sensitivity scenario: {directory}")
    if nested:
        if settings.get("scenario", {}).get("name") != scenario:
            raise ValueError(f"Mismatched scenario settings: {directory}")
        equal(settings.get("outer_repeats"), expected_repeats, f"{key}/configured repeats")
    else:
        equal(settings.get("selection_and_evaluation_cv", {}).get("repeats"), expected_repeats,
              f"{key}/comparison configured repeats")
    if not receipt.get("input_sha256") or not receipt.get("script_sha256"):
        raise ValueError(f"Missing input/producer identity: {directory}")
    filenames = ["settings_used.json", "metrics_summary.csv", "oof_predictions.csv",
                 "repeat_metrics.csv", "outer_predictions_long.csv"]
    audit.verify_declared(directory, receipt, filenames, required=not nested)
    metrics_frame = audit.csv(directory / "metrics_summary.csv")
    require(metrics_frame, ["dataset", "n_samples", "n_positive", "n_negative"], directory)
    if len(metrics_frame) != 1 or metrics_frame.iloc[0]["dataset"] != key:
        raise ValueError(f"Non-unique or mismatched metric row: {directory}")
    metrics = metrics_frame.iloc[0].to_dict()
    for field in ("n_samples", "n_positive", "n_negative"):
        metrics[field] = number(metrics[field], field, low=1, integer=True)
    if metrics["n_samples"] != metrics["n_positive"] + metrics["n_negative"]:
        raise ValueError(f"Class-count mismatch: {directory}")
    oof = audit.csv(directory / "oof_predictions.csv")
    predictions = audit.csv(directory / "outer_predictions_long.csv")
    repeats = audit.csv(directory / "repeat_metrics.csv")
    id_col = receipt.get("id_column", "id")
    require(oof, [id_col, "y_true", "p_oof", "oof_count"], directory)
    require(predictions, [id_col, "repeat", "outer_fold", "y_true", "p"], directory)
    require(repeats, ["dataset", "repeat", "n", "positive", "negative", "roc_auc"], directory)
    if (oof[id_col].str.strip() == "").any() or (predictions[id_col].str.strip() == "").any():
        raise ValueError(f"Blank patient identifier: {directory}")
    unique(oof, [id_col], directory)
    numeric(oof, ["y_true"], 0, 1, True)
    numeric(oof, ["oof_count"], 1, integer=True)
    numeric(oof, ["p_oof"], 0, 1)
    numeric(predictions, ["y_true"], 0, 1, True)
    numeric(predictions, ["repeat", "outer_fold"], 1, integer=True)
    numeric(predictions, ["p"], 0, 1)
    numeric(repeats, ["repeat", "n", "positive", "negative"], 1, integer=True)
    numeric(repeats, ["roc_auc"], 0, 1)
    unique(predictions, ["repeat", id_col], directory)
    unique(repeats, ["repeat"], directory)
    if nested and "scenario" in repeats and set(repeats.scenario) != {scenario}:
        raise ValueError(f"Mismatched repeat scenario: {directory}")
    if not nested and "method" in repeats and set(repeats.method) != {"fully_non_nested"}:
        raise ValueError(f"Mismatched repeat method: {directory}")
    if set(repeats.dataset) != {key}:
        raise ValueError(f"Mismatched repeat dataset labels: {directory}")
    expected = set(range(1, expected_repeats + 1))
    if set(repeats.repeat) != expected or set(predictions.repeat) != expected:
        raise ValueError(f"Missing or unexpected CV repeats: {directory}")
    if len(oof) != metrics["n_samples"] or int(oof.y_true.sum()) != metrics["n_positive"]:
        raise ValueError(f"OOF class-count mismatch: {directory}")
    if not (oof.oof_count == expected_repeats).all():
        raise ValueError(f"Incomplete averaged OOF predictions: {directory}")
    patients = oof.set_index(id_col).sort_index()
    for repeat, part in predictions.groupby("repeat"):
        part = part.set_index(id_col).sort_index()
        if not part.index.equals(patients.index) or not np.array_equal(part.y_true, patients.y_true):
            raise ValueError(f"Repeat patient/label mismatch: {directory}, repeat {repeat}")
        saved = repeats.loc[repeats.repeat == repeat].iloc[0]
        for col, val in [("n", len(part)), ("positive", int(part.y_true.sum())),
                         ("negative", int((part.y_true == 0).sum()))]:
            equal(saved[col], val, f"{key}/repeat {repeat}/{col}")
        equal(saved.roc_auc, auc(part.y_true, part.p), f"{key}/repeat {repeat}/AUC")
    means = predictions.groupby(id_col).p.mean().sort_index()
    if not np.allclose(means, patients.p_oof, atol=1e-12, rtol=0):
        raise ValueError(f"OOF probabilities are not repeat averages: {directory}")
    auc_field = "roc_auc_oof" if nested else "nonnested_roc_auc"
    equal(metrics.get(auc_field), auc(patients.y_true, patients.p_oof), f"{key}/{auc_field}")
    return SavedUnit(directory, receipt, settings, metrics, oof, predictions, repeats, id_col)


def validate_pair(nested, nonnested, key, audit, scenario):
    field = "main_run_receipt_sha256" if scenario == "primary" else "nested_source_receipt_sha256"
    if nonnested.settings.get(field) != sha256(nested.directory / "run_receipt.json"):
        raise ValueError(f"Nested/comparison source-receipt mismatch: {key}/{scenario}")
    if nested.receipt["input_sha256"] != nonnested.receipt["input_sha256"]:
        raise ValueError(f"Nested/comparison input mismatch: {key}/{scenario}")
    cv = nonnested.settings.get("selection_and_evaluation_cv", {})
    for a, b in [("splits", "outer_splits_used"), ("repeats", "outer_repeats"), ("random_state", "random_state")]:
        equal(cv.get(a), nested.settings.get(b), f"{key}/paired CV setting {a}")
    if nested.id_col != nonnested.id_col:
        raise ValueError(f"Patient ID schema differs: {key}/{scenario}")
    cols = ["repeat", nested.id_col, "outer_fold", "y_true"]
    a = nested.predictions[cols].sort_values(cols[:2]).reset_index(drop=True)
    b = nonnested.predictions[cols].sort_values(cols[:2]).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError(f"Nested/non-nested patients, labels, or CV folds differ: {key}/{scenario}")
    for field in ("n_samples", "n_positive", "n_negative"):
        equal(nested.metrics[field], nonnested.metrics[field], f"{key}/{field}")
    equal(nested.metrics["roc_auc_oof"], nonnested.metrics["nested_roc_auc"], f"{key}/nested AUC")
    delta = float(nonnested.metrics["nonnested_roc_auc"]) - float(nonnested.metrics["nested_roc_auc"])
    equal(delta, nonnested.metrics["delta_auc_nonnested_minus_nested"], f"{key}/delta AUC")
    return delta


def omission_summaries(individual):
    """Unweighted cohort summaries; absent cohorts never contribute zeros."""
    individual = individual.copy()
    require(individual, ["dataset", "delta_auc"], "individual results")
    unique(individual, ["dataset"], "individual results")
    if not set(individual.dataset) <= set(PRE):
        raise ValueError("Only intended main Pre cohorts can enter omission summaries")
    numeric(individual, ["delta_auc"], -1, 1)
    values = dict(zip(individual.dataset, individual.delta_auc))
    present = [key for key in PRE if key in values]
    scope = "all_six_main_cohorts" if len(present) == len(PRE) else "available_subset"
    scenarios = [("all_available", "", present)]
    scenarios.extend(("omit_one", key, [k for k in present if k != key]) for key in present)
    scenarios.append(("lcms_only", "dataset7_pre" if "dataset7_pre" in present else "",
                      [k for k in present if k != "dataset7_pre"]))
    rows = []
    for summary, excluded, keys in scenarios:
        v = np.array([values[k] for k in keys])
        rows.append({"summary": summary, "excluded_dataset": excluded,
                     "source_scope": scope, "n_source_cohorts": len(present),
                     "n_retained": len(keys), "retained_datasets": ";".join(keys),
                     "status": "available" if len(keys) else "no_retained_cohorts",
                     "mean_delta": float(v.mean()) if len(v) else np.nan,
                     "median_delta": float(np.median(v)) if len(v) else np.nan,
                     "minimum_delta": float(v.min()) if len(v) else np.nan,
                     "maximum_delta": float(v.max()) if len(v) else np.nan})
    return pd.DataFrame(rows)


def missingness(unit, key, audit):
    frame = audit.csv(unit.directory / "feature_manifest.csv")
    require(frame, ["feature", "included_in_analysis", "n_missing_after_rules"], unit.directory)
    unique(frame, ["feature"], unit.directory)
    flags = frame.included_in_analysis.str.lower()
    if not set(flags) <= {"true", "false"}:
        raise ValueError(f"Invalid included_in_analysis flag: {unit.directory}")
    numeric(frame, ["n_missing_after_rules"], 0, unit.metrics["n_samples"], True)
    kept = frame.loc[flags == "true"]
    if (frame.loc[flags == "false", "n_missing_after_rules"] != unit.metrics["n_samples"]).any():
        raise ValueError(f"Excluded feature is not all missing: {unit.directory}")
    if (frame.feature.str.strip() == "").any():
        raise ValueError(f"Blank feature name: {unit.directory}")
    if "dataset" in frame and set(frame.dataset) != {key}:
        raise ValueError(f"Feature-manifest dataset mismatch: {unit.directory}")
    if kept.empty:
        raise ValueError(f"No retained features: {unit.directory}")
    n = unit.metrics["n_samples"]
    return {"dataset": key, "n_samples": n, "input_features": len(frame),
            "all_missing_excluded": len(frame) - len(kept), "remaining_features": len(kept),
            "missing_cells_input": int(frame.n_missing_after_rules.sum()),
            "missing_cells_remaining": int(kept.n_missing_after_rules.sum()),
            "missing_rate_input": float(frame.n_missing_after_rules.sum() / (n * len(frame))),
            "missing_rate_remaining": float(kept.n_missing_after_rules.sum() / (n * len(kept))),
            "remaining_features_with_missing": int((kept.n_missing_after_rules > 0).sum())}


def extension_inventory(root, audit):
    """Explicit adapter for the historical R4.8 extension; never auto-detected."""
    root = Path(root).resolve()
    contract = audit.json(root / "run_contract.json")
    if not str(contract.get("owner", "")).startswith("KNN_Revision_R4_8_"):
        raise ValueError("Unrecognized historical KNN extension contract")
    scope = contract.get("scope", [])
    new_scope = contract.get("new_fit_scope", [])
    if not scope or len(scope) != len(set(scope)) or not set(scope) <= set(PRE) or not set(new_scope) <= set(scope):
        raise ValueError("Invalid historical KNN contract scope")
    summary = audit.json(root / "summary/summary_receipt.json")
    if summary.get("status") != "completed" or summary.get("completed_datasets") != len(scope):
        raise ValueError("Incomplete historical KNN extension summary")
    files = ["knn_comparison.csv", "repeat_comparison.csv", "unit_status.csv"]
    audit.verify_declared(root / "summary", summary, files, required=True)
    states = audit.csv(root / "summary/unit_status.csv")
    require(states, ["dataset", "status"], root)
    unique(states, ["dataset"], root)
    if set(states.dataset) != set(scope):
        raise ValueError("KNN extension status/contract scope mismatch")
    knn, comparison, origins = {}, {}, {}
    for row in states.to_dict("records"):
        key = row["dataset"]
        origin = "new_r4_8_sensitivity_fit" if key in new_scope else "reused_from_first_revision_no_model_fit"
        if row["status"] != ("PASS_NEW" if key in new_scope else "PASS_REUSED"):
            raise ValueError(f"Incomplete KNN extension unit: {key}")
        for phase in ("nested", "comparison"):
            base = root / "units" / key / phase
            receipt = audit.json(base / "extension_unit_receipt.json")
            for field, expected in [("dataset", key), ("phase", phase), ("status", "completed"),
                                    ("fingerprint", contract.get("fingerprint")), ("owner", contract["owner"]), ("origin", origin)]:
                if receipt.get(field) != expected:
                    raise ValueError(f"KNN extension {field} mismatch: {base}")
            unit = base / "supplement/sensitivity/knn" / key if phase == "nested" else base
            for name in ("run_receipt.json", "settings_used.json", "metrics_summary.csv", "oof_predictions.csv",
                         "repeat_metrics.csv", "outer_predictions_long.csv"):
                path = audit.path(unit / name)
                if receipt.get("files", {}).get(path.relative_to(base).as_posix()) != sha256(path):
                    raise ValueError(f"KNN extension artifact hash mismatch: {path}")
            (knn if phase == "nested" else comparison)[key] = unit
        origins[key] = origin
    return knn, comparison, origins


def build(main_root, compare_root, out, knn_root=None, sensitivity_root=None, expected_repeats=5,
          knn_extension_root=None):
    if knn_extension_root is not None and (knn_root is not None or sensitivity_root is not None):
        raise ValueError("Choose native KNN roots or --knn-extension-root, never both")
    knn_requested = knn_root is not None or knn_extension_root is not None
    if (knn_root is None) != (sensitivity_root is None):
        raise ValueError("Supply both --knn-root and --sensitivity-root, or neither")
    expected_repeats = number(expected_repeats, "expected repeats", 1, integer=True)
    out = Path(out).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Choose a new, empty output directory: {out}")
    roots = {"main": Path(main_root).resolve(), "compare": Path(compare_root).resolve()}
    if knn_root is not None:
        roots.update(knn=Path(knn_root).resolve(), sensitivity=Path(sensitivity_root).resolve())
    if knn_extension_root is not None:
        roots["knn_extension"] = Path(knn_extension_root).resolve()
    if any(out == root or out in root.parents or root in out.parents for root in roots.values()):
        raise ValueError("Output directory must be separate from every source run root")
    audit = Audit()
    main = inventory(roots["main"], "primary", "nested", audit)
    compare = inventory(roots["compare"], "primary", "compare", audit)
    if not main or not compare:
        raise ValueError("No completed primary/comparison datasets in the supplied run roots")
    if set(compare) != set(main):
        raise ValueError("Primary and comparison inventories differ; supply matching completed run roots")
    knn = inventory(roots["knn"], "knn", "nested", audit) if knn_root else {}
    sens = inventory(roots["sensitivity"], "knn", "sensitivity", audit) if knn_root else {}
    if knn_root is not None and (not knn or not sens):
        raise ValueError("Supplied KNN roots contain no completed KNN scenario units")
    origins = {}
    if knn_extension_root is not None:
        knn, sens, origins = extension_inventory(roots["knn_extension"], audit)
    if set(sens) != set(knn) or not set(sens) <= set(compare):
        raise ValueError("KNN comparison lacks matching nested or primary artifacts")
    if not (set(knn) | set(sens)) <= set(PRE):
        raise ValueError("KNN scope must be a subset of the six intended Pre cohorts")
    units = {}
    provenance = []
    feature_rows = []
    individual = []
    repeat_rows = []
    knn_rows = []
    for key in CANONICAL:
        if key not in main:
            continue
        nested = load_unit(main[key], key, "primary", True, audit, expected_repeats)
        units[key] = nested
        feature_rows.append(missingness(nested, key, audit))
        provenance.append({"dataset": key, "analysis": "primary_nested", "producer_sha256": nested.receipt["script_sha256"],
                           "producer_output_hashes": "available" if "output_sha256" in nested.receipt else "not_recorded_by_original_producer"})
        if key not in compare:
            continue
        nonnested = load_unit(compare[key], key, "primary", False, audit, expected_repeats)
        delta = validate_pair(nested, nonnested, key, audit, "primary")
        provenance.append({"dataset": key, "analysis": "primary_nonnested", "producer_sha256": nonnested.receipt["script_sha256"], "producer_output_hashes": "verified"})
        if key not in PRE:
            continue
        row = {"dataset": key, **{f: nested.metrics[f] for f in ("n_samples", "n_positive", "n_negative")},
               "nested_auc": float(nonnested.metrics["nested_roc_auc"]),
               "nonnested_auc": float(nonnested.metrics["nonnested_roc_auc"]), "delta_auc": delta}
        individual.append(row)
        saved_repeats = {}
        for repeat in range(1, expected_repeats + 1):
            a = nested.repeats.set_index("repeat").loc[repeat, "roc_auc"]
            b = nonnested.repeats.set_index("repeat").loc[repeat, "roc_auc"]
            rr = {"dataset": key, "n_samples": row["n_samples"], "repeat": repeat,
                  "primary_nested_auc": float(a), "primary_nonnested_auc": float(b),
                  "primary_delta_auc": float(b - a), "knn_nested_auc": np.nan,
                  "knn_nonnested_auc": np.nan, "knn_delta_auc": np.nan,
                  "knn_status": "not_requested" if not knn_requested else "not_available"}
            repeat_rows.append(rr)
            saved_repeats[repeat] = rr
        if key not in sens:
            continue
        kn = load_unit(knn[key], key, "knn", True, audit, expected_repeats)
        sn = load_unit(sens[key], key, "knn", False, audit, expected_repeats)
        kd = validate_pair(kn, sn, key, audit, "knn")
        cols = ["repeat", nested.id_col, "outer_fold", "y_true"]
        if kn.id_col != nested.id_col or not kn.predictions[cols].sort_values(cols[:2]).reset_index(drop=True).equals(
                nested.predictions[cols].sort_values(cols[:2]).reset_index(drop=True)):
            raise ValueError(f"KNN/primary population, labels or CV folds differ: {key}")
        for field, val in [("primary_nested_roc_auc", row["nested_auc"]),
                           ("primary_nonnested_roc_auc", row["nonnested_auc"]),
                           ("primary_delta_auc_nonnested_minus_nested", delta)]:
            equal(sn.metrics.get(field), val, f"{key}/KNN reference {field}")
        kr = {"dataset": key, "sensitivity_setting": "KNN preprocessing", "n_samples": row["n_samples"],
              "n_positive": row["n_positive"], "n_negative": row["n_negative"],
              "primary_delta_auc": delta, "knn_nested_auc": float(sn.metrics["nested_roc_auc"]),
              "knn_nonnested_auc": float(sn.metrics["nonnested_roc_auc"]),
              "knn_delta_auc": kd, "change_delta_auc": kd - delta}
        for source, dest in [("nested_average_precision", "knn_nested_ap"), ("nonnested_average_precision", "knn_nonnested_ap"),
                             ("nested_brier", "knn_nested_brier"), ("nonnested_brier", "knn_nonnested_brier")]:
            kr[dest] = number(sn.metrics.get(source), source, 0, 1)
        knn_rows.append(kr)
        for repeat in range(1, expected_repeats + 1):
            a = float(kn.repeats.set_index("repeat").loc[repeat, "roc_auc"])
            b = float(sn.repeats.set_index("repeat").loc[repeat, "roc_auc"])
            saved_repeats[repeat].update(knn_nested_auc=a, knn_nonnested_auc=b, knn_delta_auc=b - a, knn_status="available")
        for unit, analysis in [(kn, "knn_nested"), (sn, "knn_nonnested")]:
            provenance.append({"dataset": key, "analysis": analysis + ("/" + origins[key] if key in origins else ""), "producer_sha256": unit.receipt["script_sha256"],
                               "producer_output_hashes": "available" if "output_sha256" in unit.receipt else "not_recorded_by_original_producer"})
    if not individual:
        raise ValueError("No paired main Pre cohort available for revision-2 reporting")
    # Mixing producer identities within the same run/scenario is not automatic.
    for analysis in {row["analysis"] for row in provenance}:
        if len({row["producer_sha256"] for row in provenance if row["analysis"] == analysis}) != 1:
            raise ValueError(f"Mixed producer versions in {analysis}; supply a coherent run root")
    present = {row["dataset"] for row in individual}
    coverage = pd.DataFrame([{"dataset": key, "main_pre_cohort": key in PRE,
                              "nested_status": "available" if key in main else "not_available",
                              "comparison_status": "available" if key in compare else "not_available",
                              "knn_status": "not_applicable" if key not in PRE else "not_requested" if not knn_requested
                              else "available" if key in sens else "nested_only" if key in knn else "not_available"}
                             for key in CANONICAL])
    tables = {"README": pd.DataFrame({"notes": NOTES}), "Coverage": coverage,
              "Individual_main_pre": pd.DataFrame(individual),
              "Dataset_omission": omission_summaries(pd.DataFrame(individual)),
              "Paired_repeats": pd.DataFrame(repeat_rows),
              "KNN_preprocessing": pd.DataFrame(knn_rows, columns=["dataset", "sensitivity_setting", "n_samples", "n_positive", "n_negative",
                    "primary_delta_auc", "knn_nested_auc", "knn_nonnested_auc", "knn_delta_auc", "change_delta_auc",
                    "knn_nested_ap", "knn_nonnested_ap", "knn_nested_brier", "knn_nonnested_brier"]),
              "Missingness": pd.DataFrame(feature_rows), "Provenance": pd.DataFrame(provenance)}
    audit.assert_unchanged()
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "Revision2_reporting.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(64, max(12, max(len(str(c.value or "")) for c in column) + 2))
    receipt = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "model_fitting_performed": False,
               "primary_scope": "all_six_main_cohorts" if present == set(PRE) else "available_subset",
               "expected_main_pre": list(PRE), "available_main_pre": [k for k in PRE if k in present],
               "missing_main_pre": [k for k in PRE if k not in present],
               "knn_scope": "not_requested" if not knn_requested else "all_six_main_cohorts" if set(sens) == set(PRE) else "available_subset",
               "available_knn_pre": [k for k in PRE if k in sens],
               "settings": {"source_roots": {k: str(v) for k, v in roots.items()}, "expected_repeats": expected_repeats,
                            "delta_definition": "non-nested minus nested", "auc_unit": "averaged patient OOF prediction"},
               "script_sha256": sha256(Path(__file__)), "input_sha256": audit.files,
               "output_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "openpyxl": openpyxl.__version__},
               "notes": NOTES}
    (out / "reporting_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=MAIN_RUN_ROOT)
    parser.add_argument("--compare-root", type=Path, default=COMPARE_RUN_ROOT)
    parser.add_argument("--knn-root", type=Path, default=KNN_RUN_ROOT)
    parser.add_argument("--sensitivity-root", type=Path, default=SENSITIVITY_RUN_ROOT)
    parser.add_argument("--knn-extension-root", type=Path, default=KNN_EXTENSION_ROOT,
                        help="Explicit historical KNN_Revision_R4_8 output root (alternative to native KNN roots)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--expected-repeats", type=int, default=EXPECTED_REPEATS)
    args = parser.parse_args(argv)
    try:
        receipt = build(args.main_root, args.compare_root, args.out, args.knn_root,
                        args.sensitivity_root, args.expected_repeats, args.knn_extension_root)
    except Exception as exc:
        parser.exit(1, f"[STOP] {type(exc).__name__}: {exc}\n")
    print(f"[DONE] {args.out}\n[SCOPE] {receipt['primary_scope']}; "
          f"{len(receipt['available_main_pre'])}/6 main Pre cohorts; no model fitting")


if __name__ == "__main__":
    main()
