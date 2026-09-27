#!/usr/bin/env python3
"""Bin-wise Wilson intervals from completed primary OOF predictions; no fitting.

This read-only path also works with archived producer receipts. It preserves
recorded source identities and does not disable analysis producer-hash checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from Metrics_Calibration import _calibration_coordinates
from Revision2_reporting import Audit, PRE, inventory, load_unit, validate_pair, sha256

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_RUN_ROOT = SCRIPT_DIR / "results_revision" / "All_data_primary"
COMPARE_RUN_ROOT = SCRIPT_DIR / "results_revision" / "Compare_method_v1"
OUTPUT_DIR = SCRIPT_DIR / "results_revision" / "Revision2_calibration"
CALIBRATION_BINS = 10  # Same quantile-bin setting as the original study.
EXPECTED_REPEATS = 5


def build(main_root, compare_root, out, n_bins=10, expected_repeats=5):
    roots = [Path(main_root).expanduser().resolve(), Path(compare_root).expanduser().resolve()]
    out = Path(out).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Choose a new, empty output directory: {out}")
    if any(out == root or out in root.parents or root in out.parents for root in roots):
        raise ValueError("Output directory must be separate from source run roots")
    if isinstance(expected_repeats, bool) or int(expected_repeats) != expected_repeats or expected_repeats < 1:
        raise ValueError("expected_repeats must be a positive integer")
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    audit = Audit()
    nested_dirs = inventory(roots[0], "primary", "nested", audit)
    comparison_dirs = inventory(roots[1], "primary", "compare", audit)
    if not nested_dirs or set(nested_dirs) != set(comparison_dirs):
        raise ValueError("Supply matching nonempty primary and comparison inventories")
    selected = [key for key in PRE if key in nested_dirs]
    if not selected:
        raise ValueError("No formal pre-ICI results were supplied for calibration reporting")
    parts, provenance = [], []
    for key in selected:
        nested = load_unit(nested_dirs[key], key, "primary", True, audit, expected_repeats)
        other = load_unit(comparison_dirs[key], key, "primary", False, audit, expected_repeats)
        validate_pair(nested, other, key, audit, "primary")
        for method, unit in (("fully_nested", nested), ("fully_non_nested", other)):
            bins = _calibration_coordinates(
                unit.oof["y_true"].to_numpy(int),
                unit.oof["p_oof"].to_numpy(float), n_bins,
            )
            if int(bins["n"].sum()) != unit.metrics["n_samples"] or int(bins["n_positive"].sum()) != unit.metrics["n_positive"]:
                raise ValueError(f"Calibration bin counts disagree with saved metrics: {key}/{method}")
            bins.insert(0, "dataset", key)
            bins.insert(1, "method", method)
            bins.insert(2, "probability_variant", "raw")
            bins["dataset_event_rate"] = unit.metrics["n_positive"] / unit.metrics["n_samples"]
            parts.append(bins)
            provenance.append({
                "dataset": key, "method": method,
                "producer_sha256": unit.receipt["script_sha256"],
                "original_output_hashes": "verified" if "output_sha256" in unit.receipt else "not_recorded_by_original_producer",
            })
    table = pd.concat(parts, ignore_index=True)
    coverage = pd.DataFrame([{
        "dataset": key, "status": "included" if key in selected else "not_provided",
        "coverage_scope": "full_six" if len(selected) == 6 else "available_subset",
    } for key in PRE])
    audit.assert_unchanged()
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"calibration_wilson_bins.csv": table, "coverage.csv": coverage,
               "provenance.csv": pd.DataFrame(provenance)}
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False, encoding="utf-8-sig", float_format="%.17g")
    receipt = {
        "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_fitting_performed": False, "datasets": selected,
        "coverage_scope": "full_six" if len(selected) == 6 else "available_subset",
        "n_bins_requested": int(n_bins), "expected_repeats": int(expected_repeats),
        "input_sha256": audit.files,
        "output_sha256": {name: sha256(out / name) for name in outputs},
        "reporter_sha256": sha256(Path(__file__)),
        "calibration_module_sha256": sha256(SCRIPT_DIR / "Metrics_Calibration.py"),
        "reporting_module_sha256": sha256(SCRIPT_DIR / "Revision2_reporting.py"),
        "notes": [
            "Raw probabilities averaged by patient over the saved CV repeats.",
            "Quantile bins and Wilson score formula match the original study; no continuity correction.",
            "Intervals are bin-wise descriptive, conditional on saved predictions and bins; not simultaneous or full model-development uncertainty.",
            "Archived producer identities are retained, not compared to edited current producers.",
            "Original nested producers did not record output hashes; current hashes and numerical consistency are recorded, not retrospective cryptographic authenticity.",
            "No patient identifiers or predictions are exported.",
        ],
    }
    (out / "calibration_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed read-only calibration reporting for {len(selected)}/6 formal Pre datasets: {out}")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=MAIN_RUN_ROOT)
    parser.add_argument("--compare-root", type=Path, default=COMPARE_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-bins", type=int, default=CALIBRATION_BINS)
    parser.add_argument("--expected-repeats", type=int, default=EXPECTED_REPEATS)
    args = parser.parse_args(argv)
    build(args.main_root, args.compare_root, args.out, args.n_bins, args.expected_repeats)


if __name__ == "__main__":
    main()
