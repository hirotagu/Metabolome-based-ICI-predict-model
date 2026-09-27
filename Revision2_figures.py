#!/usr/bin/env python3
"""Render current Figure S2 from validated aggregate calibration bins only.

Optional dependency: install requirements-figures.txt. No patient data, model
fitting or statistical estimation is used. Edit USER SETTINGS for Positron or
run with --input-dir, --out and optionally --font/--formats/--dpi.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
except ImportError as exc:
    raise ImportError("Figure rendering needs requirements-figures.txt; install it in your plotting environment") from exc

# USER SETTINGS
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results_revision" / "Revision2_calibration"
OUTPUT_DIR = SCRIPT_DIR / "results_revision" / "Revision2_figures"
FONT_FAMILY = "Arial"  # Must be installed; select another explicitly if needed.
FIGURE_FORMATS = ("png", "pdf", "svg")
RASTER_DPI = 600
FONT_SIZE = 10.0
AXIS_LABEL_SIZE = 13.0
TICK_SIZE = 13.0
LEGEND_SIZE = 13.0
ANNOTATION_SIZE = 10.0
MARKER_SIZE = 6.2
LINE_WIDTH = 1.35
CI_WIDTH = 0.65
CI_CAPSIZE = 1.6
CI_ALPHA = 0.55
PRE = tuple(f"dataset{i}_pre" for i in (1, 2, 3, 4, 5, 7))
METHODS = ("fully_nested", "fully_non_nested")
SUPPORTED_FORMATS = {"png", "pdf", "svg"}
SOURCE_FILES = ("calibration_wilson_bins.csv", "coverage.csv", "provenance.csv")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Empty source table: {path}")
    return frame


def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {label} columns: {sorted(missing)}")


def numeric(frame, column, low, high=None, integer=False):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < low).any()
            or (high is not None and (values > high).any())
            or (integer and not np.equal(values, np.floor(values)).all())):
        raise ValueError(f"Invalid numeric values in {column}")
    frame[column] = values.astype(int) if integer else values


def load_calibration(input_dir):
    root = Path(input_dir).expanduser().resolve()
    receipt_path = root / "calibration_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or receipt.get("model_fitting_performed") is not False:
        raise ValueError("Require a completed read-only calibration receipt")
    recorded = receipt.get("output_sha256", {})
    fingerprints = {receipt_path.name: sha256(receipt_path)}
    for name in SOURCE_FILES:
        path = root / name
        actual = sha256(path)
        if recorded.get(name) != actual:
            raise ValueError(f"Calibration artifact hash mismatch: {name}")
        fingerprints[name] = actual
    keys = receipt.get("datasets", [])
    if (not isinstance(keys, list) or not keys or len(set(keys)) != len(keys)
            or not set(keys) <= set(PRE)):
        raise ValueError("Invalid calibration dataset coverage")
    keys = [key for key in PRE if key in keys]
    scope = "full_six" if len(keys) == 6 else "available_subset"
    if receipt.get("coverage_scope") != scope:
        raise ValueError("Receipt coverage scope disagrees with datasets")
    coverage = read_csv(root / "coverage.csv")
    require(coverage, ["dataset", "status", "coverage_scope"], "coverage")
    if coverage.dataset.duplicated().any() or set(coverage.dataset) != set(PRE):
        raise ValueError("Coverage must contain each intended Pre cohort once")
    if not set(coverage.status) <= {"included", "not_provided"} or set(coverage.coverage_scope) != {scope}:
        raise ValueError("Invalid coverage table status/scope")
    if set(coverage.loc[coverage.status == "included", "dataset"]) != set(keys):
        raise ValueError("Coverage table disagrees with receipt")
    table = read_csv(root / "calibration_wilson_bins.csv")
    columns = ["dataset", "method", "probability_variant", "bin", "n", "n_positive", "predicted_mean",
               "observed_rate", "observed_rate_wilson95_low", "observed_rate_wilson95_high", "dataset_event_rate"]
    require(table, columns, "calibration bins")
    if set(table.dataset) != set(keys) or set(table.method) != set(METHODS) or set(table.probability_variant) != {"raw"}:
        raise ValueError("Unexpected dataset, method or probability variant")
    for column in ("bin", "n"):
        numeric(table, column, 1, integer=True)
    numeric(table, "n_positive", 0, integer=True)
    for column in ("predicted_mean", "observed_rate", "observed_rate_wilson95_low", "observed_rate_wilson95_high", "dataset_event_rate"):
        numeric(table, column, 0, 1)
    if table.duplicated(["dataset", "method", "bin"]).any():
        raise ValueError("Duplicate calibration dataset/method/bin")
    if (table.n_positive > table.n).any() or not np.allclose(table.observed_rate, table.n_positive / table.n, rtol=0, atol=1e-12):
        raise ValueError("Calibration bin counts disagree with observed proportions")
    lower, upper, observed = table.observed_rate_wilson95_low, table.observed_rate_wilson95_high, table.observed_rate
    if ((lower > upper) | (lower > observed + 1e-12) | (upper < observed - 1e-12)).any():
        raise ValueError("Invalid calibration interval bounds")
    totals = {}
    for key in keys:
        for method in METHODS:
            part = table.loc[(table.dataset == key) & (table.method == method)]
            if part.empty or set(part.bin) != set(range(1, len(part) + 1)):
                raise ValueError(f"Missing method or calibration bin: {key}/{method}")
            n, positive = int(part.n.sum()), int(part.n_positive.sum())
            if not 0 < positive < n or not np.allclose(part.dataset_event_rate, positive / n, rtol=0, atol=1e-12):
                raise ValueError(f"Invalid total outcome counts/event rate: {key}/{method}")
            if key in totals and totals[key] != (n, positive):
                raise ValueError(f"Methods have different dataset counts: {key}")
            totals[key] = (n, positive)
    provenance = read_csv(root / "provenance.csv")
    require(provenance, ["dataset", "method", "producer_sha256"], "provenance")
    if (provenance.duplicated(["dataset", "method"]).any()
            or set(zip(provenance.dataset, provenance.method)) != {(k, m) for k in keys for m in METHODS}):
        raise ValueError("Provenance method/dataset coverage disagrees with bins")
    return table, keys, totals, scope, fingerprints


def make_figure(table, keys, totals, scope, font_family):
    font_path = font_manager.findfont(font_manager.FontProperties(family=[font_family]), fallback_to_default=False)
    actual_font = font_manager.FontProperties(fname=font_path).get_name()
    if actual_font.casefold() != font_family.casefold():
        raise ValueError(f"Font {font_family!r} resolved to {actual_font!r}; choose an installed font explicitly")
    cols = 2 if len(keys) == 4 else min(3, len(keys))
    rows = math.ceil(len(keys) / cols)
    figsize = (9.2 if cols == 3 else 6.4 if cols == 2 else 3.8, 6.2 if rows == 2 else 3.9)
    with plt.rc_context({"font.family": actual_font, "font.size": FONT_SIZE, "axes.labelsize": AXIS_LABEL_SIZE,
                         "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
                         "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        specs = ((METHODS[0], "o", "-", "#D62728"), (METHODS[1], "s", "--", "#111111"))
        for index, key in enumerate(keys):
            row, col = divmod(index, cols)
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color="#B8B8B8", linestyle=":", linewidth=1.0, zorder=1)
            for method, marker, linestyle, color in specs:
                part = table.loc[(table.dataset == key) & (table.method == method)].sort_values("bin")
                y = part.observed_rate.to_numpy()
                ax.errorbar(part.predicted_mean, y,
                            yerr=np.vstack([np.maximum(0, y - part.observed_rate_wilson95_low),
                                            np.maximum(0, part.observed_rate_wilson95_high - y)]),
                            fmt="none", ecolor=color, elinewidth=CI_WIDTH, capsize=CI_CAPSIZE,
                            capthick=CI_WIDTH, alpha=CI_ALPHA, zorder=2, clip_on=False)
                ax.plot(part.predicted_mean, y, color=color, linestyle=linestyle, linewidth=LINE_WIDTH,
                        marker=marker, markersize=MARKER_SIZE - 1.0, clip_on=False, zorder=3)
            n, positive = totals[key]
            ax.set_title("Dataset " + key.split("_")[0].removeprefix("dataset"), pad=21)
            ax.text(.5, 1.025, f"Positive: {positive}/{n} ({100 * positive / n:.1f}%)",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Observed event rate")
            if row == rows - 1:
                ax.set_xlabel("Mean predicted probability")
        for ax in axes.flat[len(keys):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=MARKER_SIZE,
                          label=label) for label, marker, color in (("nCV", "o", "#D62728"), ("Non-nested CV", "s", "#111111"))]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .995))
        if scope != "full_six":
            fig.text(.5, .015, f"Available subset: {len(keys)}/6 intended Pre cohorts", ha="center", va="bottom", fontsize=9)
        fig.subplots_adjust(top=.84 if rows == 2 else .75, bottom=.12 if scope == "full_six" else .20,
                            left=.10 if cols == 3 else .13 if cols == 2 else .20,
                            right=.98, hspace=.60, wspace=.24 if cols == 3 else .35)
    return fig, actual_font, font_path, figsize


def build(input_dir, out, formats=FIGURE_FORMATS, dpi=RASTER_DPI, font=FONT_FAMILY):
    root, out = Path(input_dir).expanduser().resolve(), Path(out).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Choose a new empty output directory: {out}")
    if out == root or out in root.parents or root in out.parents:
        raise ValueError("Output directory must be separate from input")
    formats = tuple(formats)
    if not formats or len(formats) != len(set(formats)) or not set(formats) <= SUPPORTED_FORMATS:
        raise ValueError(f"Select unique supported formats: {sorted(SUPPORTED_FORMATS)}")
    if isinstance(dpi, bool) or int(dpi) != dpi or dpi < 50 or dpi > 1200:
        raise ValueError("dpi must be an integer from 50 to 1200")
    table, keys, totals, scope, fingerprints = load_calibration(root)
    try:
        fig, actual_font, font_path, figsize = make_figure(table, keys, totals, scope, font)
    except ValueError as exc:
        if "font" in str(exc).lower():
            raise ValueError(f"Requested font {font!r} unavailable. Choose an installed family with --font (e.g. 'DejaVu Sans').") from exc
        raise
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".calibration_figures_", dir=out.parent))
    try:
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
            for ext in formats:
                fig.savefig(stage / f"FigureS2_calibration.{ext}", dpi=dpi, facecolor="white")
        for name, digest in fingerprints.items():
            if sha256(root / name) != digest:
                raise ValueError(f"Calibration input changed during rendering: {name}")
        receipt = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "datasets": keys, "coverage_scope": scope, "model_fitting_performed": False,
                   "statistical_estimation_performed": False, "input_sha256": fingerprints,
                   "source_module_sha256": {Path(__file__).name: sha256(Path(__file__))},
                   "output_sha256": {p.name: sha256(p) for p in sorted(stage.iterdir())},
                   "settings": {"formats": list(formats), "dpi": int(dpi), "font_requested": font,
                                "font_used": actual_font, "font_sha256": sha256(font_path),
                                "figure_size_inches": list(figsize), "font_size": FONT_SIZE,
                                "axis_label_size": AXIS_LABEL_SIZE, "tick_size": TICK_SIZE,
                                "legend_size": LEGEND_SIZE, "annotation_size": ANNOTATION_SIZE,
                                "marker_size": MARKER_SIZE, "line_width": LINE_WIDTH,
                                "ci_width": CI_WIDTH, "ci_capsize": CI_CAPSIZE, "ci_alpha": CI_ALPHA},
                   "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__},
                   "notes": ["Current manuscript Figure S2; original-style red/black markers and bin-wise Wilson error bars.",
                             "Intervals are read from validated saved aggregates; no new intervals or model estimates are computed.",
                             "Bin-wise descriptive intervals are conditional on saved predictions/bins; not simultaneous or full model-development uncertainty.",
                             "Patient identifiers and predictions are neither read nor exported."]}
        (stage / "render_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out.mkdir(exist_ok=True)
        for path in stage.iterdir():
            path.replace(out / path.name)
    finally:
        plt.close(fig)
        shutil.rmtree(stage)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--font", default=FONT_FAMILY)
    parser.add_argument("--formats", nargs="+", default=FIGURE_FORMATS)
    parser.add_argument("--dpi", type=int, default=RASTER_DPI)
    args = parser.parse_args(argv)
    try:
        receipt = build(args.input_dir, args.out, args.formats, args.dpi, args.font)
    except Exception as exc:
        parser.exit(1, f"[STOP] {type(exc).__name__}: {exc}\n")
    print(f"[DONE] Figure S2: {len(receipt['datasets'])}/6 cohorts ({receipt['coverage_scope']}): {args.out}")


if __name__ == "__main__":
    main()
