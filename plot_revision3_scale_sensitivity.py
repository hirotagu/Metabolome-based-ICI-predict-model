#!/usr/bin/env python3
"""Recreate Figure S7 from the included aggregate results; no model fitting.

Run from the repository folder:
    python3 plot_revision3_scale_sensitivity.py

Input:  figure_data/FigureS7_data.csv (Datasets 3–5 only)
Output: result_revision3/FigureS_scale_sensitivity.{pdf,png,csv,json}
Use --check for input validation without rendering. Use --font to choose a font.
Re-running updates only these four figure outputs, never the source data.
"""
from __future__ import annotations

from pathlib import Path

# ======================== USER SETTINGS ========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "figure_data" / "FigureS7_data.csv"
OUTPUT_DIR = BASE_DIR / "result_revision3"
FIGURE_NAME = "FigureS_scale_sensitivity"
FIGSIZE = (13.4, 12.4)
ROW_HEIGHT_RATIOS = (1.25, 1.0)
PNG_DPI = 600
FONT_FAMILY = "Arial"
FONT_SIZE = 10.0
AXIS_LABEL_SIZE = 13.0
TICK_SIZE = 13.0
LEGEND_SIZE = 13.0
TITLE_SIZE = 13.0
PANEL_LABEL_SIZE = 12.0
MARKER_SIZE = 6.2
LINE_WIDTH = 1.35
ERROR_LINE_WIDTH = 1.10
ORIGINAL_COLOR = "#111111"
SCALE_AWARE_COLOR = "#D62728"
# ==============================================================

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

try:
    import numpy as np
    import pandas as pd
    import matplotlib
except ImportError as exc:
    raise SystemExit("Install the documented dependencies in the selected Python environment: "
                     "python3 -m pip install -r requirements-figures.txt") from exc
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

PRE = [f"dataset{d}_pre" for d in (3, 4, 5)]
POST = [f"dataset{d}_post{t}" for d in (3, 4, 5) for t in (1, 2)]
ALL_MATRICES = [f"dataset{d}_{t}" for d in (3, 4, 5)
                for t in ("pre", "post1", "post2")]
HYBRIDS = {"ranking_nested_kC_global": "Enet-nest",
           "ranking_global_kC_nested": "k/L2-nest"}
COUNTS = ("n_samples", "n_positive", "n_negative")
TOL = 2e-8  # 旧hybrid CSVは小数9桁。保存値の丸めを許容。
CAPTION = (
    "Comparison of original and scale-aware preprocessing in Datasets 3–5. "
    "Original preprocessing applies log1p to deposited log10 values after the "
    "original missing-value handling. Scale-aware preprocessing retains the "
    "deposited log10 values, preserves finite zero and negative values, and uses "
    "imputation appropriate to the input scale. Dataset 5 Post2 missing entries "
    "are restored from the source matrix to distinguish missingness from true zeros. "
    "A, Pre-treatment AUROC. B, Post-treatment AUROC. A and B use patient-averaged "
    "out-of-fold predictions; connecting lines link the two preprocessing settings "
    "within the same matrix and evaluation method. No uncertainty intervals are "
    "displayed in A or B. C, Median AUROC across 25 overlapping stratified 70:30 "
    "holdout splits; horizontal bars show the interquartile range, not confidence "
    "intervals. D, Hybrid AUROC minus fully nested AUROC under the same preprocessing "
    "setting. Enet-nest nests feature ranking but selects k/C globally; k/L2-nest "
    "uses global feature ranking and nests k/C selection. These hybrids are "
    "diagnostic comparisons and are not leakage-free performance estimates. "
    "Nested denotes fully nested CV; non-nested denotes the fully non-nested comparison. "
    "Only Datasets 3–5 are shown."
)


EXPECTED_COLUMNS = (
    "panel", "dataset", "method", "metric", "original", "scale_aware", "interval",
    "original_q1", "original_q3", "scale_aware_q1", "scale_aware_q3",
    "original_reference_auc", "scale_aware_reference_auc",
    "n_samples", "n_positive", "n_negative", "scale_aware_minus_original",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_data(data):
    """Validate the complete four-panel aggregate table before any rendering."""
    require(set(data.columns) == set(EXPECTED_COLUMNS),
            "Unexpected columns: use the documented FigureS7_data.csv aggregate schema.")
    expected = {(panel, d, method)
                for panel, datasets, methods in (
                    ("A", PRE, ("nested", "non-nested")),
                    ("B", POST, ("nested", "non-nested")),
                    ("C", ALL_MATRICES, ("repeated holdout",)),
                    ("D", PRE, tuple(HYBRIDS.values())))
                for d in datasets for method in methods}
    keys = list(data[["panel", "dataset", "method"]].itertuples(index=False, name=None))
    require(len(keys) == 33 and len(set(keys)) == 33 and set(keys) == expected,
            "Figure S7 requires exactly the 33 unique Datasets 3–5 panel/dataset/method rows.")
    numeric = [x for x in EXPECTED_COLUMNS if x not in ("panel", "dataset", "method", "metric", "interval")]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="raise")
    for col in ("original", "scale_aware", "scale_aware_minus_original", *COUNTS):
        require(np.isfinite(data[col]).all(), f"Non-finite required values in {col}.")
    require(np.allclose(data.scale_aware - data.original, data.scale_aware_minus_original,
                        atol=TOL, rtol=0), "The stored preprocessing differences are inconsistent.")
    for dataset, group in data.groupby("dataset"):
        require(len(group[list(COUNTS)].drop_duplicates()) == 1,
                f"Inconsistent cohort counts across panels: {dataset}")
        n, pos, neg = group.iloc[0][list(COUNTS)].astype(float)
        require(all(x == int(x) for x in (n, pos, neg)) and n == pos + neg and min(pos, neg) > 0,
                f"Invalid cohort/class counts: {dataset}")
    for _, row in data.iterrows():
        is_hybrid = row.panel == "D"
        lo = -1 if is_hybrid else 0
        for col in ("original", "scale_aware"):
            require(lo - TOL <= row[col] <= 1 + TOL, f"Out-of-range {col} for {row.dataset}.")
        metric = ("Hybrid AUROC minus fully nested AUROC" if is_hybrid
                  else "Median AUROC" if row.panel == "C" else "AUROC")
        require(row.metric == metric, "Unexpected metric label.")
        if row.panel == "C":
            require(row.interval == "IQR across 25 overlapping holdout splits", "Unexpected interval label.")
            for col in ("original", "scale_aware"):
                q1, q3 = row[col + "_q1"], row[col + "_q3"]
                require(np.isfinite([q1, q3]).all() and 0 <= q1 <= row[col] <= q3 <= 1,
                        f"Invalid holdout IQR: {row.dataset}/{col}")
        else:
            require(row.interval == "none", "Unexpected interval label.")
            require(row[["original_q1", "original_q3", "scale_aware_q1", "scale_aware_q3"]].isna().all(),
                    "Unexpected intervals for a panel without intervals.")
        if is_hybrid:
            for col in ("original", "scale_aware"):
                reference = row[col + "_reference_auc"]
                baseline = data.loc[(data.panel == "A") & (data.dataset == row.dataset)
                                    & (data.method == "nested"), col].iloc[0]
                require(np.isfinite(reference) and np.isclose(reference, baseline, atol=TOL, rtol=0),
                        f"Hybrid reference differs from fully nested AUC: {row.dataset}/{col}")
                require(-TOL <= reference + row[col] <= 1 + TOL, "Invalid reconstructed hybrid AUC.")
        else:
            require(row[["original_reference_auc", "scale_aware_reference_auc"]].isna().all(),
                    "Unexpected hybrid references outside panel D.")
    return data


def load_data(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    provenance_path = path.parent / "FigureS7_source.json"
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("data_file") == path.name:
            require(provenance.get("data_sha256") == hashlib.sha256(raw).hexdigest(),
                    "The aggregate CSV hash differs from FigureS7_source.json.")
        else:
            provenance = None
    data = pd.read_csv(path, float_precision="round_trip")
    return validate_data(data), provenance


def short(dataset):
    return dataset.replace("dataset", "Dataset").replace("_pre", " Pre").replace("_post", " Post")


def select_font(requested):
    """Prefer the manuscript Arial font, with an explicit recorded fallback."""
    if sys.platform == "darwin" and requested.casefold() == "arial":
        folders = (Path("/System/Library/Fonts/Supplemental"),
                   Path("/Library/Fonts"), Path.home() / "Library/Fonts")
        for folder in folders:
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.stem.casefold() in ("arial", "arial bold", "arial italic", "arial bold italic"):
                        if path.suffix.lower() in (".ttf", ".otf", ".ttc"):
                            font_manager.fontManager.addfont(str(path))
    for family in dict.fromkeys((requested, "DejaVu Sans")):
        resolved = {}
        try:
            for weight in ("normal", "bold"):
                path = font_manager.findfont(font_manager.FontProperties(family=[family], weight=weight),
                                             fallback_to_default=False)
                actual = font_manager.FontProperties(fname=path).get_name()
                require(actual.casefold() == family.casefold(), "Unexpected font substitution.")
                resolved[weight] = {"family": actual, "path": path}
        except ValueError:
            continue
        if family != requested:
            print(f"[FONT] {requested} is unavailable; using {family}. "
                  "Install Arial or use --font for the manuscript font.")
        else:
            print(f"[FONT] {family}")
        return family, resolved
    raise RuntimeError("Neither the requested font nor DejaVu Sans is available.")


def draw(data, output):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT_FAMILY],
                         "font.size": FONT_SIZE, "axes.titlesize": TITLE_SIZE,
                         "axes.labelsize": AXIS_LABEL_SIZE, "xtick.labelsize": TICK_SIZE,
                         "ytick.labelsize": TICK_SIZE, "legend.fontsize": LEGEND_SIZE,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE,
                             gridspec_kw={"height_ratios": ROW_HEIGHT_RATIOS})
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.09, top=0.90, hspace=0.36, wspace=0.68)
    titles = {"A": "Pre-treatment", "B": "Post-treatment",
              "C": "Repeated holdout", "D": "Diagnostic hybrids (Pre)"}
    specs = (("original", ORIGINAL_COLOR, "o", "white"),
             ("scale_aware", SCALE_AWARE_COLOR, "D", SCALE_AWARE_COLOR))
    plotted = []
    for panel, ax in zip("ABCD", axes.flat):
        subset = data.loc[data.panel.eq(panel)]
        datasets = PRE if panel in "AD" else POST if panel == "B" else ALL_MATRICES
        methods = list(HYBRIDS.values()) if panel == "D" else ["nested", "non-nested"]
        order = [(d, m) for d in datasets for m in (methods if panel != "C" else ["repeated holdout"])]
        yvalues, labels = [], []
        previous = None
        y = 0.0
        for dataset, method in order:
            if previous is not None and dataset != previous:
                y += 0.55 if panel == "B" else 0.32
            row = subset.loc[subset.dataset.eq(dataset) & subset.method.eq(method)].iloc[0]
            label = short(dataset)
            if panel in "AD":
                label = label.replace(" Pre", "")
            if panel != "C":
                label += "  " + method
            yvalues.append(y)
            labels.append(label)
            if panel != "C":
                ax.plot([row.original, row.scale_aware], [y, y], color="#9A9A9A", lw=LINE_WIDTH, zorder=1)
            for i, (column, color, marker, face) in enumerate(specs):
                yp = y + (-0.14 if i == 0 else 0.14) if panel == "C" else y
                value = float(row[column])
                if panel == "C":
                    lo, hi = float(row[column + "_q1"]), float(row[column + "_q3"])
                    ax.errorbar(value, yp, xerr=[[value - lo], [hi - value]], fmt="none",
                                ecolor=color, elinewidth=ERROR_LINE_WIDTH, capsize=2.2, zorder=2)
                ax.plot(value, yp, marker=marker, markersize=MARKER_SIZE, color=color,
                        markerfacecolor=face, markeredgewidth=1.1, linestyle="none", zorder=3 + i)
                plotted.append((panel, dataset, method, column, value))
            previous = dataset
            y += 1.0
        ax.set_yticks(yvalues, labels)
        ax.set_ylim(y - 0.30, -0.65)
        ax.set_title(titles[panel], loc="left", fontweight="normal", pad=13)
        ax.annotate(panel, xy=(0, 1), xycoords="axes fraction", xytext=(-28, 13),
                    textcoords="offset points", ha="left", va="baseline",
                    fontsize=PANEL_LABEL_SIZE, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", direction="out", length=3, width=0.7, color="#111111")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#111111")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        if panel == "D":
            values = subset[["original", "scale_aware"]].to_numpy()
            lower = min(-0.025, float(values.min()) - 0.02)
            upper = max(0.15, float(values.max()) + 0.02)
            ax.set_xlim(lower, upper)
            ax.xaxis.set_major_locator(MultipleLocator(0.05))
            ax.axvline(0, linestyle="--", color="#B8B8B8", lw=0.9)
            ax.set_xlabel("Hybrid AUROC − fully nested AUROC")
        else:
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.axvline(0.5, linestyle="--", color="#B8B8B8", lw=0.9)
            ax.set_xlabel("AUROC (median and IQR)" if panel == "C" else "AUROC")
    require(len(plotted) == 66, "描画点数が66点ではありません。")
    handles = [Line2D([], [], marker=m, markersize=MARKER_SIZE, color=c, markerfacecolor=f,
                      linestyle="none", markeredgewidth=1.1)
               for _, c, m, f in specs]
    fig.legend(handles, ["Original preprocessing", "Scale-aware preprocessing"],
               loc="upper center", bbox_to_anchor=(0.55, 0.976), ncol=2,
               frameon=False, columnspacing=3, handletextpad=0.7, fontsize=LEGEND_SIZE)
    pdf = output / f"{FIGURE_NAME}.pdf"
    png = output / f"{FIGURE_NAME}.png"
    fig.savefig(pdf, metadata={"Title": "Preprocessing sensitivity in Datasets 3–5",
                              "Subject": CAPTION, "Creator": Path(__file__).name})
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    plt.close(fig)
    return [pdf, png]


def main():
    global FONT_FAMILY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=INPUT_CSV, help="Figure S7 aggregate CSV")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--font", default=FONT_FAMILY)
    parser.add_argument("--check", action="store_true", help="Validate aggregate data without rendering")
    args = parser.parse_args()
    input_path = args.data.expanduser().resolve()
    data, provenance = load_data(input_path)
    print(f"[CHECK] Validated 33 aggregate rows from {input_path.name}; no patient-level input is required.")
    if args.check:
        return
    output = args.out.expanduser().resolve()
    require(input_path != output / f"{FIGURE_NAME}.csv", "Output must not overwrite the source aggregate CSV.")
    FONT_FAMILY, fonts = select_font(args.font)
    output.mkdir(parents=True, exist_ok=True)
    created = draw(data, output)
    csv = output / f"{FIGURE_NAME}.csv"
    data.to_csv(csv, index=False, encoding="utf-8-sig", float_format="%.17g")
    created.append(csv)
    receipt = {
        "status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_fits": 0, "bootstrap_runs": 0,
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "input_file": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_provenance": provenance,
        "source_checksum_verified": provenance is not None,
        "plot_rows": len(data), "point_estimates": 66, "holdout_IQR_endpoints": 36,
        "validation": "Exact panel/dataset/method coverage, cohort counts, metric bounds, stored differences, "
                      "IQR ordering and hybrid reference AUCs checked; see source_checksum_verified for the supplied provenance.",
        "figure_settings": {"size_inches": FIGSIZE, "png_dpi": PNG_DPI,
                            "row_height_ratios": ROW_HEIGHT_RATIOS,
                            "requested_font": args.font, "font_family": FONT_FAMILY, "resolved_fonts": fonts,
                            "font_size": FONT_SIZE, "axis_label_size": AXIS_LABEL_SIZE,
                            "tick_size": TICK_SIZE, "legend_size": LEGEND_SIZE,
                            "title_size": TITLE_SIZE, "panel_label_size": PANEL_LABEL_SIZE,
                            "original_color": ORIGINAL_COLOR, "scale_aware_color": SCALE_AWARE_COLOR},
        "figure_legend": CAPTION,
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in created},
    }
    record = output / f"{FIGURE_NAME}.json"
    record.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    created.append(record)
    print("[OK] Four-panel Figure S7 created (66 points); no model fitting or bootstrap rerun.")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
