"""
Compare 3 evaluation schemes on representative datasets:
  (A) Nested CV (nCV)                : read existing metrics_summary.csv (no re-run)
  (B) Non-nested CV (single-layer)   : fix FS/HP using the SAME CV used for OOF evaluation (optimistic)
  (C) Holdout (70/30)                : 1x stratified split; train-side only tuning/FS;
                                       test AUC + bootstrap CI (bootstrap on test)

Requirements:
  - Put this script in the same folder as ICI_predict_v2.py
  - Run ICI_predict_v2.py beforehand to generate nested results:
      <NESTED_RESULTS_ROOT>/<dataset_stem>/metrics_summary.csv
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from scipy.stats import wilcoxon
# ---- base pipeline module ----
# Prefer ICI_predict_v2.py, but keep backward compatibility.
try:
    import ICI_predict_v2 as base  # type: ignore
except ImportError:
    import ICI_predict as base  # type: ignore

# ============================================================
# USER SETTINGS 
# ============================================================
# Representative datasets: (xlsx_path, display_label_for_yaxis)
REPRESENTATIVE_DATASETS: List[Tuple[str, str]] = [
    # (r"your dataset1.xlsx", "Dataset 1"),
    # (r"your dataset2.xlsx", "Dataset 2"),
]

# Where nested results exist (same as OUTPUT_ROOT used in 01_Model.py)
# Needs: <NESTED_RESULTS_ROOT>/<dataset_stem>/metrics_summary.csv
NESTED_RESULTS_ROOT = r""

# Where to save compare outputs (must be writable)
COMPARE_OUTPUT_ROOT = r"output"

# Single split settings
SINGLE_SPLIT_TEST_SIZE = 0.30
SINGLE_SPLIT_RANDOM_STATE = 42

# Non-nested CV settings (recommended: match your datasets' settings, e.g. 5x3)
NONNESTED_OUTER_SPLITS_TARGET = 5
NONNESTED_OUTER_REPEATS = 5

# Plot ordering: keep as specified in REPRESENTATIVE_DATASETS
SORT_BY_NESTED_AUC = False

# Output filenames
OUT_WIDE_TABLE = "comparison_table.csv"
OUT_ABSTRACT_SUMMARY = "comparison_summary.csv"

# ============================================================
# helper functions
# ============================================================
def _read_nested_metrics(dataset_stem: str, nested_root: Path) -> Dict[str, Any]:
    candidates: List[Path] = [nested_root / dataset_stem / "metrics_summary.csv"]

    out_root = getattr(base, "OUTPUT_ROOT", None)
    if out_root:
        try:
            candidates.append(Path(str(out_root)) / dataset_stem / "metrics_summary.csv")
        except Exception:
            pass

    mpath = None
    for p in candidates:
        if p.exists():
            mpath = p
            break

    if mpath is None:
        tried = "\n".join([f"- {p}" for p in candidates])
        raise FileNotFoundError(
            f"Missing nested metrics for dataset '{dataset_stem}'. Tried:\n{tried}\n"
            f"Run Model first, and ensure NESTED_RESULTS_ROOT is correct."
        )

    df = pd.read_csv(mpath, encoding="utf-8-sig")
    r = df.iloc[0].to_dict()
    return dict(
        nested_auc=float(r["roc_auc_oof"]),
        nested_ci_lo=float(r["roc_auc_ci95_lo"]),
        nested_ci_hi=float(r["roc_auc_ci95_hi"]),
        n_samples=int(r.get("n_samples", np.nan)),
        n_features=int(r.get("n_features", np.nan)),
        metrics_path=str(mpath),
    )

def _save_pred_csv(path: Path, ids: pd.Series, y: np.ndarray, p: np.ndarray, split: str) -> None:
    df = pd.DataFrame({
        base.ID_COL: ids.values,
        "y": y.astype(int),
        "p": p.astype(float),
        "split": split,
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")

def _fit_predict_pipeline_on_split(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    dataset_name: str,
    tag: str,
) -> np.ndarray:
    """Fit full pipeline on train only (tuning/FS/preprocess), predict test."""
    log_lines: List[str] = []

    Xtr = X.iloc[tr_idx]
    ytr = y[tr_idx]
    Xte = X.iloc[te_idx]

    inner_cv = base.make_inner_cv(ytr, f"{dataset_name} inner({tag})", log_lines)

    en_C, en_l1, _ = base.tune_elasticnet(
        Xtr, ytr, feature_names,
        inner_cv=inner_cv,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines,
    )

    ranking, _ = base.stability_select_and_rank(
        Xtr, ytr, feature_names,
        inner_cv=inner_cv,
        en_C=en_C,
        en_l1=en_l1,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines,
    )

    k_sel, l2_C, _ = base.select_k_by_1se(
        Xtr, ytr, ranking,
        inner_cv=inner_cv,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines,
    )

    feats = ranking[:k_sel]
    prep = base.MetaboPreprocessor(feats, missing_rate_threshold=base.MISSING_RATE_THRESHOLD).fit(Xtr[feats])
    clf = base.fit_logreg_with_refit(
        prep.transform(Xtr[feats]), ytr,
        kind="l2", C=l2_C, l1_ratio=None,
        dataset_name=dataset_name,
        stage=f"split-fit({tag})",
        log_lines=log_lines,
    )

    p_te = clf.predict_proba(prep.transform(Xte[feats]))[:, 1]
    return p_te

def run_single_split( # holdout
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
) -> Dict[str, Any]:
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=SINGLE_SPLIT_TEST_SIZE,
        random_state=SINGLE_SPLIT_RANDOM_STATE,
        stratify=y,
    )
    tr_idx = np.asarray(tr_idx, dtype=int)
    te_idx = np.asarray(te_idx, dtype=int)

    p_te = _fit_predict_pipeline_on_split(
        X, y, feature_names,
        tr_idx=tr_idx, te_idx=te_idx,
        dataset_name=dataset_name,
        tag="single_split",
    )

    y_te = y[te_idx]
    auc = float(roc_auc_score(y_te, p_te))
    ci_lo, ci_hi = base.bootstrap_auc_ci(
        y_te, p_te, n_boot=base.BOOTSTRAP_N, seed=base.RANDOM_STATE
    )

    return dict(
        split_auc=auc,
        split_ci_lo=float(ci_lo),
        split_ci_hi=float(ci_hi),
        tr_idx=tr_idx,
        te_idx=te_idx,
        p_te=p_te,
    )

def run_non_nested_cv_leaky( # non-nested-CV
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
) -> Dict[str, Any]:
    log_lines: List[str] = []
    cv_splits_used = base.adjusted_n_splits(
        y, NONNESTED_OUTER_SPLITS_TARGET, f"{dataset_name} nonnested(single-layer)", log_lines
    )
    cv_single = RepeatedStratifiedKFold(
        n_splits=cv_splits_used,
        n_repeats=NONNESTED_OUTER_REPEATS,
        random_state=base.RANDOM_STATE,
    )

    en_C, en_l1, _ = base.tune_elasticnet(
        X, y, feature_names,
        inner_cv=cv_single,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines,
    )
    ranking, _ = base.stability_select_and_rank(
        X, y, feature_names,
        inner_cv=cv_single,
        en_C=en_C,
        en_l1=en_l1,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines,
    )
    k_sel, l2_C, _ = base.select_k_by_1se(
        X, y, ranking,
        inner_cv=cv_single,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines,
    )
    feats_fixed = ranking[:k_sel]

    n = len(y)
    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=int)

    for cv_iter, (tr_idx, te_idx) in enumerate(cv_single.split(X, y), start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        te_idx = np.asarray(te_idx, dtype=int)

        Xtr = X.iloc[tr_idx]
        ytr = y[tr_idx]
        Xte = X.iloc[te_idx]

        prep = base.MetaboPreprocessor(feats_fixed, missing_rate_threshold=base.MISSING_RATE_THRESHOLD).fit(Xtr[feats_fixed])
        clf = base.fit_logreg_with_refit(
            prep.transform(Xtr[feats_fixed]), ytr,
            kind="l2", C=l2_C, l1_ratio=None,
            dataset_name=dataset_name,
            stage=f"nonnested single-layer iter={cv_iter}",
            log_lines=log_lines,
        )
        p_te = clf.predict_proba(prep.transform(Xte[feats_fixed]))[:, 1]
        oof_sum[te_idx] += p_te
        oof_cnt[te_idx] += 1

    p_oof = oof_sum / oof_cnt
    auc = float(roc_auc_score(y, p_oof))
    ci_lo, ci_hi = base.bootstrap_auc_ci(y, p_oof, n_boot=base.BOOTSTRAP_N, seed=base.RANDOM_STATE)

    return dict(
        nonnested_auc=auc,
        nonnested_ci_lo=float(ci_lo),
        nonnested_ci_hi=float(ci_hi),
        p_oof=p_oof,
        fixed_en_C=float(en_C),
        fixed_en_l1=float(en_l1),
        fixed_k=int(k_sel),
        fixed_l2_C=float(l2_C),
        cv_splits_used=int(cv_splits_used),
        cv_repeats=int(NONNESTED_OUTER_REPEATS),
    )

def _ci_width(lo: float, hi: float) -> float:
    return float(hi - lo)

def _format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def _summary_row(name: str, values: np.ndarray, fmt: str = "{:.3f}") -> Dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {
        "item": name,
        "mean": fmt.format(float(np.mean(v))),
        "min": fmt.format(float(np.min(v))),
        "max": fmt.format(float(np.max(v))),
        "p_raw": "",
    }

def make_1panel_plot(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    if df.empty:
        raise ValueError("No data to plot.")

    dfp = df.copy()
    if SORT_BY_NESTED_AUC:
        dfp = dfp.sort_values("nested_auc", ascending=False).reset_index(drop=True)

    dataset_labels = dfp["dataset_label"].tolist()
    n_ds = len(dataset_labels)

    # (legend_name, auc_col, lo_col, hi_col, marker, short_tick_label)
    methods = [
        ("Nested CV",                    "nested_auc",    "nested_ci_lo",    "nested_ci_hi",    "o", "nCV"),
        ("Non-nested CV (single-layer)", "nonnested_auc", "nonnested_ci_lo", "nonnested_ci_hi", "s", "CV"),
        ("holdout (70/30)",              "split_auc",     "split_ci_lo",     "split_ci_hi",     "^", "holdout"),
    ]
    m = len(methods)

    # ---- layout knobs ----
    gap = 1 
    method_fontsize = 9
    dataset_fontsize = 10
    dataset_label_y = -0.17 
    bottom_margin = 0.30 
    right_margin = 0.82 
    # ----------------------

    def xpos(i: int, j: int) -> int:
        return i * (m + gap) + j

    # x tick: method labels only
    xticks = [xpos(i, j) for i in range(n_ds) for j in range(m)]
    xtick_labels = [methods[j][5] for i in range(n_ds) for j in range(m)]

    fig_w = max(9.0, 0.55 * len(xticks) + 2.0)
    fig_h = 5.2
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # plot points + CI (all black)
    for j, (name, col_auc, col_lo, col_hi, marker, _) in enumerate(methods):
        auc = dfp[col_auc].to_numpy(dtype=float)
        lo = dfp[col_lo].to_numpy(dtype=float)
        hi = dfp[col_hi].to_numpy(dtype=float)

        xs = np.array([xpos(i, j) for i in range(n_ds)], dtype=float)
        yerr = np.vstack([auc - lo, hi - auc])

        ax.errorbar(
            xs, auc, yerr=yerr,
            fmt=marker, linestyle="none",
            color="black", ecolor="black",
            markerfacecolor="black", markeredgecolor="black",
            markersize=5, capsize=2,
            label=name,
        )

        # numeric labels (AUC) just above upper CI
        for x, a, h in zip(xs, auc, hi):
            ax.text(x, min(h + 0.015, 1.045), f"{a:.2f}",
                    ha="center", va="bottom", fontsize=8, color="black")

    # baseline
    ax.axhline(0.5, linestyle="--", linewidth=1, color="black")

    ax.set_ylabel("ROC-AUC (95% CI)")
    ax.set_xlabel("")

    ax.set_ylim(0.0, 1.1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=method_fontsize)
    ax.tick_params(axis="x", pad=2)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    # dataset labels (once per group) + optional separators
    for i, dlab in enumerate(dataset_labels):
        center = i * (m + gap) + (m - 1) / 2.0
        ax.text(center, dataset_label_y, dlab,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=dataset_fontsize, color="black")
        # separator line between dataset groups (薄く)
        if i > 0:
            sep_x = i * (m + gap) - 0.5
            ax.axvline(sep_x, color="black", linewidth=0.6, alpha=0.2)

    # legend outside right (avoid overlap)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)

    fig.subplots_adjust(bottom=bottom_margin, right=right_margin)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    if not REPRESENTATIVE_DATASETS:
        raise RuntimeError("REPRESENTATIVE_DATASETS is empty. Set (xlsx_path, label) at the top.")

    nested_root = Path(NESTED_RESULTS_ROOT)
    out_root = Path(COMPARE_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for xlsx_path, dataset_label in REPRESENTATIVE_DATASETS:
        dataset_stem = Path(xlsx_path).stem
        print("=" * 80)
        print(f"[COMPARE] {dataset_stem}  ({dataset_label})")
        print("=" * 80)

        # load
        X, y, feature_names, ids = base.read_xlsx_dataset(xlsx_path)

        # (A) nested: read metrics
        nm = _read_nested_metrics(dataset_stem, nested_root)

        # per-dataset dir
        ds_out = out_root / dataset_stem
        ds_out.mkdir(parents=True, exist_ok=True)

        # (C) holdout
        ss = run_single_split(X, y, feature_names, dataset_name=dataset_stem)
        te_idx = ss["te_idx"]
        _save_pred_csv(ds_out / "single_split_test_pred.csv", ids.iloc[te_idx], y[te_idx], ss["p_te"], "test")

        # (B) non-nested CV (single-layer)
        nn = run_non_nested_cv_leaky(X, y, feature_names, dataset_name=dataset_stem)
        _save_pred_csv(ds_out / "nonnested_oof_pred.csv", ids, y, nn["p_oof"], "oof")

        # save meta
        (ds_out / "meta_non_nested.json").write_text(
            json.dumps({k: v for k, v in nn.items() if k != "p_oof"}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        (ds_out / "meta_single_split.json").write_text(
            json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ss.items() if k not in {"p_te"}},
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        row = dict(
            dataset_stem=dataset_stem,
            dataset_label=dataset_label,
            xlsx_path=xlsx_path,
            n_samples=int(len(y)),
            n_features=int(X.shape[1]),

            # nCV
            nested_auc=nm["nested_auc"],
            nested_ci_lo=nm["nested_ci_lo"],
            nested_ci_hi=nm["nested_ci_hi"],

            # CV (non-nested)
            nonnested_auc=nn["nonnested_auc"],
            nonnested_ci_lo=nn["nonnested_ci_lo"],
            nonnested_ci_hi=nn["nonnested_ci_hi"],

            # holdout
            split_auc=ss["split_auc"],
            split_ci_lo=ss["split_ci_lo"],
            split_ci_hi=ss["split_ci_hi"],

            # (legacy) AUC deltas
            delta_nonnested_minus_nested=nn["nonnested_auc"] - nm["nested_auc"],
            delta_split_minus_nested=ss["split_auc"] - nm["nested_auc"],
        )
        rows.append(row)

        print(
            f"Nested     AUC={row['nested_auc']:.3f} [{row['nested_ci_lo']:.3f}, {row['nested_ci_hi']:.3f}]\n"
            f"Non-nested AUC={row['nonnested_auc']:.3f} [{row['nonnested_ci_lo']:.3f}, {row['nonnested_ci_hi']:.3f}]\n"
            f"Holdout    AUC={row['split_auc']:.3f} [{row['split_ci_lo']:.3f}, {row['split_ci_hi']:.3f}]"
        )

    df = pd.DataFrame(rows)

    # ---- add CI widths + delta metrics (requested) ----
    df["nested_ci_width"] = df.apply(lambda r: _ci_width(r["nested_ci_lo"], r["nested_ci_hi"]), axis=1)
    df["nonnested_ci_width"] = df.apply(lambda r: _ci_width(r["nonnested_ci_lo"], r["nonnested_ci_hi"]), axis=1)
    df["split_ci_width"] = df.apply(lambda r: _ci_width(r["split_ci_lo"], r["split_ci_hi"]), axis=1)

    # clear names for delta metrics
    df["delta_auc_cv_minus_ncv"] = df["nonnested_auc"] - df["nested_auc"]
    df["delta_auc_holdout_minus_ncv"] = df["split_auc"] - df["nested_auc"]
    df["delta_ci_width_cv_minus_ncv"] = df["nonnested_ci_width"] - df["nested_ci_width"]
    df["delta_ci_width_holdout_minus_ncv"] = df["split_ci_width"] - df["nested_ci_width"]

    # ---- save dataset-level tables ----
    out_csv_main = out_root / "comparison_subset_auc.csv"
    out_csv_wide = out_root / OUT_WIDE_TABLE

    # Use a tidy, explicit column order for the wide table
    wide_cols = [
        "dataset_stem", "dataset_label", "xlsx_path", "n_samples", "n_features",

        # nCV
        "nested_auc", "nested_ci_lo", "nested_ci_hi", "nested_ci_width",

        # CV
        "nonnested_auc", "nonnested_ci_lo", "nonnested_ci_hi", "nonnested_ci_width",

        # holdout
        "split_auc", "split_ci_lo", "split_ci_hi", "split_ci_width",

        # deltas
        "delta_auc_cv_minus_ncv",
        "delta_auc_holdout_minus_ncv",
        "delta_ci_width_cv_minus_ncv",
        "delta_ci_width_holdout_minus_ncv",

        # legacy deltas (keep at end)
        "delta_nonnested_minus_nested",
        "delta_split_minus_nested",
    ]
    df_wide = df[wide_cols].copy()

    df_wide.to_csv(out_csv_main, index=False, encoding="utf-8-sig")
    df_wide.to_csv(out_csv_wide, index=False, encoding="utf-8-sig")

    # ---- abstract-ready summary + Wilcoxon ----
    summary_rows: List[Dict[str, Any]] = []

    summary_rows.append(_summary_row("AUC_nCV", df["nested_auc"].to_numpy()))
    summary_rows.append(_summary_row("AUC_CV", df["nonnested_auc"].to_numpy()))
    summary_rows.append(_summary_row("AUC_holdout", df["split_auc"].to_numpy()))

    # delta metrics
    row_delta_auc_cv = _summary_row("ΔAUC_CV_minus_nCV", df["delta_auc_cv_minus_ncv"].to_numpy())
    summary_rows.append(row_delta_auc_cv)

    summary_rows.append(_summary_row("ΔAUC_holdout_minus_nCV", df["delta_auc_holdout_minus_ncv"].to_numpy()))
    summary_rows.append(_summary_row("ΔCIwidth_CV_minus_nCV", df["delta_ci_width_cv_minus_ncv"].to_numpy()))

    row_delta_width_holdout = _summary_row("ΔCIwidth_holdout_minus_nCV", df["delta_ci_width_holdout_minus_ncv"].to_numpy())
    summary_rows.append(row_delta_width_holdout)

    # Wilcoxon signed-rank tests (two-sided), zero_method='pratt'
    # 1) AUC: CV vs nCV
    p_auc_raw = float(wilcoxon(
        df["nonnested_auc"].to_numpy(dtype=float),
        df["nested_auc"].to_numpy(dtype=float),
        alternative="two-sided",
        zero_method="pratt",
    ).pvalue)

    # 2) CI width: holdout vs nCV
    p_width_raw = float(wilcoxon(
        df["split_ci_width"].to_numpy(dtype=float),
        df["nested_ci_width"].to_numpy(dtype=float),
        alternative="two-sided",
        zero_method="pratt",
    ).pvalue)

    # attach formatted p-values to the corresponding delta rows
    for r in summary_rows:
        if r["item"] == "ΔAUC_CV_minus_nCV":
            r["p_raw"] = _format_p(p_auc_raw)
        if r["item"] == "ΔCIwidth_holdout_minus_nCV":
            r["p_raw"] = _format_p(p_width_raw)

    df_sum = pd.DataFrame(summary_rows, columns=["item", "mean", "min", "max", "p_raw"])
    out_sum = out_root / OUT_ABSTRACT_SUMMARY
    df_sum.to_csv(out_sum, index=False, encoding="utf-8-sig")

    # ---- plot ----
    out_png = out_root / "comparison_panel.png"
    out_pdf = out_root / "comparison_panel.pdf"
    make_1panel_plot(df, out_png, out_pdf)

    print("\n[OK] Saved:")
    print(f"- {out_csv_main}")
    print(f"- {out_csv_wide}")
    print(f"- {out_sum}")
    print(f"- {out_png}")
    print(f"- {out_pdf}")

if __name__ == "__main__":
    main()