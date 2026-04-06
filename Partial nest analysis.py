"""
Partial nesting study (nCV vs partial-nest vs CV) for ICI metabolomics ML pipeline
===============================================================================
Decompose the AUC optimism gap between:
  - nCV (fully nested; already computed by ICI_predict_v2.py)
  - CV  (single-layer / non-nested; optimistically biased; same as Compare validation script)
by comparing 4 schemes:
  1) nCV       : read existing nested results (metrics_summary.csv + tuning_summary.csv)
  2) Enet-nest : ENet tuning + stability ranking are nested per outer fold,
                 but L2 spec (k, C) is fixed from leaky CV (same as CV)
  3) L2-nest   : ENet tuning + ranking are fixed from leaky CV (same as CV),
                 but L2 selection (k, C) is nested per outer fold
  4) CV        : ENet tuning + ranking + L2 selection are fixed from leaky CV
                 (Compare validation script compatible)

Requirements
  - Put this script in the same folder as ICI_predict_v2.py.
  - Run ICI_predict_v2.py beforehand (so that nCV results exist per dataset).
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
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
# 6 datasets (xlsx) and their corresponding nCV result directories.
# Recommended format (per-dataset nCV dir):
#   (".../AAA.xlsx", ".../AAA")  where the directory contains:
#     - metrics_summary.csv
#     - tuning_summary.csv
# If you leave the 2nd element empty (""), this script falls back to:
#   <NESTED_RESULTS_ROOT>/<dataset_stem>/metrics_summary.csv
# <- fill with the nCV result folder for RCC-UC_pre if not under NESTED_RESULTS_ROOT
DATASETS: List[Tuple[str, str]] = [
    (
        r".xlsx",
        r"", 
    ),
    (
        r"",
        r"",
    ),
]

# Fallback root folder (used only when the per-dataset nCV dir is set to ""):
NESTED_RESULTS_ROOT = r""  # <- fill if you want to use fallback behavior

# Where to write this script's outputs
COMPARE_OUTPUT_ROOT = r""
COMPARE_SUBDIR = "compare_partialnest"
LEGACY_COMPARE_ROOT = r"" # for sanity check(set to "" to disable)


# ============================================================
# PLOT SETTINGS (Fig. 1)
# ============================================================

FIG1_FILENAME = "figure_nest_compare.tiff"   # TIFF, 300 dpi
FIG1_DPI = 300
JITTER_WIDTH = 0.10
POINT_SIZE = 28
MEAN_LINE_HALF_WIDTH = 0.18
MEAN_LINEWIDTH = 1.8   # "not too thick" per request


# ============================================================
# Small utilities
# ============================================================

class PrecomputedCV:
    """A CV splitter that yields a precomputed list of splits."""
    def __init__(self, splits: List[Tuple[np.ndarray, np.ndarray]]):
        self._splits = [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in splits]
    def split(self, X, y=None, groups=None):
        for tr, te in self._splits:
            yield tr, te
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._splits)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_nested_auc_and_k(
    dataset_stem: str,
    nested_dir: Path,
) -> Tuple[float, np.ndarray]:
    """Read nCV AUC and per-outer k list from ICI_predict outputs."""
    metrics_path = nested_dir / "metrics_summary.csv"
    tuning_path = nested_dir / "tuning_summary.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing nCV metrics_summary.csv: {metrics_path}")
    if not tuning_path.exists():
        raise FileNotFoundError(f"Missing nCV tuning_summary.csv: {tuning_path}")

    mdf = pd.read_csv(metrics_path)
    if "roc_auc_oof" not in mdf.columns:
        raise ValueError(f"'roc_auc_oof' not found in {metrics_path}")
    auc = float(mdf.loc[0, "roc_auc_oof"])

    tdf = pd.read_csv(tuning_path)
    if "k_selected" not in tdf.columns:
        raise ValueError(f"'k_selected' not found in {tuning_path}")
    k_list = tdf["k_selected"].to_numpy(dtype=int)
    return auc, k_list

def _wilcoxon_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0.0):
        return 1.0
    try:
        stat = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
        return float(stat.pvalue)
    except ValueError:
        # e.g., all differences are zero after removing zeros
        return 1.0

def _k_summary_stats(k: np.ndarray) -> Dict[str, float]:
    k = np.asarray(k, dtype=float)
    return {
        "k_mean": float(np.mean(k)),
        "k_median": float(np.median(k)),
        "k_min": float(np.min(k)),
        "k_max": float(np.max(k)),
    }

# ============================================================
# Core computations
# ============================================================

def _compute_outer_splits_list(X: pd.DataFrame, y: np.ndarray, dataset_name: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    log_lines: List[str] = []
    outer_splits_used = base.adjusted_n_splits(y, base.OUTER_SPLITS_TARGET, f"{dataset_name} outer", log_lines)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=int(outer_splits_used),
        n_repeats=int(base.OUTER_REPEATS),
        random_state=int(base.RANDOM_STATE),
    )
    return list(outer_cv.split(X, y))

def _compute_global_leaky_spec(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    cv_single: PrecomputedCV,
    dataset_name: str,
) -> Dict[str, Any]:
    """Compute Compare-compatible leaky spec on full data: ENet(C,l1), ranking, and L2(k,C)."""
    log_lines: List[str] = []

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

    return {
        "en_C": float(en_C),
        "en_l1": float(en_l1),
        "ranking": ranking,
        "k": int(k_sel),
        "l2_C": float(l2_C),
        "features": ranking[: int(k_sel)],
    }

def _oof_predict_with_fixed_spec(
    X: pd.DataFrame,
    y: np.ndarray,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    features: List[str],
    l2_C: float,
    stage_prefix: str,
) -> np.ndarray:
    n = len(y)
    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=int)

    for it, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        Xtr = X.iloc[tr_idx][features]
        ytr = y[tr_idx]
        Xte = X.iloc[te_idx][features]

        prep = base.MetaboPreprocessor(features, missing_rate_threshold=base.MISSING_RATE_THRESHOLD).fit(Xtr)
        clf = base.fit_logreg_with_refit(
            prep.transform(Xtr), ytr,
            kind="l2", C=l2_C, l1_ratio=None,
            dataset_name=dataset_name,
            stage=f"{stage_prefix} iter={it}",
            log_lines=[],
        )
        p_te = clf.predict_proba(prep.transform(Xte))[:, 1]
        oof_sum[te_idx] += p_te
        oof_cnt[te_idx] += 1

    if np.any(oof_cnt == 0):
        raise RuntimeError(f"{dataset_name}: some samples were never predicted (unexpected).")

    return oof_sum / oof_cnt

def _oof_predict_L2_nested(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    ranking_fixed: List[str],
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=int)
    k_list: List[int] = []

    for outer_iter, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        Xtr_full = X.iloc[tr_idx]
        ytr = y[tr_idx]
        Xte_full = X.iloc[te_idx]

        log_lines: List[str] = []
        inner_cv = base.make_inner_cv(ytr, f"{dataset_name} inner (L2-nest outer={outer_iter})", log_lines)

        k_sel, l2_C, _ = base.select_k_by_1se(
            Xtr_full, ytr, ranking_fixed,
            inner_cv=inner_cv,
            dataset_name=dataset_name,
            outer_iter=outer_iter,
            log_lines=log_lines,
        )
        k_list.append(int(k_sel))

        feats = ranking_fixed[: int(k_sel)]

        prep = base.MetaboPreprocessor(feats, missing_rate_threshold=base.MISSING_RATE_THRESHOLD).fit(Xtr_full[feats])
        clf = base.fit_logreg_with_refit(
            prep.transform(Xtr_full[feats]), ytr,
            kind="l2", C=float(l2_C), l1_ratio=None,
            dataset_name=dataset_name,
            stage=f"L2-nest outerfit outer={outer_iter}",
            log_lines=log_lines,
        )

        p_te = clf.predict_proba(prep.transform(Xte_full[feats]))[:, 1]
        oof_sum[te_idx] += p_te
        oof_cnt[te_idx] += 1

    if np.any(oof_cnt == 0):
        raise RuntimeError(f"{dataset_name}: some samples were never predicted (unexpected).")

    return (oof_sum / oof_cnt), np.asarray(k_list, dtype=int)

def _oof_predict_ENet_nested(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    fixed_k: int,
    fixed_l2_C: float,
    dataset_name: str,
) -> np.ndarray:
    n = len(y)
    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=int)

    for outer_iter, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        Xtr_full = X.iloc[tr_idx]
        ytr = y[tr_idx]
        Xte_full = X.iloc[te_idx]

        log_lines: List[str] = []
        inner_cv = base.make_inner_cv(ytr, f"{dataset_name} inner (Enet-nest outer={outer_iter})", log_lines)

        en_C, en_l1, _ = base.tune_elasticnet(
            Xtr_full, ytr, feature_names,
            inner_cv=inner_cv,
            dataset_name=dataset_name,
            outer_iter=outer_iter,
            log_lines=log_lines,
        )

        ranking, _ = base.stability_select_and_rank(
            Xtr_full, ytr, feature_names,
            inner_cv=inner_cv,
            en_C=float(en_C),
            en_l1=float(en_l1),
            dataset_name=dataset_name,
            outer_iter=outer_iter,
            log_lines=log_lines,
        )

        feats = ranking[: int(fixed_k)]
        prep = base.MetaboPreprocessor(feats, missing_rate_threshold=base.MISSING_RATE_THRESHOLD).fit(Xtr_full[feats])

        clf = base.fit_logreg_with_refit(
            prep.transform(Xtr_full[feats]), ytr,
            kind="l2", C=float(fixed_l2_C), l1_ratio=None,
            dataset_name=dataset_name,
            stage=f"Enet-nest outerfit outer={outer_iter}",
            log_lines=log_lines,
        )

        p_te = clf.predict_proba(prep.transform(Xte_full[feats]))[:, 1]
        oof_sum[te_idx] += p_te
        oof_cnt[te_idx] += 1

    if np.any(oof_cnt == 0):
        raise RuntimeError(f"{dataset_name}: some samples were never predicted (unexpected).")

    return oof_sum / oof_cnt


# ============================================================
# Plot
# ============================================================

def make_fig1_auc_jitter_mean(df_auc: pd.DataFrame, out_path: Path) -> None:
    order = ["nCV", "Enet-nest", "L2-nest", "CV"]
    col_map = {
        "nCV": "AUC_nCV",
        "Enet-nest": "AUC_Enet-nest",
        "L2-nest": "AUC_L2-nest",
        "CV": "AUC_CV",
    }
    rng = np.random.RandomState(int(base.RANDOM_STATE))

    ys_all = []
    for name in order:
        ys_all.extend(df_auc[col_map[name]].to_numpy(dtype=float).tolist())
    ys_all = np.asarray(ys_all, dtype=float)
    y_min = max(0.0, float(np.min(ys_all) - 0.05))
    y_max = min(1.0, float(np.max(ys_all) + 0.05))
    if y_max - y_min < 0.15:
        # keep some vertical room
        mid = (y_min + y_max) / 2.0
        y_min = max(0.0, mid - 0.075)
        y_max = min(1.0, mid + 0.075)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))

    for i, name in enumerate(order):
        y = df_auc[col_map[name]].to_numpy(dtype=float)
        x0 = float(i)
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(y))
        xs = x0 + jitter

        ax.scatter(xs, y, s=POINT_SIZE, c="black", edgecolors="black", linewidths=0.0, alpha=1.0)

        mean_y = float(np.mean(y))
        ax.hlines(mean_y, x0 - MEAN_LINE_HALF_WIDTH, x0 + MEAN_LINE_HALF_WIDTH,
                  colors="black", linewidth=MEAN_LINEWIDTH)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(FIG1_DPI), format="tiff")
    plt.close(fig)

# ============================================================
# Main
# ============================================================
def main() -> None:
    nested_root = Path(NESTED_RESULTS_ROOT) if str(NESTED_RESULTS_ROOT).strip() else None
    out_root = Path(COMPARE_OUTPUT_ROOT) / COMPARE_SUBDIR
    _ensure_dir(out_root)

    legacy_root = Path(LEGACY_COMPARE_ROOT) if str(LEGACY_COMPARE_ROOT).strip() else None

    rows_auc: List[Dict[str, Any]] = []
    rows_k: List[Dict[str, Any]] = []
    sanity_rows: List[Dict[str, Any]] = []

    for item in DATASETS:
        # item = (xlsx_path, ncv_dir) where ncv_dir contains metrics_summary.csv and tuning_summary.csv
        xlsx_path = str(item[0])
        ncv_dir_raw = str(item[1]) if len(item) > 1 else ""
        dataset_stem = Path(xlsx_path).stem
        dataset_name = dataset_stem

        print("=" * 80)
        print(f"[DATASET] {dataset_name}")
        print("=" * 80)

        # Load data
        X, y, feature_names, ids = base.read_xlsx_dataset(xlsx_path)
        n, p = X.shape

        # Read nCV from existing ICI_predict outputs
        if ncv_dir_raw.strip():
            ncv_dir = Path(ncv_dir_raw)
        else:
            if nested_root is None:
                raise ValueError(
                    f"{dataset_name}: per-dataset nCV dir is empty and NESTED_RESULTS_ROOT is also empty. "
                    "Please set either the per-dataset nCV dir or NESTED_RESULTS_ROOT."
                )
            ncv_dir = nested_root / dataset_stem

        auc_ncv, k_ncv_list = _read_nested_auc_and_k(dataset_stem, ncv_dir)

        # Precompute outer splits (fixed across methods)
        outer_splits_list = _compute_outer_splits_list(X, y, dataset_name)
        cv_single = PrecomputedCV(outer_splits_list)

        # Global leaky spec (Compare-compatible)
        leaky = _compute_global_leaky_spec(X, y, feature_names, cv_single, dataset_name)
        ranking_fixed = leaky["ranking"]
        k_fixed = int(leaky["k"])
        l2_C_fixed = float(leaky["l2_C"])
        feats_fixed = leaky["features"]

        # ---- CV (fully non-nested; Compare-compatible)
        p_oof_cv = _oof_predict_with_fixed_spec(
            X, y, outer_splits_list, dataset_name,
            features=feats_fixed, l2_C=l2_C_fixed,
            stage_prefix="CV(single-layer)",
        )
        auc_cv = float(roc_auc_score(y, p_oof_cv))
        k_cv_list = np.asarray([k_fixed] * len(outer_splits_list), dtype=int)

        # ---- L2-nest (ranking fixed leaky; L2 selection nested)
        p_oof_l2nest, k_l2nest_list = _oof_predict_L2_nested(
            X, y, feature_names, outer_splits_list, ranking_fixed, dataset_name
        )
        auc_l2nest = float(roc_auc_score(y, p_oof_l2nest))

        # ---- Enet-nest (ENet ranking nested; L2 spec fixed leaky)
        p_oof_enetnest = _oof_predict_ENet_nested(
            X, y, feature_names, outer_splits_list, fixed_k=k_fixed, fixed_l2_C=l2_C_fixed, dataset_name=dataset_name
        )
        auc_enetnest = float(roc_auc_score(y, p_oof_enetnest))
        k_enetnest_list = np.asarray([k_fixed] * len(outer_splits_list), dtype=int)

        rows_auc.append({
            "dataset_stem": dataset_stem,
            "xlsx_path": xlsx_path,
            "n_samples": int(n),
            "n_features": int(p),
            "AUC_nCV": float(auc_ncv),
            "AUC_Enet-nest": float(auc_enetnest),
            "AUC_L2-nest": float(auc_l2nest),
            "AUC_CV": float(auc_cv),
        })

        rows_k.append({
            "dataset_stem": dataset_stem,
            **{f"nCV_{k}": v for k, v in _k_summary_stats(k_ncv_list).items()},
            **{f"Enet-nest_{k}": v for k, v in _k_summary_stats(k_enetnest_list).items()},
            **{f"L2-nest_{k}": v for k, v in _k_summary_stats(k_l2nest_list).items()},
            **{f"CV_{k}": v for k, v in _k_summary_stats(k_cv_list).items()},
        })

        # sanity check vs legacy Compare outputs (CV AUC)
        if legacy_root is not None:
            legacy_wide = legacy_root / "comparison_auc_ci_width_wide.csv"
            if legacy_wide.exists():
                try:
                    df_legacy = pd.read_csv(legacy_wide)
                    if "dataset_stem" in df_legacy.columns and "nonnested_auc" in df_legacy.columns:
                        hit = df_legacy.loc[df_legacy["dataset_stem"].astype(str) == str(dataset_stem)]
                        if len(hit) > 0:
                            legacy_auc = float(hit.iloc[0]["nonnested_auc"])
                            sanity_rows.append({
                                "dataset_stem": dataset_stem,
                                "legacy_CV_auc": legacy_auc,
                                "new_CV_auc": float(auc_cv),
                                "diff_new_minus_legacy": float(auc_cv - legacy_auc),
                            })
                except Exception as e:
                    print(f"[WARN] legacy sanity check skipped for {dataset_stem}: {e}")

        print(f"[AUC] nCV={auc_ncv:.4f} | Enet-nest={auc_enetnest:.4f} | L2-nest={auc_l2nest:.4f} | CV={auc_cv:.4f}")

    # ----------------------------
    # Save tables
    # ----------------------------
    df_auc = pd.DataFrame(rows_auc)
    df_auc = df_auc.sort_values("dataset_stem").reset_index(drop=True)
    df_auc.to_csv(out_root / "auc_by_dataset.csv", index=False, encoding="utf-8-sig")

    df_k = pd.DataFrame(rows_k)
    df_k = df_k.sort_values("dataset_stem").reset_index(drop=True)
    df_k.to_csv(out_root / "k_summary.csv", index=False, encoding="utf-8-sig")

    # ----------------------------
    # Wilcoxon vs nCV
    # ----------------------------
    deltas = {
        "Enet-nest": (df_auc["AUC_Enet-nest"] - df_auc["AUC_nCV"]).to_numpy(dtype=float),
        "L2-nest": (df_auc["AUC_L2-nest"] - df_auc["AUC_nCV"]).to_numpy(dtype=float),
        "CV": (df_auc["AUC_CV"] - df_auc["AUC_nCV"]).to_numpy(dtype=float),
    }

    comp_rows: List[Dict[str, Any]] = []
    pvals_raw: List[float] = []

    for name, d in deltas.items():
        p_raw = _wilcoxon_safe(d)
        pvals_raw.append(p_raw)
        comp_rows.append({
            "comparison": f"{name} vs nCV",
            "delta_auc_mean": float(np.mean(d)),
            "delta_auc_median": float(np.median(d)),
            "p_value": float(p_raw),
        })

    df_w = pd.DataFrame(comp_rows)
    df_w.to_csv(out_root / "wilcoxon_vs_nCV.csv", index=False, encoding="utf-8-sig")

    # ----------------------------
    # sanity check output
    # ----------------------------
    if sanity_rows:
        df_sanity = pd.DataFrame(sanity_rows).sort_values("dataset_stem").reset_index(drop=True)
        df_sanity.to_csv(out_root / "legacy_compare_sanity_check.csv", index=False, encoding="utf-8-sig")

    # ----------------------------
    # Fig
    # ----------------------------
    make_fig1_auc_jitter_mean(df_auc, out_root / FIG1_FILENAME)

    print("=" * 80)
    print("[DONE]")
    print(f"Outputs saved to: {out_root}")
    print("=" * 80)

if __name__ == "__main__":
    main()
