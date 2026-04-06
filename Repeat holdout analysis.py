"""
Sensitivity analysis for repeated holdout only.
What this script does
---------------------
- Reuses the existing pipeline settings from Compare_method_v2.py / ICI_predict_v2.py
- Runs stratified 70/30 holdout repeatedly (default: 25 repeats)
- Summarizes repeated-holdout ROC-AUC and 95% CI per dataset
- Compares repeated-holdout median CI width vs existing nested CV CI width
  using Wilcoxon signed-rank test (nominal p only)

Assumptions
-----------
- Put this script in the SAME directory that contains the per-dataset nested-result
  folders, e.g.:
      <SCRIPT_DIR>/RCC-UC_pre/metrics_summary.csv
      <SCRIPT_DIR>/NSCLC_pre/metrics_summary.csv
      ...
- Put this script in the SAME directory as Compare_method_v2.py and ICI_predict_v2.py
- Edit DATASETS below, then run once for Pre and once for Post
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Reuse the current comparison / pipeline implementation as much as possible
try:
    import Compare_method_v2 as cmp  # type: ignore
except ImportError as e:
    raise ImportError(
        "Compare_method_v2.py could not be imported. "
        "Place this script in the same folder as Compare_method_v2.py."
    ) from e

base = cmp.base


# ============================================================
# USER SETTINGS
# ============================================================

# Edit here and run once for Pre, once for Post.
# Format: (xlsx_path, display_label_for_table)
DATASETS: List[Tuple[str, str]] = [
    # (r".xlsx", "Dataset 1"),
    # (r".xlsx", "Dataset 2"),
]
RUN_NAME = "holdout_repeat_post" # Output subfolder name, created under the SAME directory as this script.
HOLDOUT_REPEATS = 25 # Repeated holdout settings

# Reuse holdout settings from Compare_method_v2.py
HOLDOUT_TEST_SIZE = cmp.SINGLE_SPLIT_TEST_SIZE
HOLDOUT_SPLIT_RANDOM_STATE = cmp.SINGLE_SPLIT_RANDOM_STATE

# Bootstrap seeds are offset across repeats, while keeping the current base seed.
BOOTSTRAP_SEED_BASE = int(base.RANDOM_STATE)

# Output filenames
OUT_PER_SPLIT = "holdout_repeat_per_split.csv"
OUT_SUMMARY = "holdout_repeat_summary.csv"
OUT_STATS = "holdout_repeat_vs_ncv_stats.csv"

# ============================================================
# Helpers
# ============================================================
def _ci_width(lo: float, hi: float) -> float:
    return float(hi - lo)

def _format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def _choose_representative_repeat(df_rep: pd.DataFrame) -> pd.Series:
    med_width = float(df_rep["ci_width"].median())
    mean_auc = float(df_rep["auc"].mean())
    tmp = df_rep.copy()
    tmp["_abs_width_dev"] = (tmp["ci_width"] - med_width).abs()
    tmp["_abs_auc_dev"] = (tmp["auc"] - mean_auc).abs()
    tmp = tmp.sort_values(["_abs_width_dev", "_abs_auc_dev", "repeat"], ascending=[True, True, True])
    return tmp.iloc[0]

def run_repeated_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
) -> Dict[str, Any]:
    splitter = StratifiedShuffleSplit(
        n_splits=HOLDOUT_REPEATS,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_SPLIT_RANDOM_STATE,
    )

    rows: List[Dict[str, Any]] = []

    for rep_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        te_idx = np.asarray(te_idx, dtype=int)

        p_te = cmp._fit_predict_pipeline_on_split(
            X, y, feature_names,
            tr_idx=tr_idx,
            te_idx=te_idx,
            dataset_name=dataset_name,
            tag=f"repeat_holdout_{rep_idx}",
        )

        y_te = y[te_idx]
        auc = float(roc_auc_score(y_te, p_te))
        ci_lo, ci_hi = base.bootstrap_auc_ci(
            y_te,
            p_te,
            n_boot=base.BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED_BASE + rep_idx,
        )
        width = _ci_width(ci_lo, ci_hi)

        rows.append({
            "repeat": int(rep_idx),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_test_pos": int(np.sum(y_te == 1)),
            "n_test_neg": int(np.sum(y_te == 0)),
            "auc": auc,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "ci_width": width,
        })

    df_rep = pd.DataFrame(rows)
    rep_row = _choose_representative_repeat(df_rep)

    summary = {
        "holdout_repeat_auc_mean": float(df_rep["auc"].mean()),
        "holdout_repeat_auc_min": float(df_rep["auc"].min()),
        "holdout_repeat_auc_max": float(df_rep["auc"].max()),
        "holdout_repeat_auc_sd": float(df_rep["auc"].std(ddof=1)) if len(df_rep) > 1 else 0.0,
        "holdout_repeat_ci_lo_repr": float(rep_row["ci_lo"]),
        "holdout_repeat_ci_hi_repr": float(rep_row["ci_hi"]),
        "holdout_repeat_ci_width_median": float(df_rep["ci_width"].median()),
        "holdout_repeat_ci_width_min": float(df_rep["ci_width"].min()),
        "holdout_repeat_ci_width_max": float(df_rep["ci_width"].max()),
        "holdout_repeat_ci_width_mean": float(df_rep["ci_width"].mean()),
        "representative_repeat": int(rep_row["repeat"]),
        "representative_repeat_auc": float(rep_row["auc"]),
        "representative_repeat_ci_width": float(rep_row["ci_width"]),
    }

    return {
        "per_split": df_rep,
        "summary": summary,
    }

# ============================================================
# Main
# ============================================================
def main() -> None:
    if not DATASETS:
        raise RuntimeError("DATASETS is empty. Edit DATASETS at the top of the script.")

    script_dir = Path(__file__).resolve().parent
    nested_root = script_dir
    out_root = script_dir / RUN_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    all_repeat_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for xlsx_path, dataset_label in DATASETS:
        dataset_stem = Path(xlsx_path).stem
        print("=" * 80)
        print(f"[HOLDOUT-REPEAT] {dataset_stem}  ({dataset_label})")
        print("=" * 80)

        X, y, feature_names, ids = base.read_xlsx_dataset(xlsx_path)
        nm = cmp._read_nested_metrics(dataset_stem, nested_root)
        rh = run_repeated_holdout(X, y, feature_names, dataset_name=dataset_stem)

        ds_out = out_root / dataset_stem
        ds_out.mkdir(parents=True, exist_ok=True)

        df_rep = rh["per_split"].copy()
        df_rep.insert(0, "dataset_label", dataset_label)
        df_rep.insert(0, "dataset_stem", dataset_stem)
        df_rep.insert(0, "xlsx_path", xlsx_path)
        df_rep.to_csv(ds_out / OUT_PER_SPLIT, index=False, encoding="utf-8-sig")
        all_repeat_rows.append(df_rep)

        summary = rh["summary"]
        nested_ci_width = _ci_width(nm["nested_ci_lo"], nm["nested_ci_hi"])
        row = {
            "dataset_stem": dataset_stem,
            "dataset_label": dataset_label,
            "xlsx_path": xlsx_path,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "nested_auc": float(nm["nested_auc"]),
            "nested_ci_lo": float(nm["nested_ci_lo"]),
            "nested_ci_hi": float(nm["nested_ci_hi"]),
            "nested_ci_width": nested_ci_width,
            **summary,
            "delta_ci_width_holdoutRepeat_minus_nCV": float(summary["holdout_repeat_ci_width_median"] - nested_ci_width),
            "delta_auc_holdoutRepeat_minus_nCV": float(summary["holdout_repeat_auc_mean"] - float(nm["nested_auc"])),
            "nested_metrics_path": str(nm["metrics_path"]),
        }
        summary_rows.append(row)

        meta = {
            "dataset_stem": dataset_stem,
            "dataset_label": dataset_label,
            "run_name": RUN_NAME,
            "holdout_repeats": HOLDOUT_REPEATS,
            "holdout_test_size": HOLDOUT_TEST_SIZE,
            "holdout_split_random_state": HOLDOUT_SPLIT_RANDOM_STATE,
            "bootstrap_n": int(base.BOOTSTRAP_N),
            "bootstrap_seed_base": int(BOOTSTRAP_SEED_BASE),
            "representative_repeat": int(summary["representative_repeat"]),
            "representative_repeat_rule": "CI width closest to median CI width; tie-break by AUC closest to mean, then repeat index",
        }
        (ds_out / "meta_holdout_repeat.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    df_all_rep = pd.concat(all_repeat_rows, axis=0, ignore_index=True)
    df_all_rep.to_csv(out_root / OUT_PER_SPLIT, index=False, encoding="utf-8-sig")

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_root / OUT_SUMMARY, index=False, encoding="utf-8-sig")

    widths_hold = df_sum["holdout_repeat_ci_width_median"].to_numpy(dtype=float)
    widths_ncv = df_sum["nested_ci_width"].to_numpy(dtype=float)
    deltas = df_sum["delta_ci_width_holdoutRepeat_minus_nCV"].to_numpy(dtype=float)

    p_raw = float(wilcoxon(widths_hold, widths_ncv).pvalue)

    stats_row = {
        "run_name": RUN_NAME,
        "n_datasets": int(len(df_sum)),
        "holdout_repeats": int(HOLDOUT_REPEATS),
        "test_size": float(HOLDOUT_TEST_SIZE),
        "mean_nested_ci_width": float(np.mean(widths_ncv)),
        "mean_holdout_repeat_median_ci_width": float(np.mean(widths_hold)),
        "mean_delta_ci_width_holdoutRepeat_minus_nCV": float(np.mean(deltas)),
        "min_delta_ci_width_holdoutRepeat_minus_nCV": float(np.min(deltas)),
        "max_delta_ci_width_holdoutRepeat_minus_nCV": float(np.max(deltas)),
        "p_nominal": p_raw,
        "p_nominal_fmt": _format_p(p_raw),
    }
    pd.DataFrame([stats_row]).to_csv(out_root / OUT_STATS, index=False, encoding="utf-8-sig")

    print("\n[OK] Saved:")
    print(f"- {out_root / OUT_PER_SPLIT}")
    print(f"- {out_root / OUT_SUMMARY}")
    print(f"- {out_root / OUT_STATS}")

if __name__ == "__main__":
    main()