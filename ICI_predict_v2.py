"""
Metabolomics ML Pipeline
- Input: xlsx (1 sheet), 1 row = 1 sample
- Columns:
    - id (sample identifier)
    - Responder (label 0/1; 1 = positive)
    - others: features (>=0), with 0 meaning missing
- Goal: performance-focused pipeline with strict nested CV (outer test never used in tuning/FS)
"""

from __future__ import annotations
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, brier_score_loss
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


# =========================
# USER SETTINGS 
# =========================
INPUT_FILES = [r"your dataset.xlsx"] # your input dataset
OUTPUT_ROOT = r"output" #output folder

ID_COL = "id" # patients ID
LABEL_COL = "Responder"  # 0/1, 1 = positive
RANDOM_STATE = 42

# Nested Cross Validation settings
OUTER_SPLITS_TARGET = 5
OUTER_REPEATS = 5
INNER_SPLITS_TARGET = 3
INNER_REPEATS = 5

# Preprocess / Feature selection
MISSING_RATE_THRESHOLD = 0.50     # drop if > 50% missing (0 treated as missing)
MAX_K = 10  # Top-k

# Hyperparameter grids
EN_C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1, 3]
EN_L1_RATIO_GRID = [0, 0.05, 0.2, 0.5, 0.8, 0.95]
L2_C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3]

# Convergence handling
EN_MAX_ITER_INIT = 5000
EN_MAX_ITER_LIMIT = 20000
L2_MAX_ITER_INIT = 2000
L2_MAX_ITER_LIMIT = 10000
TOL = 1e-4

# Metrics / plots
BOOTSTRAP_N = 2000
CALIBRATION_BINS_TARGET = 5  # auto-adjust downward for small n
DCA_PT_MIN = 0.30
DCA_PT_MAX = 0.70
DCA_PT_STEP = 0.05

FIG_FORMAT = "tiff"
FIG_DPI = 300
# =========================
# Utilities
# =========================
def clean_and_make_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def to_csv_utf8sig(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    df.to_csv(path, index=index, encoding="utf-8-sig")


def to_tsv_utf8sig(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    df.to_csv(path, sep="\t", index=index, encoding="utf-8-sig")


def adjusted_n_splits(y: np.ndarray, target_splits: int, context: str, log_lines: List[str]) -> int:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"{context}: Only one class present (classes={classes}). Cannot perform CV.")
    min_class = int(counts.min())
    if min_class < 2:
        raise ValueError(f"{context}: min class count < 2 (min_class={min_class}). Cannot perform CV.")
    used = min(target_splits, min_class)
    if used < target_splits:
        msg = f"[WARN] {context}: requested n_splits={target_splits}, using n_splits={used} (min_class={min_class})"
        print(msg)
        log_lines.append(msg)
    else:
        msg = f"[INFO] {context}: using n_splits={used} (min_class={min_class})"
        print(msg)
        log_lines.append(msg)
    return used


# =========================
# Preprocessor
# =========================
class MetaboPreprocessor:
    def __init__(self, feature_names: List[str], missing_rate_threshold: float = 0.50):
        self.feature_names = list(feature_names)
        self.missing_rate_threshold = float(missing_rate_threshold)

        # learned
        self._kept_after_missing: List[str] = []
        self._impute_values: pd.Series | None = None
        self._kept_features: List[str] = []
        self._mean_: pd.Series | None = None
        self._std_: pd.Series | None = None

    @staticmethod
    def _to_float_df(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def fit(self, X: pd.DataFrame) -> "MetaboPreprocessor":
        X = X[self.feature_names].copy()
        X = self._to_float_df(X)
        X = X.mask(X < 0, np.nan)
        X = X.replace(0, np.nan)

        # missing-rate filter
        miss_rate = X.isna().mean(axis=0)
        miss_rank = miss_rate.sort_values()

        kept = miss_rank.index[miss_rank <= self.missing_rate_threshold].tolist()

        if len(kept) == 0:
            kept = [miss_rank.index[0]]

        self._kept_after_missing = kept
        Xk = X[kept].copy()
        Xk = X[kept].copy()

        # compute impute values: half of min positive (non-missing) per feature
        min_pos = Xk.min(axis=0, skipna=True)
        min_pos = min_pos.fillna(1e-12)
        impute = (min_pos / 2.0).astype(float)
        # guard: if min_pos == 0, impute stays 0 -> would become missing again; set tiny
        impute = impute.where(impute > 0, other=1e-12)
        self._impute_values = impute
        Xk = Xk.fillna(impute)
        Xlog = np.log1p(Xk) # # log1p

        # constant filter (variance == 0)
        var = Xlog.var(axis=0, ddof=0)
        keep2 = var.index[var > 0].tolist()
        if len(keep2) == 0:
            keep2 = Xlog.columns.tolist()
        self._kept_features = keep2
        Xf = Xlog[keep2].copy()

        # z-scoring
        mean_ = Xf.mean(axis=0)
        std_ = Xf.std(axis=0, ddof=0)
        std_ = std_.where(std_ > 0, other=1.0)
        self._mean_ = mean_
        self._std_ = std_

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._impute_values is None or self._mean_ is None or self._std_ is None:
            raise RuntimeError("MetaboPreprocessor is not fitted.")

        # allow extra columns, but require all columns used at fit time
        missing_cols = [c for c in self._kept_after_missing if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required columns: {missing_cols}")

        X = X[self._kept_after_missing].copy()
        X = self._to_float_df(X)
        X = X.mask(X < 0, np.nan)
        X = X.replace(0, np.nan)
        X = X.fillna(self._impute_values)

        Xlog = np.log1p(X)

        # keep constant-filtered columns
        Xlog = Xlog[self._kept_features]

        Xz = (Xlog - self._mean_) / self._std_
        return Xz.to_numpy(dtype=float)

    @property
    def kept_features_(self) -> List[str]:
        return list(self._kept_features)


# =========================
# Model package for deployment
# =========================
@dataclass
class FinalModelPackage:
    selected_features: List[str]     # Top-k features (columns to read)
    preprocessor: MetaboPreprocessor
    classifier: LogisticRegression
    threshold: float                 # Youden threshold from CV OOF
    metadata: Dict

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.selected_features].copy()
        Xp = self.preprocessor.transform(X)
        return self.classifier.predict_proba(Xp)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)


# =========================
# Fitting helpers (with refit on non-convergence)
# =========================
def fit_logreg_with_refit(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    C: float,
    l1_ratio: Optional[float],
    dataset_name: str,
    stage: str,
    log_lines: List[str],
) -> LogisticRegression:
    if kind == "elasticnet":
        max_iter_init = EN_MAX_ITER_INIT
        max_iter_limit = EN_MAX_ITER_LIMIT
        solver = "saga"
        penalty = "elasticnet"
        if l1_ratio is None:
            raise ValueError("elasticnet requires l1_ratio")
    elif kind == "l2":
        max_iter_init = L2_MAX_ITER_INIT
        max_iter_limit = L2_MAX_ITER_LIMIT
        solver = "liblinear"
        penalty = "l2"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    max_iter = max_iter_init

    while True:
        model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=float(C),
            l1_ratio=None if kind != "elasticnet" else float(l1_ratio),
            class_weight="balanced",
            max_iter=int(max_iter),
            tol=float(TOL),
            random_state=RANDOM_STATE,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
            conv_warnings = [wi for wi in w if issubclass(wi.category, ConvergenceWarning)]

        if not conv_warnings:
            return model

        msg = (
            f"[WARN] {dataset_name} | {stage}: ConvergenceWarning at max_iter={max_iter}. "
            f"Refit with max_iter={min(max_iter*2, max_iter_limit)}"
        )
        log_lines.append(msg)

        if max_iter >= max_iter_limit:
            msg2 = f"[WARN] {dataset_name} | {stage}: reached max_iter_limit={max_iter_limit}; proceed without convergence."
            log_lines.append(msg2)
            return model

        max_iter = min(max_iter * 2, max_iter_limit)

# =========================
# Core: EN tuning, stability selection, k-selection
# =========================
def make_inner_cv(y_train: np.ndarray, context: str, log_lines: List[str]) -> RepeatedStratifiedKFold:
    n_splits = adjusted_n_splits(y_train, INNER_SPLITS_TARGET, context, log_lines)
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=INNER_REPEATS,
        random_state=RANDOM_STATE,
    )

def tune_elasticnet(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    inner_cv: RepeatedStratifiedKFold,
    dataset_name: str,
    outer_iter: int,
    log_lines: List[str],
) -> Tuple[float, float, pd.DataFrame]:
    results = []
    best_mean = -np.inf
    best_params = (EN_C_GRID[0], EN_L1_RATIO_GRID[0])

    splits = list(inner_cv.split(X, y))
    n_eval = len(splits)

    for C in EN_C_GRID:
        for l1 in EN_L1_RATIO_GRID:
            aucs = []
            for fold_i, (tr, va) in enumerate(splits):
                Xtr = X.iloc[tr]
                Xva = X.iloc[va]
                ytr = y[tr]
                yva = y[va]

                prep = MetaboPreprocessor(feature_names, missing_rate_threshold=MISSING_RATE_THRESHOLD).fit(Xtr)
                Xtr_p = prep.transform(Xtr)
                Xva_p = prep.transform(Xva)

                clf = fit_logreg_with_refit(
                    Xtr_p, ytr,
                    kind="elasticnet", C=C, l1_ratio=l1,
                    dataset_name=dataset_name,
                    stage=f"EN-tune outer={outer_iter} innerfold={fold_i}",
                    log_lines=log_lines
                )
                p = clf.predict_proba(Xva_p)[:, 1]
                try:
                    auc = roc_auc_score(yva, p)
                except ValueError:
                    auc = np.nan
                aucs.append(auc)

            mean_auc = float(np.nanmean(np.asarray(aucs, dtype=float)))
            results.append({"C": C, "l1_ratio": l1, "mean_auc": mean_auc, "n_folds": n_eval})

            if mean_auc > best_mean:
                best_mean = mean_auc
                best_params = (C, l1)

    res_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    msg = f"[INFO] {dataset_name} | outer={outer_iter}: best EN params: C={best_params[0]}, l1_ratio={best_params[1]}, meanAUC={best_mean:.4f}"
    log_lines.append(msg)
    print(msg)
    return best_params[0], best_params[1], res_df

def stability_select_and_rank(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    inner_cv: RepeatedStratifiedKFold,
    en_C: float,
    en_l1: float,
    dataset_name: str,
    outer_iter: int,
    log_lines: List[str],
) -> Tuple[List[str], pd.DataFrame]:
    """
    Stability selection WITHOUT hard thresholding:
    - compute non-zero frequency and mean(|coef|)
    - rank ALL features by:
        1) nonzero_freq (desc)
        2) mean_abs_coef (desc)
        3) feature name (asc) for tie-break
    """
    p = len(feature_names)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    splits = list(inner_cv.split(X, y))
    n_splits_total = len(splits)

    nonzero_counts = np.zeros(p, dtype=int)
    abscoef_sum = np.zeros(p, dtype=float)

    for fold_i, (tr, va) in enumerate(splits):
        Xtr = X.iloc[tr]
        ytr = y[tr]

        prep = MetaboPreprocessor(feature_names, missing_rate_threshold=MISSING_RATE_THRESHOLD).fit(Xtr)
        Xtr_p = prep.transform(Xtr)

        clf = fit_logreg_with_refit(
            Xtr_p, ytr,
            kind="elasticnet", C=en_C, l1_ratio=en_l1,
            dataset_name=dataset_name,
            stage=f"EN-stability outer={outer_iter} innerfold={fold_i}",
            log_lines=log_lines
        )

        kept = prep.kept_features_
        coef = clf.coef_.ravel()

        coef_vec = np.zeros(p, dtype=float)
        for j, fname in enumerate(kept):
            coef_vec[name_to_idx[fname]] = coef[j]

        nonzero_counts += (coef_vec != 0.0).astype(int)
        abscoef_sum += np.abs(coef_vec)

    freq = nonzero_counts / float(n_splits_total)
    mean_abs = abscoef_sum / float(n_splits_total)

    df = pd.DataFrame({
        "feature": feature_names,
        "nonzero_freq": freq,
        "mean_abs_coef": mean_abs,
    }).sort_values(
        ["nonzero_freq", "mean_abs_coef", "feature"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    ranking = df["feature"].tolist()
    return ranking, df

def select_k_by_1se(
    X: pd.DataFrame,
    y: np.ndarray,
    ranking: List[str],
    inner_cv: RepeatedStratifiedKFold,
    dataset_name: str,
    outer_iter: int,
    log_lines: List[str],
) -> Tuple[int, float, pd.DataFrame]:
    max_k = min(MAX_K, len(ranking))
    if max_k < 1:
        raise ValueError("No features available for k-selection.")

    splits = list(inner_cv.split(X, y))
    n_eval = len(splits)

    per_k_rows = []

    for k in range(1, max_k + 1):
        feats_k = ranking[:k]
        best_mean = -np.inf
        best_se = np.inf
        best_C = L2_C_GRID[0]

        for C in L2_C_GRID:
            aucs = []
            for fold_i, (tr, va) in enumerate(splits):
                Xtr = X.iloc[tr][feats_k]
                Xva = X.iloc[va][feats_k]
                ytr = y[tr]
                yva = y[va]

                prep = MetaboPreprocessor(feats_k, missing_rate_threshold=MISSING_RATE_THRESHOLD).fit(Xtr)
                Xtr_p = prep.transform(Xtr)
                Xva_p = prep.transform(Xva)

                clf = fit_logreg_with_refit(
                    Xtr_p, ytr,
                    kind="l2", C=C, l1_ratio=None,
                    dataset_name=dataset_name,
                    stage=f"L2-kselect outer={outer_iter} k={k} C={C} innerfold={fold_i}",
                    log_lines=log_lines
                )
                p = clf.predict_proba(Xva_p)[:, 1]
                try:
                    auc = roc_auc_score(yva, p)
                except ValueError:
                    auc = np.nan
                aucs.append(auc)

            aucs = np.asarray(aucs, dtype=float)
            mean_auc = float(np.nanmean(aucs))
            sd = float(np.nanstd(aucs, ddof=1)) if np.sum(~np.isnan(aucs)) > 1 else 0.0
            se = sd / np.sqrt(np.sum(~np.isnan(aucs))) if np.sum(~np.isnan(aucs)) > 0 else np.inf

            if mean_auc > best_mean:
                best_mean = mean_auc
                best_se = se
                best_C = C

        per_k_rows.append({
            "outer_iter": outer_iter,
            "k": k,
            "best_C": best_C,
            "mean_auc": best_mean,
            "se_auc": best_se,
            "n_folds": n_eval,
        })

    per_k_df = pd.DataFrame(per_k_rows)

    # 1SE rule
    best_idx = per_k_df["mean_auc"].idxmax()
    best_mean = float(per_k_df.loc[best_idx, "mean_auc"])
    best_se = float(per_k_df.loc[best_idx, "se_auc"])
    threshold = best_mean - best_se

    eligible = per_k_df[per_k_df["mean_auc"] >= threshold].sort_values("k")
    chosen_row = eligible.iloc[0]
    chosen_k = int(chosen_row["k"])
    chosen_C = float(chosen_row["best_C"])

    msg = (
        f"[INFO] {dataset_name} | outer={outer_iter}: k-selection best_mean={best_mean:.4f}, best_se={best_se:.4f}, "
        f"1SE_threshold={threshold:.4f} -> chosen k={chosen_k}, C={chosen_C}"
    )
    print(msg)
    log_lines.append(msg)

    return chosen_k, chosen_C, per_k_df

# =========================
# Evaluation utilities
# =========================
def youden_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, p)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thr[idx])

def bootstrap_auc_ci(y_true: np.ndarray, p: np.ndarray, n_boot: int, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    attempts = 0
    while len(aucs) < n_boot and attempts < n_boot * 10:
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            attempts += 1
            continue
        aucs.append(roc_auc_score(yt, p[idx]))
        attempts += 1
    if len(aucs) < max(100, n_boot // 10):
        raise RuntimeError("Bootstrap failed too often due to single-class resamples.")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)

def decision_curve(y_true: np.ndarray, p: np.ndarray, pt_grid: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y_true)
    prev = y_true.mean()

    rows = []
    for pt in pt_grid:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())

        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        nb_none = 0.0

        rows.append({"pt": pt, "net_benefit_model": nb, "net_benefit_all": nb_all, "net_benefit_none": nb_none})

    return pd.DataFrame(rows)

# =========================
# Plotting
# =========================
def save_roc_plot(y_true: np.ndarray, p: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, linewidth=2, color="black", label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, format=FIG_FORMAT)
    plt.close(fig)


def save_pr_plot(y_true: np.ndarray, p: np.ndarray, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rec, prec, linewidth=2, color="black", label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-AUC")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, format=FIG_FORMAT)
    plt.close(fig)


def save_confusion_matrix_plot(y_true: np.ndarray, p: np.ndarray, thr: float, out_path: Path) -> None:
    y_pred = (p >= thr).astype(int)
    cm_counts = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm_prop = cm_counts / cm_counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm_prop, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

    for i in range(2):
        for j in range(2):
            prop = cm_prop[i, j]
            cnt = cm_counts[i, j]
            ax.text(j, i, f"{prop:.2f}\n({cnt})",
                    ha="center", va="center",
                    color="white" if prop > 0.5 else "black", fontsize=11)

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Responder", "Non-responder"])
    ax.set_yticklabels(["Responder", "Non-responder"])
    ax.text(0.5, -0.12, f"thr={thr:.2f}", transform=ax.transAxes, ha="center", va="top")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion within true label")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, format=FIG_FORMAT)
    plt.close(fig)


def save_calibration_plot(y_true: np.ndarray, p: np.ndarray, n_bins: int, out_path: Path) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, p)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="black")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration")
    ax.text(0.98, 0.02, f"Brier={brier:.3f}\nBins={n_bins}", transform=ax.transAxes,
            ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, format=FIG_FORMAT)
    plt.close(fig)
    return float(brier)


def save_dca_plot(dca_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(dca_df["pt"], dca_df["net_benefit_model"], linewidth=2, color="black", label="Model")
    ax.plot(dca_df["pt"], dca_df["net_benefit_all"], linestyle="--", linewidth=1.5, color="gray", label="Treat all")
    ax.plot(dca_df["pt"], dca_df["net_benefit_none"], linestyle=":", linewidth=1.5, color="gray", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("DCA")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, format=FIG_FORMAT)
    plt.close(fig)


def save_k_curve_plot(k_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(k_summary["k"], k_summary["mean_auc_across_outer"], linewidth=2, color="black")
    ax.fill_between(
        k_summary["k"],
        k_summary["mean_auc_across_outer"] - k_summary["se_auc_across_outer"],
        k_summary["mean_auc_across_outer"] + k_summary["se_auc_across_outer"],
        alpha=0.2,
        color="gray",
        linewidth=0
    )
    ax.set_xlabel("k")
    ax.set_ylabel("ROC-AUC (inner CV, best C)")
    ax.set_title("k vs ROC-AUC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, format=FIG_FORMAT)
    plt.close(fig)

# =========================
# Data I/O
# =========================
def read_xlsx_dataset(path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.Series]:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    if ID_COL not in df.columns:
        raise ValueError(f"Missing ID column '{ID_COL}' in {path}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column '{LABEL_COL}' in {path}")

    if df[ID_COL].duplicated().any():
        dup = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()[:10]
        raise ValueError(f"Duplicate IDs found in {path}. Examples: {dup}")

    y = df[LABEL_COL].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Label column must be 0/1 only. Found: {np.unique(y)}")

    feature_cols = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
    X = df[feature_cols].copy()

    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().all(axis=None):
        raise ValueError(f"All features are NaN after numeric conversion in {path}")
    ids = df[ID_COL].copy()
    return X, y, feature_cols, ids

# =========================
# Main per-dataset runner
# =========================
def run_one_dataset(xlsx_path: str) -> None:
    dataset_name = Path(xlsx_path).stem
    out_dir = Path(OUTPUT_ROOT) / dataset_name
    clean_and_make_dir(out_dir)

    log_lines: List[str] = []
    log_lines.append(f"[INFO] dataset={dataset_name}")
    log_lines.append(f"[INFO] input={xlsx_path}")

    print("=" * 80)
    print(f"[START] {dataset_name}")
    print("=" * 80)

    X, y, feature_names, ids = read_xlsx_dataset(xlsx_path)
    n, p = X.shape
    log_lines.append(f"[INFO] n_samples={n}, n_features={p}")

    outer_splits = adjusted_n_splits(y, OUTER_SPLITS_TARGET, f"{dataset_name} outer", log_lines)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE
    )

    oof_sum = np.zeros(n, dtype=float)
    oof_count = np.zeros(n, dtype=int)

    tuning_rows = []
    k_curve_rows = []
    topk_counter: Dict[str, int] = {f: 0 for f in feature_names}
    outer_topk_rows = []

    n_outer_iters = outer_splits * OUTER_REPEATS

    for outer_iter, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        te_idx = np.asarray(te_idx, dtype=int)

        Xtr_full = X.iloc[tr_idx]
        ytr = y[tr_idx]
        Xte_full = X.iloc[te_idx]

        inner_cv = make_inner_cv(ytr, f"{dataset_name} inner (outer_iter={outer_iter})", log_lines)

        en_C, en_l1, _ = tune_elasticnet(
            Xtr_full, ytr, feature_names,
            inner_cv=inner_cv,
            dataset_name=dataset_name,
            outer_iter=outer_iter,
            log_lines=log_lines,
        )

        ranking, stability_df = stability_select_and_rank(
            Xtr_full, ytr, feature_names,
            inner_cv=inner_cv,
            en_C=en_C,
            en_l1=en_l1,
            dataset_name=dataset_name,
            outer_iter=outer_iter,
            log_lines=log_lines,
        )

        k_sel, l2_C, per_k_df = select_k_by_1se(
            Xtr_full, ytr, ranking,
            inner_cv=inner_cv,
            dataset_name=dataset_name,
            outer_iter=outer_iter,
            log_lines=log_lines
        )
        k_curve_rows.append(per_k_df)

        feats_k = ranking[:k_sel]
        for f in feats_k:
            if f in topk_counter:
                topk_counter[f] += 1

        outer_topk_rows.append({
            "outer_iter": outer_iter,
            "k": k_sel,
            "features": ";".join(feats_k)
        })

        prep = MetaboPreprocessor(feats_k, missing_rate_threshold=MISSING_RATE_THRESHOLD).fit(Xtr_full[feats_k])
        Xtr_p = prep.transform(Xtr_full[feats_k])
        clf = fit_logreg_with_refit(
            Xtr_p, ytr,
            kind="l2", C=l2_C, l1_ratio=None,
            dataset_name=dataset_name,
            stage=f"L2-outerfit outer={outer_iter}",
            log_lines=log_lines
        )

        Xte_p = prep.transform(Xte_full[feats_k])
        p_te = clf.predict_proba(Xte_p)[:, 1]

        oof_sum[te_idx] += p_te
        oof_count[te_idx] += 1

        tuning_rows.append({
            "outer_iter": outer_iter,
            "outer_splits_used": outer_splits,
            "inner_splits_used": inner_cv.get_n_splits(),
            "en_C": en_C,
            "en_l1_ratio": en_l1,
            "k_selected": k_sel,
            "l2_C": l2_C,
            "n_ranked_features": len(ranking),
        })

        print(f"[{dataset_name}] outer_iter {outer_iter}/{n_outer_iters} done")

    if (oof_count == 0).any():
        raise RuntimeError("Some samples were never predicted in outer CV (unexpected).")

    p_oof = oof_sum / oof_count
    oof_df = pd.DataFrame({
        ID_COL: ids.values,
        "y_true": y,
        "p_oof": p_oof,
        "oof_count": oof_count
    })
    to_csv_utf8sig(oof_df, out_dir / "oof_predictions.csv", index=False)


    roc_auc = roc_auc_score(y, p_oof)
    pr_auc = average_precision_score(y, p_oof)
    ci_lo, ci_hi = bootstrap_auc_ci(y, p_oof, n_boot=BOOTSTRAP_N, seed=RANDOM_STATE)
    thr = youden_threshold(y, p_oof)

    n_bins = int(CALIBRATION_BINS_TARGET)
    while n_bins > 5 and n < n_bins * 5:
        n_bins -= 1
    msg = f"[INFO] {dataset_name}: calibration bins used = {n_bins}"
    log_lines.append(msg)
    print(msg)

    metrics_df = pd.DataFrame([{
        "dataset": dataset_name,
        "n_samples": n,
        "n_features": p,
        "roc_auc_oof": roc_auc,
        "roc_auc_ci95_lo": ci_lo,
        "roc_auc_ci95_hi": ci_hi,
        "pr_auc_oof": pr_auc,
        "youden_threshold": thr,
    }])
    to_csv_utf8sig(metrics_df, out_dir / "metrics_summary.csv")

    tuning_df = pd.DataFrame(tuning_rows)
    to_csv_utf8sig(tuning_df, out_dir / "tuning_summary.csv")

    k_curve_df = pd.concat(k_curve_rows, ignore_index=True)
    to_csv_utf8sig(k_curve_df, out_dir / "k_curve_per_outer.csv")

    k_summary = (
        k_curve_df.groupby("k")["mean_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_auc_across_outer", "std": "std_auc_across_outer", "count": "n_outer"})
    )
    k_summary["se_auc_across_outer"] = k_summary["std_auc_across_outer"] / np.sqrt(k_summary["n_outer"])
    to_csv_utf8sig(k_summary, out_dir / "k_curve_summary.csv")

    feat_freq = pd.DataFrame({
        "feature": list(topk_counter.keys()),
        "selected_count": list(topk_counter.values()),
        "selected_rate": [v / float(n_outer_iters) for v in topk_counter.values()],
    }).sort_values(["selected_rate", "feature"], ascending=[False, True]).reset_index(drop=True)
    to_csv_utf8sig(feat_freq, out_dir / "feature_selection_frequency.csv")

    outer_topk_df = pd.DataFrame(outer_topk_rows)
    to_csv_utf8sig(outer_topk_df, out_dir / "outer_topk_features.csv")

    save_roc_plot(y, p_oof, out_dir / f"ROC-AUC.{FIG_FORMAT}")
    save_pr_plot(y, p_oof, out_dir / f"PR-AUC.{FIG_FORMAT}")
    save_confusion_matrix_plot(y, p_oof, thr, out_dir / f"Confusion_matrix.{FIG_FORMAT}")
    brier = save_calibration_plot(y, p_oof, n_bins, out_dir / f"Calibration.{FIG_FORMAT}")

    pt_grid = np.round(np.arange(DCA_PT_MIN, DCA_PT_MAX + 1e-9, DCA_PT_STEP), 2)
    dca_df = decision_curve(y, p_oof, pt_grid)
    to_csv_utf8sig(dca_df, out_dir / "dca.csv")
    save_dca_plot(dca_df, out_dir / f"DCA.{FIG_FORMAT}")

    save_k_curve_plot(k_summary, out_dir / f"k_vs_ROC-AUC.{FIG_FORMAT}")

    # Final model (deployment): fit on all data, save only
    inner_cv_full = make_inner_cv(y, f"{dataset_name} final-inner", log_lines)
    en_C_final, en_l1_final, _ = tune_elasticnet(
        X, y, feature_names,
        inner_cv=inner_cv_full,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines
    )
    ranking_final, _ = stability_select_and_rank(
        X, y, feature_names,
        inner_cv=inner_cv_full,
        en_C=en_C_final,
        en_l1=en_l1_final,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines
    )
    k_final, l2_C_final, _ = select_k_by_1se(
        X, y, ranking_final,
        inner_cv=inner_cv_full,
        dataset_name=dataset_name,
        outer_iter=0,
        log_lines=log_lines
    )

    feats_final = ranking_final[:k_final]

    prep_final = MetaboPreprocessor(feats_final, missing_rate_threshold=MISSING_RATE_THRESHOLD).fit(X[feats_final])
    Xp_final = prep_final.transform(X[feats_final])
    clf_final = fit_logreg_with_refit(
        Xp_final, y,
        kind="l2", C=l2_C_final, l1_ratio=None,
        dataset_name=dataset_name,
        stage="L2-finalfit",
        log_lines=log_lines
    )

    metadata = {
        "dataset": dataset_name,
        "id_col": ID_COL,
        "label_col": LABEL_COL,
        "random_state": RANDOM_STATE,
        "outer_splits_used": outer_splits,
        "outer_repeats": OUTER_REPEATS,
        "inner_splits_target": INNER_SPLITS_TARGET,
        "inner_repeats": INNER_REPEATS,
        "missing_rate_threshold": MISSING_RATE_THRESHOLD,
        "max_k": MAX_K,
        "en_C_grid": EN_C_GRID,
        "en_l1_ratio_grid": EN_L1_RATIO_GRID,
        "l2_C_grid": L2_C_GRID,
        "final_en_C": en_C_final,
        "final_en_l1_ratio": en_l1_final,
        "final_k": k_final,
        "final_l2_C": l2_C_final,
        "youden_threshold_from_oof": thr,
        "oof_roc_auc": roc_auc,
        "oof_roc_auc_ci95": [ci_lo, ci_hi],
        "oof_pr_auc": pr_auc,
        "brier_oof": brier,
    }

    package = FinalModelPackage(
        selected_features=feats_final,
        preprocessor=prep_final,
        classifier=clf_final,
        threshold=thr,
        metadata=metadata
    )

    joblib.dump(package, out_dir / "final_model.joblib")

    final_feat_list = pd.DataFrame({
        "rank": np.arange(1, len(feats_final) + 1),
        "feature": feats_final
    })
    to_tsv_utf8sig(final_feat_list, out_dir / "final_feature_list.tsv")

    used_features_after_prep = prep_final.kept_features_
    coef = clf_final.coef_.ravel()
    coef_df = pd.DataFrame({
        "feature_after_preprocess": used_features_after_prep,
        "coef": coef
    }).sort_values("coef", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    coef_df.loc[len(coef_df)] = {"feature_after_preprocess": "INTERCEPT", "coef": float(clf_final.intercept_.ravel()[0])}
    to_tsv_utf8sig(coef_df, out_dir / "final_coefficients.tsv")

    write_text(out_dir / "final_threshold_youden.txt", f"{thr:.6f}\n")
    write_text(out_dir / "final_model_spec.json", json.dumps(metadata, indent=2, ensure_ascii=False))
    write_text(out_dir / "run_log.txt", "\n".join(log_lines) + "\n")

    print(f"[DONE] {dataset_name} -> {out_dir}")

def main() -> None:
    if not INPUT_FILES:
        raise RuntimeError("INPUT_FILES is empty. Please set INPUT_FILES at the top of the script.")
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    for xlsx in INPUT_FILES:
        run_one_dataset(xlsx)

if __name__ == "__main__":
    main()