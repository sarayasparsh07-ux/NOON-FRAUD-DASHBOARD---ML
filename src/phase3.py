"""Phase 3 — scoring frozen classifiers on the real referee test set.

Nothing in this module fits, tunes, calibrates or selects anything. Every function either inspects a file,
applies a model's own frozen prediction rule, or computes metrics from predictions that already exist.
Positive class = fraud (is_fraud = 1).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .config import BINARY_COLS, FEATURES, PAYMENT_METHODS, SCHEMA, TARGET
from .compare import sha256, wilson

INT_DICT_COLS = ["account_age_days", "hour_of_day", "num_orders_last_7d", "prev_refunds"]
FLOAT_DICT_COLS = ["order_amount", "delivery_distance_km"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def cells_digest(nb_path: Path, phase) -> str:
    """SHA-256 of the cells of one phase (source + outputs). phase=None → untagged (Phase 1) cells."""
    nb = json.loads(Path(nb_path).read_text())
    cells = [c for c in nb["cells"] if c.get("metadata", {}).get("phase") == phase]
    return hashlib.sha256(json.dumps([[c["cell_type"], c["source"], c.get("outputs", [])] for c in cells],
                                     sort_keys=True).encode()).hexdigest()


def verify_files(recorded: dict, root: Path) -> pd.DataFrame:
    rows = []
    for rel, digest in recorded.items():
        p = Path(root) / rel
        rows.append({"file": rel, "present": p.exists(), "unchanged": p.exists() and sha256(p) == digest})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Referee integrity (data-dictionary checks only; no labels are used beyond counting them)
# ---------------------------------------------------------------------------
def referee_integrity(ref: pd.DataFrame) -> pd.DataFrame:
    chk = []

    def add(name, ok, detail):
        chk.append({"check": name, "result": "Passed" if ok else "FAILED", "detail": detail})

    add("Rows", True, f"{len(ref):,}")
    add("Columns: names and order match the data dictionary", list(ref.columns) == list(SCHEMA), ", ".join(ref.columns))
    add("No missing values", int(ref.isna().sum().sum()) == 0, f"{int(ref.isna().sum().sum())} missing cells")
    add("No fully duplicated records (11 columns)", int(ref.duplicated().sum()) == 0, f"{int(ref.duplicated().sum())} duplicates")
    add("Integer columns hold whole numbers", all(pd.api.types.is_integer_dtype(ref[c]) for c in INT_DICT_COLS),
        ", ".join(f"{c}: {ref[c].dtype}" for c in INT_DICT_COLS))
    add("Number columns are numeric", all(pd.api.types.is_numeric_dtype(ref[c]) for c in FLOAT_DICT_COLS),
        ", ".join(f"{c}: {ref[c].dtype}" for c in FLOAT_DICT_COLS))
    add("payment_method ∈ {card, wallet, cod, applepay}", set(ref.payment_method.unique()) <= set(PAYMENT_METHODS),
        str(sorted(ref.payment_method.unique())))
    add("Binary flags and is_fraud ∈ {0, 1}", all(set(ref[c].unique()) <= {0, 1} for c in BINARY_COLS + [TARGET]),
        "; ".join(f"{c}: {sorted(ref[c].unique())}" for c in BINARY_COLS + [TARGET]))
    add("hour_of_day within 0–23", bool(ref.hour_of_day.between(0, 23).all()), f"{ref.hour_of_day.min()}–{ref.hour_of_day.max()}")
    nonneg = [c for c in FLOAT_DICT_COLS + INT_DICT_COLS if (ref[c] < 0).any()]
    add("No negative amounts, ages, counts or distances", not nonneg, "none negative" if not nonneg else str(nonneg))
    add("order_amount > 0", bool((ref.order_amount > 0).all()), f"min {ref.order_amount.min()}")
    return pd.DataFrame(chk)


def label_counts(ref: pd.DataFrame) -> dict:
    n, k = len(ref), int(ref[TARGET].sum())
    return {"rows": n, "legitimate (is_fraud = 0)": n - k, "fraud (is_fraud = 1)": k, "fraud prevalence": k / n,
            "distinct feature rows (10 predictors)": int(len(ref.drop_duplicates(FEATURES))),
            "feature rows shared by a legit and a fraud record": int(ref.groupby(FEATURES)[TARGET].nunique().gt(1).sum())}


def range_vs_training(ref: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: how many referee rows fall outside the range seen in my Phase 1 training data."""
    rows = []
    for c in FLOAT_DICT_COLS + INT_DICT_COLS:
        lo, hi = train[c].min(), train[c].max()
        out = (ref[c] < lo) | (ref[c] > hi)
        rows.append({"feature": c, "my training min": lo, "my training max": hi, "referee min": ref[c].min(),
                     "referee max": ref[c].max(), "referee rows outside my training range": int(out.sum()),
                     "of which fraud": int(ref.loc[out, TARGET].sum())})
    return pd.DataFrame(rows)


def exact_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Number of distinct 10-feature rows present in both frames (exact match)."""
    ka = set(map(tuple, a[FEATURES].astype(str).to_numpy()))
    kb = set(map(tuple, b[FEATURES].astype(str).to_numpy()))
    return len(ka & kb)


# ---------------------------------------------------------------------------
# Scoring with each model's own frozen rule
# ---------------------------------------------------------------------------
def score_fraudmodel(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated fraud probability and the model's own predict() (stored threshold)."""
    p = np.asarray(model.predict_proba(X[model.feature_order]), dtype=float)
    pred = np.asarray(model.predict(X[model.feature_order]), dtype=int)
    return p, pred


def score_noon(bundle: dict, X: pd.DataFrame, prep) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Noon's own rule: helper encoding reindexed to the stored columns, then model.predict().
    Probability is taken from the column whose class label is 1."""
    Xe = prep(X, columns=bundle["columns"])
    m = bundle["model"]
    pos = list(m.classes_).index(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = m.predict_proba(Xe)[:, pos]
        pred = np.asarray(m.predict(Xe), dtype=int)
    return p, pred, Xe


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_row(y, score, pred, name: str, rule: str) -> dict:
    y, score, pred = np.asarray(y).astype(int), np.asarray(score, float), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec_lo, rec_hi = wilson(int(tp), int(tp + fn))
    pre_lo, pre_hi = wilson(int(tp), int(tp + fp))
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {
        "model": name, "prediction rule": rule,
        "recall": tp / (tp + fn), "recall 95% CI low": rec_lo, "recall 95% CI high": rec_hi,
        "precision": prec, "precision 95% CI low": pre_lo, "precision 95% CI high": pre_hi,
        "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "flagged as fraud": int(tp + fp), "false-positive rate": fp / (fp + tn), "specificity": tn / (tn + fp),
        "average precision (PR-AUC)": average_precision_score(y, score), "ROC-AUC": roc_auc_score(y, score),
        "accuracy (supplementary)": (tp + tn) / len(y), "n": int(len(y)), "n fraud": int(y.sum()),
    }


def sklearn_crosscheck(y, pred, row: dict) -> dict:
    """Independent recomputation with scikit-learn's own metric functions."""
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "model": row["model"],
        "recall (sklearn)": recall_score(y, pred), "precision (sklearn)": precision_score(y, pred, zero_division=0),
        "F1 (sklearn)": f1_score(y, pred), "accuracy (sklearn)": accuracy_score(y, pred),
        "confusion [[TN, FP], [FN, TP]]": cm.tolist(),
        "all match": bool(np.isclose(row["recall"], recall_score(y, pred))
                          and np.isclose(row["precision"], precision_score(y, pred, zero_division=0))
                          and np.isclose(row["F1"], f1_score(y, pred))
                          and cm.tolist() == [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
                          and cm.sum() == len(y)),
    }


def bootstrap_ranking_ci(y, score, B: int, seed: int) -> dict:
    """Stratified bootstrap (resample fraud and legit rows separately) for AP and ROC-AUC."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ap, auc = [], []
    for _ in range(B):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        ap.append(average_precision_score(y[idx], score[idx]))
        auc.append(roc_auc_score(y[idx], score[idx]))
    return {"AP 95% CI low": np.percentile(ap, 2.5), "AP 95% CI high": np.percentile(ap, 97.5),
            "ROC-AUC 95% CI low": np.percentile(auc, 2.5), "ROC-AUC 95% CI high": np.percentile(auc, 97.5)}


def paired_bootstrap(y, pred_a, pred_b, B: int, seed: int) -> pd.DataFrame:
    """Paired, class-stratified bootstrap of (B − A) for recall, precision, F1 and FP on the same rows."""
    y, a, b = np.asarray(y), np.asarray(pred_a).astype(bool), np.asarray(pred_b).astype(bool)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def stats(idx, p):
        yy, pp = y[idx], p[idx]
        tp = np.sum(pp & (yy == 1)); fp = np.sum(pp & (yy == 0)); fn = np.sum(~pp & (yy == 1))
        return {"recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else np.nan,
                "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "FP": float(fp)}

    full = np.arange(len(y))
    est = {k: stats(full, b)[k] - stats(full, a)[k] for k in ("recall", "precision", "F1", "FP")}
    draws = {k: [] for k in est}
    for _ in range(B):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        sa, sb = stats(idx, a), stats(idx, b)
        for k in draws:
            draws[k].append(sb[k] - sa[k])
    return pd.DataFrame([{"metric": k, "difference (B − A)": est[k], "95% CI low": np.nanpercentile(draws[k], 2.5),
                          "95% CI high": np.nanpercentile(draws[k], 97.5)} for k in est])


def mcnemar(y, pred_a, pred_b) -> pd.DataFrame:
    """Exact McNemar (binomial) on discordant pairs, separately for fraud rows and legitimate rows."""
    from scipy.stats import binomtest
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    rows = []
    for name, mask in (("fraud rows (detections)", y == 1), ("legitimate rows (false alarms)", y == 0)):
        only_a = int(((a == 1) & (b == 0) & mask).sum())
        only_b = int(((a == 0) & (b == 1) & mask).sum())
        both = int(((a == 1) & (b == 1) & mask).sum())
        n = only_a + only_b
        rows.append({"rows": name, "flagged by both": both, "flagged only by A": only_a, "flagged only by B": only_b,
                     "exact McNemar p": binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0})
    return pd.DataFrame(rows)
