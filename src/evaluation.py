"""Metrics, uncertainty and calibration utilities. Positive class = fraud (is_fraud = 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": threshold,
        "recall": rec,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc_ap": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "specificity": spec,
        "balanced_accuracy": 0.5 * (rec + spec),
        "accuracy": (tp + tn) / len(y_true),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "flagged": int(tp + fp), "n": int(len(y_true)), "n_fraud": int(y_true.sum()),
        "brier": brier_score_loss(y_true, proba),
        "mean_predicted_proba": float(np.mean(proba)),
    }


def verify_metrics(y_true, proba, threshold, m: dict) -> bool:
    """Independent recomputation of the headline metrics with scikit-learn's own functions."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    checks = [
        np.isclose(m["recall"], recall_score(y_true, pred, zero_division=0)),
        np.isclose(m["precision"], precision_score(y_true, pred, zero_division=0)),
        np.isclose(m["f1"], f1_score(y_true, pred, zero_division=0)),
        m["TP"] + m["FP"] + m["TN"] + m["FN"] == len(y_true),
        m["TP"] + m["FN"] == int(np.sum(y_true)),
    ]
    return bool(all(checks))


def threshold_sweep(y_true, proba, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, t)
        rows.append({k: m[k] for k in ["threshold", "precision", "recall", "f1", "fpr", "FP", "FN", "TP", "flagged"]})
    return pd.DataFrame(rows)


def stratified_bootstrap_indices(y, B: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(B):
        yield np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]


def fast_ap(y, p):
    """Tie-aware average precision, identical to sklearn's step-wise definition (fast for bootstrap loops)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    idx = np.r_[np.flatnonzero(np.diff(ps)), len(ys) - 1]
    tps = np.cumsum(ys)[idx]
    fps = idx + 1 - tps
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _point(y, p, t, metric):
    """Fast numpy metrics (verified against scikit-learn in the notebook audit)."""
    pred = p >= t
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    if metric == "pr_auc_ap":
        return fast_ap(y, p)
    if metric == "recall":
        return tp / (tp + fn) if tp + fn else 0.0
    if metric == "precision":
        return tp / (tp + fp) if tp + fp else 0.0
    if metric == "f1":
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    if metric == "FP":
        return fp
    if metric == "FN":
        return fn
    raise ValueError(metric)


METRICS_FOR_CI = ["pr_auc_ap", "recall", "precision", "f1", "FP", "FN"]


def bootstrap_ci(y, p, t, B: int, seed: int, metrics=METRICS_FOR_CI) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    samples = {m: [] for m in metrics}
    for idx in stratified_bootstrap_indices(y, B, seed):
        for m in metrics:
            samples[m].append(_point(y[idx], p[idx], t, m))
    return pd.DataFrame([{"metric": m, "estimate": _point(y, p, t, m),
                          "ci_low": np.percentile(samples[m], 2.5), "ci_high": np.percentile(samples[m], 97.5)}
                         for m in metrics])


def paired_bootstrap_diff(y, p_a, t_a, p_b, t_b, B: int, seed: int, metrics=METRICS_FOR_CI) -> pd.DataFrame:
    """Difference (B − A) with the SAME resampled rows for both models (paired), stratified by class."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = {m: [] for m in metrics}
    for idx in stratified_bootstrap_indices(y, B, seed):
        for m in metrics:
            diffs[m].append(_point(y[idx], p_b[idx], t_b, m) - _point(y[idx], p_a[idx], t_a, m))
    rows = []
    for m in metrics:
        d = np.asarray(diffs[m])
        est = _point(y, p_b, t_b, m) - _point(y, p_a, t_a, m)
        rows.append({"metric": m, "diff_B_minus_A": est, "ci_low": np.percentile(d, 2.5),
                     "ci_high": np.percentile(d, 97.5), "share_of_resamples_B_better":
                     float(np.mean(d < 0)) if m in ("FP", "FN") else float(np.mean(d > 0))})
    return pd.DataFrame(rows)


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    """Exact McNemar tests, run separately on fraud rows (detections) and legitimate rows (flags)."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    out = {}
    for name, mask in (("fraud_rows", y == 1), ("legit_rows", y == 0)):
        n01 = int(((a[mask] == 0) & (b[mask] == 1)).sum())   # only B flags
        n10 = int(((a[mask] == 1) & (b[mask] == 0)).sum())   # only A flags
        n = n01 + n10
        out[f"{name}: only_B_flags"] = n01
        out[f"{name}: only_A_flags"] = n10
        out[f"{name}: p_value"] = binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
    return out


def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values().index
    m = len(pvals)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return pd.Series(adj)[pvals.index]


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Quantile bins (equal counts) — uniform bins would put almost every prediction in bin 1."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    t = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    return t.reset_index()


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * (t["mean_predicted"] - t["observed_rate"]).abs()).sum() / t["n"].sum())


def brier_skill(y, p) -> float:
    prev = float(np.mean(y))
    return 1 - brier_score_loss(y, p) / (prev * (1 - prev))


def cost_optimal_threshold(y, p, fn_cost: float, fp_cost: float = 1.0):
    """Hypothetical cost scenario: threshold minimising fn_cost·FN + fp_cost·FP on the given data."""
    ts = np.round(np.linspace(0.01, 0.99, 99), 2)
    y = np.asarray(y)
    costs = [fn_cost * (((p < t) & (y == 1)).sum()) + fp_cost * (((p >= t) & (y == 0)).sum()) for t in ts]
    i = int(np.argmin(costs))
    return float(ts[i]), float(costs[i])
