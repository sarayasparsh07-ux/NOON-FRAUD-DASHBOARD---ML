"""Shared experiment logic so the primary run, seed repeats and prevalence scenarios are identical."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import ARM_ORDER, AUG_RATIOS, FEATURES, RATIO_TIE_TOL, TARGET
from .evaluation import compute_metrics
from .gan import FraudCTGAN, independent_marginal_rows, random_oversample_rows, smote_nc_rows
from .modeling import build_training_set, fit_and_score, max_f1_threshold, stratified_split
from .seed_generator import generate_transactions

GENERATOR_ARMS = ["B_ctgan", "D_smote_nc", "E_random_oversampling", "F_independent_marginals"]


def make_pools(X_tr, y_tr, seed, n_max, ctgan_rows=None):
    """Synthetic-fraud pools for every generator arm, all built from TRAINING data only.
    Smaller augmentation ratios use the first k rows of each pool (nested subsets)."""
    ftr = X_tr[y_tr.values == 1]
    pools = {
        "D_smote_nc": smote_nc_rows(X_tr, y_tr, n_max, seed),
        "E_random_oversampling": random_oversample_rows(ftr, n_max, seed),
        "F_independent_marginals": independent_marginal_rows(ftr, n_max, seed),
    }
    if ctgan_rows is not None:
        pools["B_ctgan"] = ctgan_rows.iloc[:n_max].reset_index(drop=True)
    return pools


def train_arm(arm, ratio, family, params, seed, X_tr, y_tr, pools, X_val, y_val):
    n_train_fraud = int(y_tr.sum())
    if arm == "A_baseline":
        return fit_and_score(family, params, seed, X_tr[FEATURES], y_tr, X_val, y_val)
    if arm == "C_class_weighted":
        return fit_and_score(family, params, seed, X_tr[FEATURES], y_tr, X_val, y_val, class_weight="balanced")
    n_syn = int(round(ratio * n_train_fraud))
    Xf, yf = build_training_set(X_tr, y_tr, pools[arm].iloc[:n_syn])
    return fit_and_score(family, params, seed, Xf, yf, X_val, y_val)


def select_ratios(family, params, seed, X_tr, y_tr, pools, X_val, y_val, arms=GENERATOR_ARMS, ratios=AUG_RATIOS):
    """Validation average precision for every (arm, ratio); pick the best per arm (ties → smaller ratio)."""
    rows = []
    for arm in arms:
        for r in ratios:
            _, _, ap = train_arm(arm, r, family, params, seed, X_tr, y_tr, pools, X_val, y_val)
            rows.append({"arm": arm, "ratio": r, "n_synthetic": int(round(r * y_tr.sum())), "val_ap": ap})
    grid = pd.DataFrame(rows)
    chosen = {}
    for arm, g in grid.groupby("arm"):
        best = g["val_ap"].max()
        chosen[arm] = float(g[g["val_ap"] >= best - RATIO_TIE_TOL].sort_values("ratio")["ratio"].iloc[0])
    return grid, chosen


def run_arms(family, params, seed, X_tr, y_tr, X_val, y_val, X_te, y_te, pools, ratios):
    """Fit every arm, choose its threshold on validation (max F1), score the test set once."""
    out = {}
    for arm in ARM_ORDER:
        if arm in GENERATOR_ARMS and arm not in pools:
            continue
        model, p_val, ap_val = train_arm(arm, ratios.get(arm), family, params, seed, X_tr, y_tr, pools, X_val, y_val)
        t = max_f1_threshold(y_val, p_val)
        p_te = model.predict_proba(X_te[FEATURES])[:, 1]
        out[arm] = dict(model=model, threshold=t, ratio=ratios.get(arm), p_val=p_val, p_test=p_te,
                        val=compute_metrics(y_val, p_val, t), test=compute_metrics(y_te, p_te, t),
                        test_at_05=compute_metrics(y_te, p_te, 0.5))
    return out


def run_pipeline(seed, prevalence, n_rows, family, params, gan_epochs, reselect_ratios=True, fixed_ratios=None):
    """Whole pipeline from data generation to test metrics (used for seed repeats & prevalence scenarios)."""
    df, meta = generate_transactions(n_rows, prevalence, seed)
    tr, va, te = stratified_split(df, seed)
    X, y = df[FEATURES], df[TARGET]
    X_tr, y_tr, X_val, y_val, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X.iloc[te], y.iloc[te]
    n_max = int(round(max(AUG_RATIOS) * y_tr.sum()))
    gan = FraudCTGAN(seed, epochs=gan_epochs).fit(X_tr[y_tr.values == 1])
    ctgan_rows, log = gan.sample_valid(n_max)
    pools = make_pools(X_tr, y_tr, seed, n_max, ctgan_rows)
    if reselect_ratios:
        _, ratios = select_ratios(family, params, seed, X_tr, y_tr, pools, X_val, y_val)
    else:
        ratios = fixed_ratios
    res = run_arms(family, params, seed, X_tr, y_tr, X_val, y_val, X_te, y_te, pools, ratios)
    rows = []
    for arm, r in res.items():
        rows.append({"seed": seed, "prevalence": prevalence, "n_rows": n_rows, "arm": arm, "ratio": r["ratio"],
                     "n_train_fraud": int(y_tr.sum()), "n_test_fraud": int(y_te.sum()),
                     "gan_fit_seconds": gan.fit_seconds, **{k: v for k, v in r["test"].items()}})
    return pd.DataFrame(rows), res, dict(df=df, meta=meta, split=(tr, va, te), gan=gan, sampling_log=log)
