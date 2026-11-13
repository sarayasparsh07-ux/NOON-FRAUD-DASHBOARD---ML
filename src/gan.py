"""Synthetic-fraud generators and their fidelity diagnostics.

* ``FraudCTGAN``      — the genuine GAN (ctgan 0.12.1), fitted only on training-partition fraud,
                         with documented rejection sampling for constraint violations.
* ``smote_nc_rows``   — conventional resampling control (imbalanced-learn SMOTE-NC).
* ``random_oversample_rows`` — duplication control.
* ``independent_marginal_rows`` — ablation: each column resampled independently, so marginals
                         are right but all joint structure is destroyed.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (CLIP_TO_FLOOR, CTGAN_DISCRETE, CTGAN_PARAMS, FEATURES, HARD_BOUNDS, PAYMENT_METHODS,
                     PLAUSIBLE_BOUNDS, ROUNDING, BINARY_COLS, INT_COLS, FLOAT_COLS)
from .seed_generator import enforce_dtypes


# ---------------------------------------------------------------------------
# CTGAN
# ---------------------------------------------------------------------------
@dataclass
class SamplingLog:
    requested: int = 0
    raw_generated: int = 0
    rejected_hard: dict = field(default_factory=dict)
    rejected_plausibility: dict = field(default_factory=dict)
    clipped_to_floor: dict = field(default_factory=dict)
    rounded_to_integer: dict = field(default_factory=dict)
    rounded_to_2dp: dict = field(default_factory=dict)
    retained: int = 0

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected_hard.values()) + sum(self.rejected_plausibility.values())

    def as_frame(self) -> pd.DataFrame:
        rows = [("Rows requested", self.requested), ("Rows drawn from the generator", self.raw_generated)]
        rows += [(f"Corrected — clipped to domain floor (training point mass): {k}", v) for k, v in self.clipped_to_floor.items()]
        rows += [(f"Corrected — rounded to integer: {k}", v) for k, v in self.rounded_to_integer.items()]
        rows += [(f"Corrected — rounded to 2 dp: {k}", v) for k, v in self.rounded_to_2dp.items()]
        rows += [(f"Rejected — hard constraint: {k}", v) for k, v in self.rejected_hard.items()]
        rows += [(f"Rejected — outside assumed plausible range: {k}", v) for k, v in self.rejected_plausibility.items()]
        rows += [("Valid rows produced", self.raw_generated - self.n_rejected),
                 ("Valid surplus not needed (final batch)", self.raw_generated - self.n_rejected - self.retained),
                 ("Rows retained", self.retained)]
        return pd.DataFrame(rows, columns=["step", "rows"])


class FraudCTGAN:
    """Thin wrapper around ``ctgan.CTGAN`` that records training evidence."""

    def __init__(self, seed: int, **overrides):
        from ctgan import CTGAN  # imported lazily so the rest of the package works without torch
        import torch
        torch.set_num_threads(2)
        self.params = {**CTGAN_PARAMS, **overrides}
        self.seed = seed
        self.model = CTGAN(**self.params)
        self.model.set_random_state(seed)
        self.fit_seconds = None
        self.n_train = None

    @staticmethod
    def _prepare(fraud_X: pd.DataFrame) -> pd.DataFrame:
        X = fraud_X[FEATURES].copy()
        X["account_age_days"] = X["account_age_days"].astype(float)   # modelled as continuous
        return X

    def fit(self, fraud_X: pd.DataFrame) -> "FraudCTGAN":
        X = self._prepare(fraud_X)
        self.n_train = len(X)
        assert self.params["batch_size"] % self.params["pac"] == 0, "batch_size must be divisible by pac"
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, discrete_columns=CTGAN_DISCRETE)
        self.fit_seconds = time.time() - t0
        return self

    @property
    def loss_values(self) -> pd.DataFrame:
        return self.model.loss_values.copy()

    def sample_valid(self, n: int, batch: int = 2000, max_rounds: int = 50):
        """Draw rows until ``n`` satisfy all constraints. Returns (rows, SamplingLog)."""
        log = SamplingLog(requested=n)
        kept = []
        n_kept = 0
        for _ in range(max_rounds):
            raw = self.model.sample(batch)
            log.raw_generated += len(raw)
            clean = postprocess_generated(raw, log)
            kept.append(clean)
            n_kept += len(clean)
            if n_kept >= n:
                break
        out = pd.concat(kept, ignore_index=True).iloc[:n].reset_index(drop=True)
        log.retained = len(out)
        if log.retained < n:
            raise RuntimeError(f"Only {log.retained} valid rows after {max_rounds} rounds")
        return out, log

    def save(self, path) -> None:
        self.model.save(str(path))


def postprocess_generated(raw: pd.DataFrame, log: SamplingLog | None = None, clip_floor: dict | None = None) -> pd.DataFrame:
    """Documented constraint handling for generated rows.

    1. Clip to the domain floor ONLY for columns where the training data has a genuine point mass
       at that floor (``CLIP_TO_FLOOR``: account_age_days = 0 for ~14% of training fraud) — CTGAN's
       Gaussian modes spill below such a spike, and rejecting them would strip the newest accounts.
    2. Round integer columns and 2-dp columns to schema precision.
    3. REJECT any row that still violates a hard constraint or an assumed plausibility bound.
    """
    clip_floor = CLIP_TO_FLOOR if clip_floor is None else clip_floor
    df = raw[FEATURES].copy()
    for c, floor in clip_floor.items():
        n_clip = int((df[c] < floor).sum())
        df[c] = df[c].clip(lower=floor)
        if log is not None:
            log.clipped_to_floor[c] = log.clipped_to_floor.get(c, 0) + n_clip
    for c in INT_COLS:
        if not pd.api.types.is_integer_dtype(df[c]):
            changed = int((df[c] != np.round(df[c])).sum())
            df[c] = np.round(df[c].astype(float))
            if log is not None:
                log.rounded_to_integer[c] = log.rounded_to_integer.get(c, 0) + changed
    for c, d in ROUNDING.items():
        changed = int((df[c] != np.round(df[c], d)).sum())
        df[c] = np.round(df[c].astype(float), d)
        if log is not None:
            log.rounded_to_2dp[c] = log.rounded_to_2dp.get(c, 0) + changed
    bad = pd.Series(False, index=df.index)
    for c, (lo, hi) in HARD_BOUNDS.items():
        m = (df[c] < lo) if lo is not None else pd.Series(False, index=df.index)
        if hi is not None:
            m |= df[c] > hi
        m &= ~bad
        if log is not None:
            log.rejected_hard[c] = log.rejected_hard.get(c, 0) + int(m.sum())
        bad |= m
    m = ~df["payment_method"].isin(PAYMENT_METHODS) & ~bad
    if log is not None:
        log.rejected_hard["payment_method"] = log.rejected_hard.get("payment_method", 0) + int(m.sum())
    bad |= m
    for c in BINARY_COLS:
        m = ~df[c].isin([0, 1]) & ~bad
        if log is not None:
            log.rejected_hard[c] = log.rejected_hard.get(c, 0) + int(m.sum())
        bad |= m
    for c, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        m = ((df[c] < lo) | (df[c] > hi)) & ~bad
        if log is not None:
            log.rejected_plausibility[c] = log.rejected_plausibility.get(c, 0) + int(m.sum())
        bad |= m
    return enforce_dtypes(df.loc[~bad].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Non-GAN comparison samplers (all fitted on training-partition data only)
# ---------------------------------------------------------------------------
def random_oversample_rows(fraud_X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fraud_X), n)
    return fraud_X[FEATURES].iloc[idx].reset_index(drop=True)


def independent_marginal_rows(fraud_X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = {c: fraud_X[c].to_numpy()[rng.integers(0, len(fraud_X), n)] for c in FEATURES}
    return enforce_dtypes(pd.DataFrame(out)[FEATURES])


def smote_nc_rows(X_train: pd.DataFrame, y_train: pd.Series, n: int, seed: int, k: int = 5) -> pd.DataFrame:
    """SMOTE-NC synthetic fraud rows only. hour_of_day, payment_method and binaries are
    treated as categorical so they stay valid; interpolated counts are rounded to integers."""
    from imblearn.over_sampling import SMOTENC
    cat_cols = ["payment_method", "hour_of_day"] + BINARY_COLS
    X = X_train[FEATURES].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    n_pos = int(y_train.sum())
    sm = SMOTENC(categorical_features=[FEATURES.index(c) for c in cat_cols],
                 sampling_strategy={1: n_pos + n}, k_neighbors=k, random_state=seed)
    Xr, yr = sm.fit_resample(X, y_train)
    assert (Xr.iloc[: len(X)].to_numpy() == X.to_numpy()).all(), "SMOTE-NC reordered the originals"
    syn = Xr.iloc[len(X):].reset_index(drop=True)
    assert len(syn) == n and (yr.iloc[len(X):] == 1).all()
    for c in ["hour_of_day"] + BINARY_COLS:
        syn[c] = syn[c].astype(int)
    for c in ["account_age_days", "num_orders_last_7d", "prev_refunds"]:
        syn[c] = np.round(syn[c].astype(float))
    for c, d in ROUNDING.items():
        syn[c] = np.round(syn[c].astype(float), d)
    return enforce_dtypes(syn[FEATURES])


# ---------------------------------------------------------------------------
# Fidelity diagnostics (synthetic vs. the training-fraud reference)
# ---------------------------------------------------------------------------
SIGNATURES = {
    "Account takeover\nage > 180 d · device change\n· location mismatch":
        lambda d: (d.account_age_days > 180) & (d.device_change == 1) & (d.location_mismatch == 1),
    "Promo abuse\nage < 15 d · discount\n· basket < AED 50":
        lambda d: (d.account_age_days < 15) & (d.discount_applied == 1) & (d.order_amount < 50),
    "Refund abuse\n≥ 2 prior refunds\n· ≥ 2 orders in 7 d":
        lambda d: (d.prev_refunds >= 2) & (d.num_orders_last_7d >= 2),
    "Stolen card\ncard · location mismatch\n· age < 60 d":
        lambda d: (d.payment_method == "card") & (d.location_mismatch == 1) & (d.account_age_days < 60),
    "COD fake order\nCOD · age < 60 d\n· basket > AED 100":
        lambda d: (d.payment_method == "cod") & (d.account_age_days < 60) & (d.order_amount > 100),
}


def signature_shares(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({name: {sig: float(f(df).mean()) for sig, f in SIGNATURES.items()}
                         for name, df in frames.items()})


def marginal_distances(real: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import ks_2samp, wasserstein_distance
    rows = []
    for c in FEATURES:
        if c in ("payment_method",) + tuple(BINARY_COLS):
            p = real[c].value_counts(normalize=True)
            q = syn[c].value_counts(normalize=True)
            tvd = 0.5 * p.subtract(q, fill_value=0).abs().sum()
            rows.append(dict(column=c, kind="categorical", ks_stat=np.nan, wasserstein=np.nan, tvd=tvd,
                             real_summary=", ".join(f"{k}:{v:.2f}" for k, v in p.sort_index().items()),
                             syn_summary=", ".join(f"{k}:{v:.2f}" for k, v in q.sort_index().items())))
        else:
            ks = ks_2samp(real[c], syn[c]).statistic
            wd = wasserstein_distance(real[c], syn[c])
            rows.append(dict(column=c, kind="numeric", ks_stat=ks, wasserstein=wd, tvd=np.nan,
                             real_summary=f"median {real[c].median():.1f}, mean {real[c].mean():.1f}",
                             syn_summary=f"median {syn[c].median():.1f}, mean {syn[c].mean():.1f}"))
    return pd.DataFrame(rows)


def sdmetrics_scores(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """SDMetrics QualityReport (pair thresholds set to 0 so weak pairs are not dropped),
    DiagnosticReport and LogisticDetection."""
    from sdmetrics.reports import QualityReport, DiagnosticReport
    from sdmetrics.single_table import LogisticDetection
    cols = {c: {"sdtype": "numerical"} for c in ["order_amount", "account_age_days", "delivery_distance_km",
                                                    "num_orders_last_7d", "prev_refunds", "hour_of_day"]}
    cols.update({c: {"sdtype": "categorical"} for c in ["payment_method"] + BINARY_COLS})
    meta = {"tables": {"fraud": {"columns": cols}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr = QualityReport()
        qr.real_correlation_threshold = 0
        qr.real_association_threshold = 0
        qr.generate({"fraud": real[FEATURES]}, {"fraud": syn[FEATURES]}, meta, verbose=False)
        props = qr.get_properties()
        dr = DiagnosticReport()
        dr.generate({"fraud": real[FEATURES]}, {"fraud": syn[FEATURES]}, meta, verbose=False)
        dprops = dr.get_properties()
        det = LogisticDetection.compute(real[FEATURES], syn[FEATURES], meta["tables"]["fraud"])
    out = {"overall_quality": qr.get_score()}
    out.update({f"quality: {r.Property}": r.Score for r in props.itertuples()})
    out.update({f"diagnostic: {r.Property}": r.Score for r in dprops.itertuples()})
    out["LogisticDetection (1 = indistinguishable)"] = det
    return out


def _encode_for_distance(df: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    num = ["order_amount", "account_age_days", "delivery_distance_km", "num_orders_last_7d", "prev_refunds"]
    Z = np.log1p(df[num].astype(float).clip(lower=0))
    R = np.log1p(ref[num].astype(float).clip(lower=0))
    Z = (Z - R.mean()) / R.std(ddof=0)
    hour = df["hour_of_day"].to_numpy()
    cyc = np.c_[np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    pm = np.c_[[(df["payment_method"] == m).to_numpy(float) for m in PAYMENT_METHODS]].T
    return np.c_[Z.to_numpy(), cyc, pm, df[BINARY_COLS].to_numpy(float)]


def distance_to_closest_record(syn: pd.DataFrame, real: pd.DataFrame, exclude_self: bool = False) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    R = _encode_for_distance(real, real)
    S = _encode_for_distance(syn, real)
    nn = NearestNeighbors(n_neighbors=2 if exclude_self else 1).fit(R)
    d, _ = nn.kneighbors(S)
    return d[:, 1] if exclude_self else d[:, 0]


def c2st(real: pd.DataFrame, syn: pd.DataFrame, seed: int, n_perm: int = 0):
    """Classifier two-sample test: can a model tell training fraud from generated fraud?

    Balanced sets (larger side subsampled), exact duplicates across sources removed,
    5-fold stratified CV. Returns AUCs for LR and HGB plus permutation-importance of HGB.
    AUC ≈ 0.5 → indistinguishable; ≈ 1.0 → trivially separable.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

    rng = np.random.default_rng(seed)
    r = real[FEATURES].drop_duplicates()
    s = syn[FEATURES].drop_duplicates()
    s = s.merge(r, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    n = min(len(r), len(s))
    r = r.iloc[rng.permutation(len(r))[:n]]
    s = s.iloc[rng.permutation(len(s))[:n]]
    X = pd.concat([r, s], ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]
    num = [c for c in FEATURES if c != "payment_method"]
    pre_lr = ColumnTransformer([("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), num),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), ["payment_method"])])
    pre_tree = ColumnTransformer([("num", "passthrough", num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), ["payment_method"])])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    models = {"Logistic regression": make_pipeline(pre_lr, LogisticRegression(max_iter=2000)),
              "Gradient boosting": make_pipeline(pre_tree, HistGradientBoostingClassifier(random_state=seed, max_iter=200))}
    res = {"n_per_source": n}
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        res[f"AUC — {name}"] = roc_auc_score(y, p)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    m = models["Gradient boosting"].fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=seed)
    res["importance"] = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    return res
