"""Phase 2 — comparing synthetic fraud datasets (my CTGAN output vs Noon's synthetic fraud).

Design rules
------------
* Nothing here creates, imputes or invents Noon data. Noon's file is read as supplied; every
  harmonisation step is limited to meaning-preserving fixes (e.g. "Card" -> "card", 1.0 -> 1),
  is logged with a row count, and anything ambiguous raises an error instead of being guessed.
* Validity is judged only against the DATA DICTIONARY (hard constraints). My Phase 1 plausibility
  bounds are my own assumptions, so values outside them are reported as "outside my assumed
  range", never as invalid.
* Neither dataset is treated as ground truth. Every comparison is symmetric unless a library
  forces a reference role, in which case both directions are reported.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import (BINARY_COLS, DATA, FEATURES, FLOAT_COLS, HARD_BOUNDS, INT_COLS, MODELS, PAYMENT_METHODS,
                     PLAUSIBLE_BOUNDS, SCHEMA, TARGET)

NOON_FILENAME = "B_NOON_synthetic_fraud.csv"
NOON_MODEL_FILENAME = "B_NOON_model.joblib"
REFEREE_FILENAME = "C_REFEREE_real_test.csv"

# Variable types follow the data dictionary: account_age_days, num_orders_last_7d and prev_refunds are INTEGERS
# (discrete, ordered); hour_of_day is an integer on a 24-hour clock (discrete, cyclic); only order_amount and
# delivery_distance_km are continuous.
CONTINUOUS = ["order_amount", "delivery_distance_km"]
DISCRETE_ORDERED = ["account_age_days", "num_orders_last_7d", "prev_refunds"]
CYCLIC = ["hour_of_day"]
NUMERIC_ALL = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d", "delivery_distance_km", "prev_refunds"]
CATEGORICAL = ["payment_method"] + BINARY_COLS
UNITS = {"order_amount": "AED", "account_age_days": "days", "delivery_distance_km": "km",
         "num_orders_last_7d": "orders", "prev_refunds": "claims", "hour_of_day": "hours"}

MIN_CELL = 20          # groups smaller than this are not reported in relationship profiles


# ---------------------------------------------------------------------------
# Files and provenance
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PROF_DIR_NAME = "professor_supplied"
PROFESSOR_FILES = [
    # (file, phase, purpose)
    ("B_NOON_synthetic_fraud.csv", "Phase 2", "Noon's GAN-generated synthetic fraud (comparison sample)"),
    ("A_eval_helper.py", "Phase 2 / 3 (optional)", "compare_distributions, lie_detector, score_on_referee"),
    ("B_NOON_benchmark.md", "Phase 3 reference", "Noon's reported benchmark results (Model A, Model B)"),
    ("B_NOON_model.joblib", "Phase 3", "Noon's benchmark classifier (Model B)"),
    ("C_REFEREE_real_test.csv", "Phase 3 — sealed", "Real referee test set; never used for development"),
    ("app(2).py", "Phase 4 reference", "Reference Streamlit fraud-check app"),
]


def professor_inventory(prof_dir: Path, recorded: dict) -> pd.DataFrame:
    rows = []
    for f, phase, purpose in PROFESSOR_FILES:
        pth = Path(prof_dir) / f
        ok = pth.exists()
        digest = sha256(pth) if ok else None
        rows.append({"file": f, "phase": phase, "purpose": purpose, "present": ok, "KB": round(pth.stat().st_size / 1024, 1) if ok else None,
                     "sha256 (first 16)": digest[:16] if ok else "—", "unchanged since received": ok and digest == recorded.get(f)})
    return pd.DataFrame(rows)


def phase1_artifact_map(root: Path) -> pd.DataFrame:
    """Where each Phase 1 deliverable lives (Master instruction, Part 3)."""
    items = [
        ("Seed transaction data (full synthetic transaction dataset, legit + fraud)", "data/seed_transactions.csv"),
        ("GAN-generated synthetic fraud ONLY (fraud-only sample for Phase 2)", "data/synthetic_fraud_gan.csv"),
        ("Train / validation / test split of the seed data", "data/split_indices.npz"),
        ("Trained GAN (CTGAN generator)", "models/ctgan_fraud_generator.pkl"),
        ("GAN training evidence (per-epoch losses)", "outputs/gan_diagnostics/ctgan_loss_values.csv"),
        ("Selected Phase 1 classifier (pipeline + calibrator + threshold + feature order)", "models/fraud_model.joblib"),
        ("CTGAN-augmented Phase 1 classifier (kept for Phase 3)", "models/fraud_model_gan_augmented.joblib"),
        ("Before / after classification results (test set)", "outputs/metrics/arms_test_metrics.csv"),
        ("Phase 1 metrics incl. augmentation ratios and frozen threshold", "outputs/metrics/phase1_metrics.json"),
        ("Written assumptions", "reports/assumptions_register.md"),
        ("Phase 1 report", "reports/final_report.md"),
    ]
    return pd.DataFrame([{"Phase 1 deliverable": d, "file": f, "present": (Path(root) / f).exists()} for d, f in items])


def phase1_cells_digest(nb_path: Path) -> str:
    """SHA-256 of the Phase 1 cells (source + outputs) of the notebook, ignoring any later-phase cells.
    Phase 1 cells carry no "phase" metadata; Phase 2+ cells are tagged with their phase number."""
    import json
    nb = json.loads(Path(nb_path).read_text())
    cells = [c for c in nb["cells"] if c.get("metadata", {}).get("phase") is None]
    payload = json.dumps([[c["cell_type"], c["source"], c.get("outputs", [])] for c in cells], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_manifest(manifest: dict, root: Path) -> pd.DataFrame:
    rows = []
    for rel, digest in manifest.items():
        p = root / rel
        now = sha256(p) if p.exists() else None
        rows.append({"file": rel, "present": p.exists(), "unchanged since Phase 1 submission": now == digest})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inspection, harmonisation and validity
# ---------------------------------------------------------------------------
def inspect_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Column-level description of a file exactly as supplied (before any change)."""
    rows = []
    for c in raw.columns:
        s = raw[c]
        ex = ", ".join(map(str, pd.unique(s.dropna())[:6]))
        rows.append({"column (as supplied)": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                     "unique values": int(s.nunique(dropna=True)), "example values": ex,
                     "in data dictionary": c in SCHEMA})
    return pd.DataFrame(rows)


def _log(log, step, detail, rows):
    log.append({"step": step, "detail": detail, "rows affected": int(rows)})


def harmonise(raw: pd.DataFrame, name: str):
    """Meaning-preserving harmonisation. Returns (df, log DataFrame). Raises on anything ambiguous."""
    log: list[dict] = []
    df = raw.copy()
    # 1. column names: strip / lower-case only if that yields an exact data-dictionary name
    ren = {}
    for c in df.columns:
        cand = str(c).strip()
        if cand not in SCHEMA:
            low = cand.lower().replace(" ", "_")
            cand = low if low in SCHEMA else cand
        if cand != c:
            ren[c] = cand
    if ren:
        df = df.rename(columns=ren)
        _log(log, "Column names normalised", "; ".join(f"'{k}' → '{v}'" for k, v in ren.items()), len(df))
    if df.columns.duplicated().any():
        raise ValueError(f"{name}: duplicate column names after normalisation: {list(df.columns[df.columns.duplicated()])}")
    # 2. pure row-index columns (0..n-1 or 1..n) are dropped; any other extra column is set aside
    for c in [c for c in df.columns if c not in SCHEMA]:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().all() and (np.array_equal(s.values, np.arange(len(df))) or np.array_equal(s.values, np.arange(1, len(df) + 1))):
            df = df.drop(columns=c)
            _log(log, "Dropped pure row-index column", f"'{c}' (values 0..n-1 or 1..n)", len(df))
    extra = [c for c in df.columns if c not in SCHEMA]
    if extra:
        df = df.drop(columns=extra)
        _log(log, "Set aside non-dictionary columns (never used in any comparison)", ", ".join(map(str, extra)), len(df))
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: predictor columns missing from the file: {missing}. Comparison cannot proceed for them.")
    # 3. payment method: case / whitespace / separator differences only
    s = df["payment_method"].astype("string")
    norm = s.str.strip().str.lower().str.replace(r"[\s_\-]", "", regex=True)
    fix = ((norm != s) & norm.isin(PAYMENT_METHODS)).fillna(False).astype(bool)
    if fix.any():
        mapping = (pd.DataFrame({"from": s[fix], "to": norm[fix]}).drop_duplicates()
                   .apply(lambda r: f"'{r['from']}' → '{r['to']}'", axis=1).tolist())
        df.loc[fix, "payment_method"] = norm[fix]
        _log(log, "Payment-method spelling normalised", "; ".join(mapping), fix.sum())
    df["payment_method"] = df["payment_method"].astype(object)
    # 4. binary columns given as bool / text / 0.0-1.0
    for c in BINARY_COLS + ([TARGET] if TARGET in df.columns else []):
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.astype("int64")
            _log(log, f"{c}: boolean → 0/1", "True → 1, False → 0", len(df))
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            m = s.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            if m.notna().all():
                df[c] = m.astype("int64")
                _log(log, f"{c}: text → 0/1", "true/false or '1'/'0' text values", len(df))
        elif pd.api.types.is_float_dtype(s) and s.notna().all() and s.isin([0.0, 1.0]).all():
            df[c] = s.astype("int64")
            _log(log, f"{c}: 0.0/1.0 floats → integers", "values unchanged", len(df))
    # 5. integer columns stored as floats
    for c in INT_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        bad_text = int(s.isna().sum() - df[c].isna().sum())
        if bad_text:
            _log(log, f"{c}: non-numeric entries", "kept as missing and reported as invalid", bad_text)
        if not pd.api.types.is_integer_dtype(df[c]):
            whole = s.dropna()
            if s.notna().all() and np.allclose(whole, np.round(whole)):
                df[c] = np.round(s).astype("int64")
                _log(log, f"{c}: integer-valued floats → integers", "values unchanged", len(df))
            else:
                df[c] = s
                n_frac = int((whole != np.round(whole)).sum())
                _log(log, f"{c}: contains non-integer values", "kept exactly as supplied; flagged in the validity report", n_frac)
    for c in FLOAT_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        bad = int(s.isna().sum() - df[c].isna().sum())
        if bad:
            _log(log, f"{c}: non-numeric entries", "kept as missing and reported as invalid", bad)
        df[c] = s.astype("float64")
    cols = FEATURES + ([TARGET] if TARGET in df.columns else [])
    if not log:
        _log(log, "No changes needed", "file already matches the data-dictionary format", 0)
    return df[cols].reset_index(drop=True), pd.DataFrame(log)


def hard_violation_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: one column per DATA-DICTIONARY rule, True where a row breaks it."""
    m = pd.DataFrame(index=df.index)
    m["missing value in any predictor"] = df[FEATURES].isna().any(axis=1)
    m["payment_method not in {card, wallet, cod, applepay}"] = ~df["payment_method"].isin(PAYMENT_METHODS)
    for c in BINARY_COLS:
        m[f"{c} not 0/1"] = ~df[c].isin([0, 1])
    h = pd.to_numeric(df["hour_of_day"], errors="coerce")
    m["hour_of_day not an integer 0–23"] = ~(h.between(0, 23) & (h == np.round(h)))
    for c in ["account_age_days", "num_orders_last_7d", "prev_refunds"]:
        s = pd.to_numeric(df[c], errors="coerce")
        m[f"{c} negative or non-integer"] = (s < 0) | (s != np.round(s)) | s.isna()
    m["delivery_distance_km negative"] = df["delivery_distance_km"] < 0
    m["order_amount ≤ 0"] = df["order_amount"] <= 0
    if TARGET in df.columns:
        m["is_fraud not 0/1"] = ~df[TARGET].isin([0, 1])
    return m


def validity_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    hard = hard_violation_mask(df)
    rows = [{"dataset": name, "rule type": "Data dictionary (hard)", "check": c,
             "rows": int(hard[c].sum()), "share": hard[c].mean()} for c in hard.columns]
    rows.append({"dataset": name, "rule type": "Data dictionary (hard)", "check": "ANY hard violation",
                 "rows": int(hard.any(axis=1).sum()), "share": hard.any(axis=1).mean()})
    for c, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        s = df[c]
        out = (s < lo) | (s > hi)
        rows.append({"dataset": name, "rule type": "My Phase 1 assumption (not a validity rule)",
                     "check": f"{c} outside my assumed range [{lo}, {hi}]", "rows": int(out.sum()), "share": out.mean()})
    d = df[FEATURES].duplicated()
    rows.append({"dataset": name, "rule type": "Quality", "check": "exact duplicate rows (predictors)",
                 "rows": int(d.sum()), "share": d.mean()})
    return pd.DataFrame(rows)


def cross_duplicates(a: pd.DataFrame, b: pd.DataFrame) -> int:
    ka = set(pd.util.hash_pandas_object(a[FEATURES], index=False))
    kb = pd.util.hash_pandas_object(b[FEATURES], index=False)
    return int(kb.isin(ka).sum())


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def decimals(s: pd.Series, max_d: int = 6) -> int:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    for d in range(max_d + 1):
        if np.allclose(v, np.round(v, d), atol=1e-9, rtol=0):
            return d
    return max_d + 1          # effectively continuous


def precision_profile(frames: dict) -> pd.DataFrame:
    cols = FLOAT_COLS + INT_COLS
    return pd.DataFrame({name: {c: decimals(df[c]) for c in cols} for name, df in frames.items()}).rename_axis("column")


def common_rounding(frames: dict) -> dict:
    """Coarsest precision present in any dataset, per numeric column (removes precision artefacts)."""
    prof = precision_profile(frames)
    return {c: int(prof.loc[c].min()) for c in prof.index}


def apply_rounding(df: pd.DataFrame, rounding: dict) -> pd.DataFrame:
    out = df.copy()
    for c, d in rounding.items():
        out[c] = np.round(out[c].astype(float), d)
        if d == 0:
            out[c] = out[c].astype("int64")
    return out


# ---------------------------------------------------------------------------
# Distribution comparisons
# ---------------------------------------------------------------------------
def numeric_summary(frames: dict) -> pd.DataFrame:
    out = {}
    for name, df in frames.items():
        rows = {}
        for c in NUMERIC_ALL:
            s = df[c].astype(float)
            rows[c] = {"n": len(s), "mean": s.mean(), "sd": s.std(), "min": s.min(), "p5": s.quantile(.05),
                       "p25": s.quantile(.25), "median": s.median(), "p75": s.quantile(.75), "p95": s.quantile(.95),
                       "max": s.max(), "share = 0": (s == 0).mean()}
        out[name] = pd.DataFrame(rows).T
    return pd.concat(out, axis=1).swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def categorical_summary(frames: dict) -> pd.DataFrame:
    rows = []
    for c in CATEGORICAL:
        levels = PAYMENT_METHODS if c == "payment_method" else [0, 1]
        for lv in levels:
            rows.append({"feature": c, "level": lv, **{name: (df[c] == lv).mean() for name, df in frames.items()}})
    return pd.DataFrame(rows).set_index(["feature", "level"])


# ---------------------------------------------------------------------------
# Relationships between features
# ---------------------------------------------------------------------------
AGE_BINS = [-1, 7, 30, 180, 730, np.inf]
AGE_LABELS = ["0–7 d", "8–30 d", "31–180 d", "181–730 d", "> 730 d"]
ORDER_BINS = [-1, 0, 2, 5, np.inf]
ORDER_LABELS = ["0", "1–2", "3–5", "6+"]

RELATIONSHIPS = [
    # (title, group feature, group binning, value feature, aggregation, y label)
    ("Device-change rate by account age", "account_age_days", "age", "device_change", "mean", "Orders with device_change = 1 (%)"),
    ("Discount rate by account age", "account_age_days", "age", "discount_applied", "mean", "Orders with a discount (%)"),
    ("Median order amount by account age", "account_age_days", "age", "order_amount", "median", "Median order amount (AED)"),
    ("Location-mismatch rate by payment method", "payment_method", None, "location_mismatch", "mean", "Orders with location_mismatch = 1 (%)"),
    ("Mean prior refunds by recent order frequency", "num_orders_last_7d", "orders", "prev_refunds", "mean", "Mean previous refund claims (count)"),
    ("Night-time ordering (00:00–05:59) by account age", "account_age_days", "age", "_night", "mean", "Orders placed 00:00–05:59 (%)"),
]


X_LABELS = {"account_age_days": "Account age band (days)", "payment_method": "Payment method",
            "num_orders_last_7d": "Orders in previous 7 days (band)"}


def _grouper(df: pd.DataFrame, feat: str, binning):
    if binning == "age":
        return pd.cut(df[feat], AGE_BINS, labels=AGE_LABELS)
    if binning == "orders":
        return pd.cut(df[feat], ORDER_BINS, labels=ORDER_LABELS)
    return pd.Categorical(df[feat], categories=PAYMENT_METHODS)


def relationship_profiles(frames: dict) -> pd.DataFrame:
    rows = []
    for title, gf, binning, vf, agg, ylab in RELATIONSHIPS:
        for name, df in frames.items():
            d = df.assign(_night=df["hour_of_day"].between(0, 5).astype(int))
            g = d.groupby(_grouper(d, gf, binning), observed=False)[vf]
            vals = g.agg(agg)
            ns = g.size()
            for grp in vals.index:
                n = int(ns[grp])
                rows.append({"relationship": title, "group": str(grp), "dataset": name, "x label": X_LABELS[gf],
                             "value": float(vals[grp]) if n >= MIN_CELL else np.nan, "n": n, "y label": ylab,
                             "is_rate": agg == "mean" and vf in BINARY_COLS + ["_night"]})
    return pd.DataFrame(rows)


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    q = df[[c for c in FEATURES if c != "payment_method"]].astype(float).copy()
    for m in PAYMENT_METHODS:
        q[f"pay_{m}"] = (df["payment_method"] == m).astype(float)
    return q.corr(method="spearman").fillna(0.0)


def mean_abs_corr_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ca, cb = spearman_matrix(a), spearman_matrix(b)
    iu = np.triu_indices(len(ca), 1)
    return float(np.abs(ca.values - cb.values)[iu].mean())


def top_corr_differences(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str, k: int = 8) -> pd.DataFrame:
    ca, cb = spearman_matrix(a), spearman_matrix(b)
    rows = []
    cols = list(ca.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i].startswith("pay_") and cols[j].startswith("pay_"):
                continue      # one-hot columns of one variable are mechanically correlated
            rows.append({"feature pair": f"{cols[i]} × {cols[j]}", f"ρ {name_a}": ca.iloc[i, j],
                         f"ρ {name_b}": cb.iloc[i, j], "Δρ (B − A)": cb.iloc[i, j] - ca.iloc[i, j]})
    out = pd.DataFrame(rows)
    return out.reindex(out["Δρ (B − A)"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def row_keys(df: pd.DataFrame, cols=None) -> pd.Series:
    return pd.util.hash_pandas_object(df[cols or FEATURES], index=False)


def duplicate_profile(df: pd.DataFrame, name: str) -> dict:
    k = row_keys(df)
    counts = k.value_counts()
    return {"dataset": name, "rows": len(df), "distinct rows (10 predictors)": int(counts.size),
            "duplicate rows (repeats beyond first copy)": int(len(df) - counts.size),
            "share of rows that are repeats": (len(df) - counts.size) / len(df),
            "distinct rows that occur more than once": int((counts > 1).sum()),
            "max copies of one row": int(counts.max()),
            "Kish effective sample size": float(counts.sum() ** 2 / (counts ** 2).sum()),
            "distinct rows incl. is_fraud": int(len(df.drop_duplicates()))}


def multiplicity_table(df: pd.DataFrame) -> pd.Series:
    return row_keys(df).value_counts().value_counts().sort_index().rename("distinct rows")


def zero_truncated_poisson_fit(mult: pd.Series) -> dict:
    """Under 'n draws with replacement from N equally likely base rows', the copies of each OBSERVED row follow a
    zero-truncated Poisson(λ = n/N). Fit λ from the mean multiplicity and return expected counts per multiplicity."""
    from scipy.optimize import brentq
    k = mult.index.to_numpy(float)
    c = mult.to_numpy(float)
    n_obs, n_rows = c.sum(), (k * c).sum()
    m = n_rows / n_obs
    lam = brentq(lambda l: l / (1 - np.exp(-l)) - m, 1e-6, 50)
    from math import factorial
    pk = np.exp(-lam) * lam ** k / np.array([factorial(int(x)) for x in k]) / (1 - np.exp(-lam))
    exp_counts = pk * n_obs
    tail = 1 - pk.sum()
    chi2 = float((((c - exp_counts) ** 2) / exp_counts).sum())
    return {"lambda": lam, "implied base rows N = rows/λ": n_rows / lam, "implied base rows never drawn": n_rows / lam - n_obs,
            "expected": pd.Series(exp_counts, index=mult.index), "chi2": chi2, "dof": len(k) - 2,
            "p (χ² goodness of fit)": float(stats.chi2.sf(chi2, len(k) - 2)), "tail prob beyond max": tail}


# ---------------------------------------------------------------------------
# Feature-level comparison (types follow the data dictionary)
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def describe_numeric(frames: dict) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        for c in NUMERIC_ALL:
            s = df[c].astype(float)
            rows.append({"feature": c, "dataset": name, "type": _vtype(c), "n": len(s), "mean": s.mean(), "sd": s.std(),
                         "min": s.min(), "p10": s.quantile(.10), "p25": s.quantile(.25), "median": s.median(),
                         "p75": s.quantile(.75), "p90": s.quantile(.90), "max": s.max(), "distinct values": s.nunique()})
    return pd.DataFrame(rows)


def describe_shares(frames: dict) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        for c in CATEGORICAL:
            levels = PAYMENT_METHODS if c == "payment_method" else [1]
            for lv in levels:
                k, n = int((df[c] == lv).sum()), len(df)
                lo, hi = wilson(k, n)
                rows.append({"feature": c, "level": lv, "dataset": name, "count": k, "n": n, "share": k / n,
                             "95% CI low": lo, "95% CI high": hi})
    return pd.DataFrame(rows)


def _vtype(c):
    return ("continuous" if c in CONTINUOUS else "discrete ordered (integer)" if c in DISCRETE_ORDERED
            else "discrete cyclic (24 h)" if c in CYCLIC else "categorical" if c == "payment_method" else "binary")


def _tvd(x, y, levels):
    p = pd.Series(x).value_counts(normalize=True).reindex(levels, fill_value=0)
    q = pd.Series(y).value_counts(normalize=True).reindex(levels, fill_value=0)
    return float(0.5 * (p - q).abs().sum())


def _perm_p(x, y, stat, n_perm, rng):
    """Two-sided permutation p-value for a non-negative distance statistic (exact under ties)."""
    obs = stat(x, y)
    z = np.concatenate([x, y])
    n = len(x)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        cnt += stat(z[:n], z[n:]) >= obs - 1e-12
    return obs, (cnt + 1) / (n_perm + 1)


EFFECT_BANDS = [(0.05, "negligible"), (0.15, "small"), (0.30, "moderate"), (np.inf, "large")]   # judgement convention


def band(v):
    return next(lab for cut, lab in EFFECT_BANDS if v < cut)


def compare_features(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Feature-by-feature comparison with type-appropriate statistics.
    continuous & discrete ordered → KS D (headline) + Wasserstein-1 in units + median difference, permutation p-value
                                     (valid with the ties that integer variables produce);
    hour_of_day (cyclic)          → TVD over the 24 hours, χ² test of homogeneity;
    categorical / binary          → TVD (= |share difference| for binaries), Cramér's V, χ² test."""
    rng = np.random.default_rng(seed)
    ks = lambda u, v: stats.ks_2samp(u, v).statistic
    rows = []
    for c in NUMERIC_ALL + CATEGORICAL:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        r = {"feature": c, "type": _vtype(c), "n A": len(x), "n B": len(y)}
        if c in CONTINUOUS + DISCRETE_ORDERED:
            d, p = _perm_p(x.astype(float), y.astype(float), ks, n_perm, rng)
            r.update({"headline effect": "KS D", "effect size": d, "p-value": p, "test": f"permutation test of KS D ({n_perm} perms)",
                      "Wasserstein-1 (units)": stats.wasserstein_distance(x, y), "median A": np.median(x), "median B": np.median(y),
                      "mean A": x.mean(), "mean B": y.mean()})
        else:
            levels = list(range(24)) if c in CYCLIC else (PAYMENT_METHODS if c == "payment_method" else [0, 1])
            tab = np.array([[int((x == lv).sum()) for lv in levels], [int((y == lv).sum()) for lv in levels]])
            tab = tab[:, tab.sum(axis=0) > 0]
            chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
            r.update({"headline effect": "TVD (24 hours)" if c in CYCLIC else "TVD", "effect size": _tvd(x, y, levels),
                      "p-value": p, "test": "χ² test of homogeneity", "Cramér V": np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))),
                      "mean A": x.mean() if c != "payment_method" else np.nan, "mean B": y.mean() if c != "payment_method" else np.nan})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(0, "pair", f"{name_a} vs {name_b}")
    out["Holm-adjusted p"] = holm(out["p-value"])
    out["magnitude (judgement band)"] = out["effect size"].map(band)
    return out


def holm(p: pd.Series) -> pd.Series:
    order = np.argsort(p.values)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p.values[k]))
        adj[k] = run
    return pd.Series(adj, index=p.index)


# ---------------------------------------------------------------------------
# Lie detector (dataset-origin classifier). Model = the helper's RandomForestClassifier(n_estimators=100, random_state=0);
# features = the helper's own preprocessing (_prep: 9 numeric columns + one-hot payment_method; is_fraud never used).
# ---------------------------------------------------------------------------
def helper_fold_leakage(mine: pd.DataFrame, noon: pd.DataFrame) -> dict:
    """Reproduce the helper's folds (cv=3 for a classifier → StratifiedKFold(3), no shuffling) and count test rows whose
    exact feature vector also sits in that fold's training data."""
    from sklearn.model_selection import StratifiedKFold
    both = pd.concat([mine[FEATURES], noon[FEATURES]], ignore_index=True)
    y = np.r_[np.zeros(len(mine), int), np.ones(len(noon), int)]
    k = row_keys(both).to_numpy()
    leaked = {0: 0, 1: 0}
    tot = {0: 0, 1: 0}
    for tr, te in StratifiedKFold(3).split(both, y):
        seen = set(k[tr])
        for lab in (0, 1):
            idx = te[y[te] == lab]
            tot[lab] += len(idx)
            leaked[lab] += int(np.isin(k[idx], list(seen)).sum())
    return {"mine test rows with an exact twin in training": leaked[0], "mine test rows": tot[0],
            "Noon test rows with an exact twin in training": leaked[1], "Noon test rows": tot[1],
            "share of Noon test rows leaked": leaked[1] / tot[1]}


def _rf(seed=0):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2)


def _cv_scores(X, y, cv, groups=None, seed=0):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = _rf(seed).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"ROC-AUC": roc_auc_score(y, p), "balanced accuracy": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred)}


def lie_detector_primary(mine_u: pd.DataFrame, noon_u: pd.DataFrame, prep, seeds, n_folds=5) -> pd.DataFrame:
    """PRIMARY: unique rows of both sources (exact cross-source overlaps already removed), equal sizes by subsampling the
    larger source (repeated over seeds), shuffled StratifiedKFold, ROC-AUC primary + balanced accuracy + accuracy."""
    from sklearn.model_selection import StratifiedKFold
    n = min(len(mine_u), len(noon_u))
    rows = []
    for s in seeds:
        a = mine_u.sample(n, random_state=s) if len(mine_u) > n else mine_u
        b = noon_u.sample(n, random_state=s) if len(noon_u) > n else noon_u
        X, y = _xy(a, b, prep)
        r = _cv_scores(X, y, StratifiedKFold(n_folds, shuffle=True, random_state=s), seed=0)
        rows.append({"seed": s, "n per source": n, **r, "majority baseline": 0.5})
    return pd.DataFrame(rows)


def lie_detector_as_supplied(mine: pd.DataFrame, noon: pd.DataFrame, prep, seeds, grouped=True, n_folds=5) -> pd.DataFrame:
    """SENSITIVITY: files as supplied (duplicates kept, unequal sizes). grouped=True → StratifiedGroupKFold keyed on the
    row hash, so identical rows never straddle training and validation folds. No inferential p-value is attached."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    X, y = _xy(mine, noon, prep)
    g = row_keys(pd.concat([mine[FEATURES], noon[FEATURES]], ignore_index=True)).to_numpy()
    rows = []
    for s in seeds:
        cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=s) if grouped else StratifiedKFold(n_folds, shuffle=True, random_state=s)
        r = _cv_scores(X, y, cv, groups=g if grouped else None, seed=0)
        rows.append({"seed": s, "n mine": len(mine), "n Noon": len(noon), **r, "majority baseline": max(y.mean(), 1 - y.mean())})
    return pd.DataFrame(rows)


def _xy(a, b, prep):
    A = a[FEATURES].copy(); A["_src"] = 0
    B = b[FEATURES].copy(); B["_src"] = 1
    both = pd.concat([A, B], ignore_index=True)
    return prep(both), both["_src"].to_numpy()


def permutation_test(mine_u, noon_u, prep, seed, n_perm, n_folds=5):
    """C2ST permutation test on unique rows (balanced sample for `seed`), shuffled StratifiedKFold, ROC-AUC; NO groups."""
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    n = min(len(mine_u), len(noon_u))
    a = mine_u.sample(n, random_state=seed) if len(mine_u) > n else mine_u
    b = noon_u.sample(n, random_state=seed) if len(noon_u) > n else noon_u
    X, y = _xy(a, b, prep)
    from sklearn.ensemble import RandomForestClassifier
    score, perm, p = permutation_test_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1), X, y,
                                            cv=StratifiedKFold(n_folds, shuffle=True, random_state=seed),
                                            scoring="roc_auc", n_permutations=n_perm, random_state=seed, n_jobs=2)
    return {"observed mean CV ROC-AUC": score, "permutation p-value": p, "null mean": perm.mean(), "null 99th pct": np.percentile(perm, 99),
            "null max": perm.max(), "n_permutations": n_perm, "null": perm}


def importance_and_control(mine_u, noon_u, prep, seed, n_folds=5):
    """Held-out permutation importance (grouped back to the 10 original features) and a 'marginals-only' control in which
    every column is shuffled independently within each source (marginals kept, joint structure destroyed)."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold
    n = min(len(mine_u), len(noon_u))
    a = mine_u.sample(n, random_state=seed) if len(mine_u) > n else mine_u
    b = noon_u.sample(n, random_state=seed) if len(noon_u) > n else noon_u
    raw = pd.concat([a[FEATURES], b[FEATURES]], ignore_index=True)
    y = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    imp = {c: [] for c in FEATURES}
    from sklearn.metrics import roc_auc_score
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(raw, y):
        m = _rf(0).fit(prep(raw).iloc[tr], y[tr])
        cols = list(prep(raw).columns)
        base = roc_auc_score(y[te], m.predict_proba(prep(raw).iloc[te])[:, 1])
        rng = np.random.default_rng(seed)
        for c in FEATURES:            # permute the ORIGINAL feature (all its one-hot columns together)
            drops = []
            for _ in range(10):
                Xp = raw.iloc[te].copy()
                Xp[c] = Xp[c].to_numpy()[rng.permutation(len(te))]
                Xe = prep(Xp).reindex(columns=cols, fill_value=0)
                drops.append(base - roc_auc_score(y[te], m.predict_proba(Xe)[:, 1]))
            imp[c].append(np.mean(drops))
    imp = pd.DataFrame({"mean drop in ROC-AUC": {c: np.mean(v) for c, v in imp.items()},
                        "sd over folds": {c: np.std(v) for c, v in imp.items()}}).sort_values("mean drop in ROC-AUC", ascending=False)
    rng = np.random.default_rng(seed)
    shuf = raw.copy()
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        for c in FEATURES:
            shuf.loc[idx, c] = raw.loc[idx, c].to_numpy()[rng.permutation(len(idx))]
    full = _cv_scores(prep(raw), y, StratifiedKFold(n_folds, shuffle=True, random_state=seed))["ROC-AUC"]
    marg = _cv_scores(prep(shuf), y, StratifiedKFold(n_folds, shuffle=True, random_state=seed))["ROC-AUC"]
    return imp, {"ROC-AUC (real rows)": full, "ROC-AUC (columns shuffled within source: marginals only)": marg,
                 "AUC attributable to joint structure": full - marg}
