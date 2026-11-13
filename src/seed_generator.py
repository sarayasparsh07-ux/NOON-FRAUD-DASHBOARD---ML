"""Seed-data generator: a latent-profile mixture model.

How it works
------------
1. Each transaction first draws a hidden *profile*: one of four legitimate
   segments (including legitimate look-alikes of fraud, e.g. brand-new customers
   using a welcome promo, or genuine customers with several real refunds) or one
   of five fraud typologies (promo abuse, refund abuse, stolen card, account
   takeover, cash-on-delivery fake orders).
2. All ten features are then drawn *conditionally on that profile* from
   overlapping probability distributions, plus shared "physics" that applies to
   every order (dinner baskets are larger, farther restaurants slightly pricier,
   counts are over-dispersed, refunds accumulate with tenure).
3. The observed label is the profile's class with label noise: 3% of fraud is
   recorded as legitimate (never detected / never charged back) and 0.1% of
   legitimate orders are recorded as fraud (review errors).

Because profiles overlap and labels are noisy, no single feature or simple rule
separates the classes. Every numeric value below is a MODELLING ASSUMPTION;
sources for the anchors are in reports/assumptions_register.md.

The hidden profile is returned separately (``meta``) and is never given to the
GAN or the classifiers; it is used only for diagnostics such as recall by
fraud typology.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from .config import PLAUSIBLE_BOUNDS, SCHEMA, FEATURES, TARGET, PAYMENT_METHODS, BINARY_COLS

# Legitimate hour-of-day weights (lunch and dinner peaks, ~9% between 00:00 and 05:59)
LEGIT_HOUR_W = np.array([2.6, 1.8, 1.3, 0.9, 0.7, 0.6, 0.8, 1.4, 2.0, 2.2, 2.4, 3.5,
                         6.5, 7.0, 5.0, 3.2, 3.0, 4.0, 6.0, 8.5, 9.0, 7.5, 5.0, 3.4])

_LEGIT_MISMATCH = {"card": 0.14, "applepay": 0.13, "wallet": 0.08, "cod": 0.02}

BASE_PARAMS: dict[str, Any] = {
    "label_noise": {"fraud_recorded_as_legit": 0.03, "legit_recorded_as_fraud": 0.001},
    "global": {
        "amount_hour_mult": {"breakfast_6_10": 0.80, "dinner_18_22": 1.08},
        "amount_per_km": 0.015,   # +1.5% basket value per km beyond 3.5 km
    },
    "legit": {
        "L_established": dict(
            share=0.885,
            age={"mix": [dict(w=1.0, kind="lognorm", median=420, sigma=0.85, lo=31, hi=3000)]},
            orders7=dict(mean=1.4, shape=2.0, burst=0.0),
            hour=dict(flatten=1.0, night_mult=1.0),
            payment={"card": 0.48, "applepay": 0.30, "wallet": 0.08, "cod": 0.14},
            amount=dict(median=62, sigma=0.62),
            distance=dict(shape=2.3, scale=1.6),
            discount=dict(p=0.32, wallet_bonus=0.08),
            refunds=dict(kind="tenure", rate=0.12),
            device_change=0.035,
            mismatch=dict(_LEGIT_MISMATCH),
        ),
        "L_new_customer": dict(
            share=0.08,
            age={"mix": [dict(w=1.0, kind="exp", mean=9, lo=0, hi=30)]},
            orders7=dict(mean=0.8, shape=2.0, burst=0.0),
            hour=dict(flatten=1.0, night_mult=1.0),
            payment={"card": 0.45, "applepay": 0.27, "wallet": 0.06, "cod": 0.22},
            amount=dict(median=60, sigma=0.60),
            distance=dict(shape=2.3, scale=1.6),
            discount=dict(p=0.72, wallet_bonus=0.0),
            refunds=dict(kind="poisson", mean=0.01),
            device_change=0.03,
            mismatch={"card": 0.16, "applepay": 0.13, "wallet": 0.08, "cod": 0.02},
        ),
        "L_power_user": dict(
            share=0.02,
            age={"mix": [dict(w=1.0, kind="lognorm", median=600, sigma=0.7, lo=31, hi=3000)]},
            orders7=dict(mean=9.0, shape=3.0, burst=0.0),
            hour=dict(flatten=1.0, night_mult=1.0),
            payment={"card": 0.50, "applepay": 0.32, "wallet": 0.10, "cod": 0.08},
            amount=dict(median=95, sigma=0.70),
            distance=dict(shape=2.3, scale=1.6),
            discount=dict(p=0.30, wallet_bonus=0.05),
            refunds=dict(kind="tenure", rate=0.35),
            device_change=0.04,
            mismatch=dict(_LEGIT_MISMATCH),
        ),
        "L_genuine_complainer": dict(
            share=0.015,
            age={"mix": [dict(w=1.0, kind="lognorm", median=500, sigma=0.8, lo=31, hi=3000)]},
            orders7=dict(mean=2.0, shape=2.0, burst=0.0),
            hour=dict(flatten=1.0, night_mult=1.0),
            payment={"card": 0.48, "applepay": 0.30, "wallet": 0.10, "cod": 0.12},
            amount=dict(median=70, sigma=0.60),
            distance=dict(shape=2.3, scale=1.6),
            discount=dict(p=0.32, wallet_bonus=0.08),
            refunds=dict(kind="abuser", p_early=0.30, early_mean=1.0, base=1, extra_mean=1.5),
            device_change=0.035,
            mismatch=dict(_LEGIT_MISMATCH),
        ),
    },
    "fraud": {
        "F_promo_abuse": dict(
            share=0.32,
            age={"mix": [dict(w=1.0, kind="exp", mean=2.5, lo=0, hi=20)]},
            orders7=dict(mean=0.4, shape=2.0, burst=0.0),
            hour=dict(flatten=0.7, night_mult=1.0),
            payment={"card": 0.45, "wallet": 0.30, "applepay": 0.18, "cod": 0.07},
            amount=dict(median=38, sigma=0.40),
            distance=dict(shape=2.3, scale=1.6),
            discount=dict(p=0.96, wallet_bonus=0.0),
            refunds=dict(kind="poisson", mean=0.02),
            device_change=0.07,
            mismatch={"card": 0.18, "applepay": 0.15, "wallet": 0.10, "cod": 0.02},
        ),
        "F_refund_abuse": dict(
            share=0.27,
            age={"mix": [dict(w=1.0, kind="lognorm", median=260, sigma=0.9, lo=10, hi=3000)]},
            orders7=dict(mean=2.6, shape=2.0, burst=0.0),
            hour=dict(flatten=1.0, night_mult=1.0),
            payment={"card": 0.40, "applepay": 0.25, "wallet": 0.22, "cod": 0.13},
            amount=dict(median=90, sigma=0.55),
            distance=dict(shape=2.3, scale=1.6),
            discount=dict(p=0.45, wallet_bonus=0.0),
            refunds=dict(kind="abuser", p_early=0.15, early_mean=0.5, base=1, extra_mean=2.3),
            device_change=0.05,
            mismatch=dict(_LEGIT_MISMATCH),
        ),
        "F_stolen_card": dict(
            share=0.23,
            age={"mix": [dict(w=0.65, kind="exp", mean=6, lo=0, hi=60),
                         dict(w=0.35, kind="lognorm", median=120, sigma=0.8, lo=0, hi=3000)]},
            orders7=dict(mean=2.8, shape=1.5, burst=0.0),
            hour=dict(flatten=1.0, night_mult=3.0),
            payment={"card": 0.78, "applepay": 0.15, "wallet": 0.07, "cod": 0.0},
            amount=dict(median=135, sigma=0.60),
            distance=dict(shape=2.3, scale=1.8),
            discount=dict(p=0.22, wallet_bonus=0.0),
            refunds=dict(kind="poisson", mean=0.05),
            device_change=0.30,
            mismatch={"card": 0.55, "applepay": 0.45, "wallet": 0.30, "cod": 0.0},
        ),
        "F_account_takeover": dict(
            share=0.13,
            age={"mix": [dict(w=1.0, kind="lognorm", median=700, sigma=0.6, lo=120, hi=3000)]},
            orders7=dict(mean=1.4, shape=2.0, burst=3.0),
            hour=dict(flatten=1.0, night_mult=2.0),
            payment={"card": 0.55, "wallet": 0.30, "applepay": 0.10, "cod": 0.05},
            amount=dict(median=115, sigma=0.55),
            distance=dict(shape=2.5, scale=2.4),
            discount=dict(p=0.20, wallet_bonus=0.0),
            refunds=dict(kind="tenure", rate=0.12),
            device_change=0.80,
            mismatch={"card": 0.45, "applepay": 0.45, "wallet": 0.45, "cod": 0.05},
        ),
        "F_cod_fake_order": dict(
            share=0.05,
            age={"mix": [dict(w=1.0, kind="exp", mean=5, lo=0, hi=60)]},
            orders7=dict(mean=1.8, shape=2.0, burst=0.0),
            hour=dict(flatten=0.8, night_mult=1.0),
            payment={"card": 0.0, "wallet": 0.0, "applepay": 0.0, "cod": 1.0},
            amount=dict(median=150, sigma=0.50),
            distance=dict(shape=2.4, scale=2.3),
            discount=dict(p=0.30, wallet_bonus=0.0),
            refunds=dict(kind="poisson", mean=0.0),
            device_change=0.10,
            mismatch={"card": 0.0, "applepay": 0.0, "wallet": 0.0, "cod": 0.03},
        ),
    },
}

# ---------------------------------------------------------------------------
# Alternative-assumption scenarios for the stress test (never used for training
# or any model-development decision).
# ---------------------------------------------------------------------------
STRESS_OVERRIDES: dict[str, dict] = {
    "S0_in_distribution": {},
    "S1_population_drift": {   # legitimate customers change; fraud does not
        "legit": {
            "L_established": {"share": 0.825,
                              "payment": {"card": 0.40, "applepay": 0.30, "wallet": 0.08, "cod": 0.22},
                              "mismatch": {"card": 0.22, "applepay": 0.20, "wallet": 0.10, "cod": 0.02},
                              "device_change": 0.06, "amount": {"median": 70, "sigma": 0.62}},
            "L_new_customer": {"share": 0.14},
        }
    },
    "S2_adaptive_fraud": {     # fraudsters adapt to look more legitimate
        "fraud": {
            "F_stolen_card": {"payment": {"card": 0.40, "applepay": 0.55, "wallet": 0.05, "cod": 0.0},
                              "amount": {"median": 85, "sigma": 0.60},
                              "mismatch": {"card": 0.30, "applepay": 0.25, "wallet": 0.20, "cod": 0.0},
                              "hour": {"flatten": 1.0, "night_mult": 1.0}, "device_change": 0.15},
            "F_account_takeover": {"device_change": 0.35, "hour": {"flatten": 1.0, "night_mult": 1.0},
                                   "distance": {"shape": 2.3, "scale": 1.6},
                                   "mismatch": {"card": 0.25, "applepay": 0.25, "wallet": 0.25, "cod": 0.05}},
            "F_promo_abuse": {"age": {"mix": [dict(w=1.0, kind="exp", mean=10, lo=0, hi=45)]},
                              "amount": {"median": 58, "sigma": 0.50}, "discount": {"p": 0.85, "wallet_bonus": 0.0}},
            "F_refund_abuse": {"refunds": dict(kind="abuser", p_early=0.45, early_mean=0.4, base=1, extra_mean=1.0)},
        }
    },
    "S3_unseen_typology": {    # a fraud pattern absent from training makes up 30% of fraud
        "fraud": {
            "F_promo_abuse": {"share": 0.32 * 0.7}, "F_refund_abuse": {"share": 0.27 * 0.7},
            "F_stolen_card": {"share": 0.23 * 0.7}, "F_account_takeover": {"share": 0.13 * 0.7},
            "F_cod_fake_order": {"share": 0.05 * 0.7},
            "F_wallet_laundering": dict(
                share=0.30,
                age={"mix": [dict(w=1.0, kind="lognorm", median=200, sigma=0.7, lo=31, hi=3000)]},
                orders7=dict(mean=3.5, shape=2.0, burst=0.0),
                hour=dict(flatten=1.0, night_mult=1.0),
                payment={"card": 0.10, "applepay": 0.0, "wallet": 0.90, "cod": 0.0},
                amount=dict(median=320, sigma=0.50),
                distance=dict(shape=2.3, scale=1.2),
                discount=dict(p=0.0, wallet_bonus=0.0),
                refunds=dict(kind="poisson", mean=0.1),
                device_change=0.08,
                mismatch={"card": 0.10, "applepay": 0.0, "wallet": 0.05, "cod": 0.0},
            ),
        }
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("payment", "mismatch", "age", "refunds"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)   # whole-dict replacement for distributions
    return out


def scenario_params(name: str) -> dict:
    return _deep_merge(BASE_PARAMS, STRESS_OVERRIDES[name])


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def _hour_probs(spec: dict) -> np.ndarray:
    w = LEGIT_HOUR_W ** spec.get("flatten", 1.0)
    w = w.copy()
    w[0:6] *= spec.get("night_mult", 1.0)
    return w / w.sum()


def _sample_age(n: int, spec: dict, rng) -> np.ndarray:
    comps = spec["mix"]
    ws = np.array([c["w"] for c in comps], dtype=float)
    idx = rng.choice(len(comps), size=n, p=ws / ws.sum())
    out = np.empty(n)
    for k, c in enumerate(comps):
        m = idx == k
        nk = int(m.sum())
        if c["kind"] == "exp":
            v = rng.exponential(c["mean"], nk)
        elif c["kind"] == "lognorm":
            v = rng.lognormal(np.log(c["median"]), c["sigma"], nk)
        else:
            raise ValueError(c["kind"])
        out[m] = np.clip(v, c.get("lo", 0), c.get("hi", 3000))
    return np.floor(out).astype(np.int64)


def _sample_refunds(n: int, spec: dict, age: np.ndarray, rng) -> np.ndarray:
    kind = spec["kind"]
    if kind == "tenure":
        lam = spec["rate"] * (age / 365.0) ** 0.7
        v = rng.poisson(lam)
    elif kind == "poisson":
        v = rng.poisson(spec["mean"], n)
    elif kind == "abuser":
        early = rng.random(n) < spec["p_early"]
        v = np.where(early, rng.poisson(spec["early_mean"], n), spec["base"] + rng.poisson(spec["extra_mean"], n))
    else:
        raise ValueError(kind)
    return np.clip(v, 0, PLAUSIBLE_BOUNDS["prev_refunds"][1]).astype(np.int64)


def _sample_profile(n: int, spec: dict, glob: dict, rng) -> pd.DataFrame:
    if n == 0:
        return pd.DataFrame(columns=FEATURES)
    age = _sample_age(n, spec["age"], rng)
    o = spec["orders7"]
    lam = rng.gamma(o["shape"], o["mean"] / o["shape"], n) if o["mean"] > 0 else np.zeros(n)
    orders = rng.poisson(lam) + (rng.poisson(o["burst"], n) if o.get("burst", 0) > 0 else 0)
    orders = np.clip(orders, 0, PLAUSIBLE_BOUNDS["num_orders_last_7d"][1]).astype(np.int64)
    hour = rng.choice(24, size=n, p=_hour_probs(spec["hour"])).astype(np.int64)
    pm_p = np.array([spec["payment"].get(m, 0.0) for m in PAYMENT_METHODS], dtype=float)
    pay = rng.choice(PAYMENT_METHODS, size=n, p=pm_p / pm_p.sum())
    dist = np.clip(rng.gamma(spec["distance"]["shape"], spec["distance"]["scale"], n),
                   *PLAUSIBLE_BOUNDS["delivery_distance_km"])
    amt = rng.lognormal(np.log(spec["amount"]["median"]), spec["amount"]["sigma"], n)
    hm = np.ones(n)
    hm[(hour >= 6) & (hour <= 10)] = glob["amount_hour_mult"]["breakfast_6_10"]
    hm[(hour >= 18) & (hour <= 22)] = glob["amount_hour_mult"]["dinner_18_22"]
    amt = amt * hm * np.clip(1 + glob["amount_per_km"] * (dist - 3.5), 0.9, 1.2)
    amt = np.clip(amt, *PLAUSIBLE_BOUNDS["order_amount"])
    pdisc = np.clip(spec["discount"]["p"] + spec["discount"].get("wallet_bonus", 0.0) * (pay == "wallet"), 0, 1)
    disc = (rng.random(n) < pdisc).astype(np.int64)
    refunds = _sample_refunds(n, spec["refunds"], age, rng)
    dev = (rng.random(n) < spec["device_change"]).astype(np.int64)
    pmis = np.array([spec["mismatch"].get(m, 0.0) for m in pay])
    mis = (rng.random(n) < pmis).astype(np.int64)
    return pd.DataFrame({
        "order_amount": np.round(amt, 2), "account_age_days": age, "hour_of_day": hour,
        "num_orders_last_7d": orders, "payment_method": pay,
        "delivery_distance_km": np.round(dist, 2), "discount_applied": disc,
        "prev_refunds": refunds, "device_change": dev, "location_mismatch": mis,
    })


def generate_transactions(n: int, prevalence: float, seed: int, params: dict | None = None):
    """Return (data, meta).

    data: the 11 schema columns in data-dictionary order.
    meta: hidden diagnostics aligned row-by-row (latent_profile, true_class, label_flipped).
    """
    P = params or BASE_PARAMS
    rng = np.random.default_rng(seed)
    fn = P["label_noise"]["fraud_recorded_as_legit"]
    fp = P["label_noise"]["legit_recorded_as_fraud"]
    # choose the number of fraud-profile rows so that the *observed* rate ≈ prevalence
    n_fraud_profile = int(round(n * (prevalence - fp) / (1 - fn - fp)))
    n_legit_profile = n - n_fraud_profile
    frames, profiles, classes = [], [], []
    for cls, group, total in (("legit", P["legit"], n_legit_profile), ("fraud", P["fraud"], n_fraud_profile)):
        names = list(group)
        shares = np.array([group[k]["share"] for k in names], dtype=float)
        counts = rng.multinomial(total, shares / shares.sum())
        for name, k in zip(names, counts):
            frames.append(_sample_profile(int(k), group[name], P["global"], rng))
            profiles += [name] * int(k)
            classes += [1 if cls == "fraud" else 0] * int(k)
    data = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame({"latent_profile": profiles, "true_class": classes})
    order = rng.permutation(len(data))
    data = data.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)
    u = rng.random(len(data))
    flip = np.where(meta["true_class"] == 1, u < fn, u < fp)
    data[TARGET] = np.where(flip, 1 - meta["true_class"], meta["true_class"]).astype(np.int64)
    meta["label_flipped"] = flip.astype(np.int64)
    data = enforce_dtypes(data[list(SCHEMA)])
    return data, meta


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, t in SCHEMA.items():
        if c not in df:
            continue
        if t == "float":
            df[c] = df[c].astype("float64")
        elif t in ("int", "binary"):
            df[c] = df[c].astype("int64")
        elif t == "category":
            df[c] = df[c].astype(str)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Return a tidy table of schema/constraint checks (check, n_violations, passed)."""
    cols = list(SCHEMA) if require_target else FEATURES
    rows = []
    rows.append(("columns exactly match data dictionary (names + order)", int(list(df.columns) != cols)))
    rows.append(("missing values", int(df.isna().sum().sum())))
    for c, t in SCHEMA.items():
        if c not in df:
            continue
        s = df[c]
        if t in ("int", "binary"):
            rows.append((f"{c}: integer dtype", int(not pd.api.types.is_integer_dtype(s))))
        if t == "float":
            rows.append((f"{c}: float dtype", int(not pd.api.types.is_float_dtype(s))))
        if t == "binary":
            rows.append((f"{c}: values in {{0,1}}", int((~s.isin([0, 1])).sum())))
        if t == "category":
            rows.append((f"{c}: values in {PAYMENT_METHODS}", int((~s.isin(PAYMENT_METHODS)).sum())))
    rows.append(("hour_of_day outside 0-23", int(((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).sum())))
    for c in ["order_amount", "account_age_days", "num_orders_last_7d", "delivery_distance_km", "prev_refunds"]:
        rows.append((f"{c}: negative values", int((df[c] < 0).sum())))
    for c, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        rows.append((f"{c}: outside assumed plausible range [{lo}, {hi}]", int(((df[c] < lo) | (df[c] > hi)).sum())))
    out = pd.DataFrame(rows, columns=["check", "n_violations"])
    out["passed"] = out["n_violations"] == 0
    return out


def params_to_markdown(P: dict | None = None) -> str:
    """Render generator parameters as markdown tables for the assumptions register."""
    P = P or BASE_PARAMS
    lines = []
    for cls in ("legit", "fraud"):
        lines.append(f"\n**{'Legitimate segments' if cls == 'legit' else 'Fraud typologies'}** "
                     f"(share = share of {'legitimate' if cls == 'legit' else 'fraud'}-profile rows)\n")
        lines.append("| Profile | Share | account_age_days | num_orders_last_7d | order_amount (AED) | payment_method | "
                     "discount | prev_refunds | device_change | location_mismatch (by payment) | hour | distance (km) |")
        lines.append("|" + "---|" * 12)
        for name, s in P[cls].items():
            age = "; ".join(
                (f"{c['w']:.0%} Exp(mean {c['mean']}d)" if c["kind"] == "exp" else
                 f"{c['w']:.0%} LogN(median {c['median']}d, σ {c['sigma']})") + f" in [{c['lo']},{c['hi']}]"
                for c in s["age"]["mix"])
            o = s["orders7"]
            orders = f"Gamma-Poisson mean {o['mean']}" + (f" + Poisson({o['burst']}) burst" if o.get("burst") else "")
            pay = ", ".join(f"{k} {v:.0%}" for k, v in s["payment"].items() if v > 0)
            r = s["refunds"]
            ref = ({"tenure": f"Poisson({r.get('rate')}·(age/365)^0.7)",
                    "poisson": f"Poisson({r.get('mean')})",
                    "abuser": f"{r.get('p_early', 0):.0%} Poisson({r.get('early_mean')}), else {r.get('base')}+Poisson({r.get('extra_mean')})"}
                   [r["kind"]])
            mis = ", ".join(f"{k} {v:.0%}" for k, v in s["mismatch"].items())
            h = s["hour"]
            hour = "legit curve" + (f", flattened^{h['flatten']}" if h.get("flatten", 1) != 1 else "") + \
                   (f", ×{h['night_mult']} at 00-05" if h.get("night_mult", 1) != 1 else "")
            disc = f"{s['discount']['p']:.0%}" + (f" (+{s['discount']['wallet_bonus']:.0%} wallet)" if s['discount'].get('wallet_bonus') else "")
            lines.append(f"| {name} | {s['share']:.1%} | {age} | {orders} | LogN(median {s['amount']['median']}, σ {s['amount']['sigma']}) "
                         f"| {pay} | {disc} | {ref} | {s['device_change']:.1%} | {mis} | {hour} | Gamma({s['distance']['shape']}, {s['distance']['scale']}) |")
    g = P["global"]
    ln = P["label_noise"]
    lines.append(f"\n**Shared rules for every profile:** basket ×{g['amount_hour_mult']['breakfast_6_10']} at 06-10h and "
                 f"×{g['amount_hour_mult']['dinner_18_22']} at 18-22h; +{g['amount_per_km']:.1%} per km beyond 3.5 km; values clipped "
                 f"to the plausibility bounds in `config.PLAUSIBLE_BOUNDS`.  \n**Label noise:** "
                 f"{ln['fraud_recorded_as_legit']:.0%} of fraud recorded as legitimate; {ln['legit_recorded_as_fraud']:.1%} of legitimate recorded as fraud.")
    return "\n".join(lines)
