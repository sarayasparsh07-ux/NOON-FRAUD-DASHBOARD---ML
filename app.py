"""Noon — Fraud Detection & Transaction Screening (Phase 4 academic prototype).

Run from anywhere:   streamlit run app.py      (from the project root, as in the brief)

An analyst types in one transaction's ten predictor values and receives the verdict of the
selected Phase 1 classifier (`models/fraud_model.joblib`, arm A — baseline):
  FRAUD — FLAGGED FOR MANUAL REVIEW   when model.predict() == 1
  LEGIT — NOT FLAGGED BY THE MODEL    when model.predict() == 0

The saved FraudModel does its own preprocessing (one-hot of payment_method), Platt calibration and
thresholding, so this app passes the RAW ten columns and never re-implements any of it.
Nothing entered here is stored or sent anywhere; the app needs no database, API or internet access.
This is an academic project prototype; it is not an official Noon system.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- 1. Project root on the import path BEFORE joblib.load (the model's class lives in src/modeling.py)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import FEATURES, PAYMENT_METHODS  # noqa: E402
from src.modeling import FAMILY_LABELS, FraudModel, PlattCalibrator  # noqa: E402,F401  (makes the pickled classes importable)

MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
SEED_PATH = ROOT / "data" / "seed_transactions.csv"
SPLIT_PATH = ROOT / "data" / "split_indices.npz"
PHASE3_METRICS = ROOT / "outputs" / "phase3" / "metrics" / "C1_referee_metrics_all_models.csv"
EXPECTED_ARM = "A_baseline"

INT_FIELDS = ["account_age_days", "hour_of_day", "num_orders_last_7d", "prev_refunds"]
FLOAT_FIELDS = ["order_amount", "delivery_distance_km"]
BINARY_FIELDS = ["discount_applied", "device_change", "location_mismatch"]
YES_NO = {"No": 0, "Yes": 1}

# Labels follow the data dictionary wording
LABELS = {
    "order_amount": "Order amount (AED)",
    "account_age_days": "Account age (days)",
    "hour_of_day": "Hour of day the order was placed (0–23)",
    "num_orders_last_7d": "Orders in the last 7 days",
    "payment_method": "Payment method",
    "delivery_distance_km": "Delivery distance (km)",
    "discount_applied": "Promo / discount code used?",
    "prev_refunds": "Previous refund claims",
    "device_change": "New or changed device?",
    "location_mismatch": "Billing and delivery location differ?",
}
HELP = {
    "order_amount": "Total value of the order, in AED. Decimals allowed.",
    "account_age_days": "Days since this customer's account was created (whole number, 0 or more).",
    "hour_of_day": "Hour the order was placed, 0 = midnight to 23 = 11 pm.",
    "num_orders_last_7d": "How many orders this customer placed in the last 7 days (whole number, 0 or more).",
    "payment_method": "One of: card, wallet, cod (cash on delivery), applepay.",
    "delivery_distance_km": "Distance from the restaurant to the delivery address, in km. Decimals allowed.",
    "discount_applied": "Yes = a promo or discount code was used (stored as 1).",
    "prev_refunds": "Number of refund claims this customer has made before (whole number, 0 or more).",
    "device_change": "Yes = the order came from a new or changed device (stored as 1).",
    "location_mismatch": "Yes = billing location and delivery location differ (stored as 1).",
}
DEFAULTS = {"order_amount": 120.50, "account_age_days": 200, "hour_of_day": 13, "num_orders_last_7d": 3,
            "payment_method": "card", "delivery_distance_km": 4.5, "discount_applied": 0, "prev_refunds": 0,
            "device_change": 0, "location_mismatch": 0}

log = logging.getLogger("fraud_check_app")


class AppError(Exception):
    """An error with a message that is safe and useful to show to the analyst."""


# ---------------------------------------------------------------------------
# Model loading and checks
# ---------------------------------------------------------------------------
def load_model_uncached(path: Path = MODEL_PATH):
    """Load the saved classifier and verify it is the expected, compatible FraudModel. Raises AppError."""
    path = Path(path)
    if not path.exists():
        raise AppError(f"The model file was not found at '{path}'. Expected the selected Phase 1 classifier "
                       "at models/fraud_model.joblib inside the project folder.")
    if not path.is_file():
        raise AppError(f"'{path}' is not a file, so the classifier cannot be loaded from it.")
    try:
        model = joblib.load(path)
    except Exception as exc:  # unreadable, truncated or incompatible pickle
        log.exception("Model loading failed")
        raise AppError(f"The model file exists but could not be loaded ({type(exc).__name__}: {exc}). "
                       "Check that the project's requirements are installed and the file is not corrupted.") from exc
    check_model(model)
    return model


def check_model(model) -> None:
    """Validate the methods and attributes the app relies on. Raises AppError with a specific reason."""
    missing = [a for a in ("predict", "predict_proba", "threshold", "feature_order", "metadata") if not hasattr(model, a)]
    if missing:
        raise AppError(f"The loaded object is not a compatible fraud model: it lacks {', '.join(missing)}.")
    if not (callable(model.predict) and callable(model.predict_proba)):
        raise AppError("The loaded object's predict / predict_proba are not callable.")
    if list(model.feature_order) != FEATURES:
        raise AppError(f"The model expects columns {list(model.feature_order)}, not the ten data-dictionary predictors.")
    try:
        thr = float(model.threshold)
    except (TypeError, ValueError) as exc:
        raise AppError("The model's stored decision threshold is not a number.") from exc
    if not (0.0 < thr < 1.0) or not math.isfinite(thr):
        raise AppError(f"The model's stored decision threshold ({model.threshold}) is outside (0, 1).")
    arm = (model.metadata or {}).get("arm")
    if arm != EXPECTED_ARM:
        raise AppError(f"The loaded model is arm '{arm}', not the selected Phase 1 classifier ('{EXPECTED_ARM}').")


@st.cache_resource(show_spinner="Loading the selected classifier…")
def load_model():
    """Cached once per server process (Streamlit resource cache)."""
    return load_model_uncached(MODEL_PATH)


def compute_training_ranges() -> dict | None:
    """Soft-warning ranges = min and max of each numeric predictor in my Phase 1 TRAINING partition
    (data/seed_transactions.csv rows listed in data/split_indices.npz['train']; 35,000 rows).
    Values outside these ranges are still scored; the analyst only sees a caution."""
    try:
        seed = pd.read_csv(SEED_PATH, usecols=FLOAT_FIELDS + INT_FIELDS)
        train = seed.iloc[np.load(SPLIT_PATH)["train"]]
    except Exception:  # data folder absent: the app still works, without range cautions
        log.warning("Training data not found; soft range warnings disabled")
        return None
    return {c: (float(train[c].min()), float(train[c].max())) for c in FLOAT_FIELDS + INT_FIELDS}


@st.cache_data(show_spinner=False)
def training_ranges() -> dict | None:
    return compute_training_ranges()


# ---------------------------------------------------------------------------
# Validation, DataFrame construction and scoring
# ---------------------------------------------------------------------------
def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


def validate_transaction(tx: dict, ranges: dict | None = None):
    """Check one transaction against the data dictionary.
    Returns (clean, errors, warnings): `clean` has exact Python types; `errors` (field → message) block scoring;
    `warnings` (field → message) are soft cautions for values outside the training range and never block scoring."""
    errors, warns, clean = {}, {}, {}
    unknown = set(tx) - set(FEATURES)
    if unknown:
        errors["_form"] = f"Unexpected field(s) {sorted(unknown)}; is_fraud is the model's output and is never an input."
    for f in FEATURES:
        if f not in tx or tx[f] is None:
            errors[f] = "A value is required."
    for f in FLOAT_FIELDS:
        if f in errors:
            continue
        v = tx[f]
        if not _is_number(v):
            errors[f] = "Enter a number."
        elif f == "order_amount" and float(v) <= 0:
            errors[f] = "Order amount must be greater than 0 AED."
        elif f == "delivery_distance_km" and float(v) < 0:
            errors[f] = "Delivery distance cannot be negative."
        else:
            clean[f] = float(v)
    for f in INT_FIELDS:
        if f in errors:
            continue
        v = tx[f]
        if not _is_number(v) or float(v) != int(float(v)):
            errors[f] = "Enter a whole number."
        elif f == "hour_of_day" and not 0 <= int(v) <= 23:
            errors[f] = "Hour must be a whole number from 0 to 23."
        elif int(v) < 0:
            errors[f] = "This count cannot be negative."
        else:
            clean[f] = int(v)
    for f in BINARY_FIELDS:
        if f in errors:
            continue
        v = tx[f]
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) not in (0, 1):
            errors[f] = "Must be 0 (No) or 1 (Yes)."
        else:
            clean[f] = int(v)
    if "payment_method" not in errors:
        if tx["payment_method"] not in PAYMENT_METHODS:
            errors["payment_method"] = f"Must be one of: {', '.join(PAYMENT_METHODS)}."
        else:
            clean["payment_method"] = tx["payment_method"]
    if ranges:
        for f, (lo, hi) in ranges.items():
            if f in clean and not lo <= clean[f] <= hi:
                warns[f] = (f"Outside the range seen in the model's training data ({lo:g}–{hi:g}). "
                            "The model will still score it, but treat the result with extra caution.")
    return clean, errors, warns


def build_transaction_dataframe(clean: dict, feature_order=None) -> pd.DataFrame:
    """One-row DataFrame with exactly the model's raw predictor columns, in its order, with training dtypes."""
    order = list(feature_order) if feature_order is not None else FEATURES
    df = pd.DataFrame([{c: clean[c] for c in order}], columns=order)
    for c in FLOAT_FIELDS:
        df[c] = df[c].astype("float64")
    for c in INT_FIELDS + BINARY_FIELDS:
        df[c] = df[c].astype("int64")
    df["payment_method"] = df["payment_method"].astype(object)
    return df


def score_transaction(model, X: pd.DataFrame):
    """Verdict from model.predict() (stored threshold) and the score from model.predict_proba(). Raises AppError."""
    if "is_fraud" in X.columns or list(X.columns) != list(model.feature_order):
        raise AppError("The transaction table does not match the model's expected predictor columns.")
    try:
        verdict = np.asarray(model.predict(X)).ravel()
        score = np.asarray(model.predict_proba(X), dtype=float).ravel()
    except Exception as exc:
        log.error("Prediction failed: %s (transaction values are deliberately not logged)", type(exc).__name__)
        raise AppError(f"The model could not score this transaction ({type(exc).__name__}: {exc}).") from exc
    if verdict.shape != (len(X),) or score.shape != (len(X),) or not np.isfinite(score).all():
        raise AppError("The model returned an unexpected output shape or a non-finite score.")
    return int(verdict[0]), float(score[0])


def phase3_selected_metrics() -> dict | None:
    try:
        return pd.read_csv(PHASE3_METRICS, index_col=0).loc["My selected classifier"].to_dict()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container {max-width: 1080px; padding-top: 3.2rem;}
.proto-tag {display:inline-block; font-size:0.78rem; letter-spacing:.04em; text-transform:uppercase; color:#6B7385;
            border:1px solid #D5DAE3; border-radius:4px; padding:2px 8px; margin-bottom:6px;}
.verdict {border-radius:8px; padding:18px 22px; margin:6px 0 14px 0; border-left:8px solid;}
.verdict h3 {margin:0 0 4px 0; font-size:1.35rem; letter-spacing:.01em;}
.verdict p {margin:0; color:#2B3445;}
.verdict.fraud {background:#FDECEA; border-color:#C2372F;} .verdict.fraud h3 {color:#9E2A24;}
.verdict.legit {background:#E7F5F4; border-color:#0E9F9A;} .verdict.legit h3 {color:#0A6E6A;}
.scorebar {position:relative; height:12px; background:#E3E6EC; border-radius:6px; margin:8px 0 4px 0;}
.scorebar .fill {position:absolute; left:0; top:0; bottom:0; border-radius:6px; background:#1B2A4A;}
.scorebar .thr {position:absolute; top:-5px; bottom:-5px; width:3px; background:#F2A541;}
.scorelegend {font-size:0.82rem; color:#6B7385;}
</style>
"""


def _field_messages(ph, f, errors, warns):
    if f in errors:
        ph.error(errors[f], icon="⛔")
    elif f in warns:
        ph.warning(warns[f], icon="⚠️")


def render_result(verdict: int, score: float, threshold: float):
    if verdict == 1:
        st.markdown('<div class="verdict fraud"><h3>FRAUD — FLAGGED FOR MANUAL REVIEW</h3>'
                    "<p>The classifier flagged this transaction. Route it to an analyst for investigation; "
                    "do not block, suspend or refuse anything on the strength of this flag alone.</p></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict legit"><h3>LEGIT — NOT FLAGGED BY THE MODEL</h3>'
                    "<p>The classifier did not flag this transaction at its stored threshold. This is not proof "
                    "that the transaction is genuine.</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Fraud Score", f"{score:.3f}")
    c2.metric("Decision threshold (frozen)", f"{threshold:.3f}")
    c3.metric("Score − threshold", f"{score - threshold:+.3f}")
    st.markdown(f'<div class="scorebar"><div class="fill" style="width:{100 * score:.1f}%"></div>'
                f'<div class="thr" style="left:calc({100 * threshold:.1f}% - 1px)"></div></div>'
                f'<div class="scorelegend">Navy bar = model fraud score (0–1). Amber line = frozen threshold '
                f'{threshold:.3f}; scores at or above it are flagged.</div>', unsafe_allow_html=True)
    st.caption("The Model Fraud Score is the classifier's calibrated output on its **synthetic** validation data. "
               "It is a ranking signal, not a verified real-world probability: a score of 0.70 does **not** mean a "
               "70% chance that this transaction is fraud, and the calibration has not been established for "
               "Noon's actual transactions.")


def render_model_info(model):
    m = model.metadata
    p3 = phase3_selected_metrics()
    with st.expander("About the model and its limitations"):
        st.markdown(
            f"- **Selected model:** `models/fraud_model.joblib` — Phase 1 arm **{m.get('arm_label', m.get('arm'))}** "
            "(chosen by the pre-declared Phase 1 selection rule; not the CTGAN-augmented candidate, not Noon's model)\n"
            f"- **Model family:** {FAMILY_LABELS.get(m.get('family'), m.get('family'))} classifier (`{m.get('family')}`) with "
            f"{m.get('calibration', 'calibration')}\n"
            f"- **Frozen decision threshold:** {model.threshold:.6f} on the calibrated score "
            f"(rule: {m.get('threshold_rule')})\n"
            f"- **Training context:** {m.get('trained_on')}: synthetic transactions built from my Phase 1 "
            f"assumptions ({m.get('prevalence_of_training_world', 0):.0%} fraud). No real Noon transaction was used for training."
        )
        if p3:
            st.markdown(
                "**Performance on Noon's real referee test set (Phase 3, 20,000 transactions, 0.325% fraud):** "
                f"recall {p3['recall']:.1%} ({int(p3['TP'])} of {int(p3['n fraud'])} frauds caught), "
                f"precision {p3['precision']:.1%}, F1 {p3['F1']:.3f}, "
                f"{int(p3['FP']):,} legitimate orders wrongly flagged ({p3['false-positive rate']:.1%} of legitimate orders). "
                f"So {1 - p3['precision']:.0%} of its flags on that data were legitimate orders. Noon's benchmark model "
                "caught more fraud with far fewer false alarms."
            )
        else:
            st.markdown("_Phase 3 results file not found; real-data performance is reported in `reports/phase3_report.md`._")


def main():
    st.set_page_config(page_title="Noon Fraud Screening — academic prototype", page_icon="🔎", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<span class="proto-tag">Academic project prototype · not an official Noon system</span>', unsafe_allow_html=True)
    st.title("Noon — Fraud Detection & Transaction Screening")
    st.write("Enter one food-delivery transaction. The tool scores it with the saved machine-learning classifier "
             "selected in Phase 1 of this project and returns a verdict: **FRAUD** (flag for manual review) or **LEGIT**.")

    try:
        model = load_model()
    except AppError as exc:
        st.error(f"**The classifier could not be loaded.** {exc}")
        st.stop()

    ranges = training_ranges()
    ph = {}
    with st.form("transaction"):
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Order")
            order_amount = st.number_input(LABELS["order_amount"], value=DEFAULTS["order_amount"], step=0.50, format="%.2f", help=HELP["order_amount"])
            ph["order_amount"] = st.empty()
            payment_method = st.selectbox(LABELS["payment_method"], PAYMENT_METHODS, index=PAYMENT_METHODS.index(DEFAULTS["payment_method"]), help=HELP["payment_method"])
            ph["payment_method"] = st.empty()
            discount_applied = st.selectbox(LABELS["discount_applied"], list(YES_NO), index=DEFAULTS["discount_applied"], help=HELP["discount_applied"])
            ph["discount_applied"] = st.empty()
            hour_of_day = st.slider(LABELS["hour_of_day"], 0, 23, DEFAULTS["hour_of_day"], help=HELP["hour_of_day"])
            ph["hour_of_day"] = st.empty()
            delivery_distance_km = st.number_input(LABELS["delivery_distance_km"], value=DEFAULTS["delivery_distance_km"], step=0.1, format="%.2f", help=HELP["delivery_distance_km"])
            ph["delivery_distance_km"] = st.empty()
        with right:
            st.subheader("Customer and account")
            account_age_days = st.number_input(LABELS["account_age_days"], value=DEFAULTS["account_age_days"], step=1, format="%d", help=HELP["account_age_days"])
            ph["account_age_days"] = st.empty()
            num_orders_last_7d = st.number_input(LABELS["num_orders_last_7d"], value=DEFAULTS["num_orders_last_7d"], step=1, format="%d", help=HELP["num_orders_last_7d"])
            ph["num_orders_last_7d"] = st.empty()
            prev_refunds = st.number_input(LABELS["prev_refunds"], value=DEFAULTS["prev_refunds"], step=1, format="%d", help=HELP["prev_refunds"])
            ph["prev_refunds"] = st.empty()
            device_change = st.selectbox(LABELS["device_change"], list(YES_NO), index=DEFAULTS["device_change"], help=HELP["device_change"])
            ph["device_change"] = st.empty()
            location_mismatch = st.selectbox(LABELS["location_mismatch"], list(YES_NO), index=DEFAULTS["location_mismatch"], help=HELP["location_mismatch"])
            ph["location_mismatch"] = st.empty()
        submitted = st.form_submit_button("Check transaction", type="primary", use_container_width=True)

    if submitted:
        tx = {"order_amount": order_amount, "account_age_days": account_age_days, "hour_of_day": hour_of_day,
              "num_orders_last_7d": num_orders_last_7d, "payment_method": payment_method,
              "delivery_distance_km": delivery_distance_km, "discount_applied": YES_NO[discount_applied],
              "prev_refunds": prev_refunds, "device_change": YES_NO[device_change],
              "location_mismatch": YES_NO[location_mismatch]}
        clean, errors, warns = validate_transaction(tx, ranges)
        for f, p in ph.items():
            _field_messages(p, f, errors, warns)
        if errors:
            st.error(f"**Not scored.** Fix the {len(errors)} highlighted field(s) and submit again.")
        else:
            try:
                verdict, score = score_transaction(model, build_transaction_dataframe(clean, model.feature_order))
            except AppError as exc:
                st.error(f"**Prediction failed.** {exc}")
            else:
                st.subheader("Result")
                render_result(verdict, score, float(model.threshold))
                if warns:
                    st.warning(f"{len(warns)} value(s) fall outside the range seen in training (marked above). "
                               "The verdict is the model's output, but it is extrapolating.")

    render_model_info(model)
    st.info("**How to use this result.** This is an academic fraud-screening prototype that assists human review. "
            "A FRAUD verdict means the transaction is flagged for investigation; a LEGIT verdict means it was not flagged "
            "at the stored threshold. Neither establishes whether the transaction is actually fraudulent. Do not use it "
            "to automatically block orders, suspend accounts or deny refunds. On Noon's real test data most flags were "
            "legitimate orders (see *About the model*).", icon="ℹ️")
    st.caption("Privacy: the values you enter are used only to score this transaction in this browser session. "
               "They are not saved to disk, written to a log or sent to any external service.")


if __name__ == "__main__":
    main()
