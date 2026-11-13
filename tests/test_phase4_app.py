"""Phase 4 test suite for app.py (standard-library unittest; no extra dependency).

Run from the project root:
    python -m unittest -v tests/test_phase4_app.py

Covers: model integration (app vs direct model calls), input validation, error handling, the
Streamlit UI through streamlit.testing.v1.AppTest, and a live `streamlit run app.py` server that is
started, queried, stopped and restarted. All transactions are hand-written synthetic examples;
no row from the real referee dataset is used.
"""
from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import app  # noqa: E402  (importing does not run main(): Streamlit runs the file as __main__)

MODEL = ROOT / "models" / "fraud_model.joblib"
GAN_MODEL = ROOT / "models" / "fraud_model_gan_augmented.joblib"

# Hand-written synthetic transactions (not referee rows). Together they cover all four payment
# methods, every binary flag at 0 and 1, integer and decimal amounts/distances, zero counts,
# zero account age, zero distance, and hours 0, 2, 3, 11, 12, 20, 22, 23.
EXAMPLES = [
    dict(order_amount=245.75, account_age_days=2, hour_of_day=3, num_orders_last_7d=6, payment_method="card",
         delivery_distance_km=9.4, discount_applied=0, prev_refunds=3, device_change=1, location_mismatch=1),
    dict(order_amount=38.20, account_age_days=0, hour_of_day=22, num_orders_last_7d=0, payment_method="wallet",
         delivery_distance_km=2.15, discount_applied=1, prev_refunds=0, device_change=1, location_mismatch=0),
    dict(order_amount=310.0, account_age_days=900, hour_of_day=2, num_orders_last_7d=5, payment_method="applepay",
         delivery_distance_km=7.8, discount_applied=0, prev_refunds=1, device_change=1, location_mismatch=1),
    dict(order_amount=160.4, account_age_days=12, hour_of_day=20, num_orders_last_7d=1, payment_method="cod",
         delivery_distance_km=6.25, discount_applied=0, prev_refunds=0, device_change=0, location_mismatch=0),
    dict(order_amount=95, account_age_days=640, hour_of_day=12, num_orders_last_7d=2, payment_method="card",
         delivery_distance_km=0, discount_applied=1, prev_refunds=0, device_change=0, location_mismatch=0),
    dict(order_amount=72.35, account_age_days=45, hour_of_day=0, num_orders_last_7d=8, payment_method="wallet",
         delivery_distance_km=3.3, discount_applied=0, prev_refunds=5, device_change=0, location_mismatch=1),
    dict(order_amount=19.99, account_age_days=1, hour_of_day=11, num_orders_last_7d=0, payment_method="applepay",
         delivery_distance_km=1.05, discount_applied=1, prev_refunds=0, device_change=0, location_mismatch=0),
    dict(order_amount=520.0, account_age_days=1500, hour_of_day=23, num_orders_last_7d=12, payment_method="cod",
         delivery_distance_km=14.9, discount_applied=0, prev_refunds=2, device_change=1, location_mismatch=0),
]
VALID = app.DEFAULTS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class T1ModelIntegration(unittest.TestCase):
    """8.1 — the app's pipeline gives exactly the saved model's own outputs."""

    @classmethod
    def setUpClass(cls):
        cls.model_hash_before = sha(MODEL)
        cls.model = app.load_model_uncached()
        cls.direct = joblib.load(MODEL)            # independent load for the direct comparison

    def test_loaded_model_is_selected_baseline(self):
        m = self.model
        self.assertEqual(type(m).__name__, "FraudModel")
        self.assertEqual(m.metadata["arm"], "A_baseline")
        self.assertEqual(m.metadata["family"], "hist_gb")
        self.assertIn("Platt", m.metadata["calibration"])
        self.assertEqual(list(m.pipeline.classes_), [0, 1])            # class 1 = fraud
        self.assertEqual(list(m.feature_order), app.FEATURES)
        self.assertNotIn("is_fraud", m.feature_order)
        self.assertAlmostEqual(m.threshold, 0.296956, places=6)         # verified, not used as a constant by the app

    def test_threshold_matches_phase1_record(self):
        import json
        p1 = json.loads((ROOT / "outputs" / "metrics" / "phase1_metrics.json").read_text())
        self.assertEqual(self.model.threshold, p1["final_threshold_calibrated"])
        self.assertEqual(p1["final_arm"], self.model.metadata["arm"])

    def test_examples_cover_required_cases(self):
        df = pd.DataFrame(EXAMPLES)
        self.assertEqual(set(df.payment_method), set(app.PAYMENT_METHODS))
        for f in app.BINARY_FIELDS:
            self.assertEqual(set(df[f]), {0, 1}, f)
        self.assertTrue(any(float(v) != int(v) for v in df.order_amount))
        self.assertTrue(any(float(v) == int(v) for v in df.order_amount))
        self.assertTrue(any(float(v) != int(v) for v in df.delivery_distance_km))
        for f in ["account_age_days", "num_orders_last_7d", "prev_refunds"]:
            self.assertIn(0, set(df[f]), f)
        self.assertGreaterEqual(df.hour_of_day.nunique(), 6)

    def test_app_matches_direct_model_on_every_example(self):
        verdicts = []
        for i, ex in enumerate(EXAMPLES):
            clean, errors, _ = app.validate_transaction(ex)
            self.assertEqual(errors, {}, f"example {i}")
            X = app.build_transaction_dataframe(clean, self.model.feature_order)
            v_app, s_app = app.score_transaction(self.model, X)
            raw = pd.DataFrame([ex])[app.FEATURES]                      # the same raw input, straight to the model
            v_dir = int(self.direct.predict(raw)[0])
            s_dir = float(self.direct.predict_proba(raw)[0])
            self.assertEqual(v_app, v_dir, f"example {i}")
            self.assertEqual(s_app, s_dir, f"example {i}")              # identical, not merely close
            self.assertEqual(v_app, int(s_dir >= self.direct.threshold), f"example {i}: verdict follows stored threshold")
            verdicts.append(v_app)
        self.assertEqual(set(verdicts), {0, 1}, "examples should exercise both verdicts")

    def test_dataframe_columns_and_dtypes(self):
        clean, _, _ = app.validate_transaction(VALID)
        X = app.build_transaction_dataframe(clean, self.model.feature_order)
        self.assertEqual(list(X.columns), list(self.model.feature_order))
        self.assertNotIn("is_fraud", X.columns)
        self.assertEqual(X.shape, (1, 10))
        for c in app.FLOAT_FIELDS:
            self.assertEqual(X[c].dtype, np.float64)
        for c in app.INT_FIELDS + app.BINARY_FIELDS:
            self.assertEqual(X[c].dtype, np.int64)

    def test_predict_proba_is_one_dimensional(self):
        clean, _, _ = app.validate_transaction(VALID)
        p = self.model.predict_proba(app.build_transaction_dataframe(clean))
        self.assertEqual(np.asarray(p).shape, (1,))

    def test_model_file_unchanged(self):
        self.assertEqual(sha(MODEL), self.model_hash_before)


class T2Validation(unittest.TestCase):
    """8.2 — invalid values are rejected before prediction; decimals are accepted."""

    def bad(self, **change):
        tx = dict(VALID, **change)
        clean, errors, _ = app.validate_transaction(tx)
        return errors

    def test_valid_default(self):
        self.assertEqual(self.bad(), {})

    def test_negative_order_amount(self):
        self.assertIn("order_amount", self.bad(order_amount=-10.0))

    def test_zero_order_amount(self):
        self.assertIn("order_amount", self.bad(order_amount=0.0))

    def test_negative_account_age(self):
        self.assertIn("account_age_days", self.bad(account_age_days=-1))

    def test_hour_below_0(self):
        self.assertIn("hour_of_day", self.bad(hour_of_day=-1))

    def test_hour_above_23(self):
        self.assertIn("hour_of_day", self.bad(hour_of_day=24))

    def test_negative_order_count(self):
        self.assertIn("num_orders_last_7d", self.bad(num_orders_last_7d=-3))

    def test_negative_distance(self):
        self.assertIn("delivery_distance_km", self.bad(delivery_distance_km=-0.5))

    def test_negative_refunds(self):
        self.assertIn("prev_refunds", self.bad(prev_refunds=-2))

    def test_invalid_binary_values(self):
        for f in app.BINARY_FIELDS:
            for v in (2, -1, 0.5, "1", True, None):
                self.assertIn(f, self.bad(**{f: v}), f"{f}={v!r}")

    def test_unsupported_payment_methods(self):
        for v in ("paypal", "Card", "CARD", "", "crypto", None):
            self.assertIn("payment_method", self.bad(payment_method=v), repr(v))

    def test_non_integer_counts_rejected(self):
        for f in ["account_age_days", "num_orders_last_7d", "prev_refunds", "hour_of_day"]:
            self.assertIn(f, self.bad(**{f: 2.5}), f)

    def test_non_finite_and_text_numbers_rejected(self):
        self.assertIn("order_amount", self.bad(order_amount=float("nan")))
        self.assertIn("delivery_distance_km", self.bad(delivery_distance_km=float("inf")))
        self.assertIn("order_amount", self.bad(order_amount="120"))

    def test_decimal_amount_and_distance_accepted(self):
        clean, errors, _ = app.validate_transaction(dict(VALID, order_amount=87.35, delivery_distance_km=3.75))
        self.assertEqual(errors, {})
        self.assertEqual(clean["order_amount"], 87.35)
        self.assertEqual(clean["delivery_distance_km"], 3.75)

    def test_whole_number_floats_accepted_for_counts(self):
        clean, errors, _ = app.validate_transaction(dict(VALID, account_age_days=30.0))
        self.assertEqual(errors, {})
        self.assertIsInstance(clean["account_age_days"], int)

    def test_is_fraud_is_not_an_input(self):
        self.assertIn("_form", self.bad(is_fraud=1))

    def test_missing_field(self):
        tx = dict(VALID); tx.pop("prev_refunds")
        self.assertIn("prev_refunds", app.validate_transaction(tx)[1])

    def test_soft_warning_does_not_block(self):
        ranges = app.compute_training_ranges()
        self.assertIsNotNone(ranges)
        clean, errors, warns = app.validate_transaction(dict(VALID, delivery_distance_km=25.0, order_amount=1200.0), ranges)
        self.assertEqual(errors, {})
        self.assertIn("delivery_distance_km", warns)
        self.assertIn("order_amount", warns)

    def test_training_ranges_come_from_training_partition(self):
        ranges = app.compute_training_ranges()
        seed = pd.read_csv(ROOT / "data" / "seed_transactions.csv")
        train = seed.iloc[np.load(ROOT / "data" / "split_indices.npz")["train"]]
        for c, (lo, hi) in ranges.items():
            self.assertEqual((lo, hi), (float(train[c].min()), float(train[c].max())), c)


class T3ErrorHandling(unittest.TestCase):
    """Part 3 — clean, specific errors instead of tracebacks."""

    def test_missing_model_file(self):
        with self.assertRaisesRegex(app.AppError, "not found"):
            app.load_model_uncached(ROOT / "models" / "does_not_exist.joblib")

    def test_path_is_a_directory(self):
        with self.assertRaisesRegex(app.AppError, "not a file"):
            app.load_model_uncached(ROOT / "models")

    def test_corrupt_model_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.joblib"
            p.write_bytes(b"this is not a pickle")
            with self.assertRaisesRegex(app.AppError, "could not be loaded"):
                app.load_model_uncached(p)

    def test_incompatible_object(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dict.joblib"
            joblib.dump({"model": None, "columns": []}, p)
            with self.assertRaisesRegex(app.AppError, "not a compatible fraud model"):
                app.load_model_uncached(p)

    def test_gan_model_is_refused(self):
        with self.assertRaisesRegex(app.AppError, "B_ctgan"):
            app.load_model_uncached(GAN_MODEL)

    def test_noon_model_is_refused(self):
        with self.assertRaisesRegex(app.AppError, "not a compatible fraud model"):
            app.load_model_uncached(ROOT / "professor_supplied" / "B_NOON_model.joblib")

    def test_prediction_failure(self):
        class Broken:
            feature_order = app.FEATURES
            def predict(self, X): raise RuntimeError("boom")
            def predict_proba(self, X): raise RuntimeError("boom")
        clean, _, _ = app.validate_transaction(VALID)
        with self.assertRaisesRegex(app.AppError, "could not score"):
            app.score_transaction(Broken(), app.build_transaction_dataframe(clean))

    def test_misaligned_columns_refused(self):
        m = app.load_model_uncached()
        clean, _, _ = app.validate_transaction(VALID)
        X = app.build_transaction_dataframe(clean)
        with self.assertRaisesRegex(app.AppError, "expected predictor columns"):
            app.score_transaction(m, X[list(reversed(X.columns))])
        with self.assertRaisesRegex(app.AppError, "expected predictor columns"):
            app.score_transaction(m, X.assign(is_fraud=0))


class T4StreamlitUI(unittest.TestCase):
    """8.3 — the rendered app via Streamlit's own testing harness."""

    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        self.at.run()
        self.direct = joblib.load(MODEL)

    def text(self):
        parts = [m.value for m in self.at.markdown] + [e.value for e in self.at.error] + [w.value for w in self.at.warning]
        return "\n".join(map(str, parts))

    def metric(self, label):
        return [m for m in self.at.metric if m.label == label][0].value

    def fill(self, tx):
        ni = {w.label: w for w in self.at.number_input}
        sb = {w.label: w for w in self.at.selectbox}
        for f in ["order_amount", "account_age_days", "num_orders_last_7d", "prev_refunds", "delivery_distance_km"]:
            ni[app.LABELS[f]].set_value(tx[f])
        sb[app.LABELS["payment_method"]].set_value(tx["payment_method"])
        for f in app.BINARY_FIELDS:
            sb[app.LABELS[f]].set_value("Yes" if tx[f] else "No")
        self.at.slider[0].set_value(tx["hour_of_day"])

    def submit(self):
        self.at.button[0].click()
        self.at.run()

    def test_renders_without_exception_and_has_ten_inputs(self):
        self.assertFalse(self.at.exception)
        self.assertEqual(self.at.title[0].value, "Noon — Fraud Detection & Transaction Screening")
        labels = [w.label for w in list(self.at.number_input) + list(self.at.selectbox) + list(self.at.slider)]
        self.assertEqual(sorted(labels), sorted(app.LABELS.values()))
        self.assertNotIn("is_fraud", " ".join(labels))
        self.assertEqual(list(self.at.selectbox[0].options), app.PAYMENT_METHODS)
        self.assertEqual(len(self.at.metric), 0, "no result before submission")

    def test_submit_default_shows_result_matching_model(self):
        self.submit()
        self.assertFalse(self.at.exception)
        raw = pd.DataFrame([app.DEFAULTS])[app.FEATURES]
        v, s = int(self.direct.predict(raw)[0]), float(self.direct.predict_proba(raw)[0])
        self.assertIn("LEGIT — NOT FLAGGED BY THE MODEL" if v == 0 else "FRAUD — FLAGGED FOR MANUAL REVIEW", self.text())
        self.assertEqual(self.metric("Model Fraud Score"), f"{s:.3f}")
        self.assertEqual(self.metric("Decision threshold (frozen)"), f"{self.direct.threshold:.3f}")

    def test_fraud_example_through_the_form(self):
        tx = EXAMPLES[0]
        self.fill(tx)
        self.submit()
        raw = pd.DataFrame([tx])[app.FEATURES]
        self.assertEqual(int(self.direct.predict(raw)[0]), 1)
        self.assertIn("FRAUD — FLAGGED FOR MANUAL REVIEW", self.text())
        self.assertEqual(self.metric("Model Fraud Score"), f"{float(self.direct.predict_proba(raw)[0]):.3f}")

    def test_every_example_through_the_form(self):
        for tx in EXAMPLES:
            self.fill(tx)
            self.submit()
            raw = pd.DataFrame([tx])[app.FEATURES]
            v = int(self.direct.predict(raw)[0])
            self.assertIn("FRAUD — FLAGGED FOR MANUAL REVIEW" if v else "LEGIT — NOT FLAGGED BY THE MODEL", self.text())
            self.assertEqual(self.metric("Model Fraud Score"), f"{float(self.direct.predict_proba(raw)[0]):.3f}")

    def test_invalid_inputs_show_field_errors_and_no_result(self):
        ni = {w.label: w for w in self.at.number_input}
        ni[app.LABELS["order_amount"]].set_value(-5.0)
        ni[app.LABELS["prev_refunds"]].set_value(-1)
        ni[app.LABELS["delivery_distance_km"]].set_value(-2.0)
        self.submit()
        errs = [e.value for e in self.at.error]
        self.assertTrue(any("greater than 0 AED" in e for e in errs))
        self.assertTrue(any("cannot be negative" in e for e in errs))
        self.assertTrue(any("Delivery distance cannot be negative" in e for e in errs))
        self.assertTrue(any("Not scored" in e for e in errs))
        self.assertEqual(len(self.at.metric), 0, "no score shown for invalid input")
        self.assertNotIn("FLAGGED", self.text().replace("NOT FLAGGED BY THE MODEL", ""))

    def test_out_of_range_value_warns_but_scores(self):
        ni = {w.label: w for w in self.at.number_input}
        ni[app.LABELS["delivery_distance_km"]].set_value(28.5)
        self.submit()
        self.assertEqual(len(self.at.metric), 3)
        self.assertTrue(any("Outside the range seen in the model's training data" in w.value for w in self.at.warning))

    def test_disclaimer_and_model_info_present(self):
        body = "\n".join(str(i.value) for i in self.at.info) + self.text()
        self.assertIn("academic fraud-screening prototype", body)
        self.assertIn("Do not use it to automatically block orders", body)
        self.assertIn("Academic project prototype · not an official Noon system", body)
        self.assertEqual(self.at.expander[0].label, "About the model and its limitations")
        exp_md = "\n".join(m.value for m in self.at.expander[0].markdown)
        self.assertIn("A · Baseline", exp_md)
        self.assertIn("0.296956", exp_md)
        self.assertIn("recall 67.7%", exp_md)

    def test_score_caption_disclaims_real_world_probability(self):
        self.submit()
        caps = "\n".join(c.value for c in self.at.caption)
        self.assertIn("not a verified real-world probability", caps)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class T5LiveServer(unittest.TestCase):
    """8.3 — `streamlit run app.py` starts, serves, stops and restarts (started from another folder too)."""

    def _start(self, port, cwd):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                                 "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    def _wait_health(self, port, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                    if r.read().decode().strip() == "ok":
                        return time.time() - t0
            except Exception:
                time.sleep(0.5)
        return None

    def test_start_stop_restart(self):
        port = _free_port()
        for attempt, cwd in enumerate([str(ROOT), tempfile.gettempdir()]):   # second start from an unrelated folder
            proc = self._start(port, cwd)
            try:
                waited = self._wait_health(port)
                self.assertIsNotNone(waited, f"server did not become healthy (start {attempt + 1})")
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                    self.assertEqual(r.status, 200)
                self.assertIsNone(proc.poll(), "server exited unexpectedly")
            finally:
                proc.terminate()
                out = proc.communicate(timeout=30)[0].decode(errors="replace")
            self.assertNotIn("Traceback", out)
            self.assertIn(proc.returncode, (0, -15, 143), out[-500:])
        self.assertIsNone(self._wait_health(port, timeout=2), "server should be stopped")

    def test_professors_exact_command_from_project_root(self):
        """`streamlit run app.py` exactly as written in the brief; port/headless come from environment variables
        so the command itself is unchanged; .streamlit/config.toml (localhost only, no telemetry) is picked up."""
        import shutil
        exe = shutil.which("streamlit")
        self.assertIsNotNone(exe, "the streamlit command is not on PATH")
        port = _free_port()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", STREAMLIT_SERVER_PORT=str(port), STREAMLIT_SERVER_HEADLESS="true")
        proc = subprocess.Popen([exe, "run", "app.py"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        try:
            self.assertIsNotNone(self._wait_health(port), "server did not become healthy")
        finally:
            proc.terminate()
            out = proc.communicate(timeout=30)[0].decode(errors="replace")
        self.assertNotIn("Traceback", out)
        self.assertIn(f"localhost:{port}", out)            # config.toml applied: served on localhost only
        self.assertNotIn("Network URL", out)
        self.assertNotIn("external IP", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
