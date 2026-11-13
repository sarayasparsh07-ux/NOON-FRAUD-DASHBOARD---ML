"""Phase 4 rendered-UI check in a real browser (optional; needs Python Playwright + Chromium).

Starts `streamlit run app.py` headless, drives the form in Chromium, saves screenshots and a log:
    python tests/browser_check_phase4.py
Outputs: outputs/phase4/screenshots/*.png and outputs/phase4/browser_check_log.txt
Uses only hand-written synthetic transactions (no referee rows).
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase4"
SHOTS = OUT / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.modeling import FraudModel  # noqa: E402,F401

FRAUD_EXAMPLE = dict(order_amount=245.75, account_age_days=2, hour_of_day=3, num_orders_last_7d=6, payment_method="card",
                     delivery_distance_km=9.4, discount_applied=0, prev_refunds=3, device_change=1, location_mismatch=1)
LABELS = {"order_amount": "Order amount (AED)", "account_age_days": "Account age (days)",
          "num_orders_last_7d": "Orders in the last 7 days", "prev_refunds": "Previous refund claims",
          "delivery_distance_km": "Delivery distance (km)"}
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def set_number(page, label, value):
    box = page.get_by_label(label, exact=True)
    box.fill(str(value))
    box.press("Tab")


def choose(page, label, option):
    page.get_by_label(label, exact=True).click()                 # opens the Streamlit selectbox
    page.get_by_role("option", name=option, exact=True).click()


def submit_and_wait(page):
    page.get_by_role("button", name="Check transaction").click()
    page.wait_for_timeout(1500)
    page.wait_for_load_state("networkidle")


def main():
    model = joblib.load(ROOT / "models" / "fraud_model.joblib")
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    ok = True
    try:
        for _ in range(120):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read() == b"ok":
                    break
            except Exception:
                time.sleep(0.5)
        log(f"Server healthy on port {port}")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1500})
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.get_by_role("button", name="Check transaction").wait_for(timeout=60000)
            page.screenshot(path=str(SHOTS / "01_form_on_load.png"), full_page=True)
            log("01 form rendered: title and submit button visible")

            # (a) default transaction
            submit_and_wait(page)
            raw = pd.DataFrame([{"order_amount": 120.5, "account_age_days": 200, "hour_of_day": 13, "num_orders_last_7d": 3,
                                 "payment_method": "card", "delivery_distance_km": 4.5, "discount_applied": 0, "prev_refunds": 0,
                                 "device_change": 0, "location_mismatch": 0}])
            v, s = int(model.predict(raw)[0]), float(model.predict_proba(raw)[0])
            body = page.inner_text("body")
            want = "LEGIT — NOT FLAGGED BY THE MODEL" if v == 0 else "FRAUD — FLAGGED FOR MANUAL REVIEW"
            ok &= want in body and f"{s:.3f}" in body
            page.screenshot(path=str(SHOTS / "02_default_transaction_result.png"), full_page=True)
            log(f"02 default transaction: expected '{want}' score {s:.3f} -> shown: {want in body and f'{s:.3f}' in body}")

            # (b) fraud example typed into the form
            for f, lab in LABELS.items():
                set_number(page, lab, FRAUD_EXAMPLE[f])
            choose(page, "New or changed device?", "Yes")
            choose(page, "Billing and delivery location differ?", "Yes")
            slider = page.get_by_role("slider")
            slider.focus()
            for _ in range(13 - FRAUD_EXAMPLE["hour_of_day"]):
                slider.press("ArrowLeft")
            submit_and_wait(page)
            raw = pd.DataFrame([FRAUD_EXAMPLE])
            v, s = int(model.predict(raw)[0]), float(model.predict_proba(raw)[0])
            body = page.inner_text("body")
            want = "FRAUD — FLAGGED FOR MANUAL REVIEW" if v == 1 else "LEGIT — NOT FLAGGED BY THE MODEL"
            ok &= want in body and f"{s:.3f}" in body
            page.screenshot(path=str(SHOTS / "03_fraud_example_result.png"), full_page=True)
            log(f"03 fraud example: expected '{want}' score {s:.3f} -> shown: {want in body and f'{s:.3f}' in body}")

            # (c) invalid values
            set_number(page, "Order amount (AED)", -5)
            set_number(page, "Previous refund claims", -1)
            submit_and_wait(page)
            body = page.inner_text("body")
            shown = "Order amount must be greater than 0 AED." in body and "This count cannot be negative." in body and "Not scored" in body
            no_result = "Model Fraud Score" not in body
            ok &= shown and no_result
            page.screenshot(path=str(SHOTS / "04_invalid_input_errors.png"), full_page=True)
            log(f"04 invalid input: field errors shown {shown}; no score shown {no_result}")

            # (d) model information expander
            set_number(page, "Order amount (AED)", 120.5)
            set_number(page, "Previous refund claims", 3)
            submit_and_wait(page)                                # valid again: errors clear, verdict returns
            page.get_by_text("About the model and its limitations").click()
            page.wait_for_timeout(800)
            body = page.inner_text("body")
            info = "0.296956" in body and "recall 67.7%" in body and "must be greater than 0" not in body
            ok &= info
            page.screenshot(path=str(SHOTS / "05_model_information.png"), full_page=True)
            log(f"05 model information expander: threshold and Phase 3 results shown {info}")
            browser.close()
    finally:
        proc.terminate()
        out = proc.communicate(timeout=30)[0].decode(errors="replace")
    clean = "Traceback" not in out
    ok &= clean
    log(f"Server log contains no traceback: {clean}")
    log(f"RESULT: {'PASSED' if ok else 'FAILED'}")
    (OUT / "browser_check_log.txt").write_text("\n".join(log_lines) + "\n\n--- streamlit server output ---\n" + out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
