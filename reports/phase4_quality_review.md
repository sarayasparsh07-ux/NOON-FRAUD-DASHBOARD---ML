# Phase 4 — Quality Review (Part 10 checklist)

Each check is marked **Passed**, **Failed** or **Not tested**, with its evidence.
- **Test suite:** `tests/test_phase4_app.py`, 44 tests, all passed. Run with `python tests/run_phase4_tests.py`. Results are in `outputs/phase4/test_results_summary.md` and the full log in `outputs/phase4/unittest_log.txt`.
- **Browser check:** `tests/browser_check_phase4.py` drove the running app in headless Chromium. Log: `outputs/phase4/browser_check_log.txt`; screenshots: `outputs/phase4/screenshots/`.

## Part 10 checklist

| Check | Result | Evidence |
|---|---|---|
| The app uses `models/fraud_model.joblib` | Passed | `MODEL_PATH` is built from `app.py`'s location. `test_loaded_model_is_selected_baseline`; the browser expander shows the file name |
| The model is the original Phase 1 baseline | Passed | `check_model` requires metadata arm `A_baseline`; `test_threshold_matches_phase1_record` confirms `final_arm` and the threshold in `phase1_metrics.json` |
| GAN-augmented model not substituted | Passed | The app refuses it even if pointed at it (`test_gan_model_is_refused`: arm `B_ctgan` rejected) |
| Noon's benchmark model not substituted | Passed | Refused as incompatible (`test_noon_model_is_refused`) |
| The model's stored preprocessing pipeline is used | Passed | The app passes the raw ten columns with no `get_dummies` or re-implemented encoding. `test_app_matches_direct_model_on_every_example`: the app's score equals the direct `predict_proba` exactly |
| Stored calibration and threshold preserved | Passed | The verdict comes from `model.predict()`, and each example's verdict equals `score ≥ model.threshold` (0.296956, read from the model, not hard-coded). The model file hash is unchanged |
| All ten predictor inputs present | Passed | `test_renders_without_exception_and_has_ten_inputs`: exactly the ten labels, no `is_fraud` field |
| Data-dictionary categories and definitions respected | Passed | Dropdown options are exactly card / wallet / cod / applepay (tested). Labels and help text follow the dictionary; Yes/No maps to 1/0 (tested through the form) |
| Decimal AED and distance supported | Passed | `test_decimal_amount_and_distance_accepted`; decimal examples scored through the form; screenshot 03 (245.75 AED, 9.40 km) |
| Invalid inputs handled | Passed | 19 validation tests: negatives, hour < 0 or > 23, non-integer counts, NaN/inf/text, invalid binaries (2, −1, 0.5, "1", True, None), unsupported or mis-cased payment methods, missing fields, `is_fraud` supplied. Rejected before prediction, with field-level messages in the UI (AppTest and screenshot 04) |
| FRAUD / LEGIT matches the classifier | Passed | 8 synthetic examples (both verdicts) agree between the app and direct calls, and again through the rendered form (AppTest and browser) |
| Displayed score comes from the classifier | Passed | The displayed "Model Fraud Score" equals `predict_proba` to 3 dp in AppTest for all 8 examples, and in the browser (0.005 and 0.990) |
| Score not presented as a real-world probability | Passed | Labelled "Model Fraud Score"; the caption says it is calibrated on synthetic validation data and is not a verified real-world probability (`test_score_caption_disclaims_real_world_probability`) |
| App starts with the professor's command | Passed | `test_professors_exact_command_from_project_root` runs `streamlit run app.py` (port and headless set by environment variables only): healthy, served on localhost, no traceback |
| Tested with synthetic examples | Passed | 8 hand-written examples covering all payment methods, both values of every binary flag, integer and decimal amounts and distances, zero counts, zero account age, zero distance and 8 different hours. No referee row is used |
| Model files unchanged | Passed | All four `.joblib` files are hash-identical to `reports/phase4_baseline_manifest.json`; `test_model_file_unchanged` |
| Professor's original app unchanged | Passed | `professor_supplied/app(2).py` is hash-identical to the baseline manifest and the professor manifest |
| Phase 1, 2 and 3 outputs unchanged | Passed | Against the pre-Phase 4 manifest of 194 files, only `README.md` and `reports/rubric_traceability.md` changed (documentation updates); all other changes are new Phase 4 files |
| README instructions accurate and reproducible | Passed (with one limit) | The app-only environment (`requirements-app.txt`, 6 packages) was built from scratch in a clean virtual environment, and all 44 tests passed there, including the live `streamlit run app.py` test. A from-scratch install of the full `requirements.txt` (includes torch) was **not** repeated in Phase 4; it is the Phase 1 environment the app has run in throughout |
| No unsupported production-readiness claims | Passed | The page says "Academic project prototype · not an official Noon system"; the disclaimer forbids automatic action; real-data performance (precision 4.1%, 1,036 false positives) is shown in the expander |

## Part 8 test areas

| Area | Result | Evidence |
|---|---|---|
| 8.1 Model integration (direct vs app, identical class and score) | Passed (7 tests) | `T1ModelIntegration` |
| 8.2 Input validation | Passed (19 tests) | `T2Validation` |
| Part 3 error handling: missing file, folder path, corrupt file, incompatible object, wrong arm, prediction failure, misaligned columns | Passed (8 tests) | `T3ErrorHandling`. Each produces a specific message, not a traceback |
| 8.3 UI through Streamlit `AppTest`: renders, submits, shows result and score, field errors, soft warnings, disclaimer | Passed (8 tests) | `T4StreamlitUI` |
| 8.3 Live server: starts, serves HTTP 200, stops and restarts, also from an unrelated folder | Passed (2 tests) | `T5LiveServer` |
| Rendered UI in a real browser | Passed | Chromium via Playwright: form renders; default transaction gives LEGIT, 0.005; fraud example gives FRAUD, 0.990; invalid input shows field errors and no score; the model-information expander shows the threshold and Phase 3 results; no traceback in the server log. Screenshots were opened and inspected |
| Responsiveness | Passed | Multiple consecutive submissions in one session (AppTest: 8 in a row; browser: 4) |

## Issues found during testing and fixed

| Issue | Fix |
|---|---|
| Streamlit tried to look up the machine's external IP over the internet when it started | `server.address = "localhost"` in `.streamlit/config.toml`; the server now prints only the localhost URL (verified in the server log) |
| Usage telemetry enabled by default | `browser.gatherUsageStats = false` |
| A cloud "Deploy" button appeared in the toolbar | `client.toolbarMode = "minimal"` |
| The "prototype" tag was clipped at the top of the page | Increased top padding |
| The prediction-failure log could include exception text | It now logs only the exception type, so transaction values never reach the terminal log |
| A "24 of 25 flags legitimate" phrase was typed into the app | Now computed from the Phase 3 metrics file (96%) |

## Not tested / limitations

| Item | Status |
|---|---|
| Fresh install of the full `requirements.txt` on another machine or operating system | Not tested: one Linux environment was available. The app-only install was tested from scratch |
| Browsers other than Chromium; mobile layout | Not tested |
| Real-world reliability of verdicts | Out of scope for Phase 4. Phase 3 shows the model catches 67.7% of real fraud with 4.1% precision, so verdicts are review prompts only |
| Score calibration on real data | Not established: calibration was fitted on synthetic validation data (stated in the app) |
