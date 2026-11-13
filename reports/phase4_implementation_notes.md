# Phase 4 — Implementation Notes: the Fraud-Check App

**Requirement (`A_INTERN_BRIEF.md` §7):** "Wrap your final classifier so a Noon analyst can type in a transaction's details and get a verdict: FRAUD or LEGIT … plug your model into [app.py] and run it locally."

**Result:** `app.py` sits at the project root. `streamlit run app.py` opens a one-page form, takes the ten data-dictionary predictors and returns the verdict of `models/fraud_model.joblib`. This is the classifier selected in Phase 1 (arm A, baseline).

## Why the professor's shell could not be used unchanged

`professor_supplied/app(2).py` (kept unmodified) assumes a different model format from mine:

| Professor's shell assumes | My saved model actually does |
|---|---|
| A `{'model', 'columns'}` dictionary | A `FraudModel` object (pipeline + Platt calibrator + frozen threshold + feature order + metadata) |
| `pd.get_dummies` on the input, reindexed to stored columns | The pipeline one-hot encodes `payment_method` itself, so the app passes the **raw** ten columns |
| `model.predict` at scikit-learn's default 0.5 | `FraudModel.predict` applies the stored threshold, 0.296956, on the calibrated score |
| `predict_proba(X)[0][1]` (two columns) | `predict_proba(X)` returns **one** fraud score per row |
| Whole-number order amount, account age capped at 2,000 days, orders capped at 30, distance capped at 40 km | Decimal AED and km; no arbitrary maximum values (see validation) |

I kept the shell's two-column form layout and its `streamlit run app.py` command.

## Architecture (single file, five main functions)

| Function | Role |
|---|---|
| `load_model_uncached(path)` / `load_model()` | Loads the model by an absolute path built from `app.py`'s own location, so the folder the command is run from does not matter. The project root is added to `sys.path` and `src.modeling.FraudModel` is imported **before** `joblib.load`. `check_model` then confirms the object has `predict`, `predict_proba`, `threshold`, `feature_order` and `metadata`, that its feature order equals the data dictionary, that its threshold lies in (0, 1), and that its arm is `A_baseline`. As a result, the GAN-augmented model or Noon's model is refused even if copied into place. `load_model()` wraps this in `@st.cache_resource`, so the model loads once per server process |
| `validate_transaction(tx, ranges)` | Applies the data dictionary's hard rules and returns `(clean, errors, warnings)`. Each error is shown under its own field and blocks scoring |
| `build_transaction_dataframe(clean, feature_order)` | Builds a one-row DataFrame with exactly `model.feature_order` (no `is_fraud`) and the training dtypes: float64 for the two decimal fields, int64 for the counts and flags, object for `payment_method` |
| `score_transaction(model, X)` | Checks the column alignment, then returns `int(model.predict(X)[0])` as the verdict and `float(model.predict_proba(X)[0])` as the score. No rule of mine is added and the verdict is never overridden |
| `main()` | The page: header, form, result, model-information expander, disclaimer |

**Errors.** Each of these produces a specific, plain-language message instead of a traceback:
- missing file;
- path that is a folder;
- unreadable or corrupt file;
- incompatible object;
- missing methods;
- wrong model arm;
- invalid input;
- misaligned columns;
- prediction failure.

Load failures stop the page with `st.error`. Prediction failures are logged to the terminal by exception type only, so transaction values are never logged.

## Input rules

| Field | Widget | Hard rule (blocks scoring) |
|---|---|---|
| `order_amount` | Decimal number input (AED, 2 dp) | A finite number > 0 |
| `account_age_days`, `num_orders_last_7d`, `prev_refunds` | Integer number input | A whole number ≥ 0 |
| `hour_of_day` | Slider 0–23 | A whole number from 0 to 23 |
| `delivery_distance_km` | Decimal number input (km) | A finite number ≥ 0 |
| `payment_method` | Dropdown | Exactly `card`, `wallet`, `cod` or `applepay` (case-sensitive; no silent correction) |
| `discount_applied`, `device_change`, `location_mismatch` | Yes/No dropdown | Mapped to the integers 1 and 0; anything else rejected |

- **No upper limits.** The data dictionary gives none, so none of the shell's caps were copied.
- **Soft warnings.** A value outside the range seen in my **Phase 1 training partition** gets a caution, but the transaction is still scored. The range is the minimum and maximum of each numeric predictor over the 35,000 rows in `data/split_indices.npz['train']` of `data/seed_transactions.csv`:

  | Field | Training range |
  |---|---|
  | Order amount | 15–756.65 AED |
  | Account age | 0–3,000 days |
  | Orders in the last 7 days | 0–32 |
  | Delivery distance | 0.3–15 km |
  | Previous refunds | 0–9 |
  | Hour | 0–23 |

  If the data folder is absent, the app still runs without these cautions.

## Presenting the result honestly

- **Verdicts:** "FRAUD — FLAGGED FOR MANUAL REVIEW" when `predict` returns 1; "LEGIT — NOT FLAGGED BY THE MODEL" when it returns 0.
- **The score is labelled "Model Fraud Score"** and shown beside the frozen threshold and the score − threshold gap. A caption explains that the score is calibrated on **synthetic** validation data and is not a real-world probability.
- **The model-information expander** gives the model, its family, threshold and training context. It also shows the Phase 3 real-data results, read from `outputs/phase3/metrics/C1_referee_metrics_all_models.csv` rather than typed in: recall 67.7%, precision 4.1%, 1,036 false positives, 96% of flags legitimate.
- **A disclaimer** says the tool assists human review only: no automatic blocking, suspension or refund refusal. The page is labelled "Academic project prototype · not an official Noon system".

## Running locally and privacy

- **No network services.** No database, API or internet access is used, and no user input is stored.
- **`.streamlit/config.toml`** applies when the command is run from the project root. It:
  - serves on `localhost` only, which also skips Streamlit's external-IP lookup;
  - turns off usage telemetry;
  - hides the cloud "Deploy" button.
- **No new dependencies.** Streamlit was already pinned in `requirements.txt` (1.65.0). The tests use only the standard library's `unittest` and Streamlit's own `AppTest`. The optional browser check uses Playwright, which is not a project dependency.

## Files added in Phase 4

| File | Role |
|---|---|
| `app.py` | The application |
| `.streamlit/config.toml` | Local-only Streamlit settings |
| `tests/test_phase4_app.py` | Test suite (44 tests) |
| `tests/run_phase4_tests.py` | Runs the suite and writes the results summary |
| `tests/browser_check_phase4.py` | Optional real-browser check with screenshots |
| `outputs/phase4/` | Test logs, results summary and screenshots |
| `reports/phase4_*.md` | These notes and the quality review |
| `reports/phase4_baseline_manifest.json` | Hashes of all files before Phase 4 |
| `reports/phase4_master_instruction.txt` | Copy of the Phase 4 prompt |

`README.md` and `reports/rubric_traceability.md` were updated. No model, dataset, professor file or Phase 1–3 output was changed.
