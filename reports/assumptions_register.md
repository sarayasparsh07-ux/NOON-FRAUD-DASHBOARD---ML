# Assumptions Register — Phase 1 (Build Blind)

*Written before any model was trained. This register separates what the brief and data dictionary actually say from what I assumed, and from what remains a hypothesis. Phase 3 (real data) is where these assumptions get checked. Research sources: `Food delivery fraud GAN research.md` (deep-research report delivered with this submission).*

## A. Source facts (from the supplied documents only)

| # | Fact | Source |
|---|---|---|
| S1 | 11 columns: `order_amount` (number, AED), `account_age_days` (integer), `hour_of_day` (integer 0–23), `num_orders_last_7d` (integer), `payment_method` (card / wallet / cod / applepay), `delivery_distance_km` (number), `discount_applied` (0/1), `prev_refunds` (integer), `device_change` (0/1), `location_mismatch` (0/1, billing vs delivery location), `is_fraud` (target 0/1) | Data dictionary |
| S2 | Fraud includes stolen cards, promo-code abuse and fake "my order never arrived" refund scams | Brief §1 |
| S3 | Genuine fraud is so infrequent that the model barely sees any | Brief §2 |
| S4 | A real GAN (CTGAN/SDV recommended) must generate the extra fraud; an LLM row-dump is not acceptable | Brief §2 |
| S5 | No sample rows, statistics, fraud rate or relationships between columns are provided | Data dictionary |

Nothing else about Noon's data is known. In particular, **Noon's fraud rate, distributions and fraud rules are unknown**, and nothing below claims otherwise.

## B. Modelling assumptions, feature by feature

Evidence strength: **M** = moderate public anchor, **W** = weak/old/indirect anchor, **A** = pure assumption.

| Feature | Valid values | Legitimate behaviour (assumed) | Fraudulent behaviour (assumed) | Interactions designed in | Innocent explanations for "risky" values | Evidence | Uncertainty |
|---|---|---|---|---|---|---|---|
| `order_amount` | > 0 AED; assumed plausible 15–2,000 | Log-normal, median ≈ AED 62, mean ≈ AED 77; larger at dinner (×1.08), smaller at breakfast (×0.8); +1.5%/km beyond 3.5 km | Bimodal: promo abuse near the minimum basket (~AED 38); stolen card (~AED 135), takeover (~AED 115) and COD fakes (~AED 150) high | Amount × account age; amount × distance; amount × device/mismatch | Group/office orders, celebrations, power users (median ~AED 95) | M (UAE ticket ≈ AED 77, 2025–26); fraud multipliers W | Mean anchored; tail shape and fraud multipliers assumed |
| `account_age_days` | integer ≥ 0; assumed ≤ 3,000 | 8% of orders from accounts ≤ 30 days; established median ≈ 420 days | **Bimodal**: promo abuse (~2.5 days), COD fakes (~5 days), stolen card (mostly < 60 days) vs account takeover (median ~700 days) | Age × device change (takeover = old + new device); age × discount | Brand-new genuine customers (welcome promos, cash payment) | Direction M (vendor case studies); shares A | High — the single most influential assumption |
| `hour_of_day` | 0–23 | Lunch (12–14) and dinner (19–21) peaks; ~9% between 00:00 and 05:59 (UAE late-night culture, Ramadan) | Stolen card ×3 and takeover ×2 weight at 00–05; promo/COD slightly flatter | Hour × account age | Shift workers, Ramadan suhoor orders | W (2014–16 payments data; UAE late-night tail M) | Night uplift is old, non-food evidence |
| `num_orders_last_7d` | integer ≥ 0; assumed ≤ 40 | Over-dispersed Gamma-Poisson, mean ≈ 1.4 (new accounts 0.8); 2% power users mean ≈ 9 | Takeover adds a burst of ~3 orders; stolen card mean ≈ 2.8; refund abusers ≈ 2.6; promo accounts ≈ 0.4 | Frequency × refunds; frequency × discount | Power users and families order daily | M (talabat: 3.8–6.7 orders/month) | Fraud bursts assumed |
| `payment_method` | card, wallet, cod, applepay | Card 48%, Apple Pay 30%, COD 14%, wallet 8% (new accounts: COD 22%) | Stolen card 78% card, 0% COD; takeover drains wallet (30%); COD fakes 100% COD; refund abusers favour wallet (refund credit) | Payment × mismatch (no billing address for COD → mismatch ≈ 2%) | Tourists/expats, cash-preferring new users | M (UAE e-commerce mix; COD single digits–teens) | Food-delivery mix not published; Apple Pay lower fraud is assumed |
| `delivery_distance_km` | ≥ 0; assumed 0.3–15 | Gamma, median ≈ 3.2 km, mean ≈ 3.7 km | Takeover orders to a new area (mean ≈ 6 km); COD fakes ≈ 5.5 km; others legit-like | Distance × amount | Office vs home addresses, suburbs | W (only London 2017 data) | Weakest anchor; carries little signal by design |
| `discount_applied` | 0/1 | 32% (+8 pp for wallet cashback); new accounts 72% | Promo abuse 96%; refund abuse 45%; others 20–30% | Discount × account age; discount × frequency | Welcome offers, loyalty promos | A (promo share of fraud from vendor data, W) | Legit promo rate assumed |
| `prev_refunds` | integer ≥ 0; assumed ≤ 15 | Poisson accumulating with tenure (≈ 0.12/yr^0.7); 1.5% "genuine complainers" with several real refunds | Refund abuse: 85% have 1 + Poisson(2.3), 15% early-career with ~0–1 | Refunds × frequency | Late or wrong deliveries do happen | Direction M; magnitudes A | High |
| `device_change` | 0/1 | 3.5% (new phones, shared family devices) | Takeover 80%; stolen card 30%; COD fakes 10%; promo 7% (device farms reuse "new" devices per account, so this flag barely sees them) | Device × mismatch × amount | New phone, app reinstall | Direction M (ATO case study); rates A | Medium |
| `location_mismatch` | 0/1 | 13–16% for card/Apple Pay (tourists, foreign cards, office deliveries), 8% wallet, 2% COD | Stolen card 55%; takeover 45%; promo/refund ≈ legit | Mismatch × payment; mismatch × device | Dubai's 18.7 M visitors (2024), expats with home-country cards | M (visitor numbers); fraud rates A | Definition ambiguous (billing vs delivery city? country?) |
| `is_fraud` | 0/1 | 97% | **3% observed** (fraud + policy abuse label); 3% of fraud recorded as legit (undetected), 0.1% of legit recorded as fraud (review errors) | — | — | M (food-delivery block rate ≈ 3.1%; IEEE-CIS 3.5%) | Confirmed card fraud alone would be ~0.1–0.5%; tested at 1% and 5% |

## C. Fraud scenarios (brief and master prompt) → how they are represented

| Scenario | Representation in the generator |
|---|---|
| A — Suspected payment abuse (changed device, mismatch, high value together) | `F_stolen_card` and `F_account_takeover` typologies |
| B — Promotional abuse (new account, discount-assisted ordering) | `F_promo_abuse` typology, with a deliberately strong legitimate look-alike (`L_new_customer`) |
| C — Refund abuse (many prior refunds + unusual ordering) | `F_refund_abuse`, with `L_genuine_complainer` as the innocent look-alike |
| D — Combined moderate signals, none decisive | Every typology draws each signal probabilistically; e.g. only 55% of stolen-card orders show a mismatch |
| COD fake orders (added) | `F_cod_fake_order` — cash removes card fraud but enables fake orders/refusals |

## D. Hypotheses (unverifiable until real data)

| # | Hypothesis | How Phase 1 treats it |
|---|---|---|
| H1 | Fraud account age is bimodal (new-account typologies vs takeover of old accounts) | Designed in; checked in EDA (§1.3) |
| H2 | No single column separates fraud well; combinations do | Checked: best single-feature AUC < 0.85; 4-leaf tree far below tuned model |
| H3 | Device change **and** location mismatch together are far riskier than either alone | Designed in; fraud-rate heatmap §1.3 |
| H4 | Promo abuse is the hardest typology to catch with these ten columns (strongest real signals — accounts per device — are absent) | Measured as recall by typology §1.8 |
| H5 | A GAN trained on fraud examples can add useful, non-duplicate fraud variety | The core Phase 1 experiment (§1.6–1.9, 1.12) |
| H6 | Night ordering is a fraud signal in the UAE | Weak by design (real late-night legitimate demand) |
| H7 | Noon's real fraud resembles this mixture | **Untestable in Phase 1** — Phase 2/3 will show how wrong it is |

## E. Experimental parameters (not claims about Noon)

50,000 seed rows; 3% prevalence (sensitivity: 1% with 100,000 rows, 5% with 50,000 rows); 70/15/15 stratified split; master seed 42. Rationale: ≥ 200 fraud cases in the test partition for stable estimates.

## F. Generator parameters (rendered from `src/seed_generator.py`)


**Legitimate segments** (share = share of legitimate-profile rows)

| Profile | Share | account_age_days | num_orders_last_7d | order_amount (AED) | payment_method | discount | prev_refunds | device_change | location_mismatch (by payment) | hour | distance (km) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L_established | 88.5% | 100% LogN(median 420d, σ 0.85) in [31,3000] | Gamma-Poisson mean 1.4 | LogN(median 62, σ 0.62) | card 48%, applepay 30%, wallet 8%, cod 14% | 32% (+8% wallet) | Poisson(0.12·(age/365)^0.7) | 3.5% | card 14%, applepay 13%, wallet 8%, cod 2% | legit curve | Gamma(2.3, 1.6) |
| L_new_customer | 8.0% | 100% Exp(mean 9d) in [0,30] | Gamma-Poisson mean 0.8 | LogN(median 60, σ 0.6) | card 45%, applepay 27%, wallet 6%, cod 22% | 72% | Poisson(0.01) | 3.0% | card 16%, applepay 13%, wallet 8%, cod 2% | legit curve | Gamma(2.3, 1.6) |
| L_power_user | 2.0% | 100% LogN(median 600d, σ 0.7) in [31,3000] | Gamma-Poisson mean 9.0 | LogN(median 95, σ 0.7) | card 50%, applepay 32%, wallet 10%, cod 8% | 30% (+5% wallet) | Poisson(0.35·(age/365)^0.7) | 4.0% | card 14%, applepay 13%, wallet 8%, cod 2% | legit curve | Gamma(2.3, 1.6) |
| L_genuine_complainer | 1.5% | 100% LogN(median 500d, σ 0.8) in [31,3000] | Gamma-Poisson mean 2.0 | LogN(median 70, σ 0.6) | card 48%, applepay 30%, wallet 10%, cod 12% | 32% (+8% wallet) | 30% Poisson(1.0), else 1+Poisson(1.5) | 3.5% | card 14%, applepay 13%, wallet 8%, cod 2% | legit curve | Gamma(2.3, 1.6) |

**Fraud typologies** (share = share of fraud-profile rows)

| Profile | Share | account_age_days | num_orders_last_7d | order_amount (AED) | payment_method | discount | prev_refunds | device_change | location_mismatch (by payment) | hour | distance (km) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F_promo_abuse | 32.0% | 100% Exp(mean 2.5d) in [0,20] | Gamma-Poisson mean 0.4 | LogN(median 38, σ 0.4) | card 45%, wallet 30%, applepay 18%, cod 7% | 96% | Poisson(0.02) | 7.0% | card 18%, applepay 15%, wallet 10%, cod 2% | legit curve, flattened^0.7 | Gamma(2.3, 1.6) |
| F_refund_abuse | 27.0% | 100% LogN(median 260d, σ 0.9) in [10,3000] | Gamma-Poisson mean 2.6 | LogN(median 90, σ 0.55) | card 40%, applepay 25%, wallet 22%, cod 13% | 45% | 15% Poisson(0.5), else 1+Poisson(2.3) | 5.0% | card 14%, applepay 13%, wallet 8%, cod 2% | legit curve | Gamma(2.3, 1.6) |
| F_stolen_card | 23.0% | 65% Exp(mean 6d) in [0,60]; 35% LogN(median 120d, σ 0.8) in [0,3000] | Gamma-Poisson mean 2.8 | LogN(median 135, σ 0.6) | card 78%, applepay 15%, wallet 7% | 22% | Poisson(0.05) | 30.0% | card 55%, applepay 45%, wallet 30%, cod 0% | legit curve, ×3.0 at 00-05 | Gamma(2.3, 1.8) |
| F_account_takeover | 13.0% | 100% LogN(median 700d, σ 0.6) in [120,3000] | Gamma-Poisson mean 1.4 + Poisson(3.0) burst | LogN(median 115, σ 0.55) | card 55%, wallet 30%, applepay 10%, cod 5% | 20% | Poisson(0.12·(age/365)^0.7) | 80.0% | card 45%, applepay 45%, wallet 45%, cod 5% | legit curve, ×2.0 at 00-05 | Gamma(2.5, 2.4) |
| F_cod_fake_order | 5.0% | 100% Exp(mean 5d) in [0,60] | Gamma-Poisson mean 1.8 | LogN(median 150, σ 0.5) | cod 100% | 30% | Poisson(0.0) | 10.0% | card 0%, applepay 0%, wallet 0%, cod 3% | legit curve, flattened^0.8 | Gamma(2.4, 2.3) |

**Shared rules for every profile:** basket ×0.8 at 06-10h and ×1.08 at 18-22h; +1.5% per km beyond 3.5 km; values clipped to the plausibility bounds in `config.PLAUSIBLE_BOUNDS`.  
**Label noise:** 3% of fraud recorded as legitimate; 0.1% of legitimate recorded as fraud.

### Stress-test scenarios (alternative assumptions, used only to score frozen models)

- **S0_in_distribution** — Fresh draw from the unchanged training generator (reference).
- **S1_population_drift** — Legitimate customers change; fraud does not.
  ```json
  {"legit": {"L_established": {"share": 0.825, "payment": {"card": 0.4, "applepay": 0.3, "wallet": 0.08, "cod": 0.22}, "mismatch": {"card": 0.22, "applepay": 0.2, "wallet": 0.1, "cod": 0.02}, "device_change": 0.06, "amount": {"median": 70, "sigma": 0.62}}, "L_new_customer": {"share": 0.14}}}
  ```
- **S2_adaptive_fraud** — Fraudsters adapt to look more legitimate.
  ```json
  {"fraud": {"F_stolen_card": {"payment": {"card": 0.4, "applepay": 0.55, "wallet": 0.05, "cod": 0.0}, "amount": {"median": 85, "sigma": 0.6}, "mismatch": {"card": 0.3, "applepay": 0.25, "wallet": 0.2, "cod": 0.0}, "hour": {"flatten": 1.0, "night_mult": 1.0}, "device_change": 0.15}, "F_account_takeover": {"device_change": 0.35, "hour": {"flatten": 1.0, "night_mult": 1.0}, "distance": {"shape": 2.3, "scale": 1.6}, "mismatch": {"card": 0.25, "applepay": 0.25, "wallet": 0.25, "cod": 0.05}}, "F_promo_abuse": {"age": {"mix": [{"w": 1.0, "kind": "exp", "mean": 10, "lo": 0, "hi": 45}]}, "amount": {"median": 58, "sigma": 0.5}, "discount": {"p": 0.85, "wallet_bonus": 0.0}}, "F_refund_abuse": {"refunds": {"kind": "abuser", "p_early": 0.45, "early_mean": 0.4, "base": 1, "extra_mean": 1.0}}}}
  ```
- **S3_unseen_typology** — A new fraud typology (wallet laundering) absent from training forms 30% of fraud.
  ```json
  {"fraud": {"F_promo_abuse": {"share": 0.22399999999999998}, "F_refund_abuse": {"share": 0.189}, "F_stolen_card": {"share": 0.161}, "F_account_takeover": {"share": 0.091}, "F_cod_fake_order": {"share": 0.034999999999999996}, "F_wallet_laundering": {"share": 0.3, "age": {"mix": [{"w": 1.0, "kind": "lognorm", "median": 200, "sigma": 0.7, "lo": 31, "hi": 3000}]}, "orders7": {"mean": 3.5, "shape": 2.0, "burst": 0.0}, "hour": {"flatten": 1.0, "night_mult": 1.0}, "payment": {"card": 0.1, "applepay": 0.0, "wallet": 0.9, "cod": 0.0}, "amount": {"median": 320, "sigma": 0.5}, "distance": {"shape": 2.3, "scale": 1.2}, "discount": {"p": 0.0, "wallet_bonus": 0.0}, "refunds": {"kind": "poisson", "mean": 0.1}, "device_change": 0.08, "mismatch": {"card": 0.1, "applepay": 0.0, "wallet": 0.05, "cod": 0.0}}}}
  ```
