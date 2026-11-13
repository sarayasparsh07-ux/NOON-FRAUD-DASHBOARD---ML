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

{{PARAMS}}

### Stress-test scenarios (alternative assumptions, used only to score frozen models)

{{STRESS}}
