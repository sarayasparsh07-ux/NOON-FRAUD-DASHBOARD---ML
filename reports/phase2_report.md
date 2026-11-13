# Phase 2 Report — My GAN vs Noon's GAN

**Sparsh Saraya · Noon ML internship challenge · Phase 2 (final)**

The authoritative specifications are `A_INTERN_BRIEF.md` and `A_INTERN_data_dictionary.md`. All numbers come from the executed notebook (`notebooks/fraud_detection_analysis.ipynb`, Phase 2 section) and the CSV files in `outputs/phase2/`. Primary results use **unique rows**; Noon's file as supplied, duplicates included, is reported as a sensitivity check. Phase 3 and Phase 4 have **not** been run.

## Data checked before any comparison

| Item | Finding |
|---|---|
| Six professor files | Present and byte-identical to the attachments (SHA-256 recorded). Kept read-only in `professor_supplied/` and never modified |
| `B_NOON_synthetic_fraud.csv` | 3,000 rows × 11 columns; schema matches the data dictionary; no missing values; all `is_fraud = 1` |
| Duplicates in Noon's file | Only **1,033 distinct rows**: 1,967 rows (65.6%) are repeats, and 833 distinct rows occur more than once (up to 9 copies). The counts fit sampling with replacement from about 1,107 base rows (zero-truncated Poisson, λ = 2.71, χ² p = 0.96). Kish effective sample size ≈ 813 |
| My CTGAN file (Phase 1) | 2,116 rows, no duplicates, no row shared with Noon's file. Distance rounded to Noon's 1 decimal place before comparison; this created no duplicates |
| Referee set | Integrity counts only: 20,000 rows, 65 fraud (0.325%). Not used for anything else |
| Phase 1 | Data, model artefacts and notebook cells verified unchanged (hash and cell digest) |

## 2.1 Distributions

| Feature (type) | My CTGAN vs Noon | What differs |
|---|---|---|
| `hour_of_day` (cyclic integer) | TVD 0.503, large | 60.1% of Noon's fraud is placed 00:00–05:59 (mine 11.9%) |
| `location_mismatch` | TVD 0.395, large | 62.9% vs 23.4% |
| `device_change` | TVD 0.338, large | 65.6% vs 31.9% |
| `account_age_days` (discrete ordered) | KS D 0.327, large | 31.0% of my rows are 0-day accounts; Noon has none. Median 147 vs 94 days |
| `num_orders_last_7d` (discrete ordered) | KS D 0.250, moderate | Median 4 vs 2; Noon has a spike at 7 orders (15.5% vs 1.8%) |
| `payment_method` | TVD 0.199, moderate | Noon: more wallet and COD, less Apple Pay |
| `delivery_distance_km` | KS D 0.185, moderate | Median 4.3 vs 3.0 km; 1.4% of Noon's rows exceed 15 km |
| `prev_refunds` (discrete ordered) | KS D 0.139, small | Noon's refunds range only 0–3 |
| `order_amount` (continuous) | KS D 0.122, small | Same mean (AED 134 vs 135), different shape |
| `discount_applied` | TVD 0.073, small | 50.3% vs 43.0% |

All differences are significant after Holm adjustment. Keeping Noon's duplicates changes no effect size by more than 0.009.

**Relationships between features.**
- In Noon's file, location mismatch is about 62–64% for **every** payment method, including cash on delivery (mine: 9%).
- Noon's device change rises with account age, to about 77% from 31 days.
- Noon's night ordering is high in every age band.
- The correlation between device change and location mismatch reverses sign: +0.33 in mine vs −0.16 in Noon's.

My **seed** fraud differs from Noon's as much as my CTGAN does, so the gap starts in my Phase 1 assumptions.

## 2.2 Fraud proportion

| Population | Numerator / denominator | Proportion |
|---|---|---|
| My seed dataset | 1,512 / 50,000 | 3.02% |
| My training partition | 1,058 / 35,000 | 3.02% |
| GAN generation volume relative to real training fraud (a volume ratio, **not** a fraud prevalence) | 2,116 GAN rows / 1,058 real training fraud rows | 200% |
| Augmented training set (selected 0.25×) | 1,322 / 35,264 | 3.75% |
| Noon's synthetic file (as supplied / distinct) | 3,000 / 3,000; 1,033 / 1,033 | 100% |
| Referee set (context only) | 65 / 20,000 | 0.325% |

Both synthetic files are fraud-only, so comparing their "fraud rates" (100% vs 100%) tells us nothing. Noon has not supplied how much synthetic fraud it added to its own training data, so its augmented composition is not estimated.

## 2.3 Lie detector

| Protocol | Accuracy | ROC-AUC | Accuracy baseline |
|---|---|---|---|
| Helper as specified (as supplied, 3 unshuffled folds) | 0.960 | — | 0.586 |
| **Primary: unique rows, 1,033 each, shuffled 5-fold, 10 repeats** | **0.920** | **0.974 ± 0.003** | 0.500 |
| Sensitivity: as supplied, grouped by row identity | 0.919 | 0.975 | 0.586 |
| Leakage illustration: as supplied, ungrouped | 0.961 | 0.996 | 0.586 |
| Reference: my seed vs Noon | 0.928 | 0.977 | 0.500 |
| Reference: my seed vs my CTGAN | 0.717 | 0.791 | 0.500 |

- **Primary permutation test** (1,000 permutations, no groups): p ≈ 0.001, which is the smallest value 1,000 permutations can give. The null distribution has mean 0.500 and maximum 0.550.
- **Features that give the source away:** account age (drop in AUC 0.096), location mismatch (0.068), hour (0.042) and device change (0.023).
- **Marginals-only control:** shuffling columns within each source lowers the AUC only to 0.949, so nearly all the separation comes from individual features.
- **The helper's 0.96 is inflated.**
  - 83.3% of Noon's test rows in its folds have an identical copy in training.
  - Its "0.50 = indistinguishable" message ignores the 0.586 baseline created by the unequal file sizes.
- **Grouped sensitivity run:** no p-value is reported, because permuting labels within row groups would be degenerate.

## 2.4 Reflection (full answers in the notebook)

- **Captured well.** My CTGAN reproduced its own training data closely (AUC 0.791 vs the seed). It also matches Noon on mean basket size, discount use falling with account age, and the share of refund-abuse patterns.
- **Supported.** Account takeover of established accounts with a device change (H1, second mode) appears in Noon's file: 22.7% of rows match my takeover signature vs 3.7% in my seed.
- **Questionable.**
  - My new-account mode (promo abuse 22.9% in my seed vs 1.2% in Noon's file).
  - Night ordering as a weak signal (H6).
  - Mismatch tied to payment method, and device change occurring together with mismatch (H3).
  - Refunds rising with order frequency.
  - My 15 km distance cap.
- **Revealed by Noon's file.** A concentrated night block, mismatch across all payment methods, busier accounts and almost no brand-new accounts. It also showed that deduplication must come before any two-sample test.
- **What I would improve.** Make the GAN treat account age as a discrete variable so it stops creating the 0-day spike, and run labelled sensitivity scenarios with wider priors. The Phase 1 assumptions stay unchanged, and Phase 3's real data decides which picture is right.

Noon's file is one generator's synthetic output, heavily duplicated and possibly rule-based. It is a benchmark, **not ground truth**.

## Limitations and outstanding items

See `phase2_limitations_and_issues.md`. In short:
- the effect-size bands are a judgement convention;
- the KS permutation p-values are floor-limited;
- the lie detector uses one model family;
- Noon's generation process is undocumented.

**Phase 3 still requires:**
- an adapter for `score_on_referee()`, since my pipeline expects raw columns and uses a 0.297 threshold;
- a check that Noon's model (pickled with scikit-learn 1.8.0, run here on 1.9.1) reproduces its benchmark.

**Phase 4 still requires** a wrapper for `app(2).py`, which accepts whole-number amounts only and calls `predict` without my threshold.
