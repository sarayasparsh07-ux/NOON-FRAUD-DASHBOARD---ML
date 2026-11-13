# Phase 3 Report — The Referee: Real Data Judges Every Model

**Sparsh Saraya · Noon ML internship challenge · Phase 3 (final)**

`A_INTERN_BRIEF.md` and `A_INTERN_data_dictionary.md` are the controlling specifications. All numbers come from the executed notebook (Phase 3 section of `notebooks/fraud_detection_analysis.ipynb`) and the files in `outputs/phase3/`.

Every model was scored on the **same 20,000 real referee transactions** using its **own frozen prediction rule**. Nothing was retrained, recalibrated, re-thresholded or selected using the referee data. Phase 4 was not part of this evaluation (it was completed afterwards; see the final note).

## Inputs checked before scoring

| Item | Finding |
|---|---|
| Referee set | 20,000 rows × 11 columns in data-dictionary order: 19,935 legitimate, 65 fraud (0.325%). No missing values, no duplicate rows, valid categories, binaries and ranges |
| My selected classifier | `fraud_model.joblib`, arm **A (baseline: seed data only)**. Matches `final_arm` in the Phase 1 metrics. Stored threshold 0.296956 on calibrated probability (Platt). Class 1 = fraud |
| My CTGAN-augmented classifier | `fraud_model_gan_augmented.joblib`, arm B. Its own threshold is 0.330321. It is **not** the selected model |
| `fraud_model_baseline.joblib` | Byte-identical to `fraud_model.joblib` and gives identical scores, so it is not reported as a separate candidate |
| Noon Model B | A dictionary holding `model`, `num`, `cat` and `columns`. The model is a RandomForest: 200 trees, balanced class weights, random_state 0, classes [0, 1], 13 encoded columns. The helper's encoding matches its stored columns exactly |
| Version warning | 202 `InconsistentVersionWarning` messages: Noon pickled with scikit-learn 1.8.0; this project runs 1.9.1. I re-scored the model in an **isolated 1.8.0 environment**: all 20,000 predictions were identical (largest probability difference 1.1 × 10⁻¹⁶). The warning does not affect the results |
| Leakage screen | 0 exact referee rows appear in my seed data, my CTGAN fraud or Noon's synthetic fraud |

## Steps 3.1 and 3.2 — Results on the real referee set

| Model | Recall | Precision | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| **My selected classifier** | 0.677 | 0.041 | 0.077 | 44 | 1,036 | 21 | 18,899 |
| My CTGAN-augmented classifier | 0.600 | 0.036 | 0.069 | 39 | 1,034 | 26 | 18,901 |
| **Noon Model B** | 0.815 | 0.427 | 0.561 | 53 | 71 | 12 | 19,864 |

| Supporting metric | My selected | My CTGAN-augmented | Noon Model B |
|---|---|---|---|
| Recall, 95% Wilson CI | 0.556–0.778 | 0.479–0.710 | 0.704–0.891 |
| Precision, 95% Wilson CI | 0.030–0.054 | 0.027–0.049 | 0.344–0.515 |
| False-positive rate | 5.20% | 5.19% | 0.36% |
| Transactions flagged | 1,080 | 1,073 | 124 |
| Average precision (bootstrap 95% CI) | 0.064 (0.036–0.126) | 0.058 (0.037–0.102) | 0.617 (0.511–0.725) |
| ROC-AUC (bootstrap 95% CI) | 0.929 (0.904–0.951) | 0.931 (0.908–0.952) | 0.997 (0.995–0.998) |

- All metrics were recomputed with scikit-learn's own functions and match.
- Every confusion matrix sums to 20,000 and includes all 65 frauds.
- Accuracy (0.947, 0.947, 0.996) is supplementary only, because always predicting "legit" would already score 0.997.
- Thresholds are model-specific and not comparable. Mine apply to calibrated probabilities built for a 3%-fraud synthetic world; Noon's model uses its native majority vote.

**Benchmark reproduction.** Noon Model B reproduces every published figure: recall 0.82, precision 0.43, F1 0.56, 71 false positives, and the confusion matrix 19,864 / 71 / 12 / 53. The professor's `score_on_referee()` agrees with my independent recomputation.

## Step 3.3 — The final comparison

- **Who detected more fraud?** Noon Model B: 53 of 65, against 44 of 65 for my selected classifier.
- **Who flagged fewer legitimate transactions?** Noon Model B: 71, against 1,036 for mine.
- **Difference in fraud recall:** +0.138 for Noon (+13.8 percentage points, 9 more frauds).
- **Difference in false positives:** 965 fewer for Noon.
- **How certain is this?** The false-positive gap is decisive: 997 legitimate orders were flagged only by mine and 32 only by Noon's (exact McNemar p < 0.001). The recall gap is real in this sample but imprecise with 65 frauds: 15 frauds were caught only by Noon and 6 only by mine (p = 0.078). The paired bootstrap 95% CI for the difference is 0.000 to 0.277.
- **My CTGAN-augmented classifier:** 5 fewer frauds caught than my selected model (39 vs 44), with essentially the same false positives (1,034 vs 1,036). All 5 discordant frauds go against it. Exact McNemar p = 0.0625, the smallest value possible with 5 pairs; bootstrap CI for the recall difference −0.138 to −0.015. **The GAN did not improve real-data performance.**

## Step 3.4 — Explaining the gap

**Real fraud looks like Noon's synthetic fraud, not mine.** These are descriptive comparisons, made after scoring. Nothing was changed as a result.

| Share of fraud rows | Real referee fraud | My seed fraud | My CTGAN fraud | Noon synthetic fraud |
|---|---|---|---|---|
| Ordered 00:00–05:59 | 47.7% | 14.2% | 11.9% | 60.1% |
| Location mismatch | 60.0% | 24.6% | 23.4% | 62.9% |
| Device change | 70.8% | 20.1% | 31.9% | 65.6% |
| Account ≤ 7 days old | 6.2% | 45.7% | 35.6% | 3.0% |

The unweighted mean of the ten feature-level discrepancy statistics (KS D or total-variation distance) is **0.099** for real fraud versus Noon's unique synthetic fraud. A simulated sampling reference (65-row draws from Noon's file compared with the file itself) gives **0.086**. For my seed fraud the mean is **0.297**, and for my CTGAN fraud **0.271**. This is descriptive evidence of similarity, not a formal multivariate equivalence test.

So the Phase 2 finding holds on real data. Noon's generator captured night-time, new-device, mismatched-location fraud on established accounts. My Phase 1 assumptions emphasised brand-new-account promo abuse, and only 6.2% of real frauds involve accounts 7 days old or younger.

**Why my classifier behaves as it does.**
- **Missed frauds:** it misses fraud my assumptions under-weighted. 71.4% of the 21 frauds it missed were night orders, against 36.4% of those it caught; in Phase 1, night ordering was deliberately a weak signal (H6).
- **False alarms:** its larger weakness is on the legitimate side. Real legitimate customers behave differently from my synthetic ones:

  | Share of legitimate rows | Real | My seed |
  |---|---|---|
  | Ordered at night | 24.5% | 8.9% |
  | At least one refund | 26.5% | 14.3% |
  | Device change | 8.0% | 3.3% |
  | ≥ 4 orders in 7 days | 35.6% | 10.6% |

  Those are the behaviours my model treats as risky. 66.9% of its false positives have a previous refund, 43.1% a device change, and 54.1% at least 4 recent orders.
- **False-positive rate:** it rose from 1.33% on my synthetic test set to 5.20% on real data. A lower fraud rate alone would have cut precision to about 11.1%. The actual 4.1% shows that real legitimate behaviour is the larger cause.

**Why Noon has the edge, and what cannot be separated.**
- According to `B_NOON_benchmark.md`, Noon's GAN was trained on real fraud, and Model B on real transactions plus that GAN fraud. My models never saw a real row.
- Model family (random forest vs gradient boosting), decision rule and training composition also differ. These effects cannot be separated without retraining, which Phase 3 forbids.
- Noon's own Model A (real data only) reportedly reached recall of only 0.14. This suggests that real data **plus** realistic synthetic fraud is what worked. Model A was not supplied, so I could not re-score it.
- My model still ranks real fraud well above chance (ROC-AUC 0.929), but at a 0.325% fraud rate its average precision is only 0.064, against Noon's 0.617.

## Limitations

- **Small fraud count:** only 65 real frauds, so recall intervals are wide (mine 0.556–0.778; Noon's 0.704–0.891).
- **One referee sample:** it may not represent Noon's full or future population, or its true fraud rate.
- **Noon's training data is undocumented**, including its overlap in time with the referee set. Noon may have an in-distribution advantage.
- **The gap analysis is descriptive** and based on 65 fraud rows; some differences are noise.
- **The referee results judge these models on this sample only.** Noon's synthetic file remains a benchmark, not ground truth.

When this report was written, Phase 4 had not started. It has since been completed: see `app.py`, `reports/phase4_implementation_notes.md` and `reports/phase4_quality_review.md`.
