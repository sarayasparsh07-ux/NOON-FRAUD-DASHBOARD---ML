# Noon Food Delivery Fraud Detection — Final Project Report

**Sparsh Saraya · Noon ML internship challenge · Phases 1–4 complete (31 Oct 2026)**

## 1. Initial fraud assumptions — and how right they were

**What I assumed.** The project was built without access to any Noon transaction data, starting from the prescribed 11-column schema and documented public-context anchors (for example, a UAE average basket of about AED 77); no confidential Noon data informed Phase 1. I modelled UAE food-delivery fraud as five typologies, each with an innocent look-alike: promo abuse (brand-new account, discount, small basket), refund abuse (several prior refunds), stolen card (card, location mismatch, large basket), account takeover (old account, new device) and cash-on-delivery fake orders. The seed had 50,000 orders labelled at a 3% fraud-and-abuse rate. That rate is an **illustrative modelling assumption**, informed partly by a published food-delivery block-rate benchmark of about 3%; it is not a measured confirmed-fraud prevalence or a Noon base rate (the real referee set has 0.325%). Every magnitude is logged as an assumption in `assumptions_register.md`.

**What the real data showed.** The 65 real referee frauds were mostly the pattern I had under-weighted, and the pattern I emphasised barely appeared:

| Share of fraud rows | Real referee fraud | My seed fraud |
|---|---|---|
| Placed 00:00–05:59 | 47.7% | 14.2% |
| Device change | 70.8% | 20.1% |
| Location mismatch | 60.0% | 24.6% |
| Account ≤ 7 days old | 6.2% | 45.7% |

I was right that account takeover exists, wrong to over-weight brand-new-account promo abuse, and also wrong about legitimate customers: real ones order at night, claim refunds, change devices and order often far more than I assumed.

## 2. The GAN and its impact

**Training.** CTGAN, chosen because it learns the joint distribution of mixed numeric and categorical columns, was trained for 600 epochs on the 1,058 training-partition fraud rows only. Generator loss went from about +0.9 (mean of the first 50 epochs) to about −2.0, and a seeded refit reproduced it exactly. It generated 2,116 fraud rows, none copied from the seed data. Its main flaw: 31.0% of its rows are 0-day accounts, against 13.8% in its training data.

**Did augmentation help? No.** Same model, threshold frozen on validation, untouched synthetic test set with 227 frauds:

| Phase 1 synthetic test | Recall | Precision | F1 | AP | False positives |
|---|---|---|---|---|---|
| A · Baseline (**selected**) | 51.1% | 54.5% | 0.527 | 0.516 | 97 |
| B · + CTGAN fraud (0.25×) | 48.9% | 57.8% | 0.530 | 0.510 | 81 |

CTGAN − Baseline: AP −0.007 (95% CI −0.020 to +0.012) and recall −0.022 (−0.048 to +0.004). The pre-declared improvement criterion was not met, so the Baseline was selected. The GAN traded recall for fewer false alarms; it did not catch more fraud.

## 3. Comparison with Noon

**Phase 2 — synthetic versus synthetic.**
- Noon's file has 3,000 rows but only 1,033 distinct ones.
- On the distinct rows, a classifier separated my CTGAN fraud from Noon's with ROC-AUC 0.974 (accuracy 0.920 on balanced samples), so the two generators describe very different fraud.
- My **seed** fraud was just as separable from Noon's (0.977), while my CTGAN was much closer to my seed (0.791). The gap therefore lay in my assumptions, not in my GAN.
- Noon's synthetic file is a benchmark, not ground truth.

**Phase 3 — real data (20,000 referee transactions, 65 frauds; every model frozen):**

| Model | Frauds caught | Recall | Precision | F1 | False positives |
|---|---|---|---|---|---|
| My selected classifier | 44 / 65 | 67.69% | 4.07% | 0.077 | 1,036 |
| My CTGAN-augmented classifier | 39 / 65 | 60.00% | 3.63% | 0.069 | 1,034 |
| Noon Model B | 53 / 65 | 81.54% | 42.74% | 0.561 | 71 |

- **Benchmark check:** Noon's published benchmark reproduced exactly.
- **Recall:** Noon caught 9 more frauds (+13.8 percentage points). With 65 frauds this gap is uncertain (paired 95% CI 0.0 to 27.7 points).
- **False alarms:** Noon flagged 965 fewer legitimate orders, a decisive difference.
- **My GAN on real data:** it caught 5 fewer frauds than my selected classifier.

**Why Noon won.**
- The mean of the ten feature-level discrepancy statistics (KS D or total-variation distance, unweighted) was 0.099 for real fraud versus Noon's unique synthetic fraud, compared with a simulated sampling reference of 0.086 and 0.297 for real fraud versus my seed fraud. This is descriptive evidence of similarity, not a formal multivariate equivalence test.
- **Different training data.** According to `B_NOON_benchmark.md`, Noon's GAN was trained on Noon's real fraud examples, and Noon Model B combined real transaction data with that GAN-generated fraud. My classifiers were trained only on synthetic seed data (plus CTGAN rows for the augmented one) and met real referee transactions only after being frozen, for evaluation.
- **False alarms.** My false-alarm rate rose from 1.33% on the synthetic test set to 5.20% on the real referee set. This is consistent with overlap between real legitimate transactions and the fraud signals encoded in my synthetic assumptions: night orders, prior refunds and device changes are all more common among real legitimate orders than among my seed's legitimate orders (24.5% vs 8.9%, 26.5% vs 14.3% and 8.0% vs 3.3%).
- **Causality cannot be isolated.** Training data, model architecture (random forest vs gradient boosting), calibration and thresholds all differ between the models, and separating their effects would need retraining, which was not allowed.

## 4. Why not just ask an LLM to "make more fraud rows"?

**Prompt-only generation.** An LLM asked to "make more fraud rows" produces plausible-looking rows from its general knowledge of what fraud tends to look like. In this prompt-only setup, the generated records are not fitted to the project's specific transaction distribution. Consequently, there is no guarantee that their frequencies, ranges or relationships between features accurately represent the target population.

**Learning and validating a distribution.** A GAN is fitted to a supplied dataset: its generator is trained against a discriminator until generated rows are hard to tell apart from the training rows. Because it is fitted, it can be validated like any model (loss curves, fidelity and two-sample tests, memorisation checks), reproduced from a seed, and trained inside an organisation on confidential examples that could never be placed in a prompt.

**Only as representative as its training data.** A GAN learns the distribution of the data it is given, nothing more. Mine was fitted to assumption-driven seed fraud and reproduced it reasonably closely (a classifier separated the two with ROC-AUC 0.791, against 0.974 for my CTGAN vs Noon's fraud), so it inherited the seed's blind spots. Noon's GAN was trained on real fraud examples.

## 5. Conclusion

The four phases show that careful method, honest evaluation and a properly trained GAN cannot replace real examples. The evidence points to the data each generator learned from, rather than the GAN technique, as the main source of the gap to Noon, although model and threshold differences cannot be ruled out. Only the real-data test exposed assumptions that synthetic metrics could not.

The Streamlit app (`streamlit run app.py`) wraps the selected classifier, `models/fraud_model.joblib`. It is an **academic screening prototype**: a FRAUD verdict flags a transaction for manual review and must never drive automatic blocking or other enforcement. On real data, 96% of its flags were legitimate orders.

*Supporting documents: `phase1_report.md`, `phase2_report.md`, `phase3_report.md`, `phase4_implementation_notes.md` and one quality review per phase (listed in `README.md`).*
