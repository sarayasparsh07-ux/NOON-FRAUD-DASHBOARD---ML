# Phase 2 — Limitations and Outstanding Issues

## Limitations of the evidence

| # | Limitation | Effect on the conclusions | How it was handled |
|---|---|---|---|
| L1 | Noon's file is **synthetic and fraud-only**. It is one generator's output, not real fraud, and it contains no legitimate rows | It shows how *Noon's* generator portrays fraud. It cannot show which picture is realistic, or Noon's real fraud rate | It is treated as a benchmark, not ground truth. Realism is left to Phase 3 |
| L2 | **Heavy duplication.** 3,000 rows, 1,033 distinct (65.6% repeats, up to 9 copies). The copy counts fit sampling with replacement from about 1,107 base rows. Kish effective size ≈ 813 | Inflates naive tests. The helper's lie detector leaks 83.3% of Noon test rows | Primary analysis uses unique rows. The as-supplied file is a sensitivity check, with grouped folds and no p-value |
| L3 | **Undocumented generation.** How Noon's generator was built and trained, and how much synthetic fraud Noon added to its own training, are unknown | The "rule-like" reading (night block, constant mismatch, refunds capped at 3) is a hypothesis. Noon's augmented training composition cannot be computed | Stated as interpretation; nothing is estimated or invented |
| L4 | **Effect-size bands** (0.05 / 0.15 / 0.30) are a judgement convention | Labels such as "moderate" are guides, not thresholds | Raw effect sizes are always reported alongside |
| L5 | **KS permutation p-values hit their floor** (2,000 permutations → minimum Holm-adjusted 0.0025). The lie-detector permutation p = 1/1,001 | The reported p-values are upper bounds, not exact values | Stated as bounds |
| L6 | **One lie-detector model family** (the helper's random forest, 100 trees) | Another model could separate the files differently | The helper's model is kept for comparability. Interpretation rests on ROC-AUC plus effect sizes, not on accuracy alone |
| L7 | **Precision harmonisation.** My `delivery_distance_km` was rounded to 1 decimal place to match Noon's file | Negligible; it removes a trivial precision signal | Checked: no duplicates created, and the original files are unchanged |
| L8 | **Typology signatures** are my own Phase 1 definitions | They are a lens on Noon's data, not Noon's labels | Labelled "my lens, not Noon's labels" |
| L9 | **Correction of the Methods Brief.** It suggests a grouped permutation test, treating account age as continuous, and about 22,000 referee rows | Applied as the user instructed: ungrouped permutation on unique rows; account age treated as discrete ordered; 20,000 referee rows counted from the file | Listed in notebook section A1 |
| L10 | **Noon's model was pickled with scikit-learn 1.8.0**; this environment runs 1.9.1 (version warning) | Could change predictions in Phase 3 | Phase 2 only inspects the model's structure. Reproducing the benchmark is a Phase 3 check |

## Outstanding items (not part of Phase 2; not executed)

| # | Item | Phase |
|---|---|---|
| O1 | Score my frozen selected model, the GAN-augmented arm and Noon's model on the **same** 20,000 referee rows. No retraining, threshold change, calibration or selection on referee data | 3 |
| O2 | `score_on_referee()` one-hot encodes the data and calls `clf.predict` at a 0.5 threshold. My pipeline needs raw columns and its frozen 0.297 threshold, so a thin adapter is needed, with metrics cross-checked against `sklearn.metrics` | 3 |
| O3 | Check that Noon's model reproduces its benchmark (recall ≈ 0.82, precision ≈ 0.43, F1 ≈ 0.56, FP = 71); investigate any gap, without tuning towards it | 3 |
| O4 | Explain the recall gap using the Phase 2 evidence (assumption differences: new-account vs takeover emphasis, night ordering, mismatch structure) | 3 |
| O5 | `app(2).py` accepts whole-number order amounts only, caps account age at 2,000 days, calls `model.predict` (ignoring my threshold) and expects a `{'model', 'columns'}` dictionary. A wrapper is needed; the original file stays unmodified | 4 |
| O6 | Labelled sensitivity scenarios with wider seed priors and a discrete-aware treatment of account age in CTGAN. These are proposals only: Phase 1 stays frozen | Future work |
