# Quality-Assurance Checklist — Phase 1 independent audit

*The Phase 1 entries below are unchanged since submission on 13 Oct. Entries marked "Not tested — not supplied / gated / Phase 4 not started" describe that date. Those items were completed later and are reviewed in `phase2_quality_review.md`, `phase3_quality_review.md` and `phase4_quality_review.md` (the fraud-check app: 44/44 tests).*

Each check is recorded as **Passed**, **Failed** or **Not tested (reason)**. "Automated" = re-derived in notebook §1.15 from saved files or predictions (`outputs/metrics/audit_checks.csv`, 22/22 passed). Nothing is marked Passed without having been performed.

## 5.1 Source-requirement audit

| Check | Result | Evidence |
|---|---|---|
| Both source documents read in full; four-phase structure and Phase-1 gate followed | Passed | `experiment_plan.md`; notebook stops at the Phase 1 gate |
| Brief Phase 1 steps 1.1–1.6 all delivered (assumptions, seed, first classifier + confusion matrix, real GAN, retrain + before/after recall/precision/F1, submission package) | Passed | Notebook §1.1–1.14; `final_report.md` |
| Master-prompt inconsistency handled ("nine predictors" vs ten in the data dictionary) | Passed | All ten predictors used; flagged in notebook Phase 0 |
| Optional `eval_helper.py` and `app.py` shell | Not tested — not supplied | Metrics computed in `src/evaluation.py`, cross-checked against scikit-learn |
| Phase 2 / 3 / 4 requirements | Not tested — gated until Noon supplies materials | — |

## 5.2 Data-integrity audit

| Check | Result | Evidence |
|---|---|---|
| Seed CSV: exact 11 columns in dictionary order, valid dtypes, categories, binaries, hour 0–23, non-negativity, plausibility bounds, no missing values | Passed (automated) | `validate_schema`, §1.2 and §1.15 |
| Seed CSV regenerates bit-for-bit from code + seed | Passed (automated) | §1.15 |
| Duplicate rows in seed | Passed — 0 | §1.2 |
| No deterministic shortcut (best single feature AUC 0.760 < 0.85; purest categorical cell 52%; 4-leaf tree AP 0.227) | Passed | §1.2 |
| `synthetic_fraud_gan.csv`: 11 columns, `is_fraud` = 1, all constraints, equals rows sampled from the fitted generator | Passed (automated) | §1.15 |
| All synthetic pools (CTGAN, SMOTE-NC, duplication, independent marginals) pass schema checks | Passed (automated) | §1.7 assertions |
| Every transformation of generated rows documented and counted (floor clip, rounding, rejection) | Passed | `ctgan_sampling_log.csv`, `constraint_policy_comparison.csv` |
| Origin of every dataset documented | Passed | README table |

## 5.3 GAN audit

| Check | Result | Evidence |
|---|---|---|
| Genuine CTGAN (ctgan 0.12.1), not rows written by hand/LLM, not SMOTE | Passed | §1.6 code, generator architecture printed |
| Trained only on training-partition fraud (index check + content-hash check vs validation/test) | Passed (automated) | §1.6, §1.15 |
| Training evidence: per-epoch losses saved and plotted; fit time; epoch-sensitivity fits | Passed | `ctgan_loss_values.csv`, `01_ctgan_loss_curves.png`, `07_epoch_sensitivity.png` |
| Saved generator reloads and samples | Passed (automated) | §1.15 |
| Seeded refit reproduces identical losses and identical synthetic rows | Passed | Separate verification run (loss identical: True; rows identical: True) |
| Fidelity measured (SDMetrics quality/diagnostic, LogisticDetection, two-sample tests, KS/TVD, correlation error, typology signatures) with baselines (real-holdout reference, no-structure ablation) | Passed | §1.7 |
| Memorisation check (0 exact copies; distance-to-closest-record) | Passed | `06_distance_to_closest_record.png` |
| Fidelity claims limited to the seed, not Noon's data | Passed | Wording in §1.7, report |

## 5.4 Leakage audit

| Check | Result | Evidence |
|---|---|---|
| Split made before any GAN/classifier fit; partitions disjoint and complete | Passed (automated) | §1.4, §1.15 |
| Preprocessing fitted inside pipelines on training data only | Passed | `src/modeling.py` (ColumnTransformer inside Pipeline) |
| Hyper-parameters tuned by CV on the training partition only | Passed | `tune_family` receives `X_tr, y_tr` only |
| Augmentation ratio, threshold, model family, final arm and calibration chosen on validation only | Passed | §1.5, §1.8, §1.9, §1.11 (test arrays never passed to selection functions) |
| Stress-test data generated after freezing and never used for decisions | Passed | §1.10 |
| SMOTE-NC / duplication / independent marginals built from training data only | Passed | `make_pools(X_tr, y_tr, …)` |
| Validation fraud used only as a real-vs-real *measuring stick* in fidelity diagnostics, never for training or selection | Passed (documented) | §1.7 helper `real_holdout_reference` |

## 5.5 Metric audit

| Check | Result | Evidence |
|---|---|---|
| Positive class = fraud; confusion-matrix orientation [[TN, FP], [FN, TP]] | Passed | `compute_metrics`, chart labels |
| Confusion matrices, recall and F1 recomputed with scikit-learn from predictions match reported values for every arm | Passed (automated) | §1.15 |
| Fast bootstrap AP equals scikit-learn AP | Passed (automated) | §1.15 |
| Thresholds reported with every operating-point metric; same test rows for all arms | Passed | `arms_test_metrics.csv` |
| Saved final model reproduces test predictions and uses the frozen threshold | Passed (automated) | §1.15 |
| Calibration changes no flagged decision | Passed (automated) | §1.11, §1.15 |
| Numbers in `final_report.md` / Word report match notebook outputs | Passed | Cross-checked against `arms_test_metrics.csv`, `paired_bootstrap_vs_baseline.csv`, `multi_seed_sign_consistency.csv` after the final run |
| Accuracy never used as a decision metric | Passed | Selection rules use AP and F1 |

## 5.6 Statistical and robustness audit

| Check | Result | Evidence |
|---|---|---|
| Improvement claims tested against the pre-declared criterion (paired CI excludes 0 AND ≥ 4/5 seeds) | Passed — criterion not met; no improvement claimed | `ctgan_improvement_claim_check.csv` |
| Paired, class-stratified bootstrap (2,000 resamples); exact McNemar on fraud and legit rows; Holm correction | Passed | §1.12 |
| Whole-pipeline repeat over 5 seeds | Passed | `multi_seed_test_metrics.csv` |
| Prevalence sensitivity (1%, 3%, 5%) | Passed | `prevalence_scenarios_test_metrics.csv` |
| Holdout, stress-test and (future) real-data results kept separate | Passed | Separate sections and folders |
| Calibration assessed with Brier, Brier skill, ECE and reliability diagram (quantile bins) | Passed | `calibration_test.csv`, `08_reliability_diagram.png` |
| Test-set size caveat (227 frauds; ±≈0.065 on recall) stated | Passed | Bootstrap intervals in §1.12 |

## 5.7 Visualisation audit

| Check | Result | Evidence |
|---|---|---|
| Every chart opened and inspected after rendering | Passed | All 36 PNGs viewed |
| Issues found and fixed: overlapping log-axis tick labels; comb-like gaps from non-integer bins; colliding x-labels on signature chart; overlapping cost-scenario annotations and a meaningless negative symlog region; precision plotted as 0 when undefined; legends overlapping bars; overlapping point labels on the prevalence and epoch charts; misleading coloured legend swatches on the two-sample-test chart; mixed "higher-is-better"/"0.5-is-best" metrics on one axis | Passed (fixed and re-rendered in the final run) | — |
| Titles, axis names, units, legends for multi-series charts, one y-axis per panel | Passed | `src/plotting.py` |
| Palette validated for colour-vision deficiency; bars carry direct value labels (contrast relief) | Passed | Validator: normal-vision floor pass; CVD 8.0 at boundary → direct labels added |
| Counts, percentages and densities labelled as what they are | Passed | y-axis labels |

## 5.8 Application audit

| Check | Result | Evidence |
|---|---|---|
| Fraud-check app (normal, boundary, invalid inputs; model-loading errors) | Not tested — Phase 4 not started (gated; `app.py` shell not supplied) | Saved model and README snippet are ready for it |

## 5.9 Reproducibility and release audit

| Check | Result | Evidence |
|---|---|---|
| Notebook executed end-to-end in a fresh kernel from the documented command | Passed | `build_and_run.py` (two complete runs) |
| Second full run reproduced the first run exactly (all classifier metrics, selections, thresholds, GAN losses and rows, seeded diagnostics) | Passed | `phase1_metrics.json` compared: 0 differences |
| SDMetrics `LogisticDetection` reproducible | Failed (minor, documented) — library uses an unseeded internal 3-fold split; varies by ≤ 0.02 between runs. Diagnostic only; no decision depends on it | Notebook §1.7 note |
| All deliverables exist and open | Passed | §1.14 inventory; Word report rendered and inspected |
| Dependencies pinned | Passed | `requirements.txt` |
| Reproduction on a different machine/OS | Not tested — single environment available | — |
