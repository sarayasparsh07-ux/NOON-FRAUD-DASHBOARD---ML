# Phase 2 — Quality-Control Audit (Part 5 checklist)

Each item is marked **Passed**, **Failed** or **Not tested (reason)**.
- **Automated** means the check is re-derived in the notebook's QC cell. Results are in `outputs/phase2/audit/QC_checks.csv` (17/17 passed in the merged run).
- **Manual** means the check was done by inspection after the final run.

## Part 5 checklist

| Part 5 check | Result | How it was verified |
|---|---|---|
| The Phase 1 fraud-only sample has been correctly identified | Passed (automated) | `data/synthetic_fraud_gan.csv`: 2,116 rows, all `is_fraud = 1`. It is the CTGAN output, not the seed |
| Noon's fraud-only sample has been correctly interpreted | Passed (automated) | 3,000 rows, all `is_fraud = 1`. Never treated as a full population or as a fraud rate |
| The required schema has been preserved | Passed (automated) | Identical 11 columns, names and order, in my file, Noon's file and the referee file. No schema changes |
| The target variable has not leaked into the lie-detector predictors | Passed (automated) | `is_fraud` is not among the 13 predictor columns produced by the helper's `_prep` |
| Duplicate observations have been investigated | Passed (automated + manual) | Recomputed from the file: 1,967 repeats; multiplicity table; sampling-with-replacement fit; copy positions; overlap with my rows (0); effect on effect sizes (≤ 0.009) and on the lie detector (leakage illustration) |
| The class-balance comparison uses correct denominators | Passed (automated) | Every proportion is stored with its numerator and denominator and recomputed |
| The helper functions have been used correctly | Passed (automated + manual) | Helper hash unchanged. `compare_distributions` and `lie_detector` run as supplied. Two of their assumptions were found not to hold and are reported: unshuffled folds leak duplicates (83.3% of Noon test rows), and "0.50 = indistinguishable" ignores the 0.586 baseline |
| The plots are readable and accurately labelled | Passed (manual) | All 9 Phase 2 charts opened after the final run. Fixed: overlapping x-labels and a coloured legend swatch (lie-detector chart); value labels on error bars (importance); doubled legend text and labels hitting diamonds (effect sizes); labels hitting expected-value markers (multiplicity); wrong legend label for the dashed line (brief features) |
| The interpretation is supported by actual results | Passed (automated + manual) | The automated check compares 29 numbers quoted in the written cells with this run's results. It caught one rounding error (primary AUC written as 0.975 instead of 0.974), which was corrected and re-run. The other numbers in the notes and `phase2_report.md` were cross-checked against the CSVs |
| No unsupported Noon statistics have been invented | Passed (manual) | Every Noon figure is computed from `B_NOON_synthetic_fraud.csv`. Noon's training composition is stated as unknown, not estimated. The benchmark's figures are quoted only as Phase 3 reference values |
| The original Phase 1 deliverables remain intact | Passed (automated) | Data and model artefacts are hash-identical; Phase 1 notebook cells match their digest before and after the merge; every changed Phase 1 file has a documented reason |

## Additional checks from the user's corrections

| Check | Result | Evidence |
|---|---|---|
| Primary permutation test: both datasets deduplicated, cross-source overlaps removed, no row-hash groups, shuffled stratified CV, ROC-AUC primary with balanced accuracy supplementary | Passed | `C.permutation_test`; 1,000 permutations; p = 1/1,001 |
| As-supplied sensitivity run: duplicates kept, `StratifiedGroupKFold` on row hash, **no** p-value | Passed | `2.3_sensitivity_as_supplied_grouped.csv`; table and chart show no p-value |
| `account_age_days` treated as discrete ordered | Passed | Integer bands and ECDF; KS with a tie-exact permutation p-value; listed as "discrete ordered (integer)" |
| Counts and baselines recomputed, not hard-coded | Passed (automated) | Baselines come from `len()`; the QC check recomputes them |
| Referee file used for integrity counts only | Passed (automated) | Read once, variable deleted, single reference in the notebook source |
| Professor files not modified | Passed (automated) | SHA-256 unchanged; folder read-only; no `__pycache__` written |
| Phase 1 not rebuilt; Phase 3/4 not executed | Passed | Only `run_phase2.py` was run; no model was scored on the referee set |

## Reproducibility

| Check | Result | Evidence |
|---|---|---|
| Phase 2 executes end to end in a fresh kernel | Passed | 3 complete runs (a test run plus 2 merged runs), 0 errors, about 15 min each |
| Results identical across runs | Passed | All lie-detector, permutation and effect-size tables were identical between runs (seeded) |
| Reproduction on another machine | Not tested — only one environment available | — |

## Post-delivery corrections (31 Oct, before Phase 3)

| Item | Action | Effect on results |
|---|---|---|
| Primary lie-detector ROC-AUC in the report versions | Re-verified from `2.3_primary_unique_balanced.csv` (10 seeds): mean 0.97447, SD 0.00274, i.e. **0.974 ± 0.003**. The current `phase2_report.md`, `Phase2_Report.docx`, the zipped copy and the notebook all show 0.974 ± 0.003. An earlier draft that showed 0.975 was corrected before delivery | None; no experiment re-run |
| Fraud-proportion row "2,116 / 1,058 = 200%" | Renamed in `phase2_report.md`, the Word report and the Phase 2 source to "GAN generation volume relative to real training fraud (a volume ratio, not a fraud prevalence)". The executed notebook and `2.2_fraud_proportion_denominators.csv` keep the original label ("My GAN-generated fraud file vs training FRAUD rows"; denominator "GAN rows / training fraud rows") because Phase 2 was deliberately not re-run for a label change | None; numbers unchanged |
