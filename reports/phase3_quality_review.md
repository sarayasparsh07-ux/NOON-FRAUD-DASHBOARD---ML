# Phase 3 — Quality Review (Part 9 checklist)

Each item is marked **Passed**, **Failed** or **Not tested (reason)**.
- **Automated** means the check is re-derived in the notebook's QC cell. Results are in `outputs/phase3/audit/E_QC_checks.csv`: 21/21 passed in the merged run.
- **Manual** means the check was done by inspection after the final run.

## Part 9 checklist

| Check | Result | Evidence |
|---|---|---|
| All three intended models identified correctly | Passed (automated) | Selected = `A_baseline` (metadata = `final_arm` in `phase1_metrics.json`); secondary = `B_ctgan`; Noon = RandomForestClassifier |
| Selected model not confused with the GAN-augmented candidate | Passed (automated) | Different arm, threshold (0.296956 vs 0.330321) and file hash; named consistently throughout |
| `fraud_model_baseline.joblib` handled correctly | Passed (automated) | Byte-identical to `fraud_model.joblib` and produces identical scores, so it is not reported as a separate model |
| No model retrained or modified | Passed (automated) | All four model files are hash-identical before and after scoring and match their Phase 1 and professor records. The Phase 3 code contains no fit, calibration or threshold-search call |
| All models scored on the same 20,000 referee records | Passed (automated) | One `X_ref` frame; every prediction array has 20,000 rows |
| All 65 fraud cases included | Passed (automated) | TP + FN = 65 for every model |
| Original schema and payment-method encoding preserved | Passed (automated) | Referee columns match the dictionary; my models receive raw columns. Noon's helper encoding equals its stored columns, and every row has exactly one payment column set |
| Each model's frozen prediction rule applied | Passed (automated) | My models: `predict` equals calibrated probability ≥ stored threshold, and the selected threshold equals the Phase 1 record. Noon: native `predict`, which equals the argmax of `predict_proba`; fraud probability taken from the column for class 1 |
| No referee labels used for tuning, calibration or selection | Passed (automated + manual) | `X_ref` excludes `is_fraud`; labels are used only to count outcomes and for the descriptive section D. No threshold or model was chosen or changed |
| Noon benchmark reproduction checked | Passed (automated) | 7/7 published figures reproduced, including the exact confusion matrix |
| Version-compatibility effect on Noon's predictions | Passed (automated) | 202 warnings recorded. Scoring in an isolated scikit-learn 1.8.0 environment gave identical predictions (largest probability difference 1.1 × 10⁻¹⁶). The project environment was not altered and the model was never re-saved |
| Helper `score_on_referee` agrees with the independent recomputation | Passed (automated) | Recall and false positives identical |
| Confusion matrices reconcile with the dataset size | Passed (automated) | TP + FP + FN + TN = 20,000 for every model |
| Recall, precision and F1 independently verified | Passed (automated) | scikit-learn `recall_score`, `precision_score`, `f1_score` and `confusion_matrix` match |
| Charts have readable labels, legends and values | Passed (manual) | All 5 Phase 3 charts opened after the final run. One colour clash was fixed: "caught by both" on the overlap chart had used the same navy as "real fraud" |
| Written conclusions supported by the actual results | Passed (automated + manual) | The automated check compares 35 numbers quoted in the written cells with the run's results. It caught two rounding errors (CTGAN F1 written as 0.068 instead of 0.069; Noon recall CI lower bound written as 0.705 instead of 0.704), which were fixed before the final run. A manual cross-check of `phase3_report.md` caught one more (a precision CI lower bound written as 0.031 instead of 0.030), also fixed. **Note:** my interim progress message gave the CTGAN F1 as 0.068; the correct value is 0.069 |
| Phase 1 and Phase 2 outputs unchanged | Passed (automated) | Phase 1 cell digest and 8 data/model files match the Phase 1 manifest. The Phase 2 cell digest and 39 Phase 2 files match `reports/phase2_freeze_manifest.json`, recorded after the two requested Phase 2 report corrections and before Phase 3. The merge script re-checks both digests |
| Professor files unchanged | Passed (automated) | SHA-256 of all six files unchanged; no `__pycache__` in the professor folder |
| Reproducible in a fresh kernel | Passed | Three fresh-kernel runs (a test run and two merged runs) produced byte-identical outputs: predictions, metric tables, confidence intervals, paired tests, gap-analysis tables and charts. Predictions also reproduce from freshly reloaded model files |
| Output inventory (Part 8) | Passed (automated) | `outputs/phase3/phase3_output_inventory.csv` maps every output to its Part 8 deliverable: 41/41 present, no unlisted files. Added in a verification pass after the same Phase 3 prompt was received again; no result changed |

## Phase 2 corrections made before Phase 3

| Item | Result |
|---|---|
| Primary lie-detector ROC-AUC | Verified from the 10-seed CSV: mean 0.97447, so **0.974 ± 0.003**. All current report versions agree. An earlier draft that showed 0.975 had already been corrected before delivery; no experiment was re-run |
| "2,116 / 1,058 = 200%" row | Renamed in the report, the Word version and the Phase 2 source to state that it is a GAN generation-volume ratio, not a fraud prevalence. The executed Phase 2 notebook and CSV keep the original label, because Phase 2 was not re-run; this is recorded in `phase2_quality_review.md` |
| `src/compare.py` change | `phase1_cells_digest` now ignores **any** phase-tagged cell instead of only Phase 2 cells, so adding Phase 3 cells cannot alter the Phase 1 check. On the current notebook it gives the identical digest (verified). Made before the Phase 2 freeze manifest was recorded |

## Not tested

| Check | Reason |
|---|---|
| Noon Model A on the referee set | Model A was not supplied; its figures are quoted from `B_NOON_benchmark.md` only |
| Reproduction on another machine or operating system | Only one environment was available |
