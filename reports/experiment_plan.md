# Phase 0 — Project Preparation and Experiment Design

*Recorded on 13 Oct 2026, before any seed data was generated or any model was trained. Decision rules below are pre-declared so that results cannot shape them.*

## 0.1 File inventory

| File | Status | Use |
|---|---|---|
| `A_INTERN_BRIEF.md` | Supplied, read in full | Assignment, four phases, deliverables, 40-mark rubric |
| `A_INTERN_data_dictionary.md` | Supplied, read in full | The only information about Noon's data: 11 column names, types, meanings |
| Master prompt (`attachment.txt`) | Supplied, read in full | Working standard for this project |
| `eval_helper.py` (optional) | **Not supplied** | All metrics are computed in `src/evaluation.py` with scikit-learn; if Noon later supplies the helper, Phase 2/3 metrics will be reconciled against it |
| `app.py` shell (Phase 4) | **Not supplied** | Phase 4 is out of scope until Phase 3 is complete; the tool will be built then |
| `B_NOON_synthetic_fraud.csv`, `B_NOON_model.joblib` | **Not supplied (by design)** | Phase 2 / 3 only |
| `C_REFEREE_real_test.csv` | **Not supplied (by design)** | Phase 3 only |

## 0.2 Environment

Python 3.13.16 (Linux x86-64, 2 CPU cores, no GPU). Library versions are pinned in `requirements.txt` and printed at the top of the notebook. Notes:

- PyTorch's CPU-only index was unreachable from the workspace, so the standard PyPI `torch` build (2.14.1) is used; it runs on CPU (`enable_gpu=False`).
- Installing `sdmetrics==0.32.0` / `rdt<2` resolved pandas to 2.3.3 (from a pre-installed 3.0.5). All code is tested on 2.3.3.
- `ctgan` 0.12.1 is licensed under the Business Source License 1.1 (source-available; academic/internal use permitted), not MIT. SDMetrics remains MIT.

## 0.3 Reproducibility

| Purpose | Seed |
|---|---|
| Seed-data generation, split, CTGAN, classifiers (primary run) | 42 |
| Multi-seed repeat (whole pipeline re-run) | 42, 7, 101, 2024, 31337 |
| Stress-test data generation | 9001, 9002, 9003, 9004 |
| Bootstrap resampling | 12345 |

Reusable logic lives in `src/` (`config.py`, `seed_generator.py`, `gan.py`, `modeling.py`, `evaluation.py`, `plotting.py`); the notebook only orchestrates and presents.

## 0.4 Experimental questions

**Primary:** Does adding CTGAN-generated fraud to the training data improve fraud detection on an untouched holdout, relative to the original classifier and non-GAN alternatives, without an unacceptable rise in false positives?

**Secondary:** (i) Did the GAN learn joint structure, or only marginals? (ii) Does any gain survive repeated seeds and other prevalence levels? (iii) How does the frozen model behave when the data-generating assumptions shift?

## 0.5 Pre-declared design

**Data.** 50,000 synthetic transactions at an illustrative 3% observed fraud rate (research anchor: Sift's 3.1% food-delivery block rate; IEEE-CIS 3.5%). Sensitivity runs at 1% (100,000 rows) and 5% (50,000 rows). Stratified 70 / 15 / 15 train / validation / test split, created **before** any GAN is fitted.

**Arms (same classifier configuration, same holdout):**

| Arm | Training data | Purpose |
|---|---|---|
| A Baseline | Original training partition | Reference |
| B CTGAN | + CTGAN-generated fraud | The assignment's method |
| C Class-weighted | Original, `class_weight="balanced"` | Non-GAN imbalance control |
| D SMOTE-NC | + SMOTE-NC fraud | Conventional resampling control |
| E Random oversampling | + duplicated fraud | Shows what "more copies" alone does |
| F Independent marginals | + fraud sampled column-by-column from training fraud | Ablation: a generator with correct marginals but **no** learned joint structure |

**Augmentation ratios** for B, D, E, F: {0.25, 0.5, 1.0, 2.0} × the number of training fraud rows. Each arm's ratio is chosen on validation average precision (ties within 0.002 → smaller ratio).

**Threshold rule (every arm):** the threshold that maximises F1 on the validation set, frozen before the test set is scored. Results at the default 0.5 are also shown, to expose the threshold shift that resampling causes.

**Classifier family:** Logistic Regression, Random Forest, Extra Trees and HistGradientBoosting are tuned by 3-fold CV on the training partition only (scoring = average precision); the family with the best validation average precision becomes the shared configuration for all arms.

**Final-model rule:** the arm with the highest validation average precision is selected; if its margin over the Baseline is below 0.01, the simpler arm is preferred (simplicity order A < C < E < D < F < B). The CTGAN-augmented model is saved separately in every case so it can be scored on Noon's real data in Phase 3.

**Primary metrics (test, fraud = positive class):** recall, precision, F1, average precision (PR-AUC), confusion matrix, specificity, balanced accuracy; ROC-AUC and accuracy as supporting metrics only.

**Uncertainty:** paired, class-stratified bootstrap (2,000 resamples) of the difference from Baseline; exact McNemar tests on fraud and legitimate rows separately; Holm correction across the five comparisons; whole-pipeline repeats over 5 seeds. An improvement is claimed only if its paired 95% interval excludes zero **and** its sign holds in at least 4 of 5 seeds.

**Calibration:** reliability diagram (quantile bins) and Brier score on test; sigmoid calibration fitted on the validation set only.

**Stress test:** three alternative-assumption datasets (population drift, adaptive fraud, unseen fraud typology) plus a fresh in-distribution draw, used only to score frozen models.

**Leakage controls (implemented as assertions in the notebook):** GAN trained only on training-partition fraud; no row overlap between partitions; preprocessing fitted inside pipelines on training data; ratios, thresholds, model family and calibration chosen without the test set; stress-test data never used for any decision.

## Phase 0 review gate

| Check | Result |
|---|---|
| Both source documents read completely | Passed |
| Schema recorded exactly (11 columns) | Passed — `src/config.py` `SCHEMA` |
| Assignment requirements and rubric mapped | Passed — see `rubric_traceability.md` |
| Missing files and dependencies identified | Passed — table 0.1 and notes 0.2 |
| Experimental questions and validation rules documented | Passed — this file |
