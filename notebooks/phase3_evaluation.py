# %% [markdown]
# ---
# # Phase 3 — The Referee: real data judges every model
#
# **Scope.** Phase 3 only, as set out in `A_INTERN_BRIEF.md`:
# * **Step 3.1:** score my classifier on the real referee set.
# * **Step 3.2:** score Noon's classifier on the same rows, and check it against Noon's published benchmark.
# * **Step 3.3:** the final bar chart.
# * **Step 3.4:** explain the gap.
#
# Phase 4 (the app) is **not** started.
#
# **Frozen.** Nothing below trains, retrains, tunes, recalibrates, re-thresholds, selects or replaces a model. Each model is applied with its own stored prediction rule. The referee labels are used only to count outcomes **after** predictions exist.
#
# **Models evaluated (same 20,000 referee rows).**
#
# | Name used throughout | File | Role |
# |---|---|---|
# | **My selected classifier** | `models/fraud_model.joblib` | **Primary.** The Phase 1 pre-declared selection (arm A, baseline: seed data only). Compared head-to-head with Noon |
# | **My CTGAN-augmented classifier** | `models/fraud_model_gan_augmented.joblib` | **Secondary.** The same model family trained with my CTGAN fraud added. Shows whether my GAN's contribution carries over to real data. It is **not** my selected model |
# | **Noon Model B** | `professor_supplied/B_NOON_model.joblib` | Noon's benchmark classifier (real data + Noon's GAN fraud) |
#
# `models/fraud_model_baseline.joblib` is checked in A3. If it is the same fitted artefact as the selected model, it is not reported as a separate candidate.
#
# Labels used: **[SOURCE FACT]**, **[FINDING]** (computed here), **[INTERPRETATION]**. These cells run in a fresh kernel after the Phase 1 and Phase 2 cells above, both verified unchanged; execution counts restart at 1.

# %%
import os, sys, json, time, platform, warnings, importlib, io, contextlib, hashlib
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True                      # never write __pycache__ into the read-only professor folder

import numpy as np
import pandas as pd
import joblib
import sklearn
from IPython.display import display, Markdown

from src.config import *
from src import plotting as P, compare as C, phase3 as R
from src.modeling import FraudModel, PlattCalibrator      # makes the saved FraudModel objects importable

PROF = ROOT / "professor_supplied"
sys.path.insert(0, str(PROF))
import A_eval_helper as H                            # professor's optional helper, imported unmodified
sys.path.remove(str(PROF))

P.style()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

P3 = OUTPUTS / "phase3"
for sub in ["audit", "predictions", "metrics", "charts", "gap_analysis"]:
    (P3 / sub).mkdir(parents=True, exist_ok=True)
B_BOOT = 2000                                         # bootstrap resamples (finite-sample uncertainty only)
T0 = time.time()
NAME_SEL, NAME_GAN, NAME_NOON = "My selected classifier", "My CTGAN-augmented classifier", "Noon Model B"
COLORS3 = {NAME_SEL: ARM_COLORS["A_baseline"], NAME_GAN: ARM_COLORS["B_ctgan"], NAME_NOON: NOON_COLOR}
warnings.filterwarnings("ignore", category=FutureWarning)
print(f"Run {time.strftime('%Y-%m-%d %H:%M')} · Python {platform.python_version()}")
display(pd.DataFrame([(p, importlib.import_module(p).__version__) for p in ["numpy", "pandas", "sklearn", "scipy", "joblib"]],
                     columns=["package", "version"]).set_index("package").T)

# %% [markdown]
# ## A. Inputs, integrity and frozen artefacts
#
# ### A1 Inventory of the seven Phase 3 inputs, and freeze checks for Phase 1 and Phase 2

# %%
prof_recorded = json.loads((ROOT / "reports" / "professor_files_manifest.json").read_text())
prof_recorded = prof_recorded.get("files", prof_recorded)
p1_manifest = json.loads((ROOT / "reports" / "phase1_submission_manifest.json").read_text())
p1_files = p1_manifest.get("files", p1_manifest)
p2_freeze = json.loads((ROOT / "reports" / "phase2_freeze_manifest.json").read_text())

inputs = [
    ("C_REFEREE_real_test.csv", PROF / "C_REFEREE_real_test.csv", "Real referee test set: the only evaluation data", prof_recorded["C_REFEREE_real_test.csv"]),
    ("B_NOON_model.joblib", PROF / "B_NOON_model.joblib", "Noon Model B (benchmark classifier)", prof_recorded["B_NOON_model.joblib"]),
    ("B_NOON_benchmark.md", PROF / "B_NOON_benchmark.md", "Noon's published results: verification targets only", prof_recorded["B_NOON_benchmark.md"]),
    ("models/fraud_model.joblib", ROOT / "models" / "fraud_model.joblib", "My selected Phase 1 classifier (primary)", p1_files["models/fraud_model.joblib"]),
    ("models/fraud_model_gan_augmented.joblib", ROOT / "models" / "fraud_model_gan_augmented.joblib", "My CTGAN-augmented classifier (secondary)", p1_files["models/fraud_model_gan_augmented.joblib"]),
    ("models/fraud_model_baseline.joblib", ROOT / "models" / "fraud_model_baseline.joblib", "Baseline copy: identity check against the selected model", p1_files["models/fraud_model_baseline.joblib"]),
    ("A_eval_helper.py", PROF / "A_eval_helper.py", "Professor's helper: _prep encoding and score_on_referee", prof_recorded["A_eval_helper.py"]),
]
inv = pd.DataFrame([{"input": n, "purpose": pur, "present": p.exists(), "KB": round(p.stat().st_size / 1024, 1),
                     "sha256 (first 16)": C.sha256(p)[:16], "matches recorded hash": C.sha256(p) == rec} for n, p, pur, rec in inputs])
display(inv)
inv.to_csv(P3 / "audit" / "A1_input_inventory.csv", index=False)
MODEL_HASHES_BEFORE = {n: C.sha256(p) for n, p, _, _ in inputs if n.endswith(".joblib")}

nb_path = ROOT / "notebooks" / "fraud_detection_analysis.ipynb"
freeze = pd.DataFrame([
    {"item": "Phase 1 notebook cells (code, text, outputs)", "unchanged": R.cells_digest(nb_path, None) == p1_manifest["phase1_notebook_cells_sha256"]},
    {"item": "Phase 1 data and model artefacts (8 files)", "unchanged": bool(R.verify_files({k: v for k, v in p1_files.items() if k.startswith(("data/", "models/"))}, ROOT)["unchanged"].all())},
    {"item": "Phase 2 notebook cells (code, text, outputs)", "unchanged": R.cells_digest(nb_path, 2) == p2_freeze["phase2_notebook_cells_sha256"]},
    {"item": f"Phase 2 outputs, data and reports ({len(p2_freeze['files'])} files)", "unchanged": bool(R.verify_files(p2_freeze["files"], ROOT)["unchanged"].all())},
    {"item": "Professor files (6)", "unchanged": bool(R.verify_files({f"professor_supplied/{k}": v for k, v in prof_recorded.items()}, ROOT)["unchanged"].all())},
])
display(freeze)
freeze.to_csv(P3 / "audit" / "A1_freeze_checks_before_scoring.csv", index=False)

# %% [markdown]
# ### A2 Referee dataset integrity
#
# The file is read once and never modified. The features (`X_ref`) and the labels (`y_ref`) are separated straight away: every model sees `X_ref` only, and `y_ref` is used to count outcomes after the predictions are fixed.

# %%
referee = pd.read_csv(PROF / "C_REFEREE_real_test.csv")
integrity = R.referee_integrity(referee)
counts = R.label_counts(referee)
display(integrity)
display(pd.Series(counts).to_frame("value"))
integrity.to_csv(P3 / "audit" / "A2_referee_integrity_checks.csv", index=False)
pd.Series(counts).to_csv(P3 / "audit" / "A2_referee_label_counts.csv")
X_ref = referee[FEATURES].copy()                      # 10 predictors, original raw form
y_ref = referee[TARGET].to_numpy().astype(int)        # used only after predictions exist
N_REF, N_FRAUD = len(referee), int(y_ref.sum())
print(f"[FINDING] {N_REF:,} rows · {N_REF - N_FRAUD:,} legitimate · {N_FRAUD} fraud · prevalence {N_FRAUD / N_REF:.3%}")

# %%
seed = pd.read_csv(DATA / "seed_transactions.csv")
split = np.load(DATA / "split_indices.npz")
train = seed.iloc[split["train"]]
noon_syn = pd.read_csv(PROF / "B_NOON_synthetic_fraud.csv")
rng_tab = R.range_vs_training(referee, train)
display(rng_tab)
rng_tab.to_csv(P3 / "audit" / "A2_referee_vs_my_training_ranges.csv", index=False)
overlap = pd.DataFrame([
    {"comparison": "Referee rows vs my seed data (all 50,000 rows)", "distinct feature rows in both": R.exact_overlap(referee, seed)},
    {"comparison": "Referee rows vs my CTGAN fraud", "distinct feature rows in both": R.exact_overlap(referee, pd.read_csv(DATA / "synthetic_fraud_gan.csv"))},
    {"comparison": "Referee rows vs Noon's synthetic fraud", "distinct feature rows in both": R.exact_overlap(referee, noon_syn)},
    {"comparison": "Referee FRAUD rows vs Noon's synthetic fraud", "distinct feature rows in both": R.exact_overlap(referee[referee[TARGET] == 1], noon_syn)},
])
display(overlap)
overlap.to_csv(P3 / "audit" / "A2_exact_row_overlap.csv", index=False)

# %% [markdown]
# Exact-row overlap is a leakage screen. If Noon's synthetic fraud contained exact copies of referee fraud rows, Noon's benchmark would be partly a memory test. The range table is descriptive: it shows where the real data goes beyond what my models ever saw in training. Neither result is used to change anything.
#
# ### A3 My saved models: structure, selection and frozen thresholds

# %%
mods = {n: joblib.load(ROOT / "models" / f) for n, f in
        [(NAME_SEL, "fraud_model.joblib"), (NAME_GAN, "fraud_model_gan_augmented.joblib"), ("baseline file", "fraud_model_baseline.joblib")]}
p1 = json.loads((OUTPUTS / "metrics" / "phase1_metrics.json").read_text())
meta = pd.DataFrame([{
    "model": n, "class": type(m).__name__, "arm (metadata)": m.metadata.get("arm"), "family": m.metadata.get("family"),
    "trained on": m.metadata.get("trained_on"), "augmentation ratio": m.metadata.get("augmentation_ratio"),
    "frozen threshold (calibrated probability)": m.threshold, "raw-score threshold": m.metadata.get("raw_threshold"),
    "threshold rule": m.metadata.get("threshold_rule"), "calibration": m.metadata.get("calibration"),
    "Platt a, b": (round(m.calibrator.a_, 6), round(m.calibrator.b_, 6)), "classes_": list(m.pipeline.classes_),
    "pipeline steps": [s for s, _ in m.pipeline.steps], "feature order = data dictionary": list(m.feature_order) == FEATURES,
    "sha256 (first 16)": C.sha256(ROOT / "models" / ("fraud_model.joblib" if n == NAME_SEL else "fraud_model_gan_augmented.joblib" if n == NAME_GAN else "fraud_model_baseline.joblib"))[:16],
} for n, m in mods.items()])
display(meta.T)
meta.to_csv(P3 / "audit" / "A3_my_model_metadata.csv", index=False)

sel, gan, base = mods[NAME_SEL], mods[NAME_GAN], mods["baseline file"]
baseline_identical = (C.sha256(ROOT / "models" / "fraud_model.joblib") == C.sha256(ROOT / "models" / "fraud_model_baseline.joblib"))
selection = pd.Series({
    "Phase 1 final_arm (phase1_metrics.json)": p1["final_arm"],
    "selected model metadata arm": sel.metadata["arm"],
    "selected threshold equals Phase 1 final_threshold_calibrated": bool(sel.threshold == p1["final_threshold_calibrated"]),
    "CTGAN model metadata arm": gan.metadata["arm"],
    "CTGAN model has its own threshold (differs from selected)": bool(gan.threshold != sel.threshold),
    "fraud_model_baseline.joblib byte-identical to fraud_model.joblib": baseline_identical,
    "fraud class label = 1 and is the 2nd column of predict_proba in both pipelines": all(list(m.pipeline.classes_) == [0, 1] for m in (sel, gan)),
})
display(selection.to_frame("value"))
selection.to_csv(P3 / "audit" / "A3_selection_verification.csv")
print(f"[FINDING] Selected = {sel.metadata['arm']} (threshold {sel.threshold:.6f}); CTGAN = {gan.metadata['arm']} (threshold {gan.threshold:.6f}). "
      f"Baseline file {'IS the same artefact as the selected model, so it is not reported separately' if baseline_identical else 'differs from the selected model'}.")

# %% [markdown]
# **How the custom `FraudModel` scores a transaction.**
# 1. `predict_proba(X)` takes the 10 **raw** columns in the stored feature order. Its fitted pipeline does its own one-hot encoding of `payment_method` (no manual encoding beforehand).
# 2. It returns the raw fraud score (`pipeline.predict_proba[:, 1]`, class 1 = fraud), passed through the stored Platt calibrator.
# 3. `predict(X)` flags fraud when that calibrated probability is ≥ the model's own stored threshold.
#
# Thresholds are **model-specific**. Mine apply to calibrated probabilities of a 3%-fraud synthetic world. Noon's random forest uses its native rule (`predict`: the majority vote of 200 trees, i.e. fraud when the averaged tree probability exceeds 0.5). The numbers are therefore not comparable with each other, and none of them was changed.

# %%
probe = X_ref.head(500)
interface = pd.DataFrame([{
    "model": n, "predict == (predict_proba ≥ stored threshold)": bool(np.array_equal(m.predict(probe), (m.predict_proba(probe) >= m.threshold).astype(int))),
    "predict_proba returns one fraud probability per row": m.predict_proba(probe).shape == (len(probe),),
    "calibrated = Platt(raw pipeline score for class 1)": bool(np.allclose(m.predict_proba(probe), m.calibrator.transform(m.pipeline.predict_proba(probe[m.feature_order])[:, 1]))),
    "referee payment methods all known to the fitted encoder": set(X_ref.payment_method.unique()) <= set(m.pipeline.named_steps["pre"].named_transformers_["pay"].categories_[0]),
} for n, m in ((NAME_SEL, sel), (NAME_GAN, gan))])
display(interface)
interface.to_csv(P3 / "audit" / "A3_fraudmodel_interface_checks.csv", index=False)

# %% [markdown]
# ### A4 Noon Model B: metadata, encoding and version compatibility

# %%
with warnings.catch_warnings(record=True) as w_load:
    warnings.simplefilter("always")
    noon = joblib.load(PROF / "B_NOON_model.joblib")
vw = [x for x in w_load if "InconsistentVersion" in type(x.message).__name__]
rf = noon["model"]
X_noon = H._prep(X_ref, columns=noon["columns"])
noon_meta = pd.Series({
    "object type": type(noon).__name__, "keys": list(noon), "estimator": type(rf).__name__,
    "n_estimators": rf.n_estimators, "class_weight": rf.class_weight, "random_state": rf.random_state, "max_depth": rf.max_depth,
    "classes_": list(rf.classes_), "fraud class (label 1) column in predict_proba": list(rf.classes_).index(1),
    "n_features_in_": rf.n_features_in_, "stored num": noon["num"], "stored cat": noon["cat"],
    "stored columns == model.feature_names_in_": list(noon["columns"]) == list(rf.feature_names_in_),
    "helper encoding == stored columns (names + order)": list(X_noon.columns) == list(noon["columns"]),
    "every referee row has exactly one payment_method column = 1": bool((X_noon.filter(like="payment_method_").sum(axis=1) == 1).all()),
    "no missing values after encoding": int(X_noon.isna().sum().sum()) == 0,
    "pickled with scikit-learn": str(vw[0].message).split("from version ")[1].split(" ")[0] if vw else "no warning",
    "project environment scikit-learn": sklearn.__version__,
    "InconsistentVersionWarning count on load": len(vw),
})
display(noon_meta.to_frame("value"))
noon_meta.astype(str).to_csv(P3 / "audit" / "A4_noon_model_metadata.csv")
print(f"[FINDING] Loading Noon's model raised {len(vw)} InconsistentVersionWarning messages (1 forest + {len(vw) - 1} trees): "
      f"pickled with {noon_meta['pickled with scikit-learn']}, loaded with {sklearn.__version__}. Whether this changes predictions is tested in B2.")

# %% [markdown]
# ## B. Scoring the same 20,000 rows (Steps 3.1 and 3.2)
#
# ### B1 Predictions with each model's own frozen rule

# %%
assert TARGET not in X_ref.columns and len(X_ref) == N_REF
p_sel, pred_sel = R.score_fraudmodel(sel, X_ref)
p_gan, pred_gan = R.score_fraudmodel(gan, X_ref)
p_base, pred_base = R.score_fraudmodel(base, X_ref)
p_noon, pred_noon, _ = R.score_noon(noon, X_ref, H._prep)
same_as_selected = bool(np.array_equal(p_base, p_sel) and np.array_equal(pred_base, pred_sel))
print(f"Baseline file reproduces the selected model's scores exactly: {same_as_selected}")
preds = pd.DataFrame({"referee_row": np.arange(N_REF), "is_fraud": y_ref,
                      "selected_proba": p_sel, "selected_pred": pred_sel, "ctgan_proba": p_gan, "ctgan_pred": pred_gan,
                      "noon_proba": p_noon, "noon_pred": pred_noon})
preds.to_csv(P3 / "predictions" / "B1_referee_predictions_all_models.csv", index=False, float_format="%.10g")
display(preds.describe().T[["mean", "min", "max"]])

# %% [markdown]
# ### B2 Noon's model: the helper's `score_on_referee`, an independent recomputation, and a cross-version check
#
# * **Helper.** `score_on_referee(rf, noon["columns"], referee)` is the professor's standard evaluation. It encodes with `_prep`, aligns to the stored columns, and calls `predict`.
# * **Cross-version.** `notebooks/phase3_noon_crossversion.py` was run once in a separate, isolated environment with **scikit-learn 1.8.0**, the version Noon pickled with. That script reads the feature columns only and never re-saves the model. The project environment (1.9.1) was not altered.

# %%
buf = io.StringIO()
with contextlib.redirect_stdout(buf), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    helper_res = H.score_on_referee(rf, noon["columns"], referee, label="Noon Model B")
print(buf.getvalue())
helper_vs_mine = pd.Series({
    "helper recall == recomputed": bool(np.isclose(helper_res["recall"], pred_noon[y_ref == 1].mean())),
    "helper false positives == recomputed": helper_res["false_positives"] == int(((pred_noon == 1) & (y_ref == 0)).sum()),
    "predict == argmax of predict_proba (native rule)": bool(np.array_equal(pred_noon, rf.classes_[np.argmax(rf.predict_proba(X_noon), axis=1)])),
})
display(helper_vs_mine.to_frame("value"))

xv_path = P3 / "audit" / "noon_predictions_sklearn_1.8.0.csv"
if xv_path.exists():
    xv = pd.read_csv(xv_path)
    xv_meta = (Path(str(xv_path) + ".meta.txt")).read_text()
    crossver = pd.Series({
        "isolated environment": xv_meta.strip().replace("\n", "; "),
        "rows compared": len(xv),
        "predictions identical (1.8.0 vs 1.9.1)": bool(np.array_equal(xv["noon_pred"].to_numpy(), pred_noon)),
        "max |probability difference|": float(np.max(np.abs(xv["noon_proba_fraud"].to_numpy() - p_noon))),
        "flagged in 1.8.0 / in 1.9.1": f"{int(xv['noon_pred'].sum())} / {int(pred_noon.sum())}",
    })
else:
    crossver = pd.Series({"cross-version check": "NOT TESTED: isolated-environment output not found"})
display(crossver.to_frame("value"))
crossver.astype(str).to_csv(P3 / "audit" / "B2_noon_cross_version_check.csv")
helper_vs_mine.to_csv(P3 / "audit" / "B2_helper_vs_recomputed.csv")

# %% [markdown]
# ## C. Results on the real referee set
#
# ### C1 Metrics, recomputed independently with scikit-learn

# %%
rows = [R.metrics_row(y_ref, p_sel, pred_sel, NAME_SEL, f"calibrated P(fraud) ≥ {sel.threshold:.4f} (stored)"),
        R.metrics_row(y_ref, p_gan, pred_gan, NAME_GAN, f"calibrated P(fraud) ≥ {gan.threshold:.4f} (stored)"),
        R.metrics_row(y_ref, p_noon, pred_noon, NAME_NOON, "RandomForest.predict (native rule, unchanged)")]
res = pd.DataFrame(rows).set_index("model")
boot = pd.DataFrame({n: R.bootstrap_ranking_ci(y_ref, s, B_BOOT, SEED) for n, s in ((NAME_SEL, p_sel), (NAME_GAN, p_gan), (NAME_NOON, p_noon))}).T
res = res.join(boot)
check = pd.DataFrame([R.sklearn_crosscheck(y_ref, pr, r) for pr, r in zip((pred_sel, pred_gan, pred_noon), rows)]).set_index("model")
headline = res[["recall", "precision", "F1", "TP", "FP", "FN", "TN"]]
display(headline.style.format({"recall": "{:.3f}", "precision": "{:.3f}", "F1": "{:.3f}"}).set_caption("Phase 3 comparison table (actual computed results)"))
display(res.drop(columns=["prediction rule"]).T)
display(check)
res.to_csv(P3 / "metrics" / "C1_referee_metrics_all_models.csv")
headline.to_csv(P3 / "metrics" / "C1_comparison_table.csv")
check.to_csv(P3 / "metrics" / "C1_sklearn_crosscheck.csv")
cms = {n: [[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]] for n, r in res.iterrows()}
_ = (P3 / "metrics" / "C1_confusion_matrices.json").write_text(json.dumps(cms, indent=1))

# %% [markdown]
# ### C2 Noon benchmark reproduction check
#
# The published figures are **verification targets**. Nothing was adjusted to hit them.

# %%
bm = {"Fraud recall": (0.82, res.loc[NAME_NOON, "recall"], 2), "Precision": (0.43, res.loc[NAME_NOON, "precision"], 2),
      "F1": (0.56, res.loc[NAME_NOON, "F1"], 2), "False positives": (71, res.loc[NAME_NOON, "FP"], 0),
      "TN": (19864, res.loc[NAME_NOON, "TN"], 0), "FN": (12, res.loc[NAME_NOON, "FN"], 0), "TP": (53, res.loc[NAME_NOON, "TP"], 0)}
repro = pd.DataFrame([{"metric": k, "published (B_NOON_benchmark.md)": v[0], "reproduced here": v[1],
                       "reproduced, rounded as published": round(float(v[1]), v[2]) if v[2] else int(v[1]),
                       "match": (round(float(v[1]), v[2]) == v[0]) if v[2] else int(v[1]) == v[0]} for k, v in bm.items()])
display(repro)
repro.to_csv(P3 / "metrics" / "C2_noon_benchmark_reproduction.csv", index=False)
print(f"[FINDING] Benchmark reproduced on every published figure: {bool(repro['match'].all())}")

# %% [markdown]
# ### C3 Step 3.3 — the final bar chart

# %%
P.p3_headline(res, COLORS3, P3 / "charts" / "3.3_final_bar_chart_recall_and_false_positives.png",
              "Step 3.3 — On the real referee set: who caught more fraud, and who flagged fewer innocent orders?")
P.p3_confusions(res, COLORS3, P3 / "charts" / "3.1_3.2_confusion_matrices.png",
                f"Confusion matrices on the same {N_REF:,} referee transactions ({N_FRAUD} real frauds)")
points = {n: (res.loc[n, "recall"], res.loc[n, "precision"], res.loc[n, "false-positive rate"]) for n in res.index}
P.p3_pr_roc(y_ref, {NAME_SEL: p_sel, NAME_GAN: p_gan, NAME_NOON: p_noon}, points, COLORS3,
            P3 / "charts" / "3.3_precision_recall_and_roc.png", "Ranking quality on the referee set (descriptive)")

d_rec = res.loc[NAME_NOON, "recall"] - res.loc[NAME_SEL, "recall"]
d_fp = int(res.loc[NAME_NOON, "FP"] - res.loc[NAME_SEL, "FP"])
answers = pd.Series({
    "Who detected more fraudulent transactions?": f"{res['TP'].idxmax()} ({int(res['TP'].max())} of {N_FRAUD})",
    "Who flagged fewer legitimate transactions?": f"{res['FP'].idxmin()} ({int(res['FP'].min()):,} false positives)",
    "Difference in fraud recall (Noon − my selected)": f"{d_rec:+.3f} ({d_rec * 100:+.1f} percentage points; {int(res.loc[NAME_NOON, 'TP'] - res.loc[NAME_SEL, 'TP']):+d} frauds)",
    "Difference in false positives (Noon − my selected)": f"{d_fp:+,}",
    "CTGAN-augmented vs my selected: recall": f"{res.loc[NAME_GAN, 'recall'] - res.loc[NAME_SEL, 'recall']:+.3f} ({int(res.loc[NAME_GAN, 'TP'] - res.loc[NAME_SEL, 'TP']):+d} frauds)",
    "CTGAN-augmented vs my selected: false positives": f"{int(res.loc[NAME_GAN, 'FP'] - res.loc[NAME_SEL, 'FP']):+,}",
})
display(answers.to_frame("answer (computed)"))
answers.to_csv(P3 / "metrics" / "C3_step_3_3_answers.csv")

# %% [markdown]
# ### C4 Uncertainty and paired comparisons (supplementary)
#
# * Only 65 frauds exist, so one detection moves recall by 1.5 percentage points.
# * Recall and precision intervals are **95% Wilson** intervals. AP and ROC-AUC intervals come from a class-stratified bootstrap (2,000 resamples).
# * Paired comparisons use the same rows for both models:
#   * an exact McNemar test on discordant decisions, run separately for fraud rows and legitimate rows;
#   * a paired, class-stratified bootstrap of the differences.
# * These intervals describe finite-sample uncertainty **within this referee set** only. They do not cover differences between this set and Noon's full transaction population, and they do not replace the required metrics.

# %%
ci = res[["recall", "recall 95% CI low", "recall 95% CI high", "precision", "precision 95% CI low", "precision 95% CI high",
          "average precision (PR-AUC)", "AP 95% CI low", "AP 95% CI high", "ROC-AUC", "ROC-AUC 95% CI low", "ROC-AUC 95% CI high"]]
display(ci.T)
ci.to_csv(P3 / "metrics" / "C4_confidence_intervals.csv")
paired = []
for a_name, a_pred, b_name, b_pred in ((NAME_SEL, pred_sel, NAME_NOON, pred_noon), (NAME_SEL, pred_sel, NAME_GAN, pred_gan)):
    mc = R.mcnemar(y_ref, a_pred, b_pred); mc.insert(0, "A vs B", f"A = {a_name} · B = {b_name}")
    pb = R.paired_bootstrap(y_ref, a_pred, b_pred, B_BOOT, SEED); pb.insert(0, "A vs B", f"A = {a_name} · B = {b_name}")
    display(mc); display(pb)
    paired.append((mc, pb))
pd.concat([m for m, _ in paired]).to_csv(P3 / "metrics" / "C4_mcnemar_exact.csv", index=False)
pd.concat([b for _, b in paired]).to_csv(P3 / "metrics" / "C4_paired_bootstrap_differences.csv", index=False)

# %% [markdown]
# ## D. Step 3.4 — Explaining the gap (descriptive evidence only)
#
# Everything in this section is **post-hoc and descriptive**. The referee labels are used to describe what kind of fraud each model caught or missed. No finding is fed back into any model, threshold or choice.
#
# ### D1 Generalisation: Phase 1 synthetic test vs the real referee set

# %%
tm = p1["test_metrics"]
gen = pd.DataFrame([{
    "model": n, "dataset": ds, "fraud prevalence": prev, "recall": rec, "precision": pre, "false-positive rate": fpr,
    "ROC-AUC": auc, "average precision": ap} for n, arm in ((NAME_SEL, "A_baseline"), (NAME_GAN, "B_ctgan"))
    for ds, prev, rec, pre, fpr, auc, ap in (
        ("Phase 1 synthetic test (7,500 rows)", tm[arm]["n_fraud"] / tm[arm]["n"], tm[arm]["recall"], tm[arm]["precision"], tm[arm]["fpr"], tm[arm]["roc_auc"], tm[arm]["pr_auc_ap"]),
        (f"Real referee set ({N_REF:,} rows)", N_FRAUD / N_REF, res.loc[n, "recall"], res.loc[n, "precision"], res.loc[n, "false-positive rate"], res.loc[n, "ROC-AUC"], res.loc[n, "average precision (PR-AUC)"]))])
# Precision expected on the referee set if the synthetic-test recall and FPR had held (prevalence effect only)
for n, arm in ((NAME_SEL, "A_baseline"), (NAME_GAN, "B_ctgan")):
    r_, f_, pi = tm[arm]["recall"], tm[arm]["fpr"], N_FRAUD / N_REF
    gen.loc[(gen.model == n) & gen.dataset.str.startswith("Real"), "precision expected from prevalence alone"] = r_ * pi / (r_ * pi + f_ * (1 - pi))
display(gen)
gen.to_csv(P3 / "gap_analysis" / "D1_synthetic_test_vs_referee.csv", index=False)

# %% [markdown]
# ### D2 What does real fraud look like? Real referee fraud compared with the three fraud sources
#
# This comparison shows which generator's picture of fraud the real fraud resembles. It uses the same type-appropriate effect sizes as Phase 2 (KS D for continuous and integer features; TVD for hour and categorical features). It is descriptive: with only 65 real frauds, small differences are noisy.

# %%
ref_fraud = referee[referee[TARGET] == 1]
ref_legit = referee[referee[TARGET] == 0]
seed_fraud = train[train[TARGET] == 1]
seed_legit = train[train[TARGET] == 0]
my_ctgan = pd.read_csv(DATA / "synthetic_fraud_gan.csv")
noon_u = noon_syn.drop_duplicates(FEATURES)
srcs = {"Real referee fraud": ref_fraud, "My seed fraud (training)": seed_fraud, "My CTGAN fraud": my_ctgan, "Noon synthetic fraud (unique rows)": noon_u}

def indicators(d):
    return {"orders placed 00:00–05:59": d.hour_of_day.between(0, 5).mean() * 100,
            "location_mismatch = 1": d.location_mismatch.mean() * 100,
            "device_change = 1": d.device_change.mean() * 100,
            "account age ≤ 7 days": (d.account_age_days <= 7).mean() * 100,
            "account age > 180 days": (d.account_age_days > 180).mean() * 100,
            "discount_applied = 1": d.discount_applied.mean() * 100,
            "≥ 1 previous refund": (d.prev_refunds >= 1).mean() * 100,
            "≥ 4 orders in last 7 days": (d.num_orders_last_7d >= 4).mean() * 100,
            "paid cash on delivery": (d.payment_method == "cod").mean() * 100}
prof_tab = pd.DataFrame({k: indicators(v) for k, v in srcs.items()})
prof_tab.loc["rows (n)"] = [len(v) for v in srcs.values()]
display(prof_tab.style.format("{:.1f}"))
prof_tab.to_csv(P3 / "gap_analysis" / "D2_fraud_indicator_profiles.csv")
P.p3_fraud_profiles(prof_tab.drop(index="rows (n)").rename(index=lambda s: s.replace(" = 1", "").replace(" in last", "\nin last").replace("ordered ", "").replace("orders placed", "ordered").replace("account age", "account age\n")),
                    {"Real referee fraud": NAVY, "My seed fraud (training)": SEED_FRAUD_COLOR, "My CTGAN fraud": MY_GAN_COLOR, "Noon synthetic fraud (unique rows)": NOON_COLOR},
                    P3 / "charts" / "3.4_real_fraud_vs_generators_profiles.png",
                    f"Step 3.4 evidence — what real fraud looks like (n = {len(ref_fraud)}) vs my seed, my CTGAN and Noon's synthetic fraud")

eff_rows = []
for name in ["My seed fraud (training)", "My CTGAN fraud", "Noon synthetic fraud (unique rows)"]:
    e = C.compare_features(ref_fraud, srcs[name], "Real referee fraud", name, n_perm=0, seed=SEED)
    e["pair"] = f"Real referee fraud vs {name}"
    eff_rows.append(e[["pair", "feature", "headline effect", "effect size", "magnitude (judgement band)"]])
eff_real = pd.concat(eff_rows, ignore_index=True)
eff_wide = eff_real.pivot(index="feature", columns="pair", values="effect size")
# Sampling-noise floor: the distance expected between 65 rows and the source they were drawn from (same distribution),
# so that effect sizes measured on only 65 real frauds are not over-read. 200 random draws from Noon's unique rows.
rng_floor = np.random.default_rng(SEED)
floor = pd.concat([C.compare_features(noon_u.iloc[rng_floor.choice(len(noon_u), len(ref_fraud), replace=False)], noon_u, "draw", "src",
                                      n_perm=0, seed=SEED).set_index("feature")["effect size"] for _ in range(200)], axis=1).mean(axis=1)
eff_wide["noise floor: 65 rows vs their own source (mean of 200 draws)"] = floor
eff_wide.loc["mean over 10 features"] = eff_wide.mean()
display(eff_wide.style.format("{:.3f}").set_caption("Distance from real referee fraud (0 = identical; lower = closer to real fraud)"))
eff_real.to_csv(P3 / "gap_analysis" / "D2_effect_sizes_real_fraud_vs_sources.csv", index=False)
eff_wide.to_csv(P3 / "gap_analysis" / "D2_effect_sizes_wide.csv")

legit_ind = pd.DataFrame({"Real referee legitimate": indicators(ref_legit), "My seed legitimate (training)": indicators(seed_legit)})
legit_ind.loc["rows (n)"] = [len(ref_legit), len(seed_legit)]
display(legit_ind.style.format("{:.1f}").set_caption("Legitimate behaviour: real vs what my seed assumed"))
legit_ind.to_csv(P3 / "gap_analysis" / "D2_legit_indicator_profiles.csv")
legit_cmp = C.compare_features(ref_legit, seed_legit, "Real referee legitimate", "My seed legitimate (training)", n_perm=0, seed=SEED)
display(legit_cmp[["feature", "headline effect", "effect size", "magnitude (judgement band)", "median A", "median B", "mean A", "mean B"]])
legit_cmp.to_csv(P3 / "gap_analysis" / "D2_real_legit_vs_my_seed_legit.csv", index=False)

# %% [markdown]
# ### D3 Which real frauds did each model catch?

# %%
f = y_ref == 1
ov = pd.DataFrame([{"Caught by both": int(((pred_sel == 1) & (pred_noon == 1) & f).sum()),
                    "Caught only by my selected classifier": int(((pred_sel == 1) & (pred_noon == 0) & f).sum()),
                    "Caught only by Noon Model B": int(((pred_sel == 0) & (pred_noon == 1) & f).sum()),
                    "Missed by both": int(((pred_sel == 0) & (pred_noon == 0) & f).sum())}])
display(ov)
ov.to_csv(P3 / "gap_analysis" / "D3_fraud_detection_overlap.csv", index=False)
P.p3_detection_overlap(ov, ["#4B5563", COLORS3[NAME_SEL], COLORS3[NAME_NOON], "#B9BFCB"],
                       P3 / "charts" / "3.4_fraud_detection_overlap.png",
                       f"The {N_FRAUD} real frauds: caught by both models, by one, or by neither")
grp = {"Caught by my selected classifier": ref_fraud[pred_sel[f] == 1], "Missed by my selected classifier": ref_fraud[pred_sel[f] == 0],
       "Caught by Noon Model B": ref_fraud[pred_noon[f] == 1], "Missed by Noon Model B": ref_fraud[pred_noon[f] == 0]}
cm_tab = pd.DataFrame({k: indicators(v) for k, v in grp.items()})
cm_tab.loc["rows (n)"] = [len(v) for v in grp.values()]
display(cm_tab.style.format("{:.1f}"))
cm_tab.to_csv(P3 / "gap_analysis" / "D3_caught_vs_missed_profiles.csv")
fp_tab = pd.DataFrame({"False positives of my selected classifier": indicators(ref_legit[pred_sel[~f] == 1]),
                       "False positives of Noon Model B": indicators(ref_legit[pred_noon[~f] == 1]),
                       "All real legitimate orders": indicators(ref_legit)})
fp_tab.loc["rows (n)"] = [int(pred_sel[~f].sum()), int(pred_noon[~f].sum()), len(ref_legit)]
display(fp_tab.style.format("{:.1f}"))
fp_tab.to_csv(P3 / "gap_analysis" / "D3_false_positive_profiles.csv")

# %% [markdown]
# ### Step 3.4 — Why does Noon perform differently?
#
# *Every number below comes from tables C1–D3. **Observed** = computed; **Interpretation** = my reading, which these data cannot prove.*
#
# **1–2. Who catches more real fraud, and who flags fewer innocent orders?**
# * *Observed.* Noon Model B catches **53 of 65** real frauds (recall 0.815). My selected classifier catches **44 of 65** (0.677). The difference is **+0.138**, or 9 more frauds for Noon.
# * *Observed.* Noon flags **71** legitimate orders; mine flags **1,036**, so Noon raises **965 fewer** false alarms. Noon's precision is 0.427 vs 0.041, and its F1 0.561 vs 0.077.
# * *Observed — how certain is this?*
#   * **False positives:** clear-cut. Of the legitimate orders only one model flagged, 997 were flagged by mine alone and 32 by Noon's alone (exact McNemar p < 0.001).
#   * **Recall:** less certain with only 65 frauds. 15 frauds were caught only by Noon and 6 only by mine (exact McNemar p = 0.078). The paired bootstrap 95% interval for Noon's recall advantage runs from 0.000 to 0.277. Noon is ahead on recall, but this referee set cannot pin down by how much.
#
# **3. Did my CTGAN augmentation help on real data?**
# * *Observed.* No. The CTGAN-augmented classifier catches **39 of 65** (recall 0.600), **5 fewer** than my selected classifier. Its false positives are essentially the same (1,034 vs 1,036). All 5 discordant frauds go against the GAN model. Exact McNemar p = 0.0625, which is the smallest p-value possible with 5 discordant pairs; the paired bootstrap interval for the recall difference runs from −0.138 to −0.015.
# * *Interpretation.* This matches Phase 1, where the pre-declared rule found no measurable improvement from CTGAN and selected the baseline. Phase 2 also showed why my GAN could not have helped much: it learned my own assumptions, and real fraud is far from them (point 5). A GAN can only re-sample the picture of fraud it was trained on.
#
# **4. Is this consistent with Phase 2?**
# * *Observed.* Yes. Phase 2 found that my fraud and Noon's differ most on night ordering, location mismatch, device change and account age. Real referee fraud sits on **Noon's side** of every one of these:
#
# | Indicator (share of fraud rows) | Real referee fraud | My seed fraud | My CTGAN fraud | Noon synthetic fraud |
# |---|---|---|---|---|
# | Ordered 00:00–05:59 | 47.7% | 14.2% | 11.9% | 60.1% |
# | Location mismatch | 60.0% | 24.6% | 23.4% | 62.9% |
# | Device change | 70.8% | 20.1% | 31.9% | 65.6% |
# | Account ≤ 7 days old | 6.2% | 45.7% | 35.6% | 3.0% |
#
# * *Observed.* Averaged over the 10 features, real fraud is **0.099** away from Noon's synthetic fraud. That is close to the **0.086** distance expected between 65 rows and their own source from sampling noise alone. Real fraud is **0.297** away from my seed fraud and **0.271** from my CTGAN fraud.
# * *Interpretation.* Noon's generator learned a picture of fraud close to the real one: established accounts, night-time orders, new device, mismatched location. My Phase 1 assumptions put most weight on brand-new-account promo abuse, which barely appears in the real fraud (6.2% of real frauds are ≤ 7 days old).
#
# **5. Why might Noon's GAN have learned what my seed data lacked?**
# * *[SOURCE FACT]* `B_NOON_benchmark.md` says Noon's GAN was trained on Noon's **real** fraud examples, and Model B was trained on real data plus that GAN fraud. My models never saw a single real row. My seed data was built from public research and my own assumptions (Phase 1 register).
# * *Observed — the fraud side.* My classifier misses exactly the fraud my assumptions under-weighted:
#   * **Night orders:** 71.4% of the 21 frauds it missed were placed 00:00–05:59, vs 36.4% of those it caught. In Phase 1 I made night ordering a deliberately weak signal (H6).
#   * **Refunds:** 56.8% of the frauds it caught had a previous refund, vs 14.3% of those it missed. It leans heavily on refund history.
# * *Observed — the legitimate side.* This is the larger problem: real legitimate customers behave differently from my synthetic ones.
#
# | Indicator (share of legitimate rows) | Real referee legitimate | My seed legitimate |
# |---|---|---|
# | Ordered 00:00–05:59 | 24.5% | 8.9% |
# | ≥ 1 previous refund | 26.5% | 14.3% |
# | Device change | 8.0% | 3.3% |
# | ≥ 4 orders in last 7 days | 35.6% | 10.6% |
#
# * *Observed.* My classifier treats these behaviours as risky. Its 1,036 false positives are concentrated in them: 66.9% have a previous refund, 43.1% a device change and 54.1% at least 4 recent orders, against 26.5%, 8.0% and 35.6% of all real legitimate orders.
# * *Observed — prevalence.* Its false-positive rate rose from 1.33% on my synthetic test set to **5.20%** on the real set. If only the fraud rate had changed (3% synthetic to 0.325% real) and recall and false-positive rate had held, precision would have fallen to about **11.1%** from base rates alone. The actual precision of 4.1% shows that real legitimate behaviour, not just rarity, is driving the false alarms.
# * *Interpretation.* The gap therefore has two sides. Noon's model learned what real fraud looks like **and** what real legitimate customers look like; mine learned both from assumptions.
#
# **6. Is it all down to the GAN? No — other differences matter and cannot be separated here.**
# * **Training data:** Noon's includes real transactions; mine has none.
# * **Model family:** Noon uses a random forest (200 trees, balanced class weights); mine is gradient boosting with Platt calibration.
# * **Decision rules:** Noon uses majority vote; mine uses a threshold of 0.297 on calibrated probabilities, tuned for a 3%-fraud synthetic world.
# * **Class composition** of training data also differs.
#
# Separating these effects would need retraining, which Phase 3 forbids. Noon's own Model A (real data only, no GAN) reportedly reached recall of only 0.14, which suggests real data **plus** realistic synthetic fraud is what worked for Noon. I could not re-score Model A, because it was not supplied.
#
# Ranking quality is a separate point. My model still ranks real fraud well above chance (ROC-AUC 0.929 vs Noon's 0.997). But at a 0.325% fraud rate, its average precision (0.064 vs 0.617) shows that too many legitimate orders score as high as real fraud.
#
# **7. What limits these conclusions?**
# * **Sample size:** there are only 65 real frauds. Recall intervals are wide (mine: 0.556–0.778; Noon: 0.704–0.891, 95% Wilson).
# * **One referee sample:** it may not represent Noon's full or future transaction population; for example, the fraud rate in production may differ from 0.325%.
# * **Unknown Noon training data:** its timing and overlap with the referee period are unknown. Noon's model may have an in-distribution advantage that a production deployment would not keep. I found no exact referee rows in Noon's synthetic file (0 overlaps), but I cannot check Noon's real training data.
# * **Descriptive analysis only:** sections D2–D3 use the referee labels after the fact. They explain the gap; they were not used to change anything, and with 65 frauds some differences are noise.
# * **Limits of the benchmark:** Noon's synthetic sample (Phase 2) is a benchmark, not ground truth. The referee results are the real evidence, but they judge **these** models on **this** sample.

# %% [markdown]
# ## E. Quality control (Part 9 checklist, automated)

# %%
audit = []
def chk(name, ok, detail=""):
    audit.append({"check": name, "result": "Passed" if ok else "FAILED", "detail": detail})

chk("All three intended models identified (selected = A_baseline per Phase 1; secondary = B_ctgan; Noon = RandomForest)",
    sel.metadata["arm"] == p1["final_arm"] == "A_baseline" and gan.metadata["arm"] == "B_ctgan" and type(rf).__name__ == "RandomForestClassifier")
chk("Selected model not confused with the CTGAN candidate (different arm, threshold and file hash)",
    sel.metadata["arm"] != gan.metadata["arm"] and sel.threshold != gan.threshold and MODEL_HASHES_BEFORE["models/fraud_model.joblib"] != MODEL_HASHES_BEFORE["models/fraud_model_gan_augmented.joblib"])
chk("Baseline file is the same fitted artefact as the selected model (not reported twice)", baseline_identical and same_as_selected)
hashes_after = {n: C.sha256(p) for n, p, _, _ in inputs if n.endswith(".joblib")}
chk("No model retrained or modified (all four model files hash-identical before and after scoring, and to their records)",
    hashes_after == MODEL_HASHES_BEFORE and bool(inv.loc[inv.input.str.endswith(".joblib"), "matches recorded hash"].all()))
chk("All models scored on the same 20,000 referee rows", all(len(a) == N_REF for a in (pred_sel, pred_gan, pred_noon)) and N_REF == 20000)
chk("All 65 fraud cases included for every model", bool((res["n fraud"] == N_FRAUD).all()) and N_FRAUD == 65 and bool(((res.TP + res.FN) == N_FRAUD).all()))
chk("Original schema preserved; Noon payment-method encoding matches its stored columns",
    list(referee.columns) == list(SCHEMA) and bool(noon_meta["helper encoding == stored columns (names + order)"])
    and bool(noon_meta["every referee row has exactly one payment_method column = 1"]))
chk("Each model's frozen prediction rule applied (stored thresholds; Noon's native predict)",
    np.array_equal(pred_sel, (p_sel >= sel.threshold).astype(int)) and np.array_equal(pred_gan, (p_gan >= gan.threshold).astype(int))
    and sel.threshold == p1["final_threshold_calibrated"] and np.array_equal(pred_noon, rf.predict(X_noon)))
src_text = (ROOT / "notebooks" / "phase3_evaluation.py").read_text()
chk("Referee labels kept out of every model input (X_ref has no is_fraud; no fit/calibrate call in Phase 3 code)",
    TARGET not in X_ref.columns and not any(t in src_text for t in ["." + "fit(", "fit_" + "transform(", "Calibrated" + "ClassifierCV", "threshold" + "_sweep(", "cost_optimal" + "_threshold("]))
chk("Noon benchmark reproduction checked (all published figures)", bool(repro["match"].all()), f"{int(repro['match'].sum())}/{len(repro)} match")
chk("Helper score_on_referee agrees with the independent recomputation", bool(helper_vs_mine.all()))
chk("Noon predictions identical under scikit-learn 1.8.0 (isolated env) and 1.9.1",
    bool(crossver.get("predictions identical (1.8.0 vs 1.9.1)", False)), str(crossver.get("max |probability difference|", "not tested")))
chk("Confusion matrices reconcile with dataset size (TP+FP+FN+TN = 20,000)", bool(((res.TP + res.FP + res.FN + res.TN) == N_REF).all()))
chk("Recall, precision and F1 independently verified with scikit-learn", bool(check["all match"].all()))
reload_ok = all(np.array_equal(joblib.load(ROOT / "models" / fn).predict(X_ref), pr) for fn, pr in (("fraud_model.joblib", pred_sel), ("fraud_model_gan_augmented.joblib", pred_gan)))
chk("Predictions reproduce from freshly reloaded model files", reload_ok)
chk("Saved prediction file reproduces the reported confusion matrices",
    all(np.array_equal(pd.read_csv(P3 / "predictions" / "B1_referee_predictions_all_models.csv")[c].to_numpy(), v) for c, v in (("selected_pred", pred_sel), ("ctgan_pred", pred_gan), ("noon_pred", pred_noon))))
p1_ok = R.cells_digest(nb_path, None) == p1_manifest["phase1_notebook_cells_sha256"] and bool(R.verify_files({k: v for k, v in p1_files.items() if k.startswith(("data/", "models/"))}, ROOT)["unchanged"].all())
p2_ok = R.cells_digest(nb_path, 2) == p2_freeze["phase2_notebook_cells_sha256"] and bool(R.verify_files(p2_freeze["files"], ROOT)["unchanged"].all())
chk("Phase 1 and Phase 2 outputs unchanged (cells, data, models, Phase 2 outputs and reports)", p1_ok and p2_ok)
chk("Professor files unchanged; no __pycache__ in the professor folder",
    bool(R.verify_files({f"professor_supplied/{k}": v for k, v in prof_recorded.items()}, ROOT)["unchanged"].all()) and not (PROF / "__pycache__").exists())
chk("All charts written", all((P3 / "charts" / f).exists() for f in ["3.3_final_bar_chart_recall_and_false_positives.png", "3.1_3.2_confusion_matrices.png",
    "3.3_precision_recall_and_roc.png", "3.4_real_fraud_vs_generators_profiles.png", "3.4_fraud_detection_overlap.png"]))
markers = ["STEP_3_4_" + "EXPLANATION", "PHASE_3_" + "SUMMARY", "TO" + "DO"]
chk("No unfilled placeholder text in the Phase 3 section", not any(m in src_text for m in markers))
r_ = res
pct = lambda x: f"{100 * x:.1f}%"
QUOTED = {
    "sel recall 0.677": f"sel recall {r_.loc[NAME_SEL, 'recall']:.3f}", "gan recall 0.600": f"gan recall {r_.loc[NAME_GAN, 'recall']:.3f}",
    "noon recall 0.815": f"noon recall {r_.loc[NAME_NOON, 'recall']:.3f}", "sel prec 0.041": f"sel prec {r_.loc[NAME_SEL, 'precision']:.3f}",
    "gan prec 0.036": f"gan prec {r_.loc[NAME_GAN, 'precision']:.3f}", "noon prec 0.427": f"noon prec {r_.loc[NAME_NOON, 'precision']:.3f}",
    "sel F1 0.077": f"sel F1 {r_.loc[NAME_SEL, 'F1']:.3f}", "gan F1 0.069": f"gan F1 {r_.loc[NAME_GAN, 'F1']:.3f}", "noon F1 0.561": f"noon F1 {r_.loc[NAME_NOON, 'F1']:.3f}",
    "CM sel 44/1036/21/18899": "CM sel {TP}/{FP}/{FN}/{TN}".format(**r_.loc[NAME_SEL, ["TP", "FP", "FN", "TN"]].astype(int).to_dict()),
    "CM gan 39/1034/26/18901": "CM gan {TP}/{FP}/{FN}/{TN}".format(**r_.loc[NAME_GAN, ["TP", "FP", "FN", "TN"]].astype(int).to_dict()),
    "CM noon 53/71/12/19864": "CM noon {TP}/{FP}/{FN}/{TN}".format(**r_.loc[NAME_NOON, ["TP", "FP", "FN", "TN"]].astype(int).to_dict()),
    "diff recall +0.138": f"diff recall {r_.loc[NAME_NOON, 'recall'] - r_.loc[NAME_SEL, 'recall']:+.3f}",
    "fewer FP 965": f"fewer FP {int(r_.loc[NAME_SEL, 'FP'] - r_.loc[NAME_NOON, 'FP'])}",
    "mcnemar legit 997/32": f"mcnemar legit {int(paired[0][0].iloc[1]['flagged only by A'])}/{int(paired[0][0].iloc[1]['flagged only by B'])}",
    "mcnemar fraud 6/15 p 0.078": f"mcnemar fraud {int(paired[0][0].iloc[0]['flagged only by A'])}/{int(paired[0][0].iloc[0]['flagged only by B'])} p {paired[0][0].iloc[0]['exact McNemar p']:.3f}",
    "boot recall 0.000–0.277": f"boot recall {paired[0][1].set_index('metric').loc['recall', '95% CI low']:.3f}–{paired[0][1].set_index('metric').loc['recall', '95% CI high']:.3f}",
    "gan mcnemar 5/0 p 0.0625": f"gan mcnemar {int(paired[1][0].iloc[0]['flagged only by A'])}/{int(paired[1][0].iloc[0]['flagged only by B'])} p {paired[1][0].iloc[0]['exact McNemar p']:.4f}",
    "gan boot −0.138 to −0.015": "gan boot {:.3f} to {:.3f}".format(*paired[1][1].set_index("metric").loc["recall", ["95% CI low", "95% CI high"]]).replace("-", "−"),
    "dist noon 0.099 floor 0.086 seed 0.297 ctgan 0.271": "dist noon {:.3f} floor {:.3f} seed {:.3f} ctgan {:.3f}".format(
        eff_wide.loc["mean over 10 features", "Real referee fraud vs Noon synthetic fraud (unique rows)"], eff_wide.loc["mean over 10 features", "noise floor: 65 rows vs their own source (mean of 200 draws)"],
        eff_wide.loc["mean over 10 features", "Real referee fraud vs My seed fraud (training)"], eff_wide.loc["mean over 10 features", "Real referee fraud vs My CTGAN fraud"]),
    "real fraud 47.7/60.0/70.8/6.2": "real fraud {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(*prof_tab.loc[["orders placed 00:00–05:59", "location_mismatch = 1", "device_change = 1", "account age ≤ 7 days"], "Real referee fraud"]),
    "seed fraud 14.2/24.6/20.1/45.7": "seed fraud {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(*prof_tab.loc[["orders placed 00:00–05:59", "location_mismatch = 1", "device_change = 1", "account age ≤ 7 days"], "My seed fraud (training)"]),
    "ctgan fraud 11.9/23.4/31.9/35.6": "ctgan fraud {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(*prof_tab.loc[["orders placed 00:00–05:59", "location_mismatch = 1", "device_change = 1", "account age ≤ 7 days"], "My CTGAN fraud"]),
    "noon fraud 60.1/62.9/65.6/3.0": "noon fraud {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(*prof_tab.loc[["orders placed 00:00–05:59", "location_mismatch = 1", "device_change = 1", "account age ≤ 7 days"], "Noon synthetic fraud (unique rows)"]),
    "missed night 71.4 caught 36.4": "missed night {:.1f} caught {:.1f}".format(cm_tab.loc["orders placed 00:00–05:59", "Missed by my selected classifier"], cm_tab.loc["orders placed 00:00–05:59", "Caught by my selected classifier"]),
    "refund caught 56.8 missed 14.3": "refund caught {:.1f} missed {:.1f}".format(cm_tab.loc["≥ 1 previous refund", "Caught by my selected classifier"], cm_tab.loc["≥ 1 previous refund", "Missed by my selected classifier"]),
    "legit real 24.5/26.5/8.0/35.6": "legit real {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(*legit_ind.loc[["orders placed 00:00–05:59", "≥ 1 previous refund", "device_change = 1", "≥ 4 orders in last 7 days"], "Real referee legitimate"]),
    "legit seed 8.9/14.3/3.3/10.6": "legit seed {:.1f}/{:.1f}/{:.1f}/{:.1f}".format(*legit_ind.loc[["orders placed 00:00–05:59", "≥ 1 previous refund", "device_change = 1", "≥ 4 orders in last 7 days"], "My seed legitimate (training)"]),
    "FP sel 66.9/43.1/54.1": "FP sel {:.1f}/{:.1f}/{:.1f}".format(*fp_tab.loc[["≥ 1 previous refund", "device_change = 1", "≥ 4 orders in last 7 days"], "False positives of my selected classifier"]),
    "fpr synth 1.33% real 5.20%": f"fpr synth {100 * tm['A_baseline']['fpr']:.2f}% real {100 * r_.loc[NAME_SEL, 'false-positive rate']:.2f}%",
    "expected precision 11.1%": f"expected precision {100 * gen.loc[(gen.model == NAME_SEL) & gen.dataset.str.startswith('Real'), 'precision expected from prevalence alone'].iloc[0]:.1f}%",
    "auc 0.929 vs 0.997": f"auc {r_.loc[NAME_SEL, 'ROC-AUC']:.3f} vs {r_.loc[NAME_NOON, 'ROC-AUC']:.3f}",
    "ap 0.064 vs 0.617": f"ap {r_.loc[NAME_SEL, 'average precision (PR-AUC)']:.3f} vs {r_.loc[NAME_NOON, 'average precision (PR-AUC)']:.3f}",
    "wilson sel 0.556–0.778 noon 0.704–0.891": "wilson sel {:.3f}–{:.3f} noon {:.3f}–{:.3f}".format(r_.loc[NAME_SEL, "recall 95% CI low"], r_.loc[NAME_SEL, "recall 95% CI high"], r_.loc[NAME_NOON, "recall 95% CI low"], r_.loc[NAME_NOON, "recall 95% CI high"]),
    "overlap 38/6/15/6": "overlap {}/{}/{}/{}".format(*ov.iloc[0].tolist()),
}
bad = {k: v for k, v in QUOTED.items() if k != v}
chk("Numbers quoted in the written explanation equal this run's results", bool(QUOTED) and not bad, str(bad) if bad else f"{len(QUOTED)} values checked")
audit = pd.DataFrame(audit)
display(audit)
audit.to_csv(P3 / "audit" / "E_QC_checks.csv", index=False)
print(f"{int((audit.result == 'Passed').sum())} of {len(audit)} checks passed · Phase 3 runtime {(time.time() - T0) / 60:.1f} min")

# %% [markdown]
# ## Phase 3 summary (for the final report)
#
# All three models were scored on the same **20,000** real referee transactions (65 frauds, 0.325%) with their own frozen prediction rules. Nothing was retrained, recalibrated or re-thresholded.
#
# | Model | Recall | Precision | F1 | TP | FP | FN | TN |
# |---|---|---|---|---|---|---|---|
# | My selected classifier (A, baseline) | 0.677 | 0.041 | 0.077 | 44 | 1,036 | 21 | 18,899 |
# | My CTGAN-augmented classifier (B) | 0.600 | 0.036 | 0.069 | 39 | 1,034 | 26 | 18,901 |
# | Noon Model B | 0.815 | 0.427 | 0.561 | 53 | 71 | 12 | 19,864 |
#
# * **Benchmark reproduction.** Noon's model reproduces its published figures exactly (confusion matrix 19,864 / 71 / 12 / 53). Its predictions are identical under scikit-learn 1.8.0 in an isolated environment and 1.9.1 in this project, so the version warning is a software notice, not a performance issue.
# * **Step 3.3 answers.** Noon catches more fraud (+9 frauds, +13.8 percentage points of recall) **and** flags far fewer innocent orders (71 vs 1,036). My CTGAN augmentation did not carry over to real data: 5 fewer frauds caught, false positives unchanged.
# * **Step 3.4 explanation.** Real fraud closely resembles Noon's synthetic fraud (mean distance 0.099, near the 0.086 noise floor) and differs from mine (0.297). It is mainly night-time, new-device, mismatched-location fraud on established accounts. Real legitimate customers also order at night, change devices, claim refunds and order often more than my seed assumed, which drives my 5.2% false-positive rate. Model family, decision rules and Noon's access to real data also differ, and these effects cannot be separated without retraining.
# * **Limitations.** 65 frauds (wide intervals: recall 0.556–0.778 for mine and 0.704–0.891 for Noon's); one referee sample; Noon's training data undocumented.
#
# Phase 4 (the app) has not been started.

# %% [markdown]
# ## F. Phase 3 output inventory (Part 8 deliverables)

# %%
DELIVERABLE = {
    "audit/A1_input_inventory.csv": "1 · Verified inventory of the seven Phase 3 inputs (presence, hash, purpose)",
    "audit/A1_freeze_checks_before_scoring.csv": "1 · Phase 1 / Phase 2 / professor-file freeze checks before scoring",
    "audit/A3_my_model_metadata.csv": "1 · Metadata of my saved models (arm, family, thresholds, calibrator, classes)",
    "audit/A3_selection_verification.csv": "1 · Selected-model verification; baseline-file identity check",
    "audit/A3_fraudmodel_interface_checks.csv": "1 · FraudModel interface checks (raw input, calibrated probability, stored threshold)",
    "audit/A4_noon_model_metadata.csv": "1 · Noon Model B metadata, encoding alignment and version warnings",
    "audit/A2_referee_integrity_checks.csv": "2 · Referee data-dictionary integrity checks",
    "audit/A2_referee_label_counts.csv": "2 · Referee counts (rows, legitimate, fraud, prevalence)",
    "audit/A2_referee_vs_my_training_ranges.csv": "2 · Referee values outside my training ranges (descriptive)",
    "audit/A2_exact_row_overlap.csv": "2 · Exact-row leakage screen (referee vs my data and Noon's synthetic data)",
    "predictions/B1_referee_predictions_all_models.csv": "3 · Row-level scores and predictions for all three models",
    "metrics/C1_comparison_table.csv": "4 · Comparison table: recall, precision, F1, TP, FP, FN, TN",
    "metrics/C1_referee_metrics_all_models.csv": "4 · Full metric set (FPR, specificity, AP, ROC-AUC, flagged, accuracy, CIs)",
    "metrics/C1_sklearn_crosscheck.csv": "4 · Independent scikit-learn recomputation",
    "metrics/C3_step_3_3_answers.csv": "4 · Step 3.3 answers (computed)",
    "metrics/C1_confusion_matrices.json": "5 · Confusion matrices [[TN, FP], [FN, TP]]",
    "charts/3.1_3.2_confusion_matrices.png": "5 · Confusion matrices for all three models",
    "charts/3.3_final_bar_chart_recall_and_false_positives.png": "6 · Required recall and false-positive chart (Step 3.3)",
    "charts/3.3_precision_recall_and_roc.png": "7 · Precision–recall and ROC curves with operating points",
    "charts/3.4_fraud_detection_overlap.png": "7 · Which real frauds each model caught",
    "charts/3.4_real_fraud_vs_generators_profiles.png": "7 · Real fraud vs my seed, my CTGAN and Noon's synthetic fraud",
    "metrics/C4_confidence_intervals.csv": "8 · Wilson (recall, precision) and bootstrap (AP, ROC-AUC) 95% intervals",
    "metrics/C4_mcnemar_exact.csv": "8 · Paired exact McNemar tests (supplementary)",
    "metrics/C4_paired_bootstrap_differences.csv": "8 · Paired bootstrap differences (supplementary)",
    "metrics/C2_noon_benchmark_reproduction.csv": "9 · Noon benchmark reproduction check",
    "audit/B2_helper_vs_recomputed.csv": "9 · Helper score_on_referee vs independent recomputation",
    "audit/B2_noon_cross_version_check.csv": "9 · Noon predictions: scikit-learn 1.8.0 (isolated) vs 1.9.1",
    "audit/noon_predictions_sklearn_1.8.0.csv": "9 · Noon predictions from the isolated scikit-learn 1.8.0 environment",
    "gap_analysis/D1_synthetic_test_vs_referee.csv": "Step 3.4 · Phase 1 synthetic test vs real referee (my models)",
    "gap_analysis/D2_fraud_indicator_profiles.csv": "Step 3.4 · Fraud indicator shares by source",
    "gap_analysis/D2_effect_sizes_wide.csv": "Step 3.4 · Distance of real fraud from each source, with noise floor",
    "gap_analysis/D2_effect_sizes_real_fraud_vs_sources.csv": "Step 3.4 · Per-feature effect sizes (long form)",
    "gap_analysis/D2_legit_indicator_profiles.csv": "Step 3.4 · Real vs my seed legitimate behaviour",
    "gap_analysis/D2_real_legit_vs_my_seed_legit.csv": "Step 3.4 · Real vs my seed legitimate effect sizes",
    "gap_analysis/D3_fraud_detection_overlap.csv": "Step 3.4 · Frauds caught by both, one or neither model",
    "gap_analysis/D3_caught_vs_missed_profiles.csv": "Step 3.4 · Profiles of caught vs missed frauds",
    "gap_analysis/D3_false_positive_profiles.csv": "Step 3.4 · Profiles of each model's false positives",
    "audit/E_QC_checks.csv": "Part 9 · Automated quality-control checks",
}
inventory = pd.DataFrame([{"file": f"outputs/phase3/{k}", "deliverable": v, "exists": (P3 / k).exists(),
                           "KB": round((P3 / k).stat().st_size / 1024, 1) if (P3 / k).exists() else None} for k, v in DELIVERABLE.items()])
inventory = pd.concat([inventory, pd.DataFrame([
    {"file": "notebooks/fraud_detection_analysis.ipynb (Phase 3 section)", "deliverable": "10 · Reproducible Phase 3 notebook section", "exists": nb_path.exists()},
    {"file": "reports/phase3_report.md", "deliverable": "11 · Written Phase 3 report", "exists": (ROOT / "reports" / "phase3_report.md").exists()},
    {"file": "reports/phase3_quality_review.md", "deliverable": "12 · Passed / failed / untested checks", "exists": (ROOT / "reports" / "phase3_quality_review.md").exists()}])], ignore_index=True)
untracked = sorted(str(p.relative_to(P3)) for p in P3.rglob("*") if p.is_file() and str(p.relative_to(P3)) not in DELIVERABLE
                   and p.name not in ("phase3_output_inventory.csv", "noon_predictions_sklearn_1.8.0.csv.meta.txt"))
display(inventory)
inventory.to_csv(P3 / "phase3_output_inventory.csv", index=False)
print(f"{int(inventory.exists.sum())}/{len(inventory)} deliverables present · files in outputs/phase3 not listed: {untracked or 'none'}")
