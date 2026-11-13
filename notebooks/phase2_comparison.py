# %% [markdown]
# ---
# # Phase 2 — The Reveal: my GAN vs Noon's GAN
#
# **Scope.** Phase 2 only: Steps 2.1 (comparison EDA), 2.2 (fraud proportion), 2.3 (lie-detector test) and 2.4 (reflection) of `A_INTERN_BRIEF.md`. Phase 3 (scoring on the real referee set) and Phase 4 (the app) are **not** executed here. The referee file is opened once, for an integrity check only, and then sealed.
#
# **Authority.** `A_INTERN_BRIEF.md` and the data dictionary are the only assignment specifications. The professor's newly supplied files are used for their stated purposes. The Phase 2 Methods Brief (deep research) is supplementary guidance, applied with the corrections listed in A1.
#
# **Analysis sets.**
#
# | Name | What it is | Role |
# |---|---|---|
# | **My CTGAN fraud** | `data/synthetic_fraud_gan.csv` — the fraud-only output of my Phase 1 GAN | The sample being compared |
# | **Noon fraud (unique rows)** | Distinct rows of `B_NOON_synthetic_fraud.csv` | **Primary** comparison sample |
# | **Noon fraud (as supplied)** | All rows of `B_NOON_synthetic_fraud.csv`, duplicates kept | Sensitivity analysis |
# | **My seed fraud** | Fraud rows of my Phase 1 training partition (what my CTGAN learned from) | Reference: separates my *assumptions* from my *GAN's* errors |
#
# Labels used: **[SOURCE FACT]**, **[ASSUMPTION]**, **[FINDING]** (computed here), **[INTERPRETATION]**. These cells run in a fresh kernel after the Phase 1 cells above, which are verified unchanged; execution counts restart at 1.

# %%
import os, sys, json, time, platform, warnings, importlib
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True                      # never write __pycache__ into the read-only professor folder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

from src.config import *
from src import plotting as P, compare as C, gan as G
from src.seed_generator import validate_schema

PROF = ROOT / "professor_supplied"
sys.path.insert(0, str(PROF))
import A_eval_helper as H                            # professor's optional helper, imported unmodified
sys.path.remove(str(PROF))

P.style()
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

P2 = OUTPUTS / "phase2"
P2_DATA = DATA / "phase2"
for sub in ["audit", "data_quality", "distributions", "relationships", "fraud_proportion", "lie_detector"]:
    (P2 / sub).mkdir(parents=True, exist_ok=True)
P2_DATA.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(10))                               # repeated balanced subsamples / CV shuffles
N_PERM = 1000                                         # permutations for the C2ST p-value
T0 = time.time()
print(f"Run {time.strftime('%Y-%m-%d %H:%M')} · Python {platform.python_version()} · master seed {SEED}")
display(pd.DataFrame([(p, importlib.import_module(p).__version__) for p in ["numpy", "pandas", "sklearn", "scipy", "matplotlib"]],
                     columns=["package", "version"]).set_index("package").T)

# %% [markdown]
# ## A. Audit before analysis
#
# ### A1. Source discrepancies and corrections (identified, not silently reconciled)
#
# | # | Discrepancy | How it is handled |
# |---|---|---|
# | 1 | The master instruction names `A_INTERN_data_dictionary(1).md` and `app(2).py`; the files received are `A_INTERN_data_dictionary.md` (Phase 1) and `app.py` | Naming only. The single data dictionary received is authoritative; `app.py` is stored unchanged as `professor_supplied/app(2).py` |
# | 2 | `B_NOON_benchmark.md` is headed "The lesson in three models" but its table contains two (A and B) | Only Models A and B are discussed; no third model is invented |
# | 3 | The benchmark's "~0.3% fraud rate" describes the **referee test set**, not Noon's synthetic file | Kept as separate denominators in Step 2.2 |
# | 4 | Only Model B is supplied as `B_NOON_model.joblib`; Model A exists only as reported numbers | No Model A evaluation is claimed |
# | 5 | Methods Brief recommended `permutation_test_score` with row-hash `groups` | **Corrected:** `groups` restricts label permutations to within groups, and identical rows share a source label, so the null would be degenerate. The permutation test runs on deduplicated rows **without** groups; the grouped as-supplied analysis reports no p-value |
# | 6 | Methods Brief treated `account_age_days` as continuous | **Corrected:** the data dictionary defines it as an integer, so it is analysed as a discrete, ordered variable (integer bands, ECDF, KS D with a permutation p-value that is exact under ties) |
# | 7 | Methods Brief back-calculated ≈22,000 referee rows from the benchmark | Superseded by the file's own count (A4) |
# | 8 | The master instruction quotes 3,000 rows, 1,967 duplicates, 1,033 distinct rows, 20,000 referee rows, and the Methods Brief quotes 2,116 student rows and a 0.586 baseline | **Not hard-coded:** every count and baseline below is recomputed from the files |

# %%
recorded = json.loads((REPORTS / "professor_files_manifest.json").read_text())["files"]
prof_inv = C.professor_inventory(PROF, recorded)
display(prof_inv)
prof_inv.to_csv(P2 / "audit" / "A2_professor_files.csv", index=False)
assert prof_inv["unchanged since received"].all(), "A professor file differs from the copy received"
print("[FINDING] All six professor files are present and byte-identical to what was received (read-only copies).")

# %% [markdown]
# ### A2. What the optional helper `A_eval_helper.py` actually computes (read and verified)
#
# | Function | What it does | Assumptions to check before trusting the number |
# |---|---|---|
# | `_prep(df, columns)` | `pd.get_dummies` on the 9 numeric columns + `payment_method`; optional `reindex` to training columns | `is_fraud` is **not** in its column list, so the target cannot leak into the lie detector |
# | `compare_distributions` | Overlaid `plt.hist(..., bins=30, density=True)` for 3 features | Bin edges are computed **separately** for each dataset, so bars are not aligned; for integer variables `density=True` gives a per-unit density, not a share of rows |
# | `lie_detector` | Concatenates mine (label 0) then Noon (label 1); `cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0), cv=3)`; mean **accuracy** | With an integer `cv` and a classifier, scikit-learn uses `StratifiedKFold(3)` **without shuffling**; classes may be unequal in size (baseline ≠ 0.5); exact duplicates can sit in both training and test folds |
# | `score_on_referee` | Phase 3: `predict` (default 0.5 threshold) on `_prep(referee)` aligned to training columns; recall, precision, F1, false positives | Not used in Phase 2 |

# %%
import joblib
noon_model = joblib.load(PROF / "B_NOON_model.joblib")
rf = noon_model["model"]
model_facts = pd.Series({
    "container type": type(noon_model).__name__, "keys": list(noon_model.keys()), "estimator": type(rf).__name__,
    "n_estimators": rf.n_estimators, "class_weight": rf.class_weight, "random_state": rf.random_state,
    "classes_ (order)": list(rf.classes_), "n input columns": rf.n_features_in_, "num (stored)": noon_model["num"],
    "cat (stored)": noon_model["cat"], "columns (stored)": noon_model["columns"],
    "feature_names_in_ == stored columns": list(rf.feature_names_in_) == list(noon_model["columns"])})
display(model_facts.to_frame("Noon benchmark model (B_NOON_model.joblib)"))
assert rf.n_estimators == 200 and rf.class_weight == "balanced" and rf.random_state == 0 and rf.n_features_in_ == 13
assert len(noon_model["num"]) == 9 and noon_model["cat"] == ["payment_method"]
print("[FINDING] Verified: Random Forest, 200 trees, class_weight='balanced', random_state=0, 13 input columns "
      "(9 numeric + payment_method one-hot into 4), classes ordered [0, 1]. It was pickled with scikit-learn 1.8.0 and this "
      f"environment runs {importlib.import_module('sklearn').__version__} (version warning on load) — a Phase 3 compatibility check.")
print("Nothing was scored with this model in Phase 2; it is reserved for Phase 3.")
del noon_model, rf

# %% [markdown]
# ### A3. Referee test set — integrity check only, then sealed
#
# **[SOURCE FACT]** `C_REFEREE_real_test.csv` is reserved for Phase 3. Here it is checked for schema, size, missing values, duplicates and label counts (to keep its denominator distinct from Noon's file in Step 2.2). It is then deleted from memory and used for nothing else: no training, augmentation, feature selection, tuning, thresholds or model selection.

# %%
_ref = pd.read_csv(PROF / C.REFEREE_FILENAME)
referee_integrity = {"rows": len(_ref), "columns": _ref.shape[1], "columns match data dictionary (names + order)": list(_ref.columns) == list(SCHEMA),
                     "missing values": int(_ref.isna().sum().sum()), "duplicate rows": int(_ref.duplicated().sum()),
                     "is_fraud = 0": int((_ref[TARGET] == 0).sum()), "is_fraud = 1": int((_ref[TARGET] == 1).sum()),
                     "fraud prevalence (fraud / all referee rows)": float(_ref[TARGET].mean()),
                     "passes data-dictionary checks": bool(validate_schema(_ref)["passed"].iloc[:13].all())}
display(pd.Series(referee_integrity).to_frame("C_REFEREE_real_test.csv (integrity only)"))
REFEREE_N, REFEREE_FRAUD = referee_integrity["rows"], referee_integrity["is_fraud = 1"]
del _ref
print("Referee data deleted from memory — sealed until Phase 3.")

# %% [markdown]
# ### A4. Phase 1 artefacts — identified and verified intact

# %%
p1_map = C.phase1_artifact_map(ROOT)
display(p1_map)
assert p1_map["present"].all()
manifest = json.loads((REPORTS / "phase1_submission_manifest.json").read_text())
ver = C.verify_manifest(manifest["files"], ROOT)
EXPECTED_CHANGES = {
    "src/config.py": "Phase 2 colour constants appended; no Phase 1 value changed",
    "src/plotting.py": "Phase 2 chart functions appended; no Phase 1 function changed",
    "README.md": "Phase 2 section and professor-file folder added; note that the helper and app were not supplied in Phase 1 updated; Phase 1 runtime statement corrected (≈26 min, not 40–50 min)",
    "reports/quality_assurance_checklist.md": "Phase 2 cross-reference added; Phase 1 entries unchanged",
    "reports/rubric_traceability.md": "Phase 2 status updated",
    "notebooks/fraud_detection_analysis.ipynb": "Phase 2 cells appended; Phase 1 cells verified unchanged by digest below",
}
changed = ver[~ver["unchanged since Phase 1 submission"]].copy()
changed["documented reason"] = changed["file"].map(EXPECTED_CHANGES).fillna("⚠ NOT DOCUMENTED")
artefacts = ver[ver["file"].str.startswith(("data/", "models/"))]
cells_ok = C.phase1_cells_digest(ROOT / "notebooks" / "fraud_detection_analysis.ipynb") == manifest["phase1_notebook_cells_sha256"]
print(f"[FINDING] Phase 1 data/model artefacts unchanged since submission: {int(artefacts['unchanged since Phase 1 submission'].sum())}/{len(artefacts)}; "
      f"Phase 1 notebook cells unchanged: {cells_ok}")
display(changed[["file", "documented reason"]])
assert artefacts["unchanged since Phase 1 submission"].all() and cells_ok and (changed["documented reason"] != "⚠ NOT DOCUMENTED").all()

# %% [markdown]
# ## B. Data quality and duplicates
#
# Both fraud-only samples are loaded **as supplied**, checked against the data dictionary, and profiled for duplicates. Noon's original file is never modified; a deduplicated analysis set is derived from it in memory.

# %%
mine = pd.read_csv(DATA / "synthetic_fraud_gan.csv")
noon_all = pd.read_csv(PROF / C.NOON_FILENAME)
seed = pd.read_csv(DATA / "seed_transactions.csv")
split = np.load(DATA / "split_indices.npz")
seed_fraud = seed.iloc[split["train"]].query("is_fraud == 1").reset_index(drop=True)

checks = pd.concat({nm: validate_schema(d).set_index("check")["n_violations"] for nm, d in
                    [("My CTGAN fraud", mine), ("Noon fraud (as supplied)", noon_all), ("My seed fraud", seed_fraud)]}, axis=1)
display(checks)
comp = pd.DataFrame({nm: {"rows": len(d), "columns": d.shape[1], "is_fraud values": sorted(d[TARGET].unique()),
                          "missing values": int(d.isna().sum().sum()), "payment levels": sorted(d.payment_method.unique())}
                     for nm, d in [("My CTGAN fraud", mine), ("Noon fraud (as supplied)", noon_all), ("My seed fraud", seed_fraud)]})
display(comp)
hard = checks.iloc[:13]
assert (hard == 0).all().all(), "A data-dictionary rule is violated"
print("Precision (decimal places) as supplied:")
display(C.precision_profile({"My CTGAN fraud": mine, "Noon fraud (as supplied)": noon_all, "My seed fraud": seed_fraud}).T)
print("[FINDING] All three samples satisfy every data-dictionary rule. The rows flagged 'outside assumed plausible range' are "
      "only relative to MY Phase 1 assumptions (e.g. delivery distance > 15 km), not errors.")

# %%
dup = pd.DataFrame([C.duplicate_profile(d, nm) for nm, d in
                    [("My CTGAN fraud", mine), ("Noon fraud (as supplied)", noon_all), ("My seed fraud", seed_fraud)]]).set_index("dataset")
display(dup.T)
dup.to_csv(P2 / "data_quality" / "B1_duplicate_profile.csv")
mult = C.multiplicity_table(noon_all)
fit = C.zero_truncated_poisson_fit(mult)
mt = pd.DataFrame({"observed distinct rows": mult, "expected under sampling with replacement": fit["expected"].round(1)})
mt.index.name = "copies of a row in Noon's file"
display(mt.T)
mt.to_csv(P2 / "data_quality" / "B2_noon_multiplicity.csv")
keys = C.row_keys(noon_all)
first_pos = keys.groupby(keys).apply(lambda s: s.index.to_numpy())
gaps = np.concatenate([np.diff(v) for v in first_pos if len(v) > 1])
overlap_mine = C.cross_duplicates(mine, noon_all)
overlap_seed = C.cross_duplicates(seed[FEATURES], noon_all)
print(f"[FINDING] Noon's file: {dup.loc['Noon fraud (as supplied)', 'rows']:,} rows, {dup.loc['Noon fraud (as supplied)', 'distinct rows (10 predictors)']:,} distinct rows, "
      f"{dup.loc['Noon fraud (as supplied)', 'duplicate rows (repeats beyond first copy)']:,} repeated rows "
      f"({dup.loc['Noon fraud (as supplied)', 'share of rows that are repeats']:.1%}); one row appears up to "
      f"{dup.loc['Noon fraud (as supplied)', 'max copies of one row']} times. Kish effective sample size ≈ "
      f"{dup.loc['Noon fraud (as supplied)', 'Kish effective sample size']:,.0f}.")
print(f"[FINDING] Copies of the same row are scattered through the file (median gap between copies {np.median(gaps):,.0f} rows; "
      f"adjacent copies {np.mean(gaps == 1):.1%}), so they are not stored in blocks.")
print(f"[FINDING] The multiplicity pattern matches 'drawn with replacement from ≈{fit['implied base rows N = rows/λ']:,.0f} base rows' "
      f"(zero-truncated Poisson, λ = {fit['lambda']:.2f}, χ² goodness-of-fit p = {fit['p (χ² goodness of fit)']:.2f}).")
print(f"[FINDING] My CTGAN file has {dup.loc['My CTGAN fraud', 'duplicate rows (repeats beyond first copy)']} duplicates. "
      f"Noon rows identical to a row of my CTGAN file: {overlap_mine}; identical to any row of my seed dataset: {overlap_seed}.")
P.p2_multiplicity(mult, fit["expected"], fit, NOON_COLOR, P2 / "data_quality" / "B3_noon_duplicate_multiplicity.png",
                  "Noon's file: how many times each distinct row appears")

# %% [markdown]
# **What the duplicates mean.**
#
# * **[FINDING]** Two thirds of Noon's rows are repeats. With continuous order amounts (1,021 distinct values in 3,000 rows), exact 10-column matches cannot arise by chance in genuine GAN sampling.
# * **[INTERPRETATION]** The repeat pattern fits a file built by drawing about 3,000 rows *with replacement* from a pool of roughly 1,100 base rows (whether GAN outputs or something else). This is consistent with the data but not proven, since Noon's generation pipeline is not documented.
# * **Implications for this analysis:**
#   1. Statistical tests assume independent rows. The 3,000 rows carry about as much information as ~800 independent ones (Kish effective sample size), so p-values on the as-supplied file would be too small. **The deduplicated rows are therefore the primary analysis set;** the as-supplied file is a sensitivity analysis.
#   2. In the lie detector, a Noon row whose identical copy sits in the training folds is recognised by memory, not by distribution, which inflates accuracy (quantified in Step 2.3).
#   3. Multiplicity re-weights Noon's distribution towards the repeated rows; every Step 2.1 statistic is shown for both versions so any conclusion that flips is visible.
#
# **Harmonisation for the primary analysis.** Noon's `delivery_distance_km` has 1 decimal and mine has 2, so mine is rounded to 1 decimal for the primary comparison and the lie detector. This removes a precision artefact without changing meaning, and the row counts after rounding are recomputed below. No other change is made.

# %%
def harmonise_precision(d):
    out = d.copy()
    out["delivery_distance_km"] = out["delivery_distance_km"].round(1)
    return out

mine_h, seed_h = harmonise_precision(mine), harmonise_precision(seed_fraud)
mine_u = mine_h.drop_duplicates(FEATURES).reset_index(drop=True)
noon_u = noon_all.drop_duplicates(FEATURES).reset_index(drop=True)
cross = C.cross_duplicates(mine_u, noon_u)
if cross:
    ku = set(C.row_keys(noon_u))
    mine_u = mine_u[~C.row_keys(mine_u).isin(ku)].reset_index(drop=True)
    noon_u = noon_u[~C.row_keys(noon_u).isin(set(C.row_keys(mine_h)))].reset_index(drop=True)
sets = pd.DataFrame({"rows as supplied": [len(mine), len(noon_all), len(seed_fraud)],
                     "rows after 1-dp distance harmonisation + deduplication": [len(mine_u), len(noon_u), len(seed_h.drop_duplicates(FEATURES))],
                     "cross-source overlaps removed": [cross, cross, 0]},
                    index=["My CTGAN fraud", "Noon fraud", "My seed fraud"])
display(sets)
sets.to_csv(P2 / "data_quality" / "B4_analysis_sets.csv")
noon_u.assign(copies_in_supplied_file=C.row_keys(noon_u).map(C.row_keys(noon_all).value_counts()).to_numpy()) \
    .to_csv(P2_DATA / "noon_unique_rows_with_multiplicity.csv", index=False)
N_MINE, N_NOON_U, N_NOON_ALL = len(mine_u), len(noon_u), len(noon_all)
NAMES = {"mine": "My CTGAN fraud", "noon": "Noon fraud (unique rows)", "seed": "My seed fraud"}
COLORS2 = {NAMES["mine"]: MY_GAN_COLOR, NAMES["noon"]: NOON_COLOR}
COLORS3 = {NAMES["seed"]: SEED_FRAUD_COLOR, NAMES["mine"]: MY_GAN_COLOR, NAMES["noon"]: NOON_COLOR}

# %% [markdown]
# ## Step 2.1 — Comparison EDA
#
# ### 2.1a The professor's `compare_distributions` (as specified, on the files as supplied)
#
# Run unmodified for consistency with the expected evaluation. Read it with the caveats from A2: separate bin edges per dataset, and a density y-axis (per AED, per day, per order) rather than a share of rows.

# %%
_show = plt.show
def _save_and_show(*a, **k):
    plt.gcf().savefig(P2 / "distributions" / "2.1a_helper_compare_distributions_as_supplied.png", dpi=150, bbox_inches="tight")
    plt.close("all")
plt.show = _save_and_show
H.compare_distributions(mine, noon_all)
plt.show = _show
display(Image(filename=str(P2 / "distributions" / "2.1a_helper_compare_distributions_as_supplied.png")))

# %% [markdown]
# ### 2.1b The three brief features, with shared bins and variable-appropriate charts
#
# `order_amount` is continuous (shared log-spaced bins). `account_age_days` and `num_orders_last_7d` are **integers** (data dictionary), so they are shown as shares per integer band or count, plus cumulative distributions (ECDFs), which need no binning. The ECDFs also show Noon's file as supplied (dashed) to make the effect of duplicates visible.

# %%
P.p2_brief_features(mine_u, noon_u, noon_all, COLORS2, P2 / "distributions" / "2.1b_brief_features.png",
                    "The three brief features: my CTGAN fraud vs Noon's fraud (unique rows; dashed = Noon as supplied)")

# %% [markdown]
# ### 2.1c Descriptive statistics (all features)

# %%
frames_desc = {NAMES["seed"]: seed_h, NAMES["mine"]: mine_u, NAMES["noon"]: noon_u, "Noon fraud (as supplied)": noon_all}
desc = C.describe_numeric(frames_desc)
piv = desc.pivot(index=["feature", "type"], columns="dataset", values=["mean", "median", "sd", "p10", "p90", "max"])
piv = piv.reindex(columns=list(frames_desc), level=1)
display(piv.style.format(precision=2).set_caption("Numeric features (units: AED, days, hours, orders, km, claims)"))
desc.to_csv(P2 / "distributions" / "2.1c_numeric_descriptives.csv", index=False)
shares = C.describe_shares(frames_desc)
sh = shares.pivot(index=["feature", "level"], columns="dataset", values="share")[list(frames_desc)]
display(sh.style.format("{:.1%}").set_caption("Categorical and binary shares (denominator: rows of each dataset)"))
shares.to_csv(P2 / "distributions" / "2.1c_shares_with_wilson_ci.csv", index=False)
extra = pd.DataFrame({nm: {"account_age_days = 0 (%)": (d.account_age_days == 0).mean() * 100,
                           "account_age_days ≤ 7 (%)": (d.account_age_days <= 7).mean() * 100,
                           "account_age_days > 365 (%)": (d.account_age_days > 365).mean() * 100,
                           "orders placed 00:00–05:59 (%)": d.hour_of_day.between(0, 5).mean() * 100,
                           "delivery_distance_km > 15 (%)": (d.delivery_distance_km > 15).mean() * 100,
                           "order_amount > AED 300 (%)": (d.order_amount > 300).mean() * 100}
                      for nm, d in frames_desc.items()})
display(extra.style.format("{:.1f}").set_caption("Selected tail and threshold shares"))
extra.to_csv(P2 / "distributions" / "2.1c_threshold_shares.csv")

# %%
P.p2_other_features(mine_u, noon_u, COLORS2, P2 / "distributions" / "2.1d_other_features.png",
                    "Remaining features: my CTGAN fraud vs Noon's fraud (unique rows)")

# %% [markdown]
# ### 2.1e Effect sizes and tests
#
# **Primary (unique rows).** Continuous features and integer features (account age, orders, refunds) use the KS distance D, with a **permutation p-value** that stays exact when integer data create ties, plus the Wasserstein-1 distance in the feature's own units. `hour_of_day` is cyclic, so it uses the total-variation distance (TVD) over the 24 hours with a χ² test. Categorical and binary features use TVD (for a binary flag, the absolute difference in shares) and Cramér's V with a χ² test. Holm-adjusted across the 10 features.
#
# **Sensitivity (as supplied).** Effect sizes only. P-values are not computed, because duplicated rows break the independence those tests assume.
#
# The bands (negligible < 0.05 ≤ small < 0.15 ≤ moderate < 0.30 ≤ large) are a judgement convention.

# %%
eff_main = C.compare_features(mine_u, noon_u, NAMES["mine"], NAMES["noon"], seed=SEED)
eff_seed = C.compare_features(seed_h, noon_u, NAMES["seed"], NAMES["noon"], seed=SEED)
eff_ref = C.compare_features(seed_h, mine_u, NAMES["seed"], NAMES["mine"], seed=SEED)
eff_sup = C.compare_features(mine_h, noon_all, NAMES["mine"], "Noon fraud (as supplied)", n_perm=0, seed=SEED)
eff_sup[["p-value", "Holm-adjusted p"]] = np.nan
eff = pd.concat([eff_main, eff_seed, eff_ref], ignore_index=True)
cols = ["pair", "feature", "type", "headline effect", "effect size", "magnitude (judgement band)", "Holm-adjusted p",
        "Wasserstein-1 (units)", "Cramér V", "median A", "median B", "mean A", "mean B"]
display(eff[cols].style.format(precision=3).format("{:.1e}", subset=["Holm-adjusted p"]))
cmp_sup = eff_main.set_index("feature")[["effect size", "magnitude (judgement band)"]].join(
    eff_sup.set_index("feature")[["effect size", "magnitude (judgement band)"]], lsuffix=" — unique (primary)", rsuffix=" — as supplied")
display(cmp_sup.style.format(precision=3, subset=[c for c in cmp_sup.columns if "size" in c]))
flips = cmp_sup[cmp_sup.iloc[:, 1] != cmp_sup.iloc[:, 3]]
print(f"[FINDING] Features whose magnitude band changes when duplicates are kept: {list(flips.index) or 'none'}")
pd.concat([eff, eff_sup], ignore_index=True).to_csv(P2 / "distributions" / "2.1e_effect_sizes_and_tests.csv", index=False)
PAIR_COLORS = {f"{NAMES['mine']} vs {NAMES['noon']}": MY_GAN_COLOR, f"{NAMES['seed']} vs {NAMES['noon']}": SEED_FRAUD_COLOR,
               f"{NAMES['seed']} vs {NAMES['mine']}": PAIR_REF_COLOR}
P.p2_effect_chart(eff, PAIR_COLORS, eff_sup, P2 / "distributions" / "2.1e_effect_sizes.png",
                  "How different is each feature? (bars: unique rows; diamonds: Noon as supplied)")

# %% [markdown]
# ### 2.1f Relationships between features
#
# Fraud is a pattern of combinations. Three views follow: conditional profiles (e.g. device-change rate by account age), correlation differences, and the share of rows matching each of my five Phase 1 typology signatures. The signatures are my own definitions, a lens built on my assumptions, not Noon's labels.

# %%
rel = C.relationship_profiles({NAMES["seed"]: seed_h, NAMES["mine"]: mine_u, NAMES["noon"]: noon_u})
rel.to_csv(P2 / "relationships" / "2.1f_relationship_profiles.csv", index=False)
P.relationship_panels(rel, COLORS3, P2 / "relationships" / "2.1f_relationship_profiles.png",
                      "Conditional relationship profiles: my seed fraud, my CTGAN fraud and Noon's fraud")
corr_tab = pd.DataFrame([{"pair": f"{a} vs {b}", "mean |Δρ| over feature pairs": C.mean_abs_corr_diff(da, db)}
                         for (a, da), (b, db) in [((NAMES["mine"], mine_u), (NAMES["noon"], noon_u)),
                                                  ((NAMES["seed"], seed_h), (NAMES["noon"], noon_u)),
                                                  ((NAMES["seed"], seed_h), (NAMES["mine"], mine_u))]])
display(corr_tab.style.format(precision=3))
top_corr = C.top_corr_differences(mine_u, noon_u, NAMES["mine"], NAMES["noon"])
display(top_corr.style.format(precision=3).set_caption("Largest Spearman-correlation differences (my CTGAN vs Noon unique)"))
corr_tab.to_csv(P2 / "relationships" / "2.1f_correlation_summary.csv", index=False)
top_corr.to_csv(P2 / "relationships" / "2.1f_top_correlation_differences.csv", index=False)
sig = G.signature_shares({NAMES["seed"]: seed_h, NAMES["mine"]: mine_u, NAMES["noon"]: noon_u})
display(sig.style.format("{:.1%}"))
sig.to_csv(P2 / "relationships" / "2.1f_typology_signatures.csv")
P.grouped_bars(sig, COLORS3, P2 / "relationships" / "2.1f_typology_signatures.png",
               "Share of rows matching each of my five Phase 1 fraud-typology signatures (my lens, not Noon's labels)",
               "Share of rows matching the signature (%)", "Typology signature (joint condition on 3 columns)", fmt="{:.1%}", figsize=(15, 5.8))

# %%
e = eff_main.sort_values("effect size", ascending=False)
print("[FINDING] Largest differences, my CTGAN fraud vs Noon fraud (unique rows):")
for _, r in e.iterrows():
    print(f"  • {r.feature:22s} {r['headline effect']:15s} {r['effect size']:.3f} ({r['magnitude (judgement band)']}), Holm p = {r['Holm-adjusted p']:.1e}")

# %% [markdown]
# ### Interpretation of Step 2.1
#
# *Figures below are from the unique-row analysis unless marked "as supplied"; every number is in the CSV files in `outputs/phase2/distributions/` and `outputs/phase2/relationships/`.*
#
# **Observed — the three features the brief names.**
# * **`order_amount`: small difference (KS D = 0.122).** Both files have almost the same mean (AED 134.0 mine vs 134.9 Noon) but are shaped differently. Noon's median is higher (114 vs 87), and 24.5% of its rows sit between AED 200 and 300 (mine: 9.9%). My CTGAN has the heavier upper tail: 11.7% of my rows are above AED 300 vs 4.0% of Noon's, and my maximum is 649 vs Noon's 470.
# * **`account_age_days`: large difference (KS D = 0.327), the biggest of the three.** Treated as a whole-day ordered variable, as the user instructed. 31.0% of my CTGAN rows are 0-day accounts, and Noon has none (its minimum is 1 day). Only 3.0% of Noon's rows are ≤ 7 days old (mine: 35.6%). Noon's median is 147 days (mine 94), and Noon has no account older than 1,107 days.
# * **`num_orders_last_7d`: moderate difference (KS D = 0.250).** Noon's fraudsters are busier (median 4 vs 2 orders; mean 3.95 vs 2.69). Only 4.2% of Noon's rows have zero orders (mine 21.9%), and Noon has a spike at exactly 7 orders: 15.5% vs 1.8%.
#
# **Observed — the other features.**
# * **`hour_of_day`: the largest single difference (TVD = 0.503).** 60.1% of Noon's fraud is placed between 00:00 and 05:59. Each hour from 00 to 04 holds 10.5–12.6%, and every other hour holds only about 1.5–3%. My CTGAN puts 11.9% of rows in that window and peaks at lunch (12:00, 11.9%) and in the evening.
# * **Binary flags.**
#   * `location_mismatch` differs strongly (TVD 0.395): 62.9% of Noon's rows vs 23.4% of mine.
#   * `device_change` also differs strongly (TVD 0.338): 65.6% vs 31.9%.
#   * `discount_applied` is close (43.0% vs 50.3%).
# * **`payment_method`: moderate difference (TVD 0.199).** Noon has more wallet (29.6% vs 17.1%) and cash-on-delivery (16.8% vs 9.5%) fraud, and less Apple Pay (10.0% vs 22.6%).
# * **`delivery_distance_km`: moderate difference (KS D 0.185).** Noon's median is 4.3 km vs 3.0, and 1.4% of Noon's rows exceed 15 km, which never happens in mine.
# * **`prev_refunds`: small difference (KS D 0.139).** Noon only uses 0–3 refunds, while 10.8% of my rows have more than 3.
#
# **Observed — relationships between features.**
# * In Noon's file, location mismatch is about 62–64% for **every** payment method, including cash on delivery. In mine it is strongly tied to the payment method: 31.7% for card but 9.0% for COD.
# * Noon's device-change rate rises with account age, from 35.5% at 0–7 days to about 77% from 31 days onward. Mine stays between 25% and 48%.
# * Noon's night-time share is high in every age band (45–65%).
# * Noon's mean refunds stay flat (0.77–0.79) across order frequency, falling to 0.58 for 6+ orders. In mine, refunds rise from 0.32 to 1.49 as order frequency rises.
# * The largest correlation differences reverse sign:
#   * `device_change × location_mismatch`: ρ = +0.33 in mine vs −0.16 in Noon's;
#   * `account_age_days × location_mismatch`: −0.19 vs +0.29.
# * Mean |Δρ| is 0.141 for my CTGAN vs Noon, 0.111 for my seed vs Noon, and 0.079 for my seed vs my CTGAN.
#
# **Observed — duplicates do not change the picture.** Keeping Noon's duplicates moves every effect size by at most 0.009, and no feature changes magnitude band. My **seed** fraud differs from Noon's about as much as my CTGAN does, and on some features more (account age 0.442, device change 0.455, orders 0.381). So the gap is mainly between my *Phase 1 assumptions* and Noon's data, not an artefact of my GAN.
#
# **Interpretation (not proven by these data).**
# * Noon's fraud looks like an **account-takeover and night-time pattern**: established accounts (median 147 days) suddenly ordering from a new device at a mismatched location in the small hours, with frequent recent orders.
# * My Phase 1 design gave most weight to **brand-new-account promo abuse** (typology signature 22.9% in my seed vs 1.2% in Noon's). My CTGAN then added the 0-day spike.
# * The near-uniform blocks in Noon's data look like rule-based generation or a model trained on rule-based data rather than on organic behaviour:
#   * hours 00–04 at about 11–12% each, then about 2% per hour;
#   * mismatch at a constant level across payment methods;
#   * refunds limited to 0–3.
# * This is a hypothesis. Noon's file is a synthetic benchmark sample, not real fraud, so it shows how *Noon's* generator sees fraud, not what real fraud looks like. Which picture is closer to reality can only be judged against the real referee data in Phase 3.

# %% [markdown]
# ## Step 2.2 — Fraud proportion, with every denominator explicit

# %%
p1 = json.loads((OUTPUTS / "metrics" / "phase1_metrics.json").read_text())
n_train = len(split["train"])
n_train_fraud = int(seed.iloc[split["train"]][TARGET].sum())
ratio_used = p1["augmentation_ratios"]["B_ctgan"]
rows = [
    ("1", "My complete Phase 1 seed dataset", "fraud rows / all seed rows", int(seed[TARGET].sum()), len(seed)),
    ("1", "My seed training partition", "fraud rows / training rows", n_train_fraud, n_train),
    ("2", "GAN generation volume relative to real training fraud (volume ratio, NOT a fraud prevalence)", "GAN rows / real training fraud rows", len(mine), n_train_fraud),
    ("2", "My GAN-generated fraud file vs ALL training rows", "GAN rows / training rows", len(mine), n_train),
    ("2", f"GAN rows actually added in Phase 1 (selected ratio {ratio_used}×)", "GAN rows used / training fraud rows",
     int(round(ratio_used * n_train_fraud)), n_train_fraud),
]
for r_ in AUG_RATIOS:
    add = int(round(r_ * n_train_fraud))
    rows.append(("3", f"Augmented training set at {r_}×{' (selected)' if r_ == ratio_used else ''}",
                 "(training fraud + GAN rows) / (training rows + GAN rows)", n_train_fraud + add, n_train + add))
rows += [
    ("4", "Noon's synthetic fraud file — as supplied", "is_fraud = 1 rows / all rows in the file", int(noon_all[TARGET].sum()), len(noon_all)),
    ("4", "Noon's synthetic fraud file — distinct rows", "is_fraud = 1 distinct rows / distinct rows", int(noon_u[TARGET].sum()), len(noon_u)),
    ("context", "Real referee test set (from the A3 integrity check; NOT Noon's synthetic file)", "fraud rows / all referee rows",
     REFEREE_FRAUD, REFEREE_N),
]
prop = pd.DataFrame(rows, columns=["item (brief §2.2)", "population", "denominator", "numerator", "denominator count"])
prop["proportion"] = prop["numerator"] / prop["denominator count"]
display(prop.style.format({"proportion": "{:.3%}", "numerator": "{:,}", "denominator count": "{:,}"}))
prop.to_csv(P2 / "fraud_proportion" / "2.2_fraud_proportion_denominators.csv", index=False)

# %% [markdown]
# **Reading Step 2.2.**
#
# * **Both synthetic fraud files are fraud-only.** Noon's file has `is_fraud = 1` in every row (100% within the file, whether counted as supplied or as distinct rows), and so does my CTGAN file. Comparing their "fraud proportions" is therefore degenerate (100% vs 100%) and says nothing about which is more realistic, or about Noon's transaction-level fraud rate, which is **not** 100%.
# * **The benchmark's "~0.3%" belongs to the real referee test set**, not to Noon's synthetic file. A3 confirms it from the file itself (shown in the table, label counts only).
# * **What *can* be compared is how much synthetic fraud each approach adds to a training set.** In Phase 1, the selected augmentation added a quarter of my training fraud count. The benchmark's warning — "If you over-generated fraud, [false positives rise]" — is about this training composition, not about the composition of the synthetic files.
# * **Missing denominators.** Noon has not supplied how many real transactions or real fraud rows Model B was trained on, nor how much of its 3,000-row (≈1,033 distinct) synthetic fraud it added. Noon's augmented training composition therefore cannot be computed, and it is not estimated here.

# %% [markdown]
# ## Step 2.3 — The lie-detector test
#
# The classifier tries to tell *which source* a fraud row came from (mine = 0, Noon = 1); it does not predict fraud. In every protocol, `is_fraud` is excluded (it is 1 everywhere and is not in the helper's feature list); features and categorical encoding come from the helper's own `_prep` (9 numeric columns + one-hot `payment_method`); and the model is the helper's `RandomForestClassifier(n_estimators=100, random_state=0)`.
#
# | Protocol | Rows | Sizes | Validation | Statistic | Role |
# |---|---|---|---|---|---|
# | **Helper (as specified)** | As supplied | Unequal | `StratifiedKFold(3)`, no shuffle | Accuracy | Professor's comparison, reported unchanged |
# | **Primary** | Unique rows, overlaps removed, distance at 1 dp | Equal (mine subsampled), 10 repeats | Shuffled `StratifiedKFold(5)` | **ROC-AUC**; balanced accuracy; accuracy | Main result; permutation p-value (1,000 permutations, **no groups**) |
# | **Sensitivity — grouped** | As supplied (duplicates kept) | Unequal | Shuffled `StratifiedGroupKFold(5)` keyed on row identity | ROC-AUC; balanced accuracy; accuracy vs majority baseline | Duplicates kept but never straddle folds; **no p-value** (a row-grouped permutation would be degenerate) |
# | **Leakage illustration** | As supplied | Unequal | Shuffled `StratifiedKFold(5)`, no groups | Same | Shows how much duplicate copies inflate the score |

# %%
t_ = time.time()
acc_helper = H.lie_detector(mine, noon_all)
base_helper = max(len(mine), len(noon_all)) / (len(mine) + len(noon_all))
leak = C.helper_fold_leakage(mine, noon_all)
print(f"Helper accuracy {acc_helper:.4f} · majority-class baseline for these sizes ({len(mine):,} vs {len(noon_all):,}) = {base_helper:.4f} · {time.time() - t_:.0f}s")
display(pd.Series(leak).to_frame("helper's own folds (StratifiedKFold(3), unshuffled)"))

# %%
t_ = time.time()
prim = C.lie_detector_primary(mine_u, noon_u, H._prep, SEEDS)
grp = C.lie_detector_as_supplied(mine_h, noon_all, H._prep, SEEDS, grouped=True)
ungrp = C.lie_detector_as_supplied(mine_h, noon_all, H._prep, SEEDS, grouped=False)
for nm, d in (("PRIMARY — unique rows, balanced", prim), ("SENSITIVITY — as supplied, grouped by row", grp),
              ("LEAKAGE ILLUSTRATION — as supplied, ungrouped", ungrp)):
    print(nm); display(d.describe().loc[["mean", "std", "min", "max"]].drop(columns="seed").style.format(precision=4))
prim.to_csv(P2 / "lie_detector" / "2.3_primary_unique_balanced.csv", index=False)
grp.to_csv(P2 / "lie_detector" / "2.3_sensitivity_as_supplied_grouped.csv", index=False)
ungrp.to_csv(P2 / "lie_detector" / "2.3_leakage_as_supplied_ungrouped.csv", index=False)
print(f"{time.time() - t_:.0f}s")

# %%
t_ = time.time()
perm = C.permutation_test(mine_u, noon_u, H._prep, seed=0, n_perm=N_PERM)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in perm.items() if k != "null"}, f"· {time.time() - t_:.0f}s")
imp, ctrl = C.importance_and_control(mine_u, noon_u, H._prep, seed=0)
display(imp.style.format(precision=4))
display(pd.Series(ctrl).to_frame("primary sample (seed 0)").style.format(precision=4))
imp.to_csv(P2 / "lie_detector" / "2.3_permutation_importance.csv")
pd.DataFrame([{k: v for k, v in perm.items() if k != "null"}]).to_csv(P2 / "lie_detector" / "2.3_permutation_test.csv", index=False)
pd.Series(ctrl).to_csv(P2 / "lie_detector" / "2.3_marginals_only_control.csv")
P.p2_perm_and_importance(perm, imp, NOON_COLOR, P2 / "lie_detector" / "2.3_permutation_test_and_importance.png",
                         "Lie detector (primary): is the separation real, and which features drive it?")

# %%
ref_seed_noon = C.lie_detector_primary(seed_h.drop_duplicates(FEATURES), noon_u, H._prep, SEEDS)
ref_seed_mine = C.lie_detector_primary(seed_h.drop_duplicates(FEATURES), mine_u, H._prep, SEEDS)
summ = pd.DataFrame([
    {"protocol": "Helper as specified\n(as supplied, cv=3 unshuffled)", "n (source 0 / Noon or 1)": f"{len(mine):,} / {len(noon_all):,}",
     "accuracy": acc_helper, "ROC-AUC": np.nan, "ROC-AUC sd": np.nan, "balanced accuracy": np.nan, "majority baseline": base_helper, "color": NOON_COLOR},
    {"protocol": "PRIMARY\n(unique rows, balanced, 10 repeats)", "n (source 0 / Noon or 1)": f"{N_NOON_U:,} / {N_NOON_U:,}",
     "accuracy": prim.accuracy.mean(), "ROC-AUC": prim["ROC-AUC"].mean(), "ROC-AUC sd": prim["ROC-AUC"].std(),
     "balanced accuracy": prim["balanced accuracy"].mean(), "majority baseline": 0.5, "color": MY_GAN_COLOR},
    {"protocol": "Sensitivity: as supplied,\ngrouped by row", "n (source 0 / Noon or 1)": f"{len(mine_h):,} / {len(noon_all):,}",
     "accuracy": grp.accuracy.mean(), "ROC-AUC": grp["ROC-AUC"].mean(), "ROC-AUC sd": grp["ROC-AUC"].std(),
     "balanced accuracy": grp["balanced accuracy"].mean(), "majority baseline": grp["majority baseline"].iloc[0], "color": "#5B8C2A"},
    {"protocol": "Leakage illustration:\nas supplied, ungrouped", "n (source 0 / Noon or 1)": f"{len(mine_h):,} / {len(noon_all):,}",
     "accuracy": ungrp.accuracy.mean(), "ROC-AUC": ungrp["ROC-AUC"].mean(), "ROC-AUC sd": ungrp["ROC-AUC"].std(),
     "balanced accuracy": ungrp["balanced accuracy"].mean(), "majority baseline": ungrp["majority baseline"].iloc[0], "color": "#9AA3B2"},
    {"protocol": "Reference: MY SEED fraud\nvs Noon (primary protocol)", "n (source 0 / Noon or 1)": f"{ref_seed_noon['n per source'].iloc[0]:,} each",
     "accuracy": ref_seed_noon.accuracy.mean(), "ROC-AUC": ref_seed_noon["ROC-AUC"].mean(), "ROC-AUC sd": ref_seed_noon["ROC-AUC"].std(),
     "balanced accuracy": ref_seed_noon["balanced accuracy"].mean(), "majority baseline": 0.5, "color": SEED_FRAUD_COLOR},
    {"protocol": "Reference: my seed fraud\nvs MY CTGAN (primary protocol)", "n (source 0 / Noon or 1)": f"{ref_seed_mine['n per source'].iloc[0]:,} each",
     "accuracy": ref_seed_mine.accuracy.mean(), "ROC-AUC": ref_seed_mine["ROC-AUC"].mean(), "ROC-AUC sd": ref_seed_mine["ROC-AUC"].std(),
     "balanced accuracy": ref_seed_mine["balanced accuracy"].mean(), "majority baseline": 0.5, "color": PAIR_REF_COLOR},
])
display(summ.drop(columns="color").style.format(precision=4, na_rep="—"))
summ.drop(columns="color").to_csv(P2 / "lie_detector" / "2.3_protocol_summary.csv", index=False)
P.p2_lie_detector(summ, P2 / "lie_detector" / "2.3_lie_detector_protocols.png",
                  "Lie detector: can a classifier tell my fraud from Noon's? (amber = accuracy baseline)")

# %% [markdown]
# ### Interpretation of Step 2.3
#
# **Observed.**
#
# | Protocol | Accuracy | ROC-AUC (mean ± SD over 10 repeats) | Balanced accuracy | Accuracy of "always guess the larger source" |
# |---|---|---|---|---|
# | Helper as specified (as supplied, 3 unshuffled folds) | 0.960 | not computed by the helper | not computed | 0.586 |
# | **Primary** (unique rows, 1,033 per source, shuffled 5-fold) | 0.920 | **0.974 ± 0.003** | 0.920 | 0.500 |
# | Sensitivity: as supplied, grouped by row | 0.919 | 0.975 ± 0.001 | 0.920 | 0.586 |
# | Leakage illustration: as supplied, ungrouped | 0.961 | 0.996 ± 0.000 | 0.955 | 0.586 |
# | Reference: my seed fraud vs Noon (primary protocol) | 0.928 | 0.977 ± 0.001 | 0.928 | 0.500 |
# | Reference: my seed fraud vs my CTGAN (primary protocol) | 0.717 | 0.791 ± 0.006 | 0.717 | 0.500 |
#
# * **Permutation test (primary only).** Observed mean cross-validated ROC-AUC is 0.976 on the seed-0 sample. Across 1,000 label permutations without groups, the null distribution has mean 0.500 and maximum 0.550, so p = 1/1,001 ≈ 0.001. That is the smallest p-value 1,000 permutations can produce, so the true p is at most about 0.001.
# * **Which features give the source away** (held-out permutation importance, drop in ROC-AUC):
#   * `account_age_days` 0.096;
#   * `location_mismatch` 0.068;
#   * `hour_of_day` 0.042;
#   * `device_change` 0.023;
#   * every other feature ≤ 0.014.
# * **Marginals-only control.** Shuffling each column within its source keeps each feature's distribution but destroys the combinations. This lowers the AUC only from 0.976 to 0.949, so about 0.026 of AUC depends on how features combine; nearly all the separation comes from single features.
#
# **Why the protocols differ, and which number to use.**
# * **The helper's 0.96 is inflated, and its own printout misleads.**
#   1. Noon's file contains each of its 1,033 distinct rows about three times on average. In the helper's own unshuffled 3-fold split, 2,500 of the 3,000 Noon test rows (83.3%) have an identical copy in the training folds, so part of the score is recall of seen rows.
#   2. The two files have unequal sizes (2,116 vs 3,000), so always guessing "Noon" already scores 0.586, not 0.50. The helper's message "0.50 = your fraud looks just like Noon's" is therefore wrong for these files.
#   3. Accuracy depends on a 0.5 threshold. ROC-AUC does not, and its chance level is 0.5 whatever the class sizes.
# * **Primary (the headline).** Deduplicating both files, removing cross-source overlaps (there were none) and balancing to 1,033 rows each removes all three problems. Repeated rows cannot leak, chance is exactly 0.5 for both accuracy and AUC, and the rows are close enough to exchangeable for a standard, ungrouped permutation p-value to be valid.
# * **Sensitivity, grouped (duplicates kept).** Folds are split by row identity, so every copy of a row stays on one side. The result matches the primary (0.975 vs 0.974), which shows deduplication did not create the result.
#   * As instructed, **no p-value** is reported here. Permuting labels while keeping row groups would be degenerate, because each group contains only one source.
#   * Its accuracy must be read against its own baseline of 0.586.
# * **Leakage illustration.** The same as-supplied data with ungrouped folds gives 0.996, which is 0.021 higher than grouped. That gap is pure duplicate leakage; it is not information about fraud.
#
# **What this means.**
# * A classifier tells my fraud from Noon's about 92% of the time on a balanced sample (AUC 0.974), so the two generators produce **very different** fraud, well above the 0.90 "very different" mark the helper's own printout uses.
# * The difference is not caused by my GAN's imperfections. My **seed** fraud is just as easy to tell apart from Noon's (AUC 0.977), while my CTGAN is much harder to tell apart from its own training data (AUC 0.791). The separation comes from different **assumptions about what fraud looks like**.
# * The classifier relies on the same features as the effect sizes: account age, location mismatch, hour and device change.
#
# **Limits of this test.**
# * An AUC near 0.5 would only mean this classifier could not tell the files apart. It would not prove they come from the same distribution, and it would not show that either looks like real fraud.
# * A high AUC shows the files are different. It does **not** show which is more realistic.
# * The test uses one model family (the helper's random forest), so other models could separate the files differently.
# * Noon's sample is smaller than its 3,000 rows suggest. It has 1,033 distinct rows, and because the copies are uneven, the 3,000 rows carry about as much information as 813 equally weighted rows (Kish effective sample size).

# %% [markdown]
# ## Step 2.4 — Reflection: my Phase 1 assumptions vs Noon's evidence
#
# The Phase 1 assumptions (`reports/assumptions_register.md`) are preserved unchanged; this section records how my understanding has moved. **Observed evidence** and **interpretation** are kept apart, and Noon's sample is a benchmark, not ground truth.

# %%
a_vs_n = pd.DataFrame({
    "my seed fraud": [seed_h.order_amount.median(), seed_h.account_age_days.median(), (seed_h.account_age_days <= 7).mean(),
                      seed_h.hour_of_day.between(0, 5).mean(), seed_h.num_orders_last_7d.mean(), seed_h.device_change.mean(),
                      seed_h.location_mismatch.mean(), seed_h.discount_applied.mean(), seed_h.prev_refunds.mean(),
                      (seed_h.payment_method == "cod").mean(), seed_h.delivery_distance_km.median()],
    "my CTGAN fraud": [mine_u.order_amount.median(), mine_u.account_age_days.median(), (mine_u.account_age_days <= 7).mean(),
                       mine_u.hour_of_day.between(0, 5).mean(), mine_u.num_orders_last_7d.mean(), mine_u.device_change.mean(),
                       mine_u.location_mismatch.mean(), mine_u.discount_applied.mean(), mine_u.prev_refunds.mean(),
                       (mine_u.payment_method == "cod").mean(), mine_u.delivery_distance_km.median()],
    "Noon fraud (unique)": [noon_u.order_amount.median(), noon_u.account_age_days.median(), (noon_u.account_age_days <= 7).mean(),
                            noon_u.hour_of_day.between(0, 5).mean(), noon_u.num_orders_last_7d.mean(), noon_u.device_change.mean(),
                            noon_u.location_mismatch.mean(), noon_u.discount_applied.mean(), noon_u.prev_refunds.mean(),
                            (noon_u.payment_method == "cod").mean(), noon_u.delivery_distance_km.median()]},
    index=["median order amount (AED)", "median account age (days)", "share of accounts ≤ 7 days", "share ordered 00:00–05:59",
           "mean orders in last 7 days", "share with device change", "share with location mismatch", "share with discount",
           "mean previous refunds", "share paid cash on delivery", "median delivery distance (km)"])
display(a_vs_n.style.format(precision=3))
a_vs_n.to_csv(P2 / "distributions" / "2.4_assumption_evidence_table.csv")

# %% [markdown]
# ### Reflection: seven questions
#
# Each answer separates **Observed** (computed above; Noon figures are unique rows) from **Interpretation**. Assumption IDs (H1–H7) and feature rows refer to `reports/assumptions_register.md`, which is unchanged.
#
# **1. What did my original GAN capture correctly?**
# * *Observed — against its own training data, the seed fraud:*
#   * `discount_applied` and `location_mismatch` were reproduced almost exactly (TVD 0.004 and 0.012);
#   * `payment_method` was close (TVD 0.053);
#   * a classifier separates CTGAN rows from seed rows only moderately (AUC 0.791), compared with 0.974 for CTGAN vs Noon.
# * *Observed — against Noon's file:*
#   * mean order amount is nearly identical (AED 134.0 vs 134.9), with a right-skewed amount distribution in both;
#   * the discount rate is close (50.3% vs 43.0%), and in both files it falls with account age (0–7 days: 77% mine vs 84% Noon; 181–730 days: 29% vs 37%);
#   * refund counts differ only a little (KS D 0.139), and the refund-abuse signature has a similar share (17.8% vs 21.7%).
# * *Interpretation.* The CTGAN learned my seed reasonably well. Where it agrees with Noon, it is mostly because my seed already agreed with Noon on those features.
#
# **2. Where do my synthetic fraud examples differ from Noon's?**
# * *Observed* (largest first):
#   * hour of day (TVD 0.503; night share 11.9% vs 60.1%);
#   * location mismatch (0.395; 23.4% vs 62.9%);
#   * device change (0.338; 31.9% vs 65.6%);
#   * account age (KS D 0.327; 0-day accounts 31.0% vs 0%; median 94 vs 147 days);
#   * orders in the last 7 days (0.250; median 2 vs 4);
#   * payment method (0.199);
#   * delivery distance (0.185).
#   A classifier separates the two files with AUC 0.974.
# * *Interpretation.* The two generators encode different ideas of what fraud looks like. My **seed** differs from Noon's by about the same amount (AUC 0.977), so the gap began with my Phase 1 assumptions, not with my GAN.
#
# **3. Which variables or combinations are represented differently?**
# * *Observed:*
#   * **Location mismatch × payment method**: about 62–64% for every method in Noon's file, including cash on delivery (COD 64% vs 9% mine).
#   * **Device change × account age**: rises to about 77% from 31 days in Noon's file; 25–48% in mine.
#   * **Device change × location mismatch**: correlation +0.33 in mine vs −0.16 in Noon's.
#   * **Account age × location mismatch**: −0.19 vs +0.29.
#   * **Refunds × order frequency**: refunds rise with frequency in mine (0.32 → 1.49) but are flat in Noon's (≈ 0.78, 0.58 for 6+ orders).
#   * **Night ordering**: high in every age band in Noon's file (45–65%), 9–13% in mine.
# * *Interpretation.* The marginals-only control shows nearly all of the separation comes from single features (only 0.026 AUC from combinations). The combination differences above are real, but they are secondary to the differences in single features.
#
# **4. Which Phase 1 assumptions appear to be supported?**
# * **Second mode of H1: takeover of established accounts with device change.** *Observed:* Noon's fraud is mostly established accounts (median 147 days; 17.0% older than a year), and 77% of accounts older than 30 days show a device change. Rows matching my takeover signature are 22.7% of Noon's file.
#   * *Interpretation:* the takeover typology I designed exists and seems more common in Noon's view than in mine (3.7% in my seed).
# * **Fraud baskets above the assumed legitimate median (AED 62).** *Observed:* Noon's fraud median is AED 114.
#   * *Caveat:* Noon supplied no legitimate rows, so this is only weak support.
# * **Discount use concentrated on young accounts.** Observed in both files (see Q1).
# * **Refund abuse as a real typology.** *Observed:* 44.6% of Noon's distinct rows have at least one previous refund, and 21.7% match my refund-abuse signature.
#
# **5. Which assumptions appear incomplete or questionable?**
# * **H1's new-account mode.** Promo abuse accounts were assumed to be about 2.5 days old, and COD fakes about 5 days. *Observed:* only 3.0% of Noon's rows are ≤ 7 days old and none are 0 days; the promo-abuse signature is 1.2% (seed 22.9%).
# * **H6 — night ordering as a deliberately weak signal.** *Observed:* 60.1% of Noon's fraud happens between 00:00 and 05:59.
# * **Mismatch tied to payment method** (COD mismatch ≈ 2%, stolen card 55%) and **H3 — device change and mismatch occurring together.** *Observed:* Noon's mismatch is flat across payment methods, and device change and mismatch are slightly negatively correlated.
# * **Refunds accumulating with order frequency, up to 9 refunds.** *Observed:* Noon's refunds are flat, and its maximum is 3.
# * **Self-imposed range caps** (distance ≤ 15 km). *Observed:* 1.4% of Noon's rows exceed 15 km (maximum 27.2).
# * **H7 — Noon's fraud resembles my mixture.** Not supported by Noon's synthetic sample.
# * *Interpretation:* these are the assumptions to re-test on real referee data in Phase 3. Noon's file is one generator's output, possibly shaped by rules, so it **challenges** these assumptions without refuting them.
#
# **6. What did Noon's examples reveal that my approach did not capture?**
# * *Observed:*
#   * a concentrated night-time block (00–04, about 11–12% per hour);
#   * mismatch common across all payment methods, including cash;
#   * busier fraudster accounts, with a spike at exactly 7 orders in the last 7 days (15.5%);
#   * longer delivery distances;
#   * almost no brand-new accounts.
# * *Also observed — a data-quality lesson.* 65.6% of Noon's rows are repeats (1,033 distinct of 3,000). Their pattern fits sampling with replacement from about 1,107 base rows (χ² p = 0.96). Without deduplication, the helper's lie detector reports an inflated 0.96 accuracy.
# * *Interpretation:* my Phase 1 research leaned on new-account promo abuse and card-not-present patterns. Noon's generator emphasises account takeover at night.
#
# **7. What would I improve, and why?**
# Phase 1 stays frozen; nothing below is applied in Phase 2.
# * **Generator fidelity (independent of Noon).**
#   * Treat `account_age_days` as a discrete, heavy-tailed variable, for example by log-transforming it before CTGAN or using conditional sampling, so the GAN stops inventing the 0-day spike (31.0% vs 13.8% in its training data).
#   * Then re-check the hour shift and the over-production of device change.
# * **Widen the seed priors instead of copying Noon.** Run scenarios with a stronger night uplift, mismatch that does not depend on payment method, a larger share of established-account takeover, and a distance cap above 15 km. Each scenario should be **labelled as a sensitivity analysis**; my original assumptions stay as they were.
# * **Let real data decide.** Phase 3's referee results should determine which picture is closer to reality. Changes made now to match Noon's synthetic file would be tuning to a benchmark, which the instructions forbid.
# * **Keep the comparison method.** Deduplicate before any two-sample test, report ROC-AUC against the correct chance level, and keep duplicate-aware sensitivity runs.

# %% [markdown]
# ## Quality control (Part 5 checklist, automated)

# %%
audit = []
def chk(name, ok, detail=""):
    audit.append({"check": name, "result": "Passed" if ok else "FAILED", "detail": detail})

chk("Phase 1 fraud-only sample correctly identified (synthetic_fraud_gan.csv: all is_fraud = 1, GAN output)",
    bool((mine[TARGET] == 1).all()), f"{len(mine):,} rows")
chk("Noon sample interpreted as fraud-only (all is_fraud = 1)", bool((noon_all[TARGET] == 1).all()), f"{len(noon_all):,} rows")
chk("Schema preserved: identical 11 columns, names and order, in all files used",
    list(mine.columns) == list(noon_all.columns) == list(SCHEMA) and referee_integrity["columns match data dictionary (names + order)"])
X_chk = H._prep(pd.concat([mine, noon_all], ignore_index=True))
chk("Target cannot leak into the lie detector (is_fraud not among predictors)", TARGET not in X_chk.columns, f"{X_chk.shape[1]} predictor columns")
chk("Duplicates investigated (counts recomputed from the file)", dup.loc["Noon fraud (as supplied)", "rows"] == len(noon_all)
    and dup.loc["Noon fraud (as supplied)", "distinct rows (10 predictors)"] == len(noon_all.drop_duplicates(FEATURES)))
chk("Primary lie detector uses equal sample sizes with no duplicate or overlapping rows",
    bool((prim["n per source"] == min(N_MINE, N_NOON_U)).all()) and not mine_u.duplicated(FEATURES).any()
    and not noon_u.duplicated(FEATURES).any() and C.cross_duplicates(mine_u, noon_u) == 0)
chk("Majority baselines recomputed from counts (not hard-coded)", np.isclose(base_helper, max(len(mine), len(noon_all)) / (len(mine) + len(noon_all))))
chk("Fraud-proportion denominators explicit and recomputable", bool((prop["proportion"] == prop["numerator"] / prop["denominator count"]).all()))
chk("Helper functions used unmodified (file hash unchanged)", C.sha256(PROF / "A_eval_helper.py") == recorded["A_eval_helper.py"])
chk("Professor files unchanged", bool(C.professor_inventory(PROF, recorded)["unchanged since received"].all()))
chk("Referee set used for integrity counts only (variable deleted; single read in the notebook source)",
    "_ref" not in globals() and (ROOT / "notebooks" / "phase2_comparison.py").read_text().count("C.REFEREE_" + "FILENAME") == 1)
rec = all(np.isclose(C.stats.ks_2samp(mine_u[c], noon_u[c]).statistic,
                     eff_main.set_index("feature").loc[c, "effect size"]) for c in ["order_amount", "account_age_days", "num_orders_last_7d"])
chk("Effect sizes recomputed independently match the reported table", rec)
chk("Relationship chart table equals a fresh recomputation", np.allclose(
    pd.read_csv(P2 / "relationships" / "2.1f_relationship_profiles.csv")["value"].fillna(-1),
    C.relationship_profiles({NAMES["seed"]: seed_h, NAMES["mine"]: mine_u, NAMES["noon"]: noon_u})["value"].fillna(-1)))
chk("Phase 1 deliverables intact (data/model artefacts + notebook cells)", bool(artefacts["unchanged since Phase 1 submission"].all()) and cells_ok)
src_text = (ROOT / "notebooks" / "phase2_comparison.py").read_text()
markers = ["STEP_2_1_" + "INTERPRETATION", "STEP_2_3_" + "INTERPRETATION", "STEP_2_4_" + "REFLECTION", "PHASE_2_" + "SUMMARY", "TO" + "DO"]
chk("No unfilled placeholder text in the Phase 2 section", not any(m in src_text for m in markers))
chk("No __pycache__ written into the professor folder", not (PROF / "__pycache__").exists())
# Numbers quoted in the written interpretation must equal this run's results (formatted as written)
em = eff_main.set_index("feature")["effect size"]
quoted = {
    "KS D = 0.122": f"KS D = {em['order_amount']:.3f}", "KS D = 0.327": f"KS D = {em['account_age_days']:.3f}",
    "KS D = 0.250": f"KS D = {em['num_orders_last_7d']:.3f}", "TVD = 0.503": f"TVD = {em['hour_of_day']:.3f}",
    "TVD 0.395": f"TVD {em['location_mismatch']:.3f}", "TVD 0.338": f"TVD {em['device_change']:.3f}",
    "TVD 0.199": f"TVD {em['payment_method']:.3f}", "KS D 0.185": f"KS D {em['delivery_distance_km']:.3f}",
    "KS D 0.139": f"KS D {em['prev_refunds']:.3f}",
    "0.974 ± 0.003": f"{prim['ROC-AUC'].mean():.3f} ± {prim['ROC-AUC'].std():.3f}",
    "grouped 0.975": f"grouped {grp['ROC-AUC'].mean():.3f}", "ungrouped 0.996": f"ungrouped {ungrp['ROC-AUC'].mean():.3f}",
    "seed-Noon 0.977": f"seed-Noon {ref_seed_noon['ROC-AUC'].mean():.3f}", "seed-CTGAN 0.791": f"seed-CTGAN {ref_seed_mine['ROC-AUC'].mean():.3f}",
    "perm 0.976": f"perm {perm['observed mean CV ROC-AUC']:.3f}", "null max 0.550": f"null max {perm['null max']:.3f}",
    "control 0.949": f"control {ctrl['ROC-AUC (columns shuffled within source: marginals only)']:.3f}",
    "joint 0.026": f"joint {ctrl['AUC attributable to joint structure']:.3f}",
    "leak 83.3%": f"leak {leak['share of Noon test rows leaked']:.1%}", "helper 0.960": f"helper {acc_helper:.3f}",
    "baseline 0.586": f"baseline {base_helper:.3f}", "distinct 1,033": f"distinct {N_NOON_U:,}",
    "night 60.1%": f"night {100 * noon_u.hour_of_day.between(0, 5).mean():.1f}%",
    "night 11.9%": f"night {100 * mine_u.hour_of_day.between(0, 5).mean():.1f}%",
    "zero-day 31.0%": f"zero-day {100 * (mine_u.account_age_days == 0).mean():.1f}%",
    "mismatch 62.9%": f"mismatch {100 * noon_u.location_mismatch.mean():.1f}%", "device 65.6%": f"device {100 * noon_u.device_change.mean():.1f}%",
    "repeats 65.6%": f"repeats {100 * (1 - N_NOON_U / len(noon_all)):.1f}%",
    "imp age 0.096": f"imp age {imp.loc['account_age_days', 'mean drop in ROC-AUC']:.3f}",
}
bad = {k: v for k, v in quoted.items() if k != v}
chk("Numbers quoted in the written interpretation equal this run's results", not bad, str(bad) if bad else f"{len(quoted)} values checked")
audit = pd.DataFrame(audit)
display(audit)
audit.to_csv(P2 / "audit" / "QC_checks.csv", index=False)
print(f"{int((audit.result == 'Passed').sum())} of {len(audit)} checks passed · Phase 2 runtime {(time.time() - T0) / 60:.1f} min")

# %% [markdown]
# ## Phase 2 summary (for the final report)
#
# **Data.**
# * Noon supplied 3,000 synthetic fraud rows (all `is_fraud = 1`, schema identical to the data dictionary). Only 1,033 are distinct: 65.6% of rows are repeats, up to 9 copies of one row. The pattern fits sampling with replacement from about 1,107 base rows.
# * My Phase 1 CTGAN file has 2,116 rows and no duplicates. No row appears in both files.
# * All primary comparisons use unique rows, and the duplicated file is kept as a sensitivity check.
#
# **2.1 Distributions.** My fraud and Noon's differ most on:
# * hour of day (60.1% of Noon's fraud is placed 00:00–05:59 vs 11.9% of mine);
# * location mismatch (62.9% vs 23.4%);
# * device change (65.6% vs 31.9%);
# * account age (no 0-day accounts in Noon's file vs 31.0% in mine; median 147 vs 94 days).
#
# Order amounts have nearly the same mean but different shapes (KS D 0.122). The relationships between features differ too, most notably mismatch being independent of payment method in Noon's data. My **seed** differs from Noon's as much as my CTGAN does.
#
# **2.2 Fraud proportion.** Both synthetic files are 100% fraud by construction, so their "fraud rate" cannot be compared.
#
# | Dataset | Fraud rows | Total rows | Fraud share |
# |---|---|---|---|
# | My seed | 1,512 | 50,000 | 3.02% |
# | My training set augmented with GAN rows (selected 0.25× ratio) | 1,322 | 35,264 | 3.75% |
# | Real referee set (context only) | 65 | 20,000 | 0.325% |
#
# Noon did not supply its augmented training composition, so it is not estimated.
#
# **2.3 Lie detector.**
# * **Primary result** (unique rows, balanced 1,033 per source, shuffled 5-fold): ROC-AUC **0.974 ± 0.003**, balanced accuracy 0.920, permutation p ≈ 0.001 (1,000 permutations).
# * **Sensitivity check** (Noon's duplicates kept, grouped folds): 0.975.
# * **The helper's 0.96 accuracy is inflated.** Duplicate rows leak across its unshuffled folds (83.3% of Noon test rows have a twin in training), and its 0.50 reference ignores the 0.586 majority baseline.
# * **References:** my seed vs Noon gives AUC 0.977, and my seed vs my CTGAN 0.791.
# * **What drives the separation:** account age, location mismatch, hour and device change.
#
# **2.4 Reflection.**
# * Noon's generator portrays mainly established-account, night-time, device-and-location-change fraud. My Phase 1 assumptions emphasised new-account promo abuse. That is a difference in **assumptions**, not a GAN failure.
# * Noon's file is a synthetic benchmark with heavy duplication, not ground truth. Which picture is closer to real fraud is a Phase 3 question.
#
# **Outstanding issues and limitations.**
# * **Noon's file.** It is fraud-only and heavily duplicated (Kish effective size about 813 rows). Noon has given no information on how it was generated or what it was trained on.
# * **Effect-size bands** (0.05 / 0.15 / 0.30) are a judgement convention.
# * **KS permutation p-values hit their floor.** 2,000 permutations give a smallest Holm-adjusted value of 0.0025. These are bounds, not exact values.
# * **One model family.** The lie detector uses only the helper's random forest.
# * **Distance precision.** `delivery_distance_km` in my file was rounded to 1 decimal place to match Noon's precision before comparison. This was checked: no duplicates were created, and the original file is unchanged.
#
# **Phase 3 and 4 are not executed here. Requirements carried forward:**
# * **Phase 3: score on the same referee rows.** Score my frozen selected model (`models/fraud_model.joblib`: pipeline, Platt calibrator, threshold 0.297), the GAN-augmented arm, and Noon's `B_NOON_model.joblib` on the same 20,000 referee rows, without any retraining, threshold changes or calibration on referee data.
# * **`score_on_referee()` does not fit my model as it stands.** It applies one-hot encoding (`_prep`) and `clf.predict` with a 0.5 threshold. My pipeline expects raw columns and its own 0.297 threshold, so a thin adapter is needed. Its outputs must be cross-checked against `sklearn.metrics`.
# * **Noon's model needs checking before use.** It was pickled with scikit-learn 1.8.0, and this environment has 1.9.1. Check that it reproduces the benchmark (recall ≈ 0.82, precision ≈ 0.43, F1 ≈ 0.56, FP = 71) and investigate any gap; do not tune towards it.
# * **Phase 4: `app(2).py` limits.**
#   * The order-amount input takes whole numbers only.
#   * Account age is capped at 2,000 days.
#   * It calls `model.predict`, which ignores my 0.297 threshold.
#   * It expects a `{'model', 'columns'}` dictionary.
#
#   It needs a small wrapper for my model, and the professor's original file must stay unmodified.
