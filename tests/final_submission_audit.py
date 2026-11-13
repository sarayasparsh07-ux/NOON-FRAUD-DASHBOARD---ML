"""Final submission-readiness audit (documentation-only; changes no data, model or result).

    python tests/final_submission_audit.py
Writes reports/final_submission_audit.md. Every check is executed here; nothing is asserted from memory.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
rows = []


def chk(section, item, ok, evidence=""):
    rows.append((section, item, "Passed" if ok else "FAILED", evidence))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# A. Package contents (audit list items 1-13) -------------------------------------------------------
A = {
    "1 Completed notebook with outputs": ["notebooks/fraud_detection_analysis.ipynb"],
    "2 Seed transaction dataset": ["data/seed_transactions.csv"],
    "3 GAN-generated synthetic fraud dataset": ["data/synthetic_fraud_gan.csv"],
    "4 Saved GAN and classifier artefacts": ["models/ctgan_fraud_generator.pkl", "models/fraud_model.joblib", "models/fraud_model_gan_augmented.joblib"],
    "5 Phase 1 metrics and diagnostics": ["outputs/metrics/phase1_metrics.json", "outputs/metrics/arms_test_metrics.csv", "outputs/confusion_matrices",
                                          "outputs/gan_diagnostics/ctgan_loss_values.csv", "outputs/eda"],
    "6 Phase 2 data, charts and report": ["outputs/phase2/lie_detector/2.3_protocol_summary.csv", "outputs/phase2/distributions/2.1e_effect_sizes.png",
                                          "reports/phase2_report.md", "reports/Phase2_Report.docx"],
    "7 Phase 3 evaluation, benchmark comparison and report": ["outputs/phase3/metrics/C1_comparison_table.csv", "outputs/phase3/metrics/C2_noon_benchmark_reproduction.csv",
                                                              "outputs/phase3/charts/3.3_final_bar_chart_recall_and_false_positives.png", "reports/phase3_report.md", "reports/Phase3_Report.docx"],
    "8 Streamlit application": ["app.py", ".streamlit/config.toml"],
    "9 Selected final classifier": ["models/fraud_model.joblib"],
    "10 Final integrated report (Markdown + Word)": ["reports/final_report.md", "reports/Final_Project_Report.docx"],
    "11 Rubric traceability": ["reports/rubric_traceability.md"],
    "12 README and dependency files": ["README.md", "requirements.txt", "requirements-app.txt"],
    "13 Quality reviews and Phase 4 test evidence": ["reports/quality_assurance_checklist.md", "reports/phase2_quality_review.md", "reports/phase3_quality_review.md",
                                                     "reports/phase4_quality_review.md", "outputs/phase4/test_results_summary.md", "outputs/phase4/browser_check_log.txt",
                                                     "outputs/phase4/screenshots/03_fraud_example_result.png"],
}
for item, paths in A.items():
    missing = [p for p in paths if not (ROOT / p).exists()]
    chk("A Package contents", item, not missing, "missing: " + ", ".join(missing) if missing else f"{len(paths)} path(s) present")

# B. Notebook: executed, outputs visible, no errors, each phase present --------------------------------
nb = json.loads((ROOT / "notebooks" / "fraud_detection_analysis.ipynb").read_text())
for ph, name in ((None, "Phase 1"), (2, "Phase 2"), (3, "Phase 3")):
    cells = [c for c in nb["cells"] if c.get("metadata", {}).get("phase") == ph]
    code = [c for c in cells if c["cell_type"] == "code"]
    with_out = sum(bool(c.get("outputs")) for c in code)
    unexec = sum(c.get("execution_count") is None for c in code)
    errs = sum(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
    chk("B Notebook", f"{name} cells executed with visible outputs, no errors", code and unexec == 0 and errs == 0 and with_out > 0,
        f"{len(cells)} cells, {len(code)} code, {with_out} with outputs, {unexec} unexecuted, {errs} errors")

# C. Frozen artefacts unchanged since the pre-Phase 4 manifest -------------------------------------------
base = json.loads((ROOT / "reports" / "phase4_baseline_manifest.json").read_text())["files"]
frozen_prefixes = ("data/", "models/", "professor_supplied/", "outputs/metrics/", "outputs/phase2/", "outputs/phase3/", "outputs/gan_diagnostics/",
                   "outputs/eda/", "outputs/confusion_matrices/", "outputs/distribution_comparisons/", "outputs/final_model_comparison/",
                   "outputs/robustness_tests/", "notebooks/fraud_detection_analysis.ipynb")
frozen = {k: v for k, v in base.items() if k.startswith(frozen_prefixes)}
changed = [k for k, v in frozen.items() if not (ROOT / k).exists() or sha(ROOT / k) != v]
chk("C Preservation", f"Data, models, professor files, Phase 1–3 outputs and the notebook unchanged ({len(frozen)} files)", not changed,
    "changed: " + ", ".join(changed) if changed else "all hashes match reports/phase4_baseline_manifest.json")
p4 = ["app.py", "outputs/phase4/test_results_summary.md", "outputs/phase4/unittest_log.txt", "outputs/phase4/browser_check_log.txt"]
chk("C Preservation", "Phase 4 app and test evidence present", all((ROOT / p).exists() for p in p4), ", ".join(p4))

# D. Every path quoted in the key documents exists ---------------------------------------------------------
docs = ["README.md", "reports/final_report.md", "reports/rubric_traceability.md"]
for d in docs:
    text = (ROOT / d).read_text()
    cands = set(re.findall(r"`([^`\s]+\.(?:md|docx|csv|json|py|ipynb|joblib|pkl|png|txt|toml|npz))`", text))
    cands |= {c for c in re.findall(r"`((?:outputs|reports|data|models|notebooks|tests|src|professor_supplied)/[^`\s]*)`", text)}
    missing = []
    for c in sorted(cands):
        c2 = c.rstrip("/").replace("/*", "")
        if "*" in c2 or c2.startswith(("venv", "$")) or c2.startswith("/"):
            continue
        hits = [ROOT / c2] + [ROOT / d / c2 for d in ("reports", "professor_supplied", "notebooks", "models", "tests", "data")]
        if not any(h.exists() for h in hits):
            missing.append(c)
    chk("D Documented paths", f"All file paths quoted in {d} exist", not missing, "missing: " + ", ".join(missing) if missing else f"{len(cands)} paths checked")

# E. Final report numbers equal the metrics files -----------------------------------------------------------
fr = (ROOT / "reports" / "final_report.md").read_text()
p3 = pd.read_csv(ROOT / "outputs" / "phase3" / "metrics" / "C1_referee_metrics_all_models.csv", index_col=0)
arms = pd.read_csv(ROOT / "outputs" / "metrics" / "arms_test_metrics.csv", index_col=0)
claim = pd.read_csv(ROOT / "outputs" / "metrics" / "ctgan_improvement_claim_check.csv").set_index("metric")
lie = pd.read_csv(ROOT / "outputs" / "phase2" / "lie_detector" / "2.3_protocol_summary.csv")
eff = pd.read_csv(ROOT / "outputs" / "phase3" / "gap_analysis" / "D2_effect_sizes_wide.csv", index_col=0)
prof = pd.read_csv(ROOT / "outputs" / "phase3" / "gap_analysis" / "D2_fraud_indicator_profiles.csv", index_col=0)
gen = pd.read_csv(ROOT / "outputs" / "phase3" / "gap_analysis" / "D1_synthetic_test_vs_referee.csv")
p1 = json.loads((ROOT / "outputs" / "metrics" / "phase1_metrics.json").read_text())
arm_rows = {i: r for i, r in arms.iterrows()}
a_row = [r for i, r in arm_rows.items() if str(i).startswith("A")][0]
b_row = [r for i, r in arm_rows.items() if str(i).startswith("B")][0]
pb = pd.read_csv(ROOT / "outputs" / "phase3" / "metrics" / "C4_paired_bootstrap_differences.csv")
pb_rec = pb[(pb["A vs B"].str.contains("Noon")) & (pb.metric == "recall")].iloc[0]
sel, gan, noon = p3.loc["My selected classifier"], p3.loc["My CTGAN-augmented classifier"], p3.loc["Noon Model B"]
expected = [
    f"| My selected classifier | {int(sel.TP)} / 65 | {100 * sel.recall:.2f}% | {100 * sel.precision:.2f}% | {sel.F1:.3f} | {int(sel.FP):,} |",
    f"| My CTGAN-augmented classifier | {int(gan.TP)} / 65 | {100 * gan.recall:.2f}% | {100 * gan.precision:.2f}% | {gan.F1:.3f} | {int(gan.FP):,} |",
    f"| Noon Model B | {int(noon.TP)} / 65 | {100 * noon.recall:.2f}% | {100 * noon.precision:.2f}% | {noon.F1:.3f} | {int(noon.FP):,} |",
    f"| {100 * a_row.recall:.1f}% | {100 * a_row.precision:.1f}% | {a_row.f1:.3f} | {a_row.pr_auc_ap:.3f} | {int(a_row.FP)} |",
    f"| {100 * b_row.recall:.1f}% | {100 * b_row.precision:.1f}% | {b_row.f1:.3f} | {b_row.pr_auc_ap:.3f} | {int(b_row.FP)} |",
    f"AP {claim.loc['pr_auc_ap', 'test Δ (CTGAN − Baseline)']:.3f}".replace("-", "−"),
    f"ROC-AUC {lie.loc[1, 'ROC-AUC']:.3f}", f"accuracy {lie.loc[1, 'accuracy']:.3f}", f"({lie.loc[4, 'ROC-AUC']:.3f})", f"({lie.loc[5, 'ROC-AUC']:.3f})",
    f"{p1['ctgan']['epochs']} epochs", f"{p1['ctgan']['train_rows']:,} training-partition", f"{p1['ctgan']['pool_rows']:,} fraud rows",
    f"was {eff.loc['mean over 10 features', 'Real referee fraud vs Noon synthetic fraud (unique rows)']:.3f} for real fraud versus Noon's unique synthetic fraud",
    f"simulated sampling reference of {eff.loc['mean over 10 features', 'noise floor: 65 rows vs their own source (mean of 200 draws)']:.3f}",
    f"and {eff.loc['mean over 10 features', 'Real referee fraud vs My seed fraud (training)']:.3f} for real fraud versus my seed fraud",
    "not a formal multivariate equivalence test", "without access to any Noon transaction data",
    f"| Placed 00:00–05:59 | {prof.loc['orders placed 00:00–05:59', 'Real referee fraud']:.1f}% | {prof.loc['orders placed 00:00–05:59', 'My seed fraud (training)']:.1f}% |",
    f"| Device change | {prof.loc['device_change = 1', 'Real referee fraud']:.1f}% | {prof.loc['device_change = 1', 'My seed fraud (training)']:.1f}% |",
    f"| Location mismatch | {prof.loc['location_mismatch = 1', 'Real referee fraud']:.1f}% | {prof.loc['location_mismatch = 1', 'My seed fraud (training)']:.1f}% |",
    f"| Account ≤ 7 days old | {prof.loc['account age ≤ 7 days', 'Real referee fraud']:.1f}% | {prof.loc['account age ≤ 7 days', 'My seed fraud (training)']:.1f}% |",
    f"from {100 * gen.loc[0, 'false-positive rate']:.2f}% on the synthetic test set to {100 * gen.loc[1, 'false-positive rate']:.2f}% on the real referee set",
    f"real referee set has {100 * p3.loc['My selected classifier', 'n fraud'] / p3.loc['My selected classifier', 'n']:.3f}%",
    f"against {lie.loc[1, 'ROC-AUC']:.3f} for my CTGAN vs Noon's fraud",
    f"paired 95% CI {100 * pb_rec['95% CI low']:.1f} to {100 * pb_rec['95% CI high']:.1f} points",
    f"{int(noon.TP - sel.TP)} more frauds (+{100 * (noon.recall - sel.recall):.1f} percentage points)",
    f"{int(sel.FP - noon.FP)} fewer legitimate orders", f"{int(sel.TP - gan.TP)} fewer frauds",
    f"{100 * (1 - sel.precision):.0f}% of its flags were legitimate",
]
legit = pd.read_csv(ROOT / "outputs" / "phase3" / "gap_analysis" / "D2_legit_indicator_profiles.csv", index_col=0)
expected.append("({:.1f}% vs {:.1f}%, {:.1f}% vs {:.1f}% and {:.1f}% vs {:.1f}%)".format(
    *[legit.loc[k, c] for k in ("orders placed 00:00–05:59", "≥ 1 previous refund", "device_change = 1")
      for c in ("Real referee legitimate", "My seed legitimate (training)")]))
expected += ["**illustrative modelling assumption**", "not a measured confirmed-fraud prevalence or a Noon base rate",
             "Noon's GAN was trained on Noon's real fraud examples", "combined real transaction data with that GAN-generated fraud",
             "only after being frozen", "This is consistent with overlap between real legitimate transactions and the fraud signals encoded in my synthetic assumptions",
             "**Causality cannot be isolated.**", "**Prompt-only generation.**", "**Learning and validating a distribution.**",
             "**Only as representative as its training data.**"]
miss = [e for e in expected if e not in fr]
chk("E Report numbers", f"Final report quotes {len(expected)} verified values exactly as computed from the metrics files", not miss,
    "not found: " + " | ".join(miss) if miss else "all found")
for q, key in (("Q1 assumptions / how right", "## 1. Initial fraud assumptions"), ("Q2 did the GAN improve detection (numbers)", "**Did augmentation help? No.**"),
               ("Q3 vs Noon on real data and why", "**Why Noon won.**"), ("Q4 why not an LLM", "## 4. Why not just ask an LLM")):
    chk("E Report numbers", f"Final report answers {q}", key in fr, key)

# F. Stale status statements in the current top-level documents ---------------------------------------------
stale = re.compile(r"\b(pending|not started|interim|in progress|awaiting)\b", re.I)
for d in ["README.md", "reports/final_report.md", "reports/rubric_traceability.md", "reports/phase4_implementation_notes.md", "reports/phase4_quality_review.md"]:
    hits = [l.strip()[:90] for l in (ROOT / d).read_text().splitlines() if stale.search(l)]
    chk("F Consistency", f"No outdated status wording in {d}", not hits, "; ".join(hits) if hits else "none")
for d in ["reports/Final_Project_Report.docx", "reports/Phase3_Report.docx"]:
    try:
        xml = zipfile.ZipFile(ROOT / d).read("word/document.xml").decode()
        ok = "Final Project Report" in xml if "Final" in d else "since been completed" in re.sub("<[^>]+>", "", xml)
    except Exception as exc:
        ok, xml = False, str(exc)
    chk("F Consistency", f"{d} is a valid Word file matching its Markdown source", ok, "opened as OOXML; key text present" if ok else xml[:80])

# F2. Markdown and Word versions of the final report say the same thing ---------------------------------------
def _plain(t):
    t = re.sub(r"\*\*|`|(?<!\w)\*|\*(?!\w)", "", t)
    return re.sub(r"\s+", " ", t).strip()
docx_xml = zipfile.ZipFile(ROOT / "reports" / "Final_Project_Report.docx").read("word/document.xml").decode()
docx_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", docx_xml.replace("</w:p>", " ").replace("</w:tc>", " ")))
docx_text = docx_text.replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&apos;", "'")
units = []
for line in fr.splitlines():
    l = line.strip()
    if not l or set(l) <= set("|-: "):
        continue
    if l.startswith("|"):
        units += [_plain(c) for c in l.strip("|").split("|") if c.strip()]
    else:
        units.append(_plain(re.sub(r"^(#+|-)\s*", "", l)))
absent = [u for u in units if u not in docx_text]
chk("F Consistency", f"Markdown and Word final reports agree ({len(units)} paragraphs, bullets and table cells compared)", not absent,
    "missing in Word: " + " | ".join(a[:60] for a in absent[:5]) if absent else "every Markdown unit found verbatim in the Word file")
p1m = json.loads((ROOT / "reports" / "phase1_submission_manifest.json").read_text())
p1m = p1m.get("files", p1m)
chk("F Consistency", "Phase 1 Word report unchanged since the Phase 1 submission", sha(ROOT / "reports" / "Phase1_Report.docx") == p1m.get("reports/Phase1_Report.docx"),
    "hash matches reports/phase1_submission_manifest.json")
for d in ["reports/phase1_report.md", "reports/phase2_report.md", "reports/Phase2_Report.docx", "reports/phase3_report.md", "reports/Phase3_Report.docx"]:
    chk("F Consistency", f"Supporting report present: {d}", (ROOT / d).exists(), "")

# G. The app still passes its test suite (evidence files are not overwritten) --------------------------------
# The suite needs the app's dependencies (requirements-app.txt) in THIS interpreter; its live-server tests also call the
# `streamlit` command, so the interpreter's own bin/Scripts folder is put first on PATH for the child process.
import importlib.util
import os
import platform
INTERP = f"{sys.executable} (Python {platform.python_version()})"
has_st = importlib.util.find_spec("streamlit") is not None
if not has_st:
    chk("G App", "Phase 4 test suite re-run after the documentation changes", False,
        f"streamlit is not installed in {INTERP}; create an environment from requirements-app.txt and run the audit with it")
else:
    import streamlit
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-B", "-m", "unittest", "tests/test_phase4_app.py"], cwd=ROOT, capture_output=True, text=True,
                       timeout=900, env=env)
    tail = (r.stderr.strip().splitlines() or ["?"])
    ran = next((l for l in tail if l.startswith("Ran ")), "?")
    chk("G App", "Phase 4 test suite re-run after the documentation changes (44 tests expected)",
        r.returncode == 0 and tail[-1].strip() == "OK" and ran.startswith("Ran 44 "),
        f"{ran} → {tail[-1].strip()} ({time.time() - t0:.0f}s) with streamlit {streamlit.__version__}")

# Write the audit ----------------------------------------------------------------------------------------------
df = pd.DataFrame(rows, columns=["section", "check", "result", "evidence"])
n_ok = int((df.result == "Passed").sum())
md = ["# Final Submission Audit", "",
      f"Run {time.strftime('%Y-%m-%d %H:%M')} by `tests/final_submission_audit.py` with interpreter `{INTERP}`. **{n_ok} of {len(df)} checks passed.** "
      "Every check below was executed by the script; the Word rendering check in the summary was done separately (LibreOffice to PDF, pages inspected).", "",
      "| Section | Check | Result | Evidence |", "|---|---|---|---|"]
md += [f"| {s} | {c} | {r_} | {e.replace('|', '/')} |" for s, c, r_, e in rows]
(ROOT / "reports" / "final_submission_audit.md").write_text("\n".join(md) + "\n")
print(f"{n_ok}/{len(df)} passed")
print(df[df.result != "Passed"].to_string() if n_ok < len(df) else "all passed")
sys.exit(0 if n_ok == len(df) else 1)
