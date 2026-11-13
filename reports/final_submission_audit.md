# Final Submission Audit

Run 2026-10-31 23:06 by `tests/final_submission_audit.py` with interpreter `/tmp/claude-0/-home-claude/521c550d-d57b-5440-a645-e0c9b8586b42/scratchpad/venv_audit/bin/python (Python 3.13.16)`. **41 of 41 checks passed.** Every check below was executed by the script; the Word rendering check in the summary was done separately (LibreOffice to PDF, pages inspected).

| Section | Check | Result | Evidence |
|---|---|---|---|
| A Package contents | 1 Completed notebook with outputs | Passed | 1 path(s) present |
| A Package contents | 2 Seed transaction dataset | Passed | 1 path(s) present |
| A Package contents | 3 GAN-generated synthetic fraud dataset | Passed | 1 path(s) present |
| A Package contents | 4 Saved GAN and classifier artefacts | Passed | 3 path(s) present |
| A Package contents | 5 Phase 1 metrics and diagnostics | Passed | 5 path(s) present |
| A Package contents | 6 Phase 2 data, charts and report | Passed | 4 path(s) present |
| A Package contents | 7 Phase 3 evaluation, benchmark comparison and report | Passed | 5 path(s) present |
| A Package contents | 8 Streamlit application | Passed | 2 path(s) present |
| A Package contents | 9 Selected final classifier | Passed | 1 path(s) present |
| A Package contents | 10 Final integrated report (Markdown + Word) | Passed | 2 path(s) present |
| A Package contents | 11 Rubric traceability | Passed | 1 path(s) present |
| A Package contents | 12 README and dependency files | Passed | 3 path(s) present |
| A Package contents | 13 Quality reviews and Phase 4 test evidence | Passed | 7 path(s) present |
| B Notebook | Phase 1 cells executed with visible outputs, no errors | Passed | 62 cells, 35 code, 35 with outputs, 0 unexecuted, 0 errors |
| B Notebook | Phase 2 cells executed with visible outputs, no errors | Passed | 43 cells, 22 code, 22 with outputs, 0 unexecuted, 0 errors |
| B Notebook | Phase 3 cells executed with visible outputs, no errors | Passed | 37 cells, 18 code, 18 with outputs, 0 unexecuted, 0 errors |
| C Preservation | Data, models, professor files, Phase 1–3 outputs and the notebook unchanged (148 files) | Passed | all hashes match reports/phase4_baseline_manifest.json |
| C Preservation | Phase 4 app and test evidence present | Passed | app.py, outputs/phase4/test_results_summary.md, outputs/phase4/unittest_log.txt, outputs/phase4/browser_check_log.txt |
| D Documented paths | All file paths quoted in README.md exist | Passed | 76 paths checked |
| D Documented paths | All file paths quoted in reports/final_report.md exist | Passed | 8 paths checked |
| D Documented paths | All file paths quoted in reports/rubric_traceability.md exist | Passed | 36 paths checked |
| E Report numbers | Final report quotes 41 verified values exactly as computed from the metrics files | Passed | all found |
| E Report numbers | Final report answers Q1 assumptions / how right | Passed | ## 1. Initial fraud assumptions |
| E Report numbers | Final report answers Q2 did the GAN improve detection (numbers) | Passed | **Did augmentation help? No.** |
| E Report numbers | Final report answers Q3 vs Noon on real data and why | Passed | **Why Noon won.** |
| E Report numbers | Final report answers Q4 why not an LLM | Passed | ## 4. Why not just ask an LLM |
| F Consistency | No outdated status wording in README.md | Passed | none |
| F Consistency | No outdated status wording in reports/final_report.md | Passed | none |
| F Consistency | No outdated status wording in reports/rubric_traceability.md | Passed | none |
| F Consistency | No outdated status wording in reports/phase4_implementation_notes.md | Passed | none |
| F Consistency | No outdated status wording in reports/phase4_quality_review.md | Passed | none |
| F Consistency | reports/Final_Project_Report.docx is a valid Word file matching its Markdown source | Passed | opened as OOXML; key text present |
| F Consistency | reports/Phase3_Report.docx is a valid Word file matching its Markdown source | Passed | opened as OOXML; key text present |
| F Consistency | Markdown and Word final reports agree (91 paragraphs, bullets and table cells compared) | Passed | every Markdown unit found verbatim in the Word file |
| F Consistency | Phase 1 Word report unchanged since the Phase 1 submission | Passed | hash matches reports/phase1_submission_manifest.json |
| F Consistency | Supporting report present: reports/phase1_report.md | Passed |  |
| F Consistency | Supporting report present: reports/phase2_report.md | Passed |  |
| F Consistency | Supporting report present: reports/Phase2_Report.docx | Passed |  |
| F Consistency | Supporting report present: reports/phase3_report.md | Passed |  |
| F Consistency | Supporting report present: reports/Phase3_Report.docx | Passed |  |
| G App | Phase 4 test suite re-run after the documentation changes (44 tests expected) | Passed | Ran 44 tests in 7.754s → OK (10s) with streamlit 1.65.0 |
