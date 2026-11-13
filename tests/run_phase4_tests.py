"""Run the Phase 4 test suite and write a readable results summary.

    python tests/run_phase4_tests.py
Writes outputs/phase4/test_results_summary.md and outputs/phase4/test_results.csv
(the full verbose log, including the deliberately triggered error logs, goes to outputs/phase4/unittest_log.txt).
"""
import csv
import io
import platform
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows = []

    def addSuccess(self, test):
        super().addSuccess(test); self.rows.append((test, "Passed", ""))

    def addFailure(self, test, err):
        super().addFailure(test, err); self.rows.append((test, "FAILED", self._exc_info_to_string(err, test).splitlines()[-1]))

    def addError(self, test, err):
        super().addError(test, err); self.rows.append((test, "ERROR", self._exc_info_to_string(err, test).splitlines()[-1]))

    def addSkip(self, test, reason):
        super().addSkip(test, reason); self.rows.append((test, "Not tested (skipped)", reason))


if __name__ == "__main__":
    import streamlit, sklearn, pandas
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_phase4_app")
    log = io.StringIO()
    t0 = time.time()
    res = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Recorder).run(suite)
    secs = time.time() - t0
    (OUT / "unittest_log.txt").write_text(log.getvalue())
    groups = {"T1ModelIntegration": "8.1 Model integration", "T2Validation": "8.2 Input validation",
              "T3ErrorHandling": "Part 3 Error handling", "T4StreamlitUI": "8.3 Streamlit UI (AppTest)",
              "T5LiveServer": "8.3 Live server (streamlit run)"}
    rows = [(groups[t.__class__.__name__], t._testMethodName, s, d) for t, s, d in res.rows]
    with open(OUT / "test_results.csv", "w", newline="") as f:
        csv.writer(f).writerows([("group", "test", "result", "detail")] + rows)
    passed = sum(r[2] == "Passed" for r in rows)
    md = [f"# Phase 4 — test results", "",
          f"Run {time.strftime('%Y-%m-%d %H:%M')} · Python {platform.python_version()} · streamlit {streamlit.__version__} · "
          f"scikit-learn {sklearn.__version__} · pandas {pandas.__version__} · {secs:.1f} s", "",
          f"**{passed} of {len(rows)} tests passed.**", "", "| Group | Test | Result |", "|---|---|---|"]
    md += [f"| {g} | `{n}` | {s}{' — ' + d if d else ''} |" for g, n, s, d in rows]
    (OUT / "test_results_summary.md").write_text("\n".join(md) + "\n")
    print(f"{passed}/{len(rows)} passed in {secs:.1f}s → {OUT / 'test_results_summary.md'}")
    sys.exit(0 if res.wasSuccessful() else 1)
