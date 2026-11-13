"""Execute the Phase 3 section in a fresh kernel and append it to the existing notebook.

Phase 1 and Phase 2 cells are never re-executed or modified: their digests are checked before and after the merge.

Usage
  python notebooks/run_phase3.py                    # execute and merge into fraud_detection_analysis.ipynb
  python notebooks/run_phase3.py --test OUT.ipynb   # execute only; write to OUT.ipynb, do not merge
Prerequisite (once): outputs/phase3/audit/noon_predictions_sklearn_1.8.0.csv from notebooks/phase3_noon_crossversion.py,
run in an isolated scikit-learn 1.8.0 environment (see that script). Without it the cross-version check is reported as NOT TESTED.
"""
import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True
from build_and_run import to_notebook          # noqa: E402
from src.phase3 import cells_digest             # noqa: E402

SRC = HERE / "phase3_evaluation.py"
MAIN = HERE / "fraud_detection_analysis.ipynb"
P1 = json.loads((HERE.parent / "reports" / "phase1_submission_manifest.json").read_text())["phase1_notebook_cells_sha256"]
P2 = json.loads((HERE.parent / "reports" / "phase2_freeze_manifest.json").read_text())["phase2_notebook_cells_sha256"]

if __name__ == "__main__":
    test_out = Path(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else None
    nb3 = to_notebook(SRC.read_text())
    for c in nb3.cells:
        c.metadata["phase"] = 3
    t0 = time.time()
    client = NotebookClient(nb3, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        if test_out:
            nbformat.write(nb3, test_out)
            print(f"Test run written to {test_out} after {(time.time() - t0) / 60:.1f} min (not merged)")
    if test_out:
        sys.exit(0)
    before = (cells_digest(MAIN, None), cells_digest(MAIN, 2))
    main = nbformat.read(MAIN, as_version=4)
    n_old = sum(1 for c in main.cells if c.metadata.get("phase") == 3)
    main.cells = [c for c in main.cells if c.metadata.get("phase") != 3] + nb3.cells
    nbformat.write(main, MAIN)
    after = (cells_digest(MAIN, None), cells_digest(MAIN, 2))
    assert before == after == (P1, P2), "Phase 1 or Phase 2 cells changed during the merge"
    print(f"Merged {len(nb3.cells)} Phase 3 cells (replaced {n_old}) into {MAIN.name} after {(time.time() - t0) / 60:.1f} min; "
          f"Phase 1 and Phase 2 cells verified unchanged.")
