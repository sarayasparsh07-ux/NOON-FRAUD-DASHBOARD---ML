"""Execute the Phase 2 section in a fresh kernel and append it to the existing notebook.

Phase 1 cells are never re-executed or modified: their digest is checked before and after the merge.

Usage
  python notebooks/run_phase2.py                    # execute and merge into fraud_detection_analysis.ipynb
  python notebooks/run_phase2.py --test OUT.ipynb   # execute only; write to OUT.ipynb, do not merge
Environment overrides (testing only): NOON_PATH, P2_OUT, P2_DATA, P2_TEST_STANDIN=1
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
from build_and_run import to_notebook          # noqa: E402
from src.compare import phase1_cells_digest     # noqa: E402

SRC = HERE / "phase2_comparison.py"
MAIN = HERE / "fraud_detection_analysis.ipynb"
MANIFEST = HERE.parent / "reports" / "phase1_submission_manifest.json"

if __name__ == "__main__":
    test_out = Path(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else None
    nb2 = to_notebook(SRC.read_text())
    for c in nb2.cells:
        c.metadata["phase"] = 2
    t0 = time.time()
    client = NotebookClient(nb2, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        if test_out:
            nbformat.write(nb2, test_out)
            print(f"Test run written to {test_out} after {(time.time() - t0) / 60:.1f} min (not merged)")
    if test_out:
        sys.exit(0)
    expected = json.loads(MANIFEST.read_text())["phase1_notebook_cells_sha256"]
    before = phase1_cells_digest(MAIN)
    main = nbformat.read(MAIN, as_version=4)
    n_old = sum(1 for c in main.cells if c.metadata.get("phase") == 2)
    main.cells = [c for c in main.cells if c.metadata.get("phase") != 2] + nb2.cells
    nbformat.write(main, MAIN)
    after = phase1_cells_digest(MAIN)
    assert before == after == expected, "Phase 1 cells changed during the merge"
    print(f"Merged {len(nb2.cells)} Phase 2 cells (replaced {n_old}) into {MAIN.name} after {(time.time() - t0) / 60:.1f} min; "
          f"Phase 1 cells verified unchanged.")
