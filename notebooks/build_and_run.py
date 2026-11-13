"""Convert the percent-format source into an .ipynb and execute it (outputs saved in the notebook).

Usage:  python notebooks/build_and_run.py            # full run (≈1 hour on 2 CPU cores)
        FRAUD_DEV=1 python notebooks/build_and_run.py  # quick smoke test
"""
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SRC = HERE / "fraud_detection_analysis.py"
OUT = HERE / ("fraud_detection_analysis_DEV.ipynb" if os.environ.get("FRAUD_DEV") == "1" else "fraud_detection_analysis.ipynb")


def to_notebook(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
    for header, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    nb = to_notebook(SRC.read_text())
    t0 = time.time()
    client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, OUT)
        print(f"Wrote {OUT} after {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0)
