"""Cross-version check for Noon's model (Phase 3, Part 4).

Noon's `B_NOON_model.joblib` was pickled with scikit-learn 1.8.0; the project environment runs 1.9.1.
This script is run ONCE in a separate, isolated environment that has scikit-learn 1.8.0 (the project
environment is not touched). It applies Noon's own prediction rule to the 10 referee feature columns and
writes probabilities and predictions; the Phase 3 notebook then compares them with the 1.9.1 results.

It never reads `is_fraud`, never fits anything, and never re-saves the model.

    python -m venv venv_sk180 && venv_sk180/bin/pip install scikit-learn==1.8.0 pandas joblib
    venv_sk180/bin/python -I notebooks/phase3_noon_crossversion.py <project_root> <output_csv>
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

root, out = Path(sys.argv[1]), Path(sys.argv[2])
prof = root / "professor_supplied"
FEATURES = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d", "payment_method",
            "delivery_distance_km", "discount_applied", "prev_refunds", "device_change", "location_mismatch"]

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bundle = joblib.load(prof / "B_NOON_model.joblib")
version_warnings = sum("InconsistentVersion" in type(x.message).__name__ for x in w)

X = pd.read_csv(prof / "C_REFEREE_real_test.csv", usecols=FEATURES)[FEATURES]        # features only
Xe = pd.get_dummies(X[bundle["num"] + bundle["cat"]], columns=bundle["cat"]).reindex(columns=bundle["columns"], fill_value=0)
m = bundle["model"]
pos = list(m.classes_).index(1)
res = pd.DataFrame({"row": np.arange(len(X)), "noon_proba_fraud": m.predict_proba(Xe)[:, pos],
                    "noon_pred": m.predict(Xe).astype(int)})
res.attrs = {}
out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out, index=False, float_format="%.17g")
Path(str(out) + ".meta.txt").write_text(f"scikit-learn {sklearn.__version__}\nnumpy {np.__version__}\npandas {pd.__version__}\n"
                                         f"InconsistentVersionWarning count on load: {version_warnings}\nrows {len(res)}\n")
print(f"scikit-learn {sklearn.__version__}: {len(res)} rows scored, {int(res.noon_pred.sum())} flagged; version warnings {version_warnings}")
