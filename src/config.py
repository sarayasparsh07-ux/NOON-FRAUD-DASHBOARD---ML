"""Central configuration: schema, paths, seeds, experiment settings and design tokens.

Everything that defines *what* the experiment is lives here, so the notebook,
the assumptions register and (later) the app all read the same values.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
OUT = {
    "eda": OUTPUTS / "eda",
    "metrics": OUTPUTS / "metrics",
    "cm": OUTPUTS / "confusion_matrices",
    "gan": OUTPUTS / "gan_diagnostics",
    "dist": OUTPUTS / "distribution_comparisons",
    "robust": OUTPUTS / "robustness_tests",
    "final": OUTPUTS / "final_model_comparison",
}

# ---------------------------------------------------------------------------
# Schema — copied exactly from A_INTERN_data_dictionary.md (source fact)
# ---------------------------------------------------------------------------
SCHEMA = {
    "order_amount": "float",          # AED
    "account_age_days": "int",
    "hour_of_day": "int",             # 0-23
    "num_orders_last_7d": "int",
    "payment_method": "category",     # card / wallet / cod / applepay
    "delivery_distance_km": "float",
    "discount_applied": "binary",
    "prev_refunds": "int",
    "device_change": "binary",
    "location_mismatch": "binary",
    "is_fraud": "binary",             # target
}
TARGET = "is_fraud"
FEATURES = [c for c in SCHEMA if c != TARGET]
PAYMENT_METHODS = ["card", "wallet", "cod", "applepay"]
BINARY_COLS = ["discount_applied", "device_change", "location_mismatch"]
INT_COLS = ["account_age_days", "hour_of_day", "num_orders_last_7d", "prev_refunds"]
FLOAT_COLS = ["order_amount", "delivery_distance_km"]
NUMERIC_COLS = FLOAT_COLS + INT_COLS
CATEGORICAL_COLS = ["payment_method"]

# Hard constraints from the data dictionary (source facts + non-negativity of counts/distances)
HARD_BOUNDS = {
    "order_amount": (0.01, None),
    "account_age_days": (0, None),
    "hour_of_day": (0, 23),
    "num_orders_last_7d": (0, None),
    "delivery_distance_km": (0.0, None),
    "prev_refunds": (0, None),
}
# Plausibility bounds — MODELLING ASSUMPTIONS (not in the data dictionary)
PLAUSIBLE_BOUNDS = {
    "order_amount": (15.0, 2000.0),        # assumed minimum basket / large group-order cap (AED)
    "account_age_days": (0, 3000),          # ~8 years of account tenure
    "num_orders_last_7d": (0, 40),
    "delivery_distance_km": (0.3, 15.0),    # assumed urban delivery radius
    "prev_refunds": (0, 15),
}
ROUNDING = {"order_amount": 2, "delivery_distance_km": 2}
# GAN post-processing: clip to the floor only where training data has a point mass there (see gan.postprocess_generated)
CLIP_TO_FLOOR = {"account_age_days": 0}

# ---------------------------------------------------------------------------
# Seeds and experiment settings (pre-declared in reports/experiment_plan.md)
# ---------------------------------------------------------------------------
SEED = 42
REPEAT_SEEDS = [42, 7, 101, 2024, 31337]
STRESS_SEEDS = {"S0_in_distribution": 9001, "S1_population_drift": 9002,
                "S2_adaptive_fraud": 9003, "S3_unseen_typology": 9004}
BOOTSTRAP_SEED = 12345
N_BOOTSTRAP = 2000

N_ROWS = 50_000
PREVALENCE = 0.03
PREVALENCE_SCENARIOS = {0.01: 100_000, 0.05: 50_000}
N_STRESS = 20_000
SPLIT = (0.70, 0.15, 0.15)

AUG_RATIOS = [0.25, 0.5, 1.0, 2.0]
RATIO_TIE_TOL = 0.002
FINAL_MODEL_TOL = 0.01
SIMPLICITY_ORDER = ["A_baseline", "C_class_weighted", "E_random_oversampling",
                    "D_smote_nc", "F_independent_marginals", "B_ctgan"]

CTGAN_PARAMS = dict(
    epochs=600, batch_size=100, pac=10, embedding_dim=128,
    generator_dim=(256, 256), discriminator_dim=(256, 256),
    generator_lr=2e-4, discriminator_lr=2e-4, discriminator_steps=1,
    log_frequency=True, verbose=False, enable_gpu=False,
)
CTGAN_DISCRETE = ["payment_method", "discount_applied", "device_change",
                  "location_mismatch", "hour_of_day", "num_orders_last_7d", "prev_refunds"]
EPOCH_SENSITIVITY = [100, 300, 1000]   # extra fits compared with the main 600-epoch model

ARMS = {
    "A_baseline": "A · Baseline",
    "B_ctgan": "B · CTGAN augmentation",
    "C_class_weighted": "C · Class-weighted",
    "D_smote_nc": "D · SMOTE-NC",
    "E_random_oversampling": "E · Random oversampling",
    "F_independent_marginals": "F · Independent marginals",
}
ARM_ORDER = ["A_baseline", "C_class_weighted", "E_random_oversampling",
             "D_smote_nc", "F_independent_marginals", "B_ctgan"]

# ---------------------------------------------------------------------------
# Design tokens (validated with the dataviz palette checker)
# ---------------------------------------------------------------------------
NAVY = "#1B2A4A"
INK = "#2B3445"
MUTED = "#6B7385"
GRID = "#E3E6EC"
AMBER = "#F2A541"        # caution indicators only (thresholds, warnings)
LEGIT = "#0E9F9A"        # teal
FRAUD = "#E5534B"        # coral
GAN = "#7C3AED"          # violet
ARM_COLORS = {
    "A_baseline": "#0E9F9A",
    "C_class_weighted": "#2F6FD6",
    "E_random_oversampling": "#C98A12",
    "D_smote_nc": "#D6457A",
    "F_independent_marginals": "#5B8C2A",
    "B_ctgan": "#7C3AED",
}
REAL_REF = "#0E9F9A"     # real (seed) training fraud in GAN-fidelity charts

# Phase 2 dataset colours (validated: teal / violet / burnt orange pass CVD, normal-vision and contrast checks)
SEED_FRAUD_COLOR = "#0E9F9A"     # my seed fraud (CTGAN training input)
MY_GAN_COLOR = "#7C3AED"         # my CTGAN synthetic fraud (same violet as Phase 1)
NOON_COLOR = "#B45309"           # Noon's synthetic fraud
PAIR_REF_COLOR = "#8A93A3"       # neutral grey for the internal reference pair
