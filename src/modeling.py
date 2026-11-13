"""Classifier pipelines, tuning, augmentation arms, thresholds and the saved model object."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, SplineTransformer

from .config import BINARY_COLS, FEATURES, PAYMENT_METHODS, SPLIT, TARGET

SKEWED = ["order_amount", "account_age_days", "delivery_distance_km", "num_orders_last_7d", "prev_refunds"]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, seed: int):
    """70/15/15 stratified train/validation/test split. Returns index arrays."""
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, train_size=SPLIT[0], stratify=df[TARGET], random_state=seed)
    va, te = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                              stratify=df[TARGET].iloc[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


# ---------------------------------------------------------------------------
# Preprocessing + candidate families
# ---------------------------------------------------------------------------
def _onehot():
    return OneHotEncoder(categories=[PAYMENT_METHODS], handle_unknown="error", sparse_output=False)


def preprocessor(family: str) -> ColumnTransformer:
    if family == "logreg":
        return ColumnTransformer([
            ("skewed", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                     SplineTransformer(n_knots=6, degree=3, knots="uniform")), SKEWED),
            ("hour", SplineTransformer(n_knots=7, degree=3, extrapolation="periodic"), ["hour_of_day"]),
            ("pay", _onehot(), ["payment_method"]),
            ("bin", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), BINARY_COLS),
        ])
    # tree families: raw numerics, one-hot payment method (no scaling needed)
    return ColumnTransformer([
        ("num", "passthrough", [c for c in FEATURES if c != "payment_method"]),
        ("pay", _onehot(), ["payment_method"]),
    ])


def make_estimator(family: str, seed: int, class_weight=None, **params):
    if family == "logreg":
        est = LogisticRegression(max_iter=5000, class_weight=class_weight, **params)
    elif family == "random_forest":
        est = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=seed, class_weight=class_weight, **params)
    elif family == "extra_trees":
        est = ExtraTreesClassifier(n_estimators=300, n_jobs=2, random_state=seed, class_weight=class_weight, **params)
    elif family == "hist_gb":
        est = HistGradientBoostingClassifier(random_state=seed, early_stopping=False, class_weight=class_weight, **params)
    else:
        raise ValueError(family)
    return Pipeline([("pre", preprocessor(family)), ("clf", est)])


SEARCH_SPACES = {
    "logreg": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "random_forest": {"clf__max_depth": [None, 12, 20], "clf__min_samples_leaf": [1, 5, 10, 20],
                      "clf__max_features": ["sqrt", 0.5]},
    "extra_trees": {"clf__max_depth": [None, 12, 20], "clf__min_samples_leaf": [1, 5, 10, 20],
                    "clf__max_features": ["sqrt", 0.5]},
    "hist_gb": {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [20, 50, 100], "clf__l2_regularization": [0.0, 0.1, 1.0],
                "clf__max_iter": [200, 400]},
}
FAMILY_LABELS = {"logreg": "Logistic regression (splines)", "random_forest": "Random forest",
                 "extra_trees": "Extra trees", "hist_gb": "HistGradientBoosting"}


def tune_family(family: str, X: pd.DataFrame, y: pd.Series, seed: int, n_iter: int = 10):
    """3-fold stratified CV on the TRAINING partition only; scoring = average precision."""
    space = SEARCH_SPACES[family]
    n_combos = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(make_estimator(family, seed), space, n_iter=min(n_iter, n_combos),
                                scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                random_state=seed, n_jobs=1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    cvres = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    return best, float(search.best_score_), cvres


# ---------------------------------------------------------------------------
# Thresholds and calibration
# ---------------------------------------------------------------------------
def max_f1_threshold(y_true, proba) -> float:
    """Pre-declared rule: the probability threshold that maximises F1 on the given (validation) data."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12), 0)
    return float(t[int(np.argmax(f1))])


class PlattCalibrator:
    """Sigmoid (Platt) recalibration of a frozen model's scores, fitted on validation data only.

    Equivalent to ``CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')`` with a
    single calibration set; written out explicitly so the fitted mapping is transparent and
    version-independent."""

    def fit(self, proba, y):
        z = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / (1 - np.clip(proba, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        self.a_, self.b_ = float(self.lr_.coef_[0, 0]), float(self.lr_.intercept_[0])
        return self

    def transform(self, proba):
        z = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / (1 - np.clip(proba, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-(self.a_ * z + self.b_)))


@dataclass
class FraudModel:
    """Everything the Phase 4 app needs: pipeline, calibrator, frozen threshold, provenance."""
    pipeline: Pipeline
    threshold: float                         # applied to CALIBRATED probability
    calibrator: PlattCalibrator | None = None
    feature_order: list = field(default_factory=lambda: list(FEATURES))
    metadata: dict = field(default_factory=dict)

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_order])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.raw_proba(X)
        return self.calibrator.transform(p) if self.calibrator is not None else p

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


# ---------------------------------------------------------------------------
# Augmentation arms
# ---------------------------------------------------------------------------
def build_training_set(X_tr: pd.DataFrame, y_tr: pd.Series, synth: pd.DataFrame | None):
    if synth is None or len(synth) == 0:
        return X_tr[FEATURES].reset_index(drop=True), y_tr.reset_index(drop=True)
    X = pd.concat([X_tr[FEATURES], synth[FEATURES]], ignore_index=True)
    y = pd.concat([y_tr.reset_index(drop=True), pd.Series(np.ones(len(synth), dtype=int))], ignore_index=True)
    return X, y


def fit_and_score(family, params, seed, X_fit, y_fit, X_val, y_val, class_weight=None):
    model = make_estimator(family, seed, class_weight=class_weight, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit)
    p_val = model.predict_proba(X_val[FEATURES])[:, 1]
    return model, p_val, float(average_precision_score(y_val, p_val))
