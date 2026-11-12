"""Stage 4 - Classifiers and baselines.

Classifiers (the proposal's "2-3 simple classifiers"):

* ``logistic_regression`` - linear, interpretable coefficients
* ``random_forest``       - non-linear, robust, little tuning
* ``gradient_boosting``   - usually the strongest on small tabular data

All three use ``class_weight="balanced"``: wrong answers are the minority class,
and without re-weighting a model can score high accuracy by never flagging anything.

Baselines (what the classifiers must beat):

* ``majority``      - always predicts the most common class ("always correct")
* ``random``        - guesses at the base rate
* ``rule:<feature>``- a single threshold on one feature, learned on training data
                      (e.g. "flag if agreement_rate < 0.6"). If a classifier cannot
                      beat the best single rule, the extra machinery isn't earning its keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]

# (feature, direction): "high" means a high value suggests a wrong answer.
RULE_BASELINES = [
    ("agreement_rate", "low"),
    ("selfcheck_says_incorrect", "high"),
    ("selfcheck_confidence", "low"),
    ("hedge_rate", "high"),
    ("n_words", "high"),
]


def make_model(name: str, seed: int = 0) -> Pipeline:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "logistic_regression":
        steps = [
            ("impute", imputer),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)),
        ]
    elif name == "random_forest":
        steps = [
            ("impute", imputer),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=1,  # data is small; process start-up would cost more than it saves
                ),
            ),
        ]
    elif name == "gradient_boosting":
        steps = [
            ("impute", imputer),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline(steps)


class ThresholdRule(ClassifierMixin, BaseEstimator):
    """Flag an answer as wrong when one feature crosses a threshold.

    The threshold is chosen on the training data to maximise F1 for the
    "wrong" class, so the rule is a fair, tuned baseline - not a straw man.
    """

    def __init__(self, feature: str, direction: str = "high"):
        self.feature = feature
        self.direction = direction

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.feature].to_numpy(dtype=float)
        values = np.where(np.isnan(values), self.fill_value_, values)
        return values if self.direction == "high" else -values

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        col = X[self.feature].to_numpy(dtype=float)
        self.fill_value_ = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
        score = self._score(X)
        best_f1, best_thr = -1.0, float(np.max(score)) + 1.0
        for thr in np.unique(score):
            pred = score >= thr
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            fn = np.sum(~pred & (y == 1))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        self.threshold_ = best_thr
        self.score_min_ = float(np.min(score))
        self.score_max_ = float(np.max(score))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._score(X) >= self.threshold_).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # A monotone rescaling of the feature: good enough for ROC/PR curves.
        span = max(self.score_max_ - self.score_min_, 1e-12)
        p = np.clip((self._score(X) - self.score_min_) / span, 0.0, 1.0)
        return np.column_stack([1 - p, p])


def make_baselines(available: list[str], seed: int = 0) -> dict[str, BaseEstimator]:
    baselines: dict[str, BaseEstimator] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "random": DummyClassifier(strategy="stratified", random_state=seed),
    }
    for feature, direction in RULE_BASELINES:
        if feature in available:
            baselines[f"rule:{feature}"] = ThresholdRule(feature, direction)
    return baselines
