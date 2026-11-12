"""Stage 5 - Evaluate: do the features actually predict wrong answers?

Protocol (chosen to avoid fooling ourselves):

1. **One held-out test set.** 25% of questions are set aside (stratified by the
   label) and never touched until the end.
2. **Model selection by cross-validation on the training set only.** The "best"
   simple and combined models are picked by mean 5-fold CV ROC-AUC, *not* by
   their test score - picking by test score would make the test score optimistic.
3. **Test-set metrics with uncertainty.** Every test metric gets a 95% bootstrap
   confidence interval, because with a few hundred questions a difference of a
   couple of points can be pure noise.
4. **Hypothesis tests.**
   * *Better than chance?* A label-permutation test of the combined model's test
     ROC-AUC (the null hypothesis is AUC = 0.5).
   * *Combined better than simple?* *Better than the best single rule?* Paired
     bootstrap of the AUC difference on the same test questions.
5. **Which features matter?** Feature-group ablations (CV on train) and
   permutation importance (on test).

Primary metric: ROC-AUC for the "wrong" class - threshold-free and unaffected by
class imbalance. Also reported: accuracy, balanced accuracy, precision / recall /
F1 for wrong answers (the proposal's "error detection" metrics), macro F1 and
PR-AUC (average precision; its chance level equals the error rate).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import EvalConfig
from .features import (
    COMBINED_GROUPS,
    FEATURE_GROUPS,
    SIMPLE_GROUPS,
    available_features,
)
from .models import make_baselines, make_model

log = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
    "precision_wrong", "recall_wrong", "f1_wrong", "f1_macro",
]
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "f1_wrong": "f1",
}


class NotEnoughDataError(ValueError):
    """Raised when there are too few wrong (or right) answers to evaluate anything."""


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": roc_auc_score(y_true, y_score) if two_classes else np.nan,
        "pr_auc": average_precision_score(y_true, y_score) if two_classes else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_wrong": precision_score(y_true, y_pred, zero_division=0),
        "recall_wrong": recall_score(y_true, y_pred, zero_division=0),
        "f1_wrong": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank formula (same value as sklearn, ~100x faster).

    Used inside bootstrap / permutation loops that compute thousands of AUCs.
    """
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_score)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> float:
    """The few metrics we bootstrap, written with plain numpy for speed."""
    if metric == "roc_auc":
        return fast_auc(y_true, y_score)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    if metric == "f1_wrong":
        return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
    if metric == "balanced_accuracy":
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        return float((tpr + tnr) / 2)
    raise ValueError(f"No fast implementation for {metric!r}")


def _bootstrap_indices(n: int, y: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            yield idx


def bootstrap_ci(y_true, y_pred, y_score, metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile-bootstrap confidence interval for one test metric."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    vals = [
        fast_metric(metric, y_true[i], y_pred[i], y_score[i])
        for i in _bootstrap_indices(len(y_true), y_true, n_boot, seed)
    ]
    if not vals:
        return (np.nan, np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_auc_difference(y_true, score_a, score_b, n_boot: int, seed: int) -> dict[str, float]:
    """Bootstrap the difference AUC(a) - AUC(b) on the same test questions."""
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    observed = fast_auc(y_true, score_a) - fast_auc(y_true, score_b)
    diffs = np.array([
        fast_auc(y_true[i], score_a[i]) - fast_auc(y_true[i], score_b[i])
        for i in _bootstrap_indices(len(y_true), y_true, n_boot, seed)
    ])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value (consistent with the 95% CI: p < 0.05 <=> CI excludes 0).
    tail = min(np.sum(diffs <= 0), np.sum(diffs >= 0))
    p = float(min(1.0, 2 * (tail + 1) / (len(diffs) + 1)))
    return {"auc_difference": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p_value": p}


def permutation_test_auc(y_true, y_score, n_perm: int, seed: int) -> dict[str, float]:
    """Is the AUC better than chance? Shuffle labels to build the null distribution."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    observed = fast_auc(y_true, y_score)
    null = np.array([fast_auc(rng.permutation(y_true), y_score) for _ in range(n_perm)])
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {"auc": float(observed), "null_mean": float(null.mean()), "p_value": p}


def _positive_score(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class ExperimentResults:
    summary: dict[str, Any]
    test_metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    ablation: pd.DataFrame
    importance: pd.DataFrame
    test_predictions: pd.DataFrame
    hypotheses: dict[str, Any]
    feature_sets: dict[str, list[str]] = field(default_factory=dict)

    def save(self, results_dir: str | Path) -> None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.test_metrics.to_csv(results_dir / "test_metrics.csv", index=False)
        self.cv_metrics.to_csv(results_dir / "cv_metrics.csv", index=False)
        self.ablation.to_csv(results_dir / "ablation.csv", index=False)
        self.importance.to_csv(results_dir / "feature_importance.csv", index=False)
        self.test_predictions.to_csv(results_dir / "test_predictions.csv", index=False)
        payload = {
            "summary": self.summary,
            "hypotheses": self.hypotheses,
            "feature_sets": self.feature_sets,
        }
        (results_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))

    @classmethod
    def load(cls, results_dir: str | Path) -> ExperimentResults:
        results_dir = Path(results_dir)
        payload = json.loads((results_dir / "summary.json").read_text())
        return cls(
            summary=payload["summary"],
            test_metrics=pd.read_csv(results_dir / "test_metrics.csv"),
            cv_metrics=pd.read_csv(results_dir / "cv_metrics.csv"),
            ablation=pd.read_csv(results_dir / "ablation.csv"),
            importance=pd.read_csv(results_dir / "feature_importance.csv"),
            test_predictions=pd.read_csv(results_dir / "test_predictions.csv"),
            hypotheses=payload["hypotheses"],
            feature_sets=payload["feature_sets"],
        )


def _check_enough_data(y: np.ndarray, cfg: EvalConfig) -> None:
    n_wrong, n_right = int(y.sum()), int(len(y) - y.sum())
    min_needed = max(2 * cfg.cv_folds, int(np.ceil(2 / cfg.test_size)) + cfg.cv_folds)
    if min(n_wrong, n_right) < min_needed:
        raise NotEnoughDataError(
            f"Only {n_wrong} wrong and {n_right} correct answers; need at least {min_needed} "
            "of each to train and test. Use more questions, a harder dataset, or a weaker "
            "model (a model that is almost always right leaves nothing to predict)."
        )


def run_experiment(df: pd.DataFrame, cfg: EvalConfig) -> ExperimentResults:
    """Run the full evaluation on a feature table from ``build_feature_table``."""
    y_all = df["is_wrong"].to_numpy(dtype=int)
    _check_enough_data(y_all, cfg)

    feature_sets = {
        "simple": available_features(df, SIMPLE_GROUPS),
        "combined": available_features(df, COMBINED_GROUPS),
    }
    if feature_sets["combined"] == feature_sets["simple"]:
        log.warning("No consistency/self-check features collected; 'combined' equals 'simple'.")
    all_feats = feature_sets["combined"]

    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, stratify=y_all, random_state=cfg.seed
    )
    y_train = train_df["is_wrong"].to_numpy(dtype=int)
    y_test = test_df["is_wrong"].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    test_rows, cv_rows = [], []
    predictions = pd.DataFrame({"id": test_df["id"].to_numpy(), "is_wrong": y_test})
    fitted: dict[str, Any] = {}

    def record_test(name, kind, feature_set, model_name, model, X_test):
        score = _positive_score(model, X_test)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred, score)
        for metric in ("roc_auc", "f1_wrong", "balanced_accuracy"):
            lo, hi = bootstrap_ci(y_test, pred, score, metric, cfg.n_bootstrap, cfg.seed)
            metrics[f"{metric}_ci_low"], metrics[f"{metric}_ci_high"] = lo, hi
        test_rows.append(
            {"name": name, "kind": kind, "feature_set": feature_set, "model": model_name, **metrics}
        )
        predictions[f"score:{name}"] = score
        predictions[f"pred:{name}"] = pred

    # ---- baselines ----------------------------------------------------------
    for name, baseline in make_baselines(all_feats, seed=cfg.seed).items():
        model = clone(baseline).fit(train_df[all_feats], y_train)
        fitted[name] = model
        record_test(name, "baseline", "-", name, model, test_df[all_feats])

    # ---- classifiers ----------------------------------------------------------
    for set_name, feats in feature_sets.items():
        for model_name in cfg.models:
            name = f"{set_name}:{model_name}"
            model = make_model(model_name, seed=cfg.seed)
            cvres = cross_validate(model, train_df[feats], y_train, cv=cv, scoring=CV_SCORING)
            row = {"name": name, "feature_set": set_name, "model": model_name}
            for metric in CV_SCORING:
                row[f"{metric}_mean"] = float(np.mean(cvres[f"test_{metric}"]))
                row[f"{metric}_std"] = float(np.std(cvres[f"test_{metric}"]))
            cv_rows.append(row)
            model.fit(train_df[feats], y_train)
            fitted[name] = model
            record_test(name, set_name, set_name, model_name, model, test_df[feats])

    test_metrics = pd.DataFrame(test_rows)
    cv_metrics = pd.DataFrame(cv_rows)

    # ---- pick the best model per feature set by CV (never by test score) -----
    best = {}
    for set_name in feature_sets:
        sub = cv_metrics[cv_metrics.feature_set == set_name]
        best[set_name] = sub.sort_values("roc_auc_mean", ascending=False).iloc[0]["name"]
    rule_rows = test_metrics[test_metrics.name.str.startswith("rule:")]
    # The best single rule is also chosen on training data (its train ROC-AUC).
    best_rule = None
    if len(rule_rows):
        train_auc = {
            n: fast_auc(y_train, _positive_score(fitted[n], train_df[all_feats]))
            for n in rule_rows.name
        }
        best_rule = max(train_auc, key=train_auc.get)

    # ---- hypothesis tests --------------------------------------------------
    combined_score = predictions[f"score:{best['combined']}"].to_numpy()
    simple_score = predictions[f"score:{best['simple']}"].to_numpy()
    hypotheses: dict[str, Any] = {
        "best_models": {**best, "rule": best_rule},
        "combined_better_than_chance": permutation_test_auc(
            y_test, combined_score, cfg.n_permutations, cfg.seed
        ),
        "simple_better_than_chance": permutation_test_auc(
            y_test, simple_score, cfg.n_permutations, cfg.seed
        ),
        "combined_vs_simple": paired_auc_difference(
            y_test, combined_score, simple_score, cfg.n_bootstrap, cfg.seed
        ),
    }
    if best_rule:
        hypotheses["combined_vs_best_rule"] = paired_auc_difference(
            y_test,
            combined_score,
            predictions[f"score:{best_rule}"].to_numpy(),
            cfg.n_bootstrap,
            cfg.seed,
        )

    # ---- ablation: which feature groups carry the signal? ---------------------
    ablation_rows = []
    available_groups = [g for g in COMBINED_GROUPS if available_features(df, [g])]
    settings = [("all features", available_groups), ("simple (free) features", SIMPLE_GROUPS)]
    settings += [(f"only {g}", [g]) for g in available_groups]
    settings += [(f"all minus {g}", [x for x in available_groups if x != g]) for g in available_groups]
    for label, groups in settings:
        feats = available_features(df, groups)
        if not feats:
            continue
        cvres = cross_validate(
            make_model(cfg.ablation_model, seed=cfg.seed),
            train_df[feats],
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "f1_wrong": "f1"},
        )
        ablation_rows.append({
            "setting": label,
            "groups": "+".join(groups),
            "n_features": len(feats),
            "roc_auc_mean": float(np.mean(cvres["test_roc_auc"])),
            "roc_auc_std": float(np.std(cvres["test_roc_auc"])),
            "f1_wrong_mean": float(np.mean(cvres["test_f1_wrong"])),
        })
    ablation = pd.DataFrame(ablation_rows)

    # ---- permutation importance of the best combined model (on test) ---------
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    perm = permutation_importance(
        fitted[best["combined"]],
        test_df[all_feats],
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=cfg.seed,
    )
    importance = pd.DataFrame({
        "feature": all_feats,
        "group": [group_of[f] for f in all_feats],
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)

    summary = {
        "n_questions": int(len(df)),
        "n_wrong": int(y_all.sum()),
        "error_rate": float(y_all.mean()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_test_wrong": int(y_test.sum()),
        "cv_folds": cfg.cv_folds,
        "seed": cfg.seed,
    }
    return ExperimentResults(
        summary=summary,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
        ablation=ablation,
        importance=importance,
        test_predictions=predictions,
        hypotheses=hypotheses,
        feature_sets=feature_sets,
    )
