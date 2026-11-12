import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from llm_reliability.config import EvalConfig
from llm_reliability.evaluate import (
    NotEnoughDataError,
    compute_metrics,
    fast_auc,
    fast_metric,
    paired_auc_difference,
    permutation_test_auc,
    run_experiment,
)
from llm_reliability.features import FEATURE_GROUPS
from llm_reliability.models import MODEL_NAMES, ThresholdRule, make_baselines, make_model


def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 80)
        s = rng.integers(0, 5, 80).astype(float)  # many ties on purpose
        assert fast_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y, p, s = rng.integers(0, 2, 100), rng.integers(0, 2, 100), rng.random(100)
    full = compute_metrics(y, p, s)
    for m in ("roc_auc", "f1_wrong", "balanced_accuracy"):
        assert fast_metric(m, y, p, s) == pytest.approx(full[m])


def test_permutation_test_detects_signal_and_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = permutation_test_auc(y, y + rng.normal(0, 0.5, 300), 300, 0)
    noise = permutation_test_auc(y, rng.random(300), 300, 0)
    assert good["p_value"] < 0.01
    assert noise["p_value"] > 0.05


def test_paired_difference_sign():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    strong, weak = y + rng.normal(0, 0.4, 300), y + rng.normal(0, 2.0, 300)
    res = paired_auc_difference(y, strong, weak, 300, 0)
    assert res["auc_difference"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.05


def test_threshold_rule_learns_direction():
    X = pd.DataFrame({"agreement_rate": [1, 1, 0.8, 0.2, 0.0, 0.1, np.nan]})
    y = np.array([0, 0, 0, 1, 1, 1, 0])
    rule = ThresholdRule("agreement_rate", "low").fit(X, y)
    assert list(rule.predict(X)[:6]) == [0, 0, 0, 1, 1, 1]
    proba = rule.predict_proba(X)
    assert proba.shape == (7, 2) and np.all((proba >= 0) & (proba <= 1))


def test_baselines_only_use_available_features():
    names = make_baselines(["n_words"])
    assert set(names) == {"majority", "random", "rule:n_words"}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_handle_missing_values(name):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((60, 3)), columns=list("abc"))
    X.iloc[::7, 1] = np.nan
    y = (X["a"] > 0.5).astype(int)
    model = make_model(name).fit(X, y)
    assert model.predict_proba(X).shape == (60, 2)


def _synthetic_features(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    df = pd.DataFrame({f: rng.random(n) for g in FEATURE_GROUPS.values() for f in g})
    df["agreement_rate"] = np.clip(np.where(y, 0.3, 0.8) + rng.normal(0, 0.2, n), 0, 1)
    df["hedge_rate"] = np.where(y, 1.0, 0.5) + rng.normal(0, 0.5, n)
    df.insert(0, "id", [f"q{i}" for i in range(n)])
    df["is_wrong"] = y.astype(int)
    return df


def test_run_experiment_outputs():
    res = run_experiment(_synthetic_features(), EvalConfig(n_bootstrap=100, n_permutations=100, cv_folds=3))
    assert set(res.test_metrics["kind"]) == {"baseline", "simple", "combined"}
    assert res.hypotheses["combined_better_than_chance"]["p_value"] < 0.05
    assert res.hypotheses["best_models"]["combined"].startswith("combined:")
    assert {"all features", "only consistency"} <= set(res.ablation["setting"])
    assert len(res.importance) == len(res.feature_sets["combined"])
    assert res.summary["n_train"] + res.summary["n_test"] == 400
    assert res.test_metrics[["roc_auc", "f1_wrong"]].notna().all().all()


def test_run_experiment_save_load_roundtrip(tmp_path):
    res = run_experiment(_synthetic_features(), EvalConfig(n_bootstrap=50, n_permutations=50, cv_folds=3))
    res.save(tmp_path)
    back = type(res).load(tmp_path)
    assert back.summary == res.summary
    assert back.hypotheses["best_models"] == res.hypotheses["best_models"]
    assert len(back.test_metrics) == len(res.test_metrics)


def test_not_enough_wrong_answers():
    df = _synthetic_features(100)
    df["is_wrong"] = 0
    df.loc[:2, "is_wrong"] = 1
    with pytest.raises(NotEnoughDataError, match="wrong"):
        run_experiment(df, EvalConfig())
