import math

import pytest

from llm_reliability.features import (
    COMBINED_GROUPS,
    FEATURE_DESCRIPTIONS,
    FEATURE_GROUPS,
    SIMPLE_GROUPS,
    EmptyResponsesError,
    available_features,
    build_feature_table,
    consistency_features,
    record_features,
    text_features,
)


def _record(main, samples=(), check=None, gold="18", tokens=100):
    return {
        "id": "q1",
        "gold_answer": gold,
        "n_steps": 2,
        "response": {"text": main, "output_tokens": tokens},
        "samples": [{"text": s} for s in samples],
        "self_check": {"text": check} if check is not None else None,
    }


def test_every_feature_is_documented():
    for feats in FEATURE_GROUPS.values():
        for f in feats:
            assert f in FEATURE_DESCRIPTIONS


def test_simple_groups_are_free():
    assert "consistency" not in SIMPLE_GROUPS and "self_check" not in SIMPLE_GROUPS
    assert set(SIMPLE_GROUPS) <= set(COMBINED_GROUPS)


def test_text_features_counts():
    f = text_features("I think it is probably 5. Clearly it is 5.\nWait, actually 6.\nAnswer: 6")
    assert f["hedge_count"] == 2  # "I think", "probably"
    assert f["confident_count"] == 1
    assert f["self_correction_count"] == 2  # "wait", "actually"
    assert f["answer_explicit"] == 1.0
    assert f["answer_missing"] == 0.0
    assert f["answer_is_integer"] == 1.0
    assert f["n_lines"] == 3
    assert f["hedge_rate"] == pytest.approx(100 * 2 / f["n_words"])


def test_hedge_words_need_word_boundaries():
    # "mighty" must not count as "might", "unlikely" is its own hedge
    f = text_features("A mighty river. Answer: 3")
    assert f["hedge_count"] == 0


def test_text_features_empty():
    f = text_features("")
    assert f["n_words"] == 0 and f["hedge_rate"] == 0
    assert f["answer_missing"] == 1.0
    assert math.isnan(f["answer_log_magnitude"])


def test_consistency_all_agree():
    f = consistency_features("Answer: 4", ["Answer: 4", "Answer: 4.0", "so 4"])
    assert f["agreement_rate"] == 1.0
    assert f["n_unique_answers"] == 1
    assert f["answer_entropy"] == 0.0
    assert f["majority_agrees"] == 1.0


def test_consistency_disagreement():
    f = consistency_features("Answer: 4", ["Answer: 5", "Answer: 5", "Answer: 4"])
    assert f["agreement_rate"] == pytest.approx(1 / 3)
    assert f["n_unique_answers"] == 2
    assert f["answer_entropy"] == pytest.approx(1.0)  # 2 vs 2 split = 1 bit
    assert f["majority_agrees"] == 1.0  # tie counts as majority


def test_consistency_without_samples_is_nan():
    f = consistency_features("Answer: 4", [])
    assert all(math.isnan(v) for v in f.values())


def test_record_features_label():
    right = record_features(_record("Answer: 18", ["Answer: 18"], "Verdict: CORRECT\nConfidence: 90"))
    wrong = record_features(_record("Answer: 17", ["Answer: 18"], "Verdict: INCORRECT\nConfidence: 20"))
    assert right["is_wrong"] == 0 and wrong["is_wrong"] == 1
    assert wrong["selfcheck_says_incorrect"] == 1.0
    assert wrong["selfcheck_confidence"] == pytest.approx(0.2)
    assert right["n_output_tokens"] == 100


def test_build_feature_table_and_available_features():
    df = build_feature_table([_record("Answer: 18", tokens=None)])
    assert list(df.columns)[0] == "id" and "is_wrong" in df.columns
    simple = available_features(df, SIMPLE_GROUPS)
    combined = available_features(df, COMBINED_GROUPS)
    assert "n_output_tokens" not in simple  # all NaN -> dropped
    assert "agreement_rate" not in combined  # no samples collected
    assert "selfcheck_confidence" not in combined


def test_build_feature_table_rejects_empty():
    with pytest.raises(EmptyResponsesError):
        build_feature_table([])
