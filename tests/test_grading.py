import math

import pytest

from llm_reliability.grading import (
    answers_match,
    extract_answer,
    is_wrong,
    normalize_number,
    parse_self_check,
)


@pytest.mark.parametrize(
    "text, value, method",
    [
        ("She makes 9 * 2 = 18 dollars.\nAnswer: 18", 18.0, "explicit"),
        ("Answer: $1,250.00", 1250.0, "explicit"),
        ("The answer is 42.", 42.0, "explicit"),
        ("**Answer:** 7", 7.0, "explicit"),
        ("so x = 3\n#### 3", 3.0, "explicit"),
        (r"Therefore \boxed{12}", 12.0, "explicit"),
        ("Answer: -5", -5.0, "explicit"),
        ("Answer: 0.5", 0.5, "explicit"),
        ("First 3 apples, then 4 more, total 7", 7.0, "fallback"),
        ("I don't know.", None, "none"),
        ("", None, "none"),
        (None, None, "none"),
    ],
)
def test_extract_answer(text, value, method):
    got, how = extract_answer(text)
    assert how == method
    if value is None:
        assert got is None
    else:
        assert got == pytest.approx(value)


def test_last_explicit_answer_wins():
    text = "Answer: 5\nWait, let me re-check.\nAnswer: 6"
    assert extract_answer(text) == (6.0, "explicit")


def test_normalize_number():
    assert normalize_number("$1,000") == 1000.0
    assert normalize_number("18.") == 18.0
    assert normalize_number("abc") is None
    assert normalize_number(None) is None


def test_answers_match():
    assert answers_match(18.0, "18")
    assert answers_match(18.0000000001, 18.0)
    assert not answers_match(17.0, "18")
    assert not answers_match(None, "18")


def test_is_wrong():
    assert is_wrong("Answer: 18", "18") == 0
    assert is_wrong("Answer: 19", "18") == 1
    assert is_wrong("no number", "18") == 1


def test_parse_self_check():
    assert parse_self_check("Verdict: INCORRECT\nConfidence: 30") == (1.0, 0.3)
    assert parse_self_check("Verdict: correct\nConfidence: 95%") == (0.0, 0.95)
    assert parse_self_check("**Verdict:** CORRECT\n**Confidence:** 150")[1] == 1.0
    s, c = parse_self_check("I am not sure")
    assert math.isnan(s) and math.isnan(c)
