"""Stage 2b - Label correctness: extract the model's final answer and compare it to gold.

The label we try to predict is ``is_wrong`` (1 = the model's answer is wrong).
We treat "wrong" as the positive class because the goal is to *flag* errors.
"""

from __future__ import annotations

import math
import re

_NUMBER = r"-?\$?\s*\d[\d,]*(?:\.\d+)?|-?\$?\s*\.\d+"
_ANSWER_LINE = re.compile(r"answer\s*(?:is)?\s*[:=]?\s*\**\s*(" + _NUMBER + r")", re.IGNORECASE)
_HASH_LINE = re.compile(r"####\s*(" + _NUMBER + r")")
_BOXED = re.compile(r"\\boxed\{\s*(" + _NUMBER + r")\s*\}")
_ANY_NUMBER = re.compile(_NUMBER)

_VERDICT = re.compile(r"verdict\s*[:\-]?\s*\**\s*(correct|incorrect)", re.IGNORECASE)
_CONFIDENCE = re.compile(r"confidence\s*[:\-]?\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE)


def normalize_number(text: str | None) -> float | None:
    """Turn strings like '$1,250.00' or ' 18 ' into floats. Returns None if impossible."""
    if text is None:
        return None
    cleaned = text.replace("$", "").replace(",", "").replace(" ", "").strip().rstrip(".")
    if cleaned in {"", "-", "."}:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def extract_answer(text: str | None) -> tuple[float | None, str]:
    """Return (numeric answer, how it was found).

    The method is one of:
      * ``"explicit"`` - the model followed the requested ``Answer: <number>`` format
        (or used ``####`` / ``\\boxed{}``)
      * ``"fallback"`` - no explicit answer; we used the last number in the text
      * ``"none"``     - no number at all
    """
    if not text:
        return None, "none"
    for pattern in (_ANSWER_LINE, _HASH_LINE, _BOXED):
        matches = pattern.findall(text)
        if matches:
            value = normalize_number(matches[-1])
            if value is not None:
                return value, "explicit"
    numbers = _ANY_NUMBER.findall(text)
    for candidate in reversed(numbers):
        value = normalize_number(candidate)
        if value is not None:
            return value, "fallback"
    return None, "none"


def answers_match(predicted: float | None, gold: float | str | None, tol: float = 1e-6) -> bool:
    """Numeric equality with a small tolerance (so 18 == 18.0)."""
    if isinstance(gold, str):
        gold = normalize_number(gold)
    if predicted is None or gold is None:
        return False
    return abs(predicted - gold) <= tol * max(1.0, abs(gold))


def is_wrong(response_text: str | None, gold_answer: str) -> int:
    """1 if the response's final answer does not match the gold answer, else 0."""
    predicted, _ = extract_answer(response_text)
    return 0 if answers_match(predicted, gold_answer) else 1


def parse_self_check(text: str | None) -> tuple[float, float]:
    """Parse the self-check reply.

    Returns (says_incorrect, confidence) where says_incorrect is 1.0/0.0 and
    confidence is in [0, 1]. Either value is NaN when it can't be parsed.
    """
    says_incorrect = math.nan
    confidence = math.nan
    if not text:
        return says_incorrect, confidence
    verdicts = _VERDICT.findall(text)
    if verdicts:
        says_incorrect = 1.0 if verdicts[-1].lower() == "incorrect" else 0.0
    conf = _CONFIDENCE.findall(text)
    if conf:
        confidence = min(max(float(conf[-1]) / 100.0, 0.0), 1.0)
    return says_incorrect, confidence
