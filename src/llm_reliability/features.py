"""Stage 3 - Features: turn each response into a row of numbers.

Features are grouped so we can ask *which kind of signal* helps. The first three
groups need only the single response we are judging ("free" signals). The last
two need extra LLM calls ("paid" signals).

=====================  ========  ==========================================================
group                  cost      intuition
=====================  ========  ==========================================================
length                 free      long, rambling answers may signal a struggle
confidence_language    free      hedging ("I think", "probably") vs. certainty ("clearly")
answer_format          free      did the model follow the requested answer format?
consistency            paid      if re-asked, does the model give the same answer?
self_check             paid      does the model say its own answer is wrong?
=====================  ========  ==========================================================

``SIMPLE`` = the free groups, ``COMBINED`` = everything. Comparing the two answers
the proposal's question "do the combined features beat simple ones?"

Every feature is documented in ``FEATURE_DESCRIPTIONS`` (and in docs/METHODOLOGY.md).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .grading import answers_match, extract_answer, parse_self_check

# ---------------------------------------------------------------------------
# Word lists. Matched case-insensitively on word boundaries.
# ---------------------------------------------------------------------------
HEDGE_PHRASES = [
    "i think", "i believe", "i guess", "probably", "likely", "unlikely", "might", "perhaps",
    "possibly", "maybe", "seems", "it appears", "appears to", "not sure", "not certain",
    "unclear", "assuming", "i assume", "roughly", "approximately", "could be", "if i understand",
]
CONFIDENT_PHRASES = [
    "definitely", "certainly", "clearly", "obviously", "surely", "undoubtedly",
    "without a doubt", "of course", "i am confident", "i'm confident", "for sure",
]
SELF_CORRECTION_PHRASES = [
    "wait", "actually", "let me re-check", "let me recheck", "let me double-check",
    "let me double check", "let me reconsider", "mistake", "correction", "oops",
    "on second thought", "recalculat", "i made an error",
]


def _phrase_pattern(phrases: list[str]) -> re.Pattern:
    # "recalculat" is a stem (recalculate / recalculating), so it gets no closing boundary.
    parts = [re.escape(p) + ("" if p == "recalculat" else r"\b") for p in phrases]
    return re.compile(r"\b(?:" + "|".join(parts) + ")", re.IGNORECASE)


HEDGE_RE = _phrase_pattern(HEDGE_PHRASES)
CONFIDENT_RE = _phrase_pattern(CONFIDENT_PHRASES)
SELF_CORRECTION_RE = _phrase_pattern(SELF_CORRECTION_PHRASES)
_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

FEATURE_GROUPS: dict[str, list[str]] = {
    "length": ["n_chars", "n_words", "n_sentences", "n_lines", "n_numbers", "n_output_tokens"],
    "confidence_language": [
        "hedge_count", "hedge_rate", "confident_count", "confident_rate", "self_correction_count",
    ],
    "answer_format": [
        "answer_explicit", "answer_missing", "answer_is_integer", "answer_log_magnitude",
    ],
    "consistency": ["agreement_rate", "n_unique_answers", "answer_entropy", "majority_agrees"],
    "self_check": ["selfcheck_says_incorrect", "selfcheck_confidence"],
}
SIMPLE_GROUPS = ["length", "confidence_language", "answer_format"]
COMBINED_GROUPS = list(FEATURE_GROUPS)

FEATURE_DESCRIPTIONS = {
    "n_chars": "Characters in the response",
    "n_words": "Words in the response",
    "n_sentences": "Sentences in the response",
    "n_lines": "Non-empty lines (roughly, reasoning steps)",
    "n_numbers": "Numbers written in the response",
    "n_output_tokens": "Output tokens billed by the API (includes hidden reasoning tokens)",
    "hedge_count": "Hedging phrases ('I think', 'probably', ...)",
    "hedge_rate": "Hedging phrases per 100 words",
    "confident_count": "Certainty phrases ('clearly', 'definitely', ...)",
    "confident_rate": "Certainty phrases per 100 words",
    "self_correction_count": "Self-correction phrases ('wait', 'actually', 'mistake', ...)",
    "answer_explicit": "1 if the model used the requested 'Answer: <number>' format",
    "answer_missing": "1 if no number could be found at all",
    "answer_is_integer": "1 if the final answer is a whole number (GSM8K answers always are)",
    "answer_log_magnitude": "log10(1 + |answer|)",
    "agreement_rate": "Share of extra samples whose answer equals the main answer",
    "n_unique_answers": "Distinct answers across main + extra samples",
    "answer_entropy": "Entropy (bits) of the answer distribution across all samples",
    "majority_agrees": "1 if the main answer is the most common answer across samples",
    "selfcheck_says_incorrect": "1 if the model judged its own answer INCORRECT",
    "selfcheck_confidence": "Model's stated confidence (0-1) that its answer is correct",
}


def features_for_groups(groups: list[str]) -> list[str]:
    return [f for g in groups for f in FEATURE_GROUPS[g]]


def text_features(text: str) -> dict[str, float]:
    """Length, confidence-language and answer-format features of one response."""
    text = text or ""
    n_words = len(_WORD_RE.findall(text))
    hedges = len(HEDGE_RE.findall(text))
    confident = len(CONFIDENT_RE.findall(text))
    per100 = 100.0 / n_words if n_words else 0.0
    answer, method = extract_answer(text)
    return {
        "n_chars": float(len(text)),
        "n_words": float(n_words),
        "n_sentences": float(len([s for s in _SENTENCE_RE.findall(text) if s.strip()])),
        "n_lines": float(len([line for line in text.splitlines() if line.strip()])),
        "n_numbers": float(len(_NUMBER_RE.findall(text))),
        "hedge_count": float(hedges),
        "hedge_rate": hedges * per100,
        "confident_count": float(confident),
        "confident_rate": confident * per100,
        "self_correction_count": float(len(SELF_CORRECTION_RE.findall(text))),
        "answer_explicit": float(method == "explicit"),
        "answer_missing": float(method == "none"),
        "answer_is_integer": float(answer is not None and abs(answer - round(answer)) < 1e-9),
        "answer_log_magnitude": math.log10(1 + abs(answer)) if answer is not None else math.nan,
    }


def _answer_key(value: float | None) -> str:
    return "none" if value is None else f"{value:.6g}"


def consistency_features(main_text: str, sample_texts: list[str]) -> dict[str, float]:
    """How consistent is the model across repeated samples of the same question?"""
    if not sample_texts:
        return {k: math.nan for k in FEATURE_GROUPS["consistency"]}
    main_answer, _ = extract_answer(main_text)
    sample_answers = [extract_answer(t)[0] for t in sample_texts]
    agree = [answers_match(a, main_answer) if main_answer is not None else False for a in sample_answers]
    keys = [_answer_key(main_answer)] + [_answer_key(a) for a in sample_answers]
    counts = Counter(keys)
    probs = np.array(list(counts.values()), dtype=float) / len(keys)
    entropy = float(-(probs * np.log2(probs)).sum())
    top_count = max(counts.values())
    return {
        "agreement_rate": float(np.mean(agree)),
        "n_unique_answers": float(len(counts)),
        "answer_entropy": entropy + 0.0,  # +0.0 turns -0.0 into 0.0
        "majority_agrees": float(counts[_answer_key(main_answer)] == top_count),
    }


def self_check_features(self_check_text: str | None) -> dict[str, float]:
    says_incorrect, confidence = parse_self_check(self_check_text)
    return {"selfcheck_says_incorrect": says_incorrect, "selfcheck_confidence": confidence}


def record_features(record: dict[str, Any]) -> dict[str, Any]:
    """All features + the label for one collected record."""
    text = record["response"]["text"]
    row: dict[str, Any] = {"id": record["id"]}
    row.update(text_features(text))
    out_tokens = record["response"].get("output_tokens")
    row["n_output_tokens"] = float(out_tokens) if out_tokens is not None else math.nan
    row.update(consistency_features(text, [s["text"] for s in record.get("samples") or []]))
    sc = record.get("self_check")
    row.update(self_check_features(sc["text"] if sc else None))

    predicted, _ = extract_answer(text)
    row["predicted_answer"] = predicted
    row["gold_answer"] = record["gold_answer"]
    row["n_steps"] = record.get("n_steps")
    row["is_wrong"] = int(not answers_match(predicted, record["gold_answer"]))
    return row


class EmptyResponsesError(ValueError):
    """Raised when there are no collected responses to turn into features."""


def build_feature_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per question: id, every feature, metadata, and the ``is_wrong`` label."""
    if not records:
        raise EmptyResponsesError(
            "No responses to build features from - run `llmrel collect` first "
            "(and check that it succeeded)"
        )
    df = pd.DataFrame([record_features(r) for r in records])
    ordered = ["id"] + [f for g in FEATURE_GROUPS.values() for f in g]
    meta = ["predicted_answer", "gold_answer", "n_steps", "is_wrong"]
    return df[ordered + meta]


def available_features(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """Features from ``groups`` that actually have data (e.g. no consistency if n_samples=0)."""
    return [f for f in features_for_groups(groups) if f in df.columns and df[f].notna().any()]
