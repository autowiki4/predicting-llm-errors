"""An offline, deterministic *fake* LLM for demos and tests.

Why it exists
-------------
Running the real experiment costs API credits and needs a key. The simulator
lets anyone run the whole pipeline end to end - collection, labelling,
features, classifiers, plots, report - in under a minute, with no network.

What it is NOT
--------------
It is not evidence about real LLMs. Its errors are generated from a hand-made
model in which uncertainty *by construction* leaks into hedging, length,
consistency and self-check behaviour. Results from a simulated run only show
that the code works; every report produced from one carries a warning banner.

How it behaves
--------------
For each question the simulator draws a hidden "probability of being wrong"
``p_wrong`` that grows with the number of reasoning steps in the reference
solution. Each call then samples an answer: correct with probability
``1 - p_wrong``, otherwise a plausible wrong number. The response text is the
reference solution (with the final number changed when wrong) plus style
choices whose frequency depends on ``p_wrong``.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np

from ..data import Question
from ..grading import extract_answer, normalize_number
from .base import LLMClient, LLMResponse

HEDGES = [
    "I think",
    "This is probably right, but",
    "If I understand the problem correctly,",
    "It seems that",
    "I'm not completely sure, but",
    "Possibly",
]
CONFIDENT = ["Clearly,", "Definitely,", "Certainly,", "Obviously,"]
SELF_CORRECTIONS = [
    "Wait, let me re-check that step.",
    "Actually, let me double-check the arithmetic.",
    "Hmm, I may have made a mistake there; recalculating.",
]
FILLER = [
    "Let me restate what the problem is asking.",
    "First, I will identify the quantities given in the problem.",
    "It helps to write down each intermediate value.",
    "Now I will combine these results.",
    "Let me make sure the units are consistent.",
]


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def _fmt(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


class SimulatedLLM(LLMClient):
    provider = "simulated"

    def __init__(
        self,
        questions: list[Question],
        seed: int = 0,
        difficulty_shift: float = 0.0,
        model: str = "simulated-llm",
    ) -> None:
        self.questions = {q.id: q for q in questions}
        self.seed = seed
        self.difficulty_shift = difficulty_shift
        self.model = model

    # ---- hidden per-question state ---------------------------------------
    def p_wrong(self, q: Question) -> float:
        rng = np.random.default_rng(_seed(self.seed, q.id, "latent"))
        z = -1.7 + 0.45 * (q.n_steps - 3) + self.difficulty_shift + rng.normal(0, 1.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _wrong_value(self, qid: str, gold: float, rng: np.random.Generator, attractor: bool) -> float:
        # The "attractor" is the single most tempting wrong answer for this question,
        # so repeated wrong samples often agree with each other - as with real LLMs.
        r = np.random.default_rng(_seed(self.seed, qid, "attractor")) if attractor else rng
        options = [gold + 1, gold - 1, gold * 2, gold / 2, gold + 10, gold - 10, gold * 1.5, gold + 5]
        options = [o for o in options if abs(o - gold) > 1e-9 and o >= 0]
        value = float(options[int(r.integers(len(options)))])
        return float(round(value)) if r.random() < 0.85 else round(value, 2)

    def _sample_answer(self, q: Question, p_wrong: float, rng: np.random.Generator) -> float:
        gold = normalize_number(q.gold_answer)
        u = rng.random()
        if u >= p_wrong:
            return gold
        return self._wrong_value(q.id, gold, rng, attractor=u < 0.65 * p_wrong)

    # ---- response rendering -------------------------------------------------
    def _render_solution(self, q: Question, answer: float, p_wrong: float, rng) -> str:
        gold = normalize_number(q.gold_answer)
        wrong = abs(answer - gold) > 1e-9
        lines = [line.strip() for line in q.gold_solution.splitlines() if line.strip()]
        if wrong and lines:
            last = lines[-1]
            gold_str = q.gold_answer
            if gold_str in last:
                idx = last.rfind(gold_str)
                last = last[:idx] + _fmt(answer) + last[idx + len(gold_str) :]
            else:
                last = f"{last} So the result is {_fmt(answer)}."
            lines[-1] = last

        out: list[str] = []
        for _ in range(rng.poisson(0.4 + 2.5 * p_wrong + (0.7 if wrong else 0.0))):
            out.append(FILLER[int(rng.integers(len(FILLER)))])
        if rng.random() < 0.08 + 0.55 * p_wrong:
            out.append(f"{HEDGES[int(rng.integers(len(HEDGES)))]} we can solve it step by step.")
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and rng.random() < 0.45 - 0.35 * p_wrong:
                line = f"{CONFIDENT[int(rng.integers(len(CONFIDENT)))]} {line[0].lower()}{line[1:]}"
            out.append(line)
            if rng.random() < (0.03 + 0.30 * p_wrong) / max(1, len(lines)) * 2:
                out.append(SELF_CORRECTIONS[int(rng.integers(len(SELF_CORRECTIONS)))])
        if rng.random() < 0.1 + 0.4 * p_wrong:
            out.append(f"{HEDGES[int(rng.integers(len(HEDGES)))]} the answer is {_fmt(answer)}.")
        if rng.random() > 0.02 + 0.06 * p_wrong:  # occasionally forget the answer format
            out.append(f"Answer: {_fmt(answer)}")
        return "\n".join(out)

    def _self_check(self, q: Question, prompt: str, p_wrong: float, rng) -> str:
        proposed_section = prompt.split("Proposed solution:", 1)[-1].split("Is the proposed", 1)[0]
        proposed, _ = extract_answer(proposed_section)
        gold = normalize_number(q.gold_answer)
        actually_wrong = proposed is None or abs(proposed - gold) > 1e-9
        p_flag = (0.30 + 0.45 * p_wrong) if actually_wrong else (0.04 + 0.25 * p_wrong)
        says_incorrect = rng.random() < p_flag
        confidence = 100 * (1 - 0.8 * p_wrong) + rng.normal(0, 10) - (25 if says_incorrect else 0)
        confidence = int(np.clip(round(confidence), 0, 100))
        verdict = "INCORRECT" if says_incorrect else "CORRECT"
        return f"Verdict: {verdict}\nConfidence: {confidence}"

    # ---- public API -----------------------------------------------------------
    def complete(self, prompt, *, purpose="answer", meta=None) -> LLMResponse:
        meta = meta or {}
        qid = meta.get("question_id")
        if qid not in self.questions:
            raise KeyError(f"SimulatedLLM needs meta['question_id'] for a known question, got {qid!r}")
        q = self.questions[qid]
        p_wrong = self.p_wrong(q)
        rng = np.random.default_rng(_seed(self.seed, qid, purpose, meta.get("sample_index", 0)))

        if purpose == "self_check":
            text = self._self_check(q, prompt, p_wrong, rng)
            thinking = rng.normal(80 * (1 + p_wrong), 20)
        else:
            answer = self._sample_answer(q, p_wrong, rng)
            text = self._render_solution(q, answer, p_wrong, rng)
            thinking = rng.normal(110 * q.n_steps * (1 + 1.5 * p_wrong), 60)
        n_words = len(re.findall(r"\S+", text))
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(len(prompt.split()) * 1.3),
            output_tokens=int(max(0, thinking) + 1.3 * n_words),
            stop_reason="end_turn",
        )
