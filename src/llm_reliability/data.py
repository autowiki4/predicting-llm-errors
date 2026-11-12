"""Stage 1 - Questions: load a public reasoning dataset with known answers.

We use GSM8K (Cobbe et al., 2021): grade-school math word problems whose answers
are single numbers. That makes correctness labelling objective and automatic -
no human grading needed - which is exactly what a one-semester project needs.

The 1,319-question test split is bundled in ``data/raw/gsm8k_test.jsonl``
(MIT licensed, see ``data/README.md``). ``download_gsm8k`` can fetch the
7,473-question train split too if you want more data.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import resolve_path

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/{split}.jsonl"
)

_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold_answer: str  # the reference final answer, e.g. "18"
    gold_solution: str  # the reference step-by-step solution (calculator tags removed)
    n_steps: int  # number of reasoning steps in the reference solution (a difficulty proxy)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_gsm8k_answer(answer_field: str) -> tuple[str, str, int]:
    """Split a GSM8K ``answer`` field into (final answer, clean solution, n_steps).

    GSM8K answers look like::

        Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
        She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
        #### 18
    """
    if "####" not in answer_field:
        raise ValueError("GSM8K answer field has no '####' final-answer marker")
    solution, final = answer_field.rsplit("####", 1)
    final = final.strip().replace(",", "")
    clean_solution = _CALC_ANNOTATION.sub("", solution).strip()
    steps = [line for line in clean_solution.splitlines() if line.strip()]
    return final, clean_solution, max(1, len(steps))


def load_gsm8k(
    path: str | Path = "data/raw/gsm8k_test.jsonl",
    n: int | None = None,
    seed: int = 0,
) -> list[Question]:
    """Load GSM8K questions. If ``n`` is given, draw a reproducible random sample."""
    full_path = resolve_path(path)
    if not full_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {full_path}. Run `llmrel download-data` first."
        )
    questions: list[Question] = []
    with open(full_path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            final, solution, n_steps = parse_gsm8k_answer(row["answer"])
            questions.append(
                Question(
                    id=f"{full_path.stem}-{idx:05d}",
                    question=row["question"].strip(),
                    gold_answer=final,
                    gold_solution=solution,
                    n_steps=n_steps,
                )
            )
    if n is not None and n < len(questions):
        rng = random.Random(seed)
        questions = sorted(rng.sample(questions, n), key=lambda q: q.id)
    return questions


def download_gsm8k(split: str = "test", dest_dir: str | Path = "data/raw") -> Path:
    """Download a GSM8K split ("train" or "test") from the official repository."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    dest = resolve_path(dest_dir) / f"gsm8k_{split}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GSM8K_URL.format(split=split), timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest
