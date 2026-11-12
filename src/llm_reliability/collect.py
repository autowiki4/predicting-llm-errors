"""Stage 2 - Responses: send every question to the LLM and save what comes back.

For each question we make up to three kinds of calls:

1. **answer**      - the main response we will judge (1 call)
2. **sample**      - ``n_samples`` extra answers to the same prompt, used to
                     measure *consistency* (does the model keep giving the same answer?)
3. **self_check**  - the model is shown its own main answer and asked whether it
                     is correct, with a 0-100 confidence (1 call)

Everything is appended to ``responses.jsonl`` one question per line, as soon as
that question finishes. Collection is therefore *resumable*: if it crashes or you
stop it, run it again and only the missing questions are requested. Raw
responses are the expensive part of the project, so they are never overwritten.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .data import Question
from .llm.base import LLMClient
from .prompts import answer_prompt, self_check_prompt

log = logging.getLogger(__name__)


class CollectionError(RuntimeError):
    """Raised when collection cannot make progress (bad key, server down, wrong model name...)."""


def calls_per_question(n_samples: int, self_check: bool) -> int:
    return 1 + n_samples + int(self_check)


def load_responses(path: str | Path) -> list[dict[str, Any]]:
    """Read a responses.jsonl file (silently skipping a half-written last line)."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping a corrupt line in %s", path)
    return records


def query_question(
    client: LLMClient, q: Question, n_samples: int, self_check: bool
) -> dict[str, Any]:
    """Make all LLM calls for one question and return a JSON-serialisable record."""
    prompt = answer_prompt(q.question)
    main = client.complete(prompt, purpose="answer", meta={"question_id": q.id})
    samples = [
        client.complete(
            prompt, purpose="sample", meta={"question_id": q.id, "sample_index": i + 1}
        )
        for i in range(n_samples)
    ]
    check = None
    if self_check:
        check = client.complete(
            self_check_prompt(q.question, main.text),
            purpose="self_check",
            meta={"question_id": q.id},
        )
    return {
        "id": q.id,
        "question": q.question,
        "gold_answer": q.gold_answer,
        "n_steps": q.n_steps,
        "provider": client.provider,
        "model": main.model,
        "response": main.to_dict(),
        "samples": [s.to_dict() for s in samples],
        "self_check": check.to_dict() if check else None,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def collect_responses(
    questions: list[Question],
    client: LLMClient,
    out_path: str | Path,
    n_samples: int = 5,
    self_check: bool = True,
    workers: int = 4,
    progress_every: int = 50,
    abort_after: int = 5,
) -> list[dict[str, Any]]:
    """Collect responses for every question not already in ``out_path``.

    Returns all records for ``questions`` (old + new). Questions whose calls fail
    are logged and skipped, so a rerun will retry them. If the first
    ``abort_after`` questions all fail, something is wrong with the setup (missing
    API key, server not running, wrong model name), so we stop instead of
    burning through every question.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["id"]: r for r in load_responses(out_path)}
    todo = [q for q in questions if q.id not in existing]
    log.info(
        "%d questions requested, %d already collected, %d to query (%d LLM calls)",
        len(questions),
        len(questions) - len(todo),
        len(todo),
        len(todo) * calls_per_question(n_samples, self_check),
    )

    failures: list[tuple[str, str]] = []
    done = 0
    pending = iter(todo)
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max(1, workers)) as pool:
        # Keep only a small window of questions in flight, so that a broken setup is
        # noticed after a handful of failed calls rather than after every question.
        in_flight: dict[Future, Question] = {}

        def submit_next() -> None:
            q = next(pending, None)
            if q is not None:
                in_flight[pool.submit(query_question, client, q, n_samples, self_check)] = q

        for _ in range(2 * max(1, workers)):
            submit_next()
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                q = in_flight.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # keep going; a rerun retries this question
                    failures.append((q.id, repr(exc)))
                    log.warning("Question %s failed: %r", q.id, exc)
                    if done == 0 and len(failures) >= min(abort_after, len(todo)):
                        for f in in_flight:
                            f.cancel()
                        raise CollectionError(
                            f"The first {len(failures)} questions all failed, so collection "
                            f"stopped. First error: {failures[0][1]}\n"
                            f"Check the setup for provider '{client.provider}': is the API key "
                            "exported (e.g. ANTHROPIC_API_KEY), is the local server running, "
                            "and is the model name right? Nothing was lost; run the command "
                            "again once it is fixed."
                        ) from exc
                else:
                    # Only the main thread writes, so lines are never interleaved.
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    existing[q.id] = record
                    done += 1
                    if progress_every and done % progress_every == 0:
                        log.info("collected %d / %d", done, len(todo))
                submit_next()

    if failures:
        log.warning(
            "%d question(s) failed and were skipped; run collect again to retry them. First error: %s",
            len(failures),
            failures[0][1],
        )
    return [existing[q.id] for q in questions if q.id in existing]
