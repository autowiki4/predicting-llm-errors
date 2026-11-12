import json

from llm_reliability.collect import calls_per_question, collect_responses, load_responses
from llm_reliability.llm.base import LLMClient, LLMResponse
from llm_reliability.llm.simulated import SimulatedLLM


class CountingClient(LLMClient):
    provider = "fake"
    model = "fake-1"

    def __init__(self, fail_ids=()):
        self.calls = 0
        self.fail_ids = set(fail_ids)

    def complete(self, prompt, *, purpose="answer", meta=None):
        self.calls += 1
        if meta["question_id"] in self.fail_ids:
            raise RuntimeError("simulated API outage")
        return LLMResponse(text="Answer: 1", model=self.model, output_tokens=5)


def test_calls_per_question():
    assert calls_per_question(5, True) == 7
    assert calls_per_question(0, False) == 1


def test_collect_writes_and_resumes(questions, tmp_path):
    qs = questions[:10]
    out = tmp_path / "responses.jsonl"
    client = CountingClient()
    records = collect_responses(qs, client, out, n_samples=2, self_check=True, workers=3)
    assert len(records) == 10 and client.calls == 10 * 4
    assert [r["id"] for r in records] == [q.id for q in qs]  # original order kept
    assert len(load_responses(out)) == 10

    # A second run makes no new calls.
    client2 = CountingClient()
    collect_responses(qs, client2, out, n_samples=2, self_check=True)
    assert client2.calls == 0


def test_failures_are_skipped_then_retried(questions, tmp_path):
    qs = questions[:5]
    out = tmp_path / "responses.jsonl"
    records = collect_responses(qs, CountingClient(fail_ids={qs[0].id}), out, n_samples=1)
    assert len(records) == 4
    records = collect_responses(qs, CountingClient(), out, n_samples=1)
    assert len(records) == 5


def test_corrupt_last_line_is_ignored(questions, tmp_path):
    out = tmp_path / "responses.jsonl"
    collect_responses(questions[:3], SimulatedLLM(questions), out, n_samples=1)
    with open(out, "a") as fh:
        fh.write('{"id": "half-writ')  # e.g. the process was killed mid-write
    assert len(load_responses(out)) == 3


def test_record_contents(questions, tmp_path):
    out = tmp_path / "r.jsonl"
    collect_responses(questions[:1], SimulatedLLM(questions), out, n_samples=2, self_check=True)
    rec = json.loads(out.read_text().splitlines()[0])
    assert set(rec) >= {"id", "question", "gold_answer", "response", "samples", "self_check"}
    assert len(rec["samples"]) == 2 and rec["self_check"]["text"].startswith("Verdict")


def test_stops_early_when_everything_fails(questions, tmp_path):
    import pytest

    from llm_reliability.collect import CollectionError

    client = CountingClient(fail_ids={q.id for q in questions})
    with pytest.raises(CollectionError, match="API key"):
        collect_responses(questions[:100], client, tmp_path / "r.jsonl", n_samples=0,
                          self_check=False, workers=1)
    assert client.calls < 20  # gave up quickly instead of trying all 100
