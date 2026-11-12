import pytest

from llm_reliability.data import load_gsm8k, parse_gsm8k_answer


def test_parse_gsm8k_answer():
    final, solution, steps = parse_gsm8k_answer(
        "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
        "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18"
    )
    assert final == "18"
    assert "<<" not in solution
    assert steps == 2


def test_parse_strips_thousands_separator():
    final, _, _ = parse_gsm8k_answer("Some step.\n#### 1,250")
    assert final == "1250"


def test_parse_requires_marker():
    with pytest.raises(ValueError):
        parse_gsm8k_answer("no final answer here")


def test_bundled_dataset_loads():
    qs = load_gsm8k()
    assert len(qs) == 1319
    assert all(q.gold_answer and q.question for q in qs)
    assert len({q.id for q in qs}) == len(qs)


def test_sampling_is_reproducible():
    a = [q.id for q in load_gsm8k(n=50, seed=3)]
    b = [q.id for q in load_gsm8k(n=50, seed=3)]
    c = [q.id for q in load_gsm8k(n=50, seed=4)]
    assert a == b and len(a) == 50
    assert a != c


def test_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-data"):
        load_gsm8k(tmp_path / "nope.jsonl")
