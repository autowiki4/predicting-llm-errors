import numpy as np

from llm_reliability.grading import extract_answer, is_wrong
from llm_reliability.llm.simulated import SimulatedLLM
from llm_reliability.prompts import answer_prompt, self_check_prompt


def test_deterministic(questions):
    llm = SimulatedLLM(questions, seed=0)
    q = questions[0]
    a = llm.complete(answer_prompt(q.question), purpose="answer", meta={"question_id": q.id})
    b = llm.complete(answer_prompt(q.question), purpose="answer", meta={"question_id": q.id})
    assert a.text == b.text and a.output_tokens == b.output_tokens


def test_error_rate_is_reasonable(questions):
    llm = SimulatedLLM(questions, seed=0)
    wrong = [
        is_wrong(llm.complete("", meta={"question_id": q.id}).text, q.gold_answer)
        for q in questions
    ]
    assert 0.1 < np.mean(wrong) < 0.45


def test_harder_shift_means_more_errors(questions):
    def rate(shift):
        llm = SimulatedLLM(questions, seed=0, difficulty_shift=shift)
        return np.mean([is_wrong(llm.complete("", meta={"question_id": q.id}).text, q.gold_answer)
                        for q in questions])
    assert rate(2.0) > rate(0.0) > rate(-2.0)


def test_answers_are_mostly_parseable(questions):
    llm = SimulatedLLM(questions, seed=0)
    found = [extract_answer(llm.complete("", meta={"question_id": q.id}).text)[1] == "explicit"
             for q in questions]
    assert np.mean(found) > 0.9


def test_self_check_format(questions):
    llm = SimulatedLLM(questions, seed=0)
    q = questions[0]
    r = llm.complete(self_check_prompt(q.question, f"Answer: {q.gold_answer}"),
                     purpose="self_check", meta={"question_id": q.id})
    assert r.text.startswith("Verdict: ") and "Confidence: " in r.text


def test_unknown_question_raises(questions):
    llm = SimulatedLLM(questions)
    try:
        llm.complete("x", meta={"question_id": "nope"})
    except KeyError:
        return
    raise AssertionError("expected KeyError")
