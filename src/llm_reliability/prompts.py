"""The exact prompts sent to the LLM.

Prompts are part of the experimental design, so they live in one place and are
saved alongside every run. Changing a prompt changes the experiment.
"""

from __future__ import annotations

ANSWER_PROMPT = """Solve the following math word problem. Show your reasoning step by step.
On the last line, write the final answer as a single number in the form:
Answer: <number>

Problem: {question}"""


SELF_CHECK_PROMPT = """You are checking someone's solution to a math word problem.

Problem: {question}

Proposed solution:
{solution}

Is the proposed final answer correct? Reply with exactly two lines:
Verdict: CORRECT or INCORRECT
Confidence: <an integer from 0 to 100 for how confident you are that the final answer is correct>"""


def answer_prompt(question: str) -> str:
    return ANSWER_PROMPT.format(question=question)


def self_check_prompt(question: str, solution: str) -> str:
    return SELF_CHECK_PROMPT.format(question=question, solution=solution)
