"""Predicting when LLMs get it wrong.

A small, reproducible pipeline that asks an LLM questions with known answers,
extracts simple features from its responses, and trains lightweight classifiers
to predict whether each answer is wrong.

Pipeline stages (each lives in its own module):

1. ``data``       - load a public QA/reasoning dataset with known answers (GSM8K)
2. ``collect``    - send questions to an LLM and save its responses
3. ``grading``    - extract the final answer and label it correct / wrong
4. ``features``   - turn each response into numeric signals
5. ``models``     - classifiers + simple baselines
6. ``evaluate``   - metrics, statistical tests, ablations, feature importance
7. ``plots`` / ``report`` - figures and a Markdown report
"""

__version__ = "0.1.0"
