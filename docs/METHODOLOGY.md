# Methodology

This document explains every design decision in the pipeline, so the experiment can be
understood, defended in a seminar, and reproduced. Section numbers follow the pipeline stages.

## 0. The question and how we will answer it

**Research question.** Can simple features from an LLM response help predict whether the
answer is correct or unreliable?

**Hypothesis (H1).** A combination of response length, consistency, confidence cues and
self-check behaviour predicts errors better than chance.

We turn this into concrete, testable claims:

| Claim | Test | Null hypothesis |
|---|---|---|
| H1a: combined features beat chance | label-permutation test on test-set ROC-AUC | AUC = 0.5 |
| H1b: free, single-response features alone beat chance | same, for the simple-feature model | AUC = 0.5 |
| H2: combined beats simple | paired bootstrap of the AUC difference | difference = 0 |
| H3: a classifier beats the best single-feature rule | paired bootstrap of the AUC difference | difference = 0 |

We test the hypothesis; we do not assume it. A null result is a legitimate finding.

## 1. Data

* **Dataset:** GSM8K (Cobbe et al., 2021), 8,792 grade-school math word problems
  (1,319 test + 7,473 train). Each has a reference step-by-step solution ending in `#### <number>`.
* **Why GSM8K:** every answer is one number, so correctness is objective and automatic. The
  problems need multi-step reasoning, so even strong models make some mistakes.
* **Sample:** by default a seeded random sample of 1,000 test questions (`data.seed`), in line
  with the proposal's 1,000-3,000.
* **Difficulty proxy:** `n_steps`, the number of lines in the reference solution. It is kept in
  the feature table for analysis but deliberately **not** used as a feature, because a deployed
  reliability check would not have the reference solution.

## 2. Responses and labels

### Prompts (`prompts.py`)
* **Answer prompt:** "Solve ... Show your reasoning step by step. On the last line, write the final
  answer ... `Answer: <number>`".
* **Self-check prompt:** shows the question and the model's own main response, and asks for
  `Verdict: CORRECT|INCORRECT` and `Confidence: 0-100`.

The prompts are saved with every run in `run_config.json`.

### Calls per question
1. **Main answer**: the response being judged.
2. **`n_samples` extra answers** (default 5) to the identical prompt. For Claude these are sampled
   with the model's default settings (current Claude models don't take a `temperature` setting); for
   OpenAI-compatible servers the main answer uses `answer_temperature` (0, greedy) and the samples
   use `sample_temperature` (0.8).
3. **Self-check**: one call.

### Label (`grading.py`)
* The final answer is extracted in priority order: an explicit `Answer: <number>` line (also
  `#### n` and `\boxed{n}`); otherwise the last number in the text; otherwise none.
* Numbers are normalised (`$1,250.00` becomes 1250) and compared with a 1e-6 relative tolerance.
* `is_wrong = 1` if the answer is missing or differs from the reference. **"Wrong" is the
  positive class**, because the practical goal is to flag answers that deserve a second look.

## 3. Features

All features are computed from the saved text, so they can be redesigned without new API calls.

| Group | Feature | Definition |
|---|---|---|
| length | `n_chars`, `n_words`, `n_sentences`, `n_lines`, `n_numbers` | counts in the main response |
| length | `n_output_tokens` | output tokens reported by the API. For Claude this includes hidden reasoning tokens, so it measures how much the model "thought" |
| confidence_language | `hedge_count`, `hedge_rate` | hedging phrases ("I think", "probably", "might", "not sure", ...) and their count per 100 words |
| confidence_language | `confident_count`, `confident_rate` | certainty phrases ("clearly", "definitely", "certainly", ...) |
| confidence_language | `self_correction_count` | "wait", "actually", "let me re-check", "mistake", ... |
| answer_format | `answer_explicit` | 1 if the requested `Answer:` format was used |
| answer_format | `answer_missing` | 1 if no number was found |
| answer_format | `answer_is_integer` | 1 if the answer is a whole number (all GSM8K answers are) |
| answer_format | `answer_log_magnitude` | log10(1 + \|answer\|) |
| consistency | `agreement_rate` | share of extra samples whose answer equals the main answer |
| consistency | `n_unique_answers` | distinct answers among main + samples |
| consistency | `answer_entropy` | Shannon entropy (bits) of that answer distribution |
| consistency | `majority_agrees` | 1 if the main answer is (one of) the most common |
| self_check | `selfcheck_says_incorrect` | 1 if the model judged its own answer INCORRECT |
| self_check | `selfcheck_confidence` | stated confidence / 100 |

Phrase lists are matched case-insensitively on word boundaries ("mighty" does not count as
"might"). They live at the top of `features.py` and are easy to extend.

**Feature sets.** *Simple* = length + confidence_language + answer_format (needs only the one
response, so it is free at deployment time). *Combined* = all five groups. Groups with no data
(for example consistency when `n_samples: 0`) are dropped automatically.

**Missing values** (for example an unparseable self-check) are imputed with the training-set
median inside each model pipeline, so test data never leaks into the imputation.

## 4. Models

| Model | Settings | Why |
|---|---|---|
| Logistic regression | standardised inputs, L2, `class_weight="balanced"` | interpretable linear baseline |
| Random forest | 200 trees, `min_samples_leaf=3`, balanced subsample weights | non-linear, robust, little tuning |
| Gradient boosting | scikit-learn `HistGradientBoostingClassifier`, lr 0.05, 200 iterations, 15 leaves, balanced | usually strongest on small tabular data |

Class re-weighting matters: with a 20% error rate, a model that never flags anything is 80%
accurate and useless.

**Baselines**
* `majority`: always predicts the most common class (correct). Recall for wrong answers = 0.
* `random`: predicts "wrong" at the base rate.
* `rule:<feature>`: one threshold on one feature (agreement rate, self-check verdict, self-check
  confidence, hedge rate, word count), chosen on the training set to maximise F1 for wrong
  answers. A classifier must beat the *best* such rule to justify itself. The best rule is chosen
  by training-set AUC.

## 5. Evaluation protocol

1. **Stratified 75/25 train/test split** (`eval.seed`). The test set is used once.
2. **5-fold stratified cross-validation on the training set** for every model and feature set.
   The best model per feature set is the one with the highest mean CV ROC-AUC. Choosing by test
   score would bias the reported test score upward.
3. **Test metrics** for every model and baseline:
   * **ROC-AUC** (primary): the probability that a random wrong answer is scored riskier than a
     random correct one. 0.5 = chance. Threshold-free and insensitive to class balance.
   * **PR-AUC** (average precision): chance level equals the error rate.
   * **Accuracy** and **balanced accuracy** (mean of per-class recall; 0.5 = chance).
   * **Precision, recall and F1 for wrong answers**: of the answers we flag, how many are really
     wrong; of the wrong answers, how many we catch.
   * **Macro F1**.
4. **Uncertainty:** 95% percentile-bootstrap CIs (1,000 resamples of the test set) for ROC-AUC,
   F1 (wrong) and balanced accuracy.
5. **Significance:**
   * *Better than chance:* shuffle the test labels 1,000 times, recompute the AUC each time, and
     report p = (number of shuffled AUCs ≥ observed + 1) / (1,000 + 1).
   * *A better than B:* resample test questions 1,000 times, compute AUC(A) − AUC(B) on each
     resample, and report the 95% CI and a two-sided bootstrap p-value. Because both models are
     scored on the same questions, the comparison is paired and more sensitive than comparing two
     separate CIs.
   * Significance level α = 0.05. With four planned tests, a Bonferroni-corrected threshold would
     be 0.0125; report both if a p-value falls between them.
6. **Which features matter:**
   * *Permutation importance* (test set, 20 repeats): the drop in AUC when one feature is shuffled.
   * *Ablation* (CV on train, using `eval.ablation_model`): each group alone, and all groups minus
     one. "Only X" shows a group's standalone value; "all minus X" shows what it adds beyond the others.

## 6. Reproducibility

* All randomness is seeded (`data.seed`, `eval.seed`; the simulator is deterministic per question).
* Raw responses are stored once and never overwritten; downstream stages are pure functions of
  `responses.jsonl` + config. Re-running `features` and `evaluate` gives identical results (this is
  tested).
* `run_config.json` records the exact config and prompts.
* The LLM itself is not deterministic: re-collecting responses will give somewhat different data.
  That is why responses are cached.

## 7. Threats to validity

* **Construct validity:** exact-match grading can mark a correct answer wrong (for example "18 dollars
  per day" parsed oddly). Spot-check about 50 wrong labels by hand and report the label error rate.
* **External validity:** one dataset, one model. A second model (stretch goal) helps.
* **Statistical power:** with 250 test questions and a 15% error rate there are about 38 wrong
  answers in the test set; CIs will be wide. More questions narrow them.
* **Leakage:** `n_steps` and the gold answer are never used as features. The self-check sees only
  the model's own response, not the reference.
* **Simulated data:** the built-in simulator is for demos and tests only. Its error model was
  written so that the features carry signal, so its results are circular by design.
