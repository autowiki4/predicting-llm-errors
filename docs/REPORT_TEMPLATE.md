# Predicting When LLMs Get It Wrong — Final Report

*Name · Junior Seminar · Date*

> Fill in the bracketed parts. Every number comes from `runs/<run_name>/report.md` and
> `runs/<run_name>/results/`. Figures are in `runs/<run_name>/figures/`.

## 1. Introduction (½ page)
- Problem: LLM answers can sound confident while being wrong. Users need to know when to double-check.
- Goal: test whether lightweight signals from a response flag unreliable answers, without building a new model.
- Research question and hypothesis (copy from the proposal).

## 2. Related work (½–1 page)
- Self-consistency / sampling agreement as an uncertainty signal.
- Verbalised confidence and self-evaluation by LLMs.
- Calibration of language models; selective prediction.
- [Add 4–6 sources from the literature review in weeks 1–2.]

## 3. Method (1 page)
- **Data:** GSM8K, [N] questions sampled with seed [s]. Why GSM8K (objective numeric answers).
- **Model studied:** [provider / model / settings], [n_samples] extra samples, self-check [on/off].
- **Labelling:** final-answer extraction + exact numeric match. Manual check of [k] labels found [x] labelling errors.
- **Features:** the five groups (table from `docs/METHODOLOGY.md` §3).
- **Classifiers and baselines:** LR, RF, gradient boosting; majority, random, single-feature rules.
- **Evaluation:** 75/25 split, 5-fold CV model selection, ROC-AUC primary, bootstrap CIs, permutation and paired-bootstrap tests.

## 4. Results (1–2 pages)
- The model's error rate: [x]% ([n] wrong answers).
- **Figure 1:** `model_comparison.png`. Baseline vs. simple vs. combined.
- **Table 1:** test metrics with CIs for the majority baseline, the best rule, the best simple model and the best combined model.
- H1a (combined > chance): AUC = [ ], p = [ ].
- H1b (simple > chance): AUC = [ ], p = [ ].
- H2 (combined > simple): ΔAUC = [ ] (95% CI [ , ]).
- H3 (combined > best rule): ΔAUC = [ ] (95% CI [ , ]).
- **Figure 2:** `feature_importance.png`; **Figure 3:** `ablation.png`. Which signals carry the information?
- Precision/recall trade-off (`roc_pr_curves.png`): e.g. "flagging the riskiest 20% of answers catches [x]% of the errors".

## 5. Discussion (½–1 page)
- Was the hypothesis supported? Which part (free vs. paid signals)?
- Practical meaning: cost (extra calls) vs. benefit (errors caught).
- Surprises and failure cases (look at a few wrong answers the model did *not* flag).

## 6. Limitations and future work
- One dataset / one model; grading strictness; sample size and CI width.
- Future: second LLM, open-ended QA datasets, token log-probabilities, a small dashboard.

## 7. Conclusion
- One paragraph that answers the research question directly.

## Reproducibility statement
Code: [repo URL], commit [hash]. Command: `llmrel run --config configs/[name].yaml`.
