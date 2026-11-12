"""Write a Markdown report that explains the results in plain language."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .evaluate import ExperimentResults
from .features import FEATURE_DESCRIPTIONS

ALPHA = 0.05


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny Markdown table writer (avoids an extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("–" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ci(row, metric) -> str:
    lo, hi = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    if lo is None or pd.isna(lo):
        return f"{row[metric]:.3f}"
    return f"{row[metric]:.3f} [{lo:.3f}, {hi:.3f}]"


def _verdict(p: float) -> str:
    return "**yes**" if p < ALPHA else "**no** (not statistically significant)"


def write_report(cfg: Config, res: ExperimentResults, figures: dict[str, Path]) -> Path:
    s, h = res.summary, res.hypotheses
    best = h["best_models"]
    tm = res.test_metrics.set_index("name")
    out = cfg.report_path
    rel = {k: Path(v).relative_to(out.parent).as_posix() for k, v in figures.items()}

    lines: list[str] = [f"# Results: {cfg.run_name}", ""]
    if cfg.llm.provider == "simulated":
        lines += [
            "> **Warning: simulated data.** These responses come from the built-in *fake* LLM,",
            "> whose errors were generated so that the features carry signal. The numbers below",
            "> show that the pipeline works; they say nothing about real LLMs.",
            "> Run `configs/claude.yaml` or `configs/ollama.yaml` for real results.",
            "",
        ]

    lines += [
        "## Setup",
        "",
        f"- **Model:** `{cfg.llm.provider}` / `{cfg.llm.model}`",
        f"- **Dataset:** `{cfg.data.path}` — {s['n_questions']} questions",
        f"- **Model error rate:** {s['error_rate']:.1%} ({s['n_wrong']} wrong answers)",
        f"- **Split:** {s['n_train']} train / {s['n_test']} test "
        f"({s['n_test_wrong']} wrong answers in test), {s['cv_folds']}-fold CV on train, seed {s['seed']}",
        f"- **Extra samples per question (consistency):** {cfg.collection.n_samples}; "
        f"**self-check:** {'on' if cfg.collection.self_check else 'off'}",
        "",
        "## Answer to the research question",
        "",
        "*Can simple features from an LLM response help predict whether the answer is wrong?*",
        "",
    ]

    comb = h["combined_better_than_chance"]
    simp = h["simple_better_than_chance"]
    cvs = h["combined_vs_simple"]
    lines += [
        f"- Combined features (`{best['combined']}`) better than chance? {_verdict(comb['p_value'])} — "
        f"test ROC-AUC {comb['auc']:.3f} vs. 0.5 (permutation p = {comb['p_value']:.4f}).",
        f"- Free single-response features only (`{best['simple']}`) better than chance? "
        f"{_verdict(simp['p_value'])} — test ROC-AUC {simp['auc']:.3f} (p = {simp['p_value']:.4f}).",
        f"- Combined better than simple? {_verdict(cvs['p_value'])} — AUC difference "
        f"{cvs['auc_difference']:+.3f}, 95% CI [{cvs['ci_low']:+.3f}, {cvs['ci_high']:+.3f}] "
        f"(bootstrap p = {cvs['p_value']:.4f}).",
    ]
    if "combined_vs_best_rule" in h:
        r = h["combined_vs_best_rule"]
        lines.append(
            f"- Combined model better than the best single rule (`{best['rule']}`)? "
            f"{_verdict(r['p_value'])} — AUC difference {r['auc_difference']:+.3f}, "
            f"95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] (bootstrap p = {r['p_value']:.4f})."
        )
    lines += [
        "",
        "How to read this: ROC-AUC is the probability that a randomly chosen wrong answer gets a",
        "higher \"risk\" score than a randomly chosen correct one (0.5 = coin flip, 1.0 = perfect).",
        "Models were chosen by cross-validation on the training set, so the test numbers are honest.",
        "",
        f"![Model comparison]({rel['model_comparison']})",
        "",
        "## Test-set results (95% bootstrap CIs)",
        "",
    ]

    table = pd.DataFrame([
        {
            "model": name,
            "kind": row["kind"],
            "ROC-AUC": _ci(row, "roc_auc"),
            "PR-AUC": row["pr_auc"],
            "accuracy": row["accuracy"],
            "balanced acc.": _ci(row, "balanced_accuracy"),
            "precision (wrong)": row["precision_wrong"],
            "recall (wrong)": row["recall_wrong"],
            "F1 (wrong)": _ci(row, "f1_wrong"),
        }
        for name, row in tm.iterrows()
    ])
    lines += [
        _table(table),
        "",
        f"PR-AUC's chance level equals the error rate ({s['error_rate']:.3f}). The majority baseline's",
        "accuracy shows why accuracy alone is misleading: it never flags anything.",
        "",
        f"![ROC and PR curves]({rel['roc_pr']})",
        "",
        "## Which features matter?",
        "",
        "Permutation importance on the test set (how much ROC-AUC drops when a feature is shuffled):",
        "",
    ]
    imp = res.importance.head(10).copy()
    imp["description"] = imp["feature"].map(FEATURE_DESCRIPTIONS)
    lines += [
        _table(imp[["feature", "group", "importance_mean", "importance_std", "description"]]),
        "",
        f"![Feature importance]({rel['feature_importance']})",
        "",
        "Feature-group ablation (cross-validated on the training set):",
        "",
        _table(res.ablation[["setting", "n_features", "roc_auc_mean", "roc_auc_std", "f1_wrong_mean"]]),
        "",
        f"![Ablation]({rel['ablation']})",
        "",
        f"![Feature distributions]({rel['distributions']})",
        "",
        f"![Confusion matrix]({rel['confusion']})",
        "",
        "## Cross-validation (training set, mean ± sd)",
        "",
        _table(res.cv_metrics[["name", "roc_auc_mean", "roc_auc_std", "f1_wrong_mean",
                               "balanced_accuracy_mean"]]),
        "",
        "## Caveats",
        "",
        "- One dataset (GSM8K math) and one model: results may not transfer to other tasks or models.",
        "- Correctness is judged by exact numeric match with the reference answer; formatting",
        "  slips count as errors.",
        "- Consistency and self-check features cost extra API calls per question.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
