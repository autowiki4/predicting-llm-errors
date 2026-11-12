"""Figures for the report and presentation.

Colours follow one fixed categorical order (blue, orange, aqua, yellow) so the
same kind of model always has the same colour across every figure. Baselines
are grey so the eye goes to the classifiers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; works on servers without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from .evaluate import ExperimentResults  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#9b9a95"
KIND_COLORS = {"baseline": GREY, "simple": BLUE, "combined": ORANGE}
GROUP_COLORS = {
    "length": BLUE,
    "confidence_language": ORANGE,
    "answer_format": AQUA,
    "consistency": YELLOW,
    "self_check": "#e87ba4",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _headline_rows(res: ExperimentResults) -> pd.DataFrame:
    """Majority baseline, best rule, best simple model, best combined model."""
    best = res.hypotheses["best_models"]
    names = [("majority", "Majority class"), ("random", "Random guess")]
    if best.get("rule"):
        names.append((best["rule"], f"Best rule ({best['rule'].split(':', 1)[1]})"))
    names += [(best["simple"], "Simple features"), (best["combined"], "Combined features")]
    tm = res.test_metrics.set_index("name")
    rows = []
    for name, label in names:
        r = tm.loc[name]
        rows.append({"name": name, "label": label, "kind": r["kind"], **r.to_dict()})
    return pd.DataFrame(rows)


def plot_model_comparison(res: ExperimentResults, path: Path) -> Path:
    """The headline chart: baseline vs simple features vs combined features."""
    rows = _headline_rows(res)
    metrics = [("roc_auc", "ROC-AUC (0.5 = chance)"), ("f1_wrong", "F1 for wrong answers")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        vals = rows[metric].to_numpy(dtype=float)
        lo = rows[f"{metric}_ci_low"].to_numpy(dtype=float)
        hi = rows[f"{metric}_ci_high"].to_numpy(dtype=float)
        err = np.vstack([np.nan_to_num(vals - lo), np.nan_to_num(hi - vals)])
        colors = [KIND_COLORS.get(k, GREY) for k in rows["kind"]]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=colors, height=0.6, xerr=err, ecolor=INK_2, capsize=3)
        for yi, v, h in zip(y, vals, hi):
            x = (h if np.isfinite(h) else v) + 0.02  # label sits just past the error bar
            ax.text(x, yi, f"{v:.2f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(y, rows["label"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        if metric == "roc_auc":
            ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1)
    fig.suptitle("Held-out test set, with 95% bootstrap intervals", x=0.01, ha="left",
                 fontsize=9, color=INK_2)
    return _save(fig, path)


def _curve_rows(res: ExperimentResults):
    rows = _headline_rows(res)
    rows = rows[rows["name"] != "random"]
    colors = {"Majority class": GREY, "Simple features": BLUE, "Combined features": ORANGE}
    for _, r in rows.iterrows():
        color = colors.get(r["label"], AQUA)
        style = ":" if r["kind"] == "baseline" else "-"
        yield r["name"], r["label"], color, style


def plot_roc_pr(res: ExperimentResults, path: Path) -> Path:
    preds = res.test_predictions
    y = preds["is_wrong"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, label, color, style in _curve_rows(res):
        score = preds[f"score:{name}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        prec, rec, _ = precision_recall_curve(y, score)
        ax1.plot(fpr, tpr, style, color=color, label=label)
        ax2.plot(rec, prec, style, color=color, label=label)
    ax1.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1)
    ax1.set(xlabel="False positive rate (correct answers flagged)",
            ylabel="True positive rate (wrong answers caught)")
    ax1.set_title("ROC curve", loc="left")
    ax2.axhline(y.mean(), color=GRID, linestyle="--", linewidth=1)
    ax2.set(xlabel="Recall (share of wrong answers caught)",
            ylabel="Precision (flags that are really wrong)")
    ax2.set_title("Precision-recall curve", loc="left")
    ax2.set_ylim(0, 1.02)
    ax1.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_feature_importance(res: ExperimentResults, path: Path, top: int = 15) -> Path:
    imp = res.importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp) + 1.4))
    colors = [GROUP_COLORS.get(g, GREY) for g in imp["group"]]
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"],
            color=colors, ecolor=INK_2, capsize=2, height=0.6)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title(f"What the best model relies on ({res.hypotheses['best_models']['combined']})",
                 loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for g, c in GROUP_COLORS.items()
               if g in set(imp["group"])]
    labels = [g.replace("_", " ") for g in GROUP_COLORS if g in set(imp["group"])]
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_ablation(res: ExperimentResults, path: Path) -> Path:
    ab = res.ablation.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(ab) + 1.4))
    colors = [ORANGE if s == "all features" else BLUE if s.startswith("simple")
              else AQUA if s.startswith("only") else YELLOW for s in ab["setting"]]
    ax.barh(ab["setting"], ab["roc_auc_mean"], xerr=ab["roc_auc_std"], color=colors,
            ecolor=INK_2, capsize=2, height=0.6)
    ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xlim(min(0.4, float((ab["roc_auc_mean"] - ab["roc_auc_std"]).min()) - 0.05), 1.0)
    ax.set_xlabel("Cross-validated ROC-AUC on the training set (mean ± sd over folds)")
    ax.set_title("Feature-group ablation", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_feature_distributions(df: pd.DataFrame, path: Path, features: list[str]) -> Path:
    features = [f for f in features if f in df.columns and df[f].notna().any()][:6]
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, features):
        data = [df.loc[df.is_wrong == k, feat].dropna().to_numpy() for k in (0, 1)]
        bp = ax.boxplot(data, tick_labels=["correct", "wrong"], patch_artist=True, widths=0.5,
                        medianprops={"color": INK}, flierprops={"markersize": 3})
        for patch, color in zip(bp["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(feat, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Feature values for correct vs. wrong answers (all questions)", x=0.01,
                 ha="left", fontsize=10, color=INK_2)
    return _save(fig, path)


def plot_confusion(res: ExperimentResults, path: Path) -> Path:
    name = res.hypotheses["best_models"]["combined"]
    preds = res.test_predictions
    cm = confusion_matrix(preds["is_wrong"], preds[f"pred:{name}"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else INK, fontsize=12)
    ax.set_xticks([0, 1], ["flagged OK", "flagged wrong"])
    ax.set_yticks([0, 1], ["actually correct", "actually wrong"])
    ax.set_title(f"Confusion matrix: {name}", loc="left", fontsize=10)
    return _save(fig, path)


def make_all_plots(res: ExperimentResults, df: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    figures_dir = Path(figures_dir)
    top_feats = list(res.importance["feature"].head(6))
    return {
        "model_comparison": plot_model_comparison(res, figures_dir / "model_comparison.png"),
        "roc_pr": plot_roc_pr(res, figures_dir / "roc_pr_curves.png"),
        "feature_importance": plot_feature_importance(res, figures_dir / "feature_importance.png"),
        "ablation": plot_ablation(res, figures_dir / "ablation.png"),
        "distributions": plot_feature_distributions(
            df, figures_dir / "feature_distributions.png", top_feats
        ),
        "confusion": plot_confusion(res, figures_dir / "confusion_matrix.png"),
    }
