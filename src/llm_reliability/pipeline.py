"""Glue code that runs the stages in order. The CLI and the notebook both use this."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .collect import collect_responses, load_responses
from .config import Config
from .data import Question, load_gsm8k
from .evaluate import ExperimentResults, run_experiment
from .features import build_feature_table
from .llm import make_client
from .plots import make_all_plots
from .prompts import ANSWER_PROMPT, SELF_CHECK_PROMPT
from .report import write_report

log = logging.getLogger(__name__)


def load_questions(cfg: Config) -> list[Question]:
    return load_gsm8k(cfg.data.path, n=cfg.data.n_questions, seed=cfg.data.seed)


def _save_run_metadata(cfg: Config) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg.to_dict(), "prompts": {"answer": ANSWER_PROMPT, "self_check": SELF_CHECK_PROMPT}}
    (cfg.run_dir / "run_config.json").write_text(json.dumps(meta, indent=2))


def stage_collect(cfg: Config) -> list[dict]:
    questions = load_questions(cfg)
    _save_run_metadata(cfg)
    client = make_client(cfg.llm, questions=questions, seed=cfg.data.seed)
    records = collect_responses(
        questions,
        client,
        cfg.responses_path,
        n_samples=cfg.collection.n_samples,
        self_check=cfg.collection.self_check,
        workers=cfg.collection.workers,
    )
    log.info("%d responses saved in %s", len(records), cfg.responses_path)
    return records


def stage_features(cfg: Config) -> pd.DataFrame:
    wanted = {q.id for q in load_questions(cfg)}
    records = [r for r in load_responses(cfg.responses_path) if r["id"] in wanted]
    df = build_feature_table(records)
    cfg.features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.features_path, index=False)
    log.info(
        "Feature table: %d rows, error rate %.1f%% -> %s",
        len(df), 100 * df.is_wrong.mean(), cfg.features_path,
    )
    return df


def stage_evaluate(cfg: Config, df: pd.DataFrame | None = None) -> tuple[ExperimentResults, Path]:
    if df is None:
        df = pd.read_csv(cfg.features_path)
    results = run_experiment(df, cfg.eval)
    results.save(cfg.results_dir)
    figures = make_all_plots(results, df, cfg.figures_dir)
    report = write_report(cfg, results, figures)
    log.info("Report written to %s", report)
    return results, report


def run_all(cfg: Config) -> tuple[ExperimentResults, Path]:
    stage_collect(cfg)
    df = stage_features(cfg)
    return stage_evaluate(cfg, df)
