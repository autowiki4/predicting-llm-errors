"""Command-line interface.

Examples::

    llmrel run --config configs/demo.yaml           # whole pipeline, offline, ~1 minute
    llmrel collect --config configs/claude.yaml     # only query the LLM (resumable)
    llmrel features --config configs/claude.yaml    # responses -> features.csv
    llmrel evaluate --config configs/claude.yaml    # features -> metrics, figures, report
    llmrel estimate --config configs/claude.yaml    # how many API calls a run will make
    llmrel download-data --split train              # fetch more GSM8K questions

``--set key=value`` overrides any config value, e.g. ``--set data.n_questions=50``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from . import pipeline
from .collect import CollectionError, calls_per_question, load_responses
from .config import load_config
from .data import download_gsm8k
from .evaluate import NotEnoughDataError
from .features import EmptyResponsesError


def _parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key] = yaml.safe_load(value)  # "50" -> 50, "true" -> True, "null" -> None
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrel", description="Predicting when LLMs get it wrong - experiment pipeline"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("run", "run every stage: collect -> features -> evaluate"),
        ("collect", "query the LLM and save raw responses (resumable)"),
        ("features", "build the feature table from saved responses"),
        ("evaluate", "train classifiers, compute metrics, write figures and report"),
        ("estimate", "print how many LLM calls a run needs, without making any"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", "-c", default="configs/demo.yaml")
        p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                       help="override a config value, e.g. --set data.n_questions=50")

    p = sub.add_parser("download-data", help="download a GSM8K split into data/raw/")
    p.add_argument("--split", choices=["train", "test"], default="test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "anthropic", "openai", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "download-data":
        print(f"Saved {download_gsm8k(args.split)}")
        return 0

    cfg = load_config(args.config, _parse_overrides(args.overrides))
    try:
        if args.command == "estimate":
            ids = {q.id for q in pipeline.load_questions(cfg)}
            done = ids & {r["id"] for r in load_responses(cfg.responses_path)}
            todo = len(ids) - len(done)
            per_q = calls_per_question(cfg.collection.n_samples, cfg.collection.self_check)
            print(f"provider={cfg.llm.provider} model={cfg.llm.model}")
            print(f"{len(ids)} questions x {per_q} calls = {len(ids) * per_q} calls in total")
            print(f"{len(done)} already collected; {todo * per_q} calls still to make")
        elif args.command == "collect":
            pipeline.stage_collect(cfg)
        elif args.command == "features":
            pipeline.stage_features(cfg)
        elif args.command == "evaluate":
            _, report = pipeline.stage_evaluate(cfg)
            print(f"Report: {report}")
        elif args.command == "run":
            _, report = pipeline.run_all(cfg)
            print(f"Report: {report}")
    except NotEnoughDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (CollectionError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except EmptyResponsesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
