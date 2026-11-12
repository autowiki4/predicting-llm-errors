"""Experiment configuration.

Every run is described by one YAML file in ``configs/``. Keeping all choices in a
config file (rather than scattered through code) is what makes a run reproducible:
the same config + the same saved responses always produce the same results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    path: str = "data/raw/gsm8k_test.jsonl"  # relative paths resolve from the repo root
    n_questions: int | None = 1000  # None = use every question in the file
    seed: int = 0


@dataclass
class LLMConfig:
    # "simulated" (offline demo), "anthropic" (Claude API) or "openai_compatible"
    # (OpenAI, Ollama, LM Studio, vLLM, ... anything speaking the OpenAI chat API).
    provider: str = "simulated"
    model: str = "simulated-llm"
    max_tokens: int = 16000
    # Anthropic only: reasoning effort ("low" | "medium" | "high" | "xhigh" | "max").
    effort: str | None = "medium"
    # Anthropic only: server-side refusal fallbacks. "auto" turns them on for models
    # that support them.
    fallbacks: str | bool = "auto"
    # OpenAI-compatible only.
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    answer_temperature: float | None = 0.0
    sample_temperature: float | None = 0.8
    # Simulated only: controls how often the fake model is wrong.
    simulated_difficulty_shift: float = 0.0
    max_retries: int = 5
    timeout_seconds: float = 600.0


@dataclass
class CollectionConfig:
    n_samples: int = 5  # extra answers per question, used for the consistency features
    self_check: bool = True  # ask the model to grade its own answer
    workers: int = 4  # parallel requests


@dataclass
class EvalConfig:
    test_size: float = 0.25
    cv_folds: int = 5
    seed: int = 42
    models: list[str] = field(
        default_factory=lambda: ["logistic_regression", "random_forest", "gradient_boosting"]
    )
    n_bootstrap: int = 1000
    n_permutations: int = 1000
    ablation_model: str = "logistic_regression"


@dataclass
class Config:
    run_name: str = "demo"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- derived paths -------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return resolve_path(self.output_dir) / self.run_name

    @property
    def responses_path(self) -> Path:
        return self.run_dir / "responses.jsonl"

    @property
    def features_path(self) -> Path:
        return self.run_dir / "features.csv"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    return copy.deepcopy(obj)


def _from_dict(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(values: dict[str, Any]) -> Config:
    return _from_dict(Config, values or {})


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config. ``overrides`` uses dotted keys, e.g. {"data.n_questions": 50}."""
    with open(resolve_path(path)) as fh:
        values = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = values
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return config_from_dict(values)
