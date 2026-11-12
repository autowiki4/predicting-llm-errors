"""Shared test fixtures."""

from __future__ import annotations

import pytest

from llm_reliability.config import Config, config_from_dict
from llm_reliability.data import load_gsm8k


@pytest.fixture(scope="session")
def questions():
    return load_gsm8k(n=300, seed=1)


@pytest.fixture
def small_config(tmp_path) -> Config:
    """A fast simulated run written to a temporary directory."""
    return config_from_dict({
        "run_name": "test-run",
        "output_dir": str(tmp_path / "runs"),
        "data": {"n_questions": 300, "seed": 1},
        "llm": {"provider": "simulated"},
        "collection": {"n_samples": 3, "self_check": True, "workers": 4},
        "eval": {"n_bootstrap": 200, "n_permutations": 200, "cv_folds": 3},
    })
