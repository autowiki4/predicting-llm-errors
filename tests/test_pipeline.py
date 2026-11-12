"""End-to-end: the whole pipeline on a small simulated run, via Python and via the CLI."""

import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from llm_reliability import pipeline
from llm_reliability.cli import main
from llm_reliability.config import REPO_ROOT, load_config

pytestmark = pytest.mark.slow


def test_run_all(small_config):
    results, report = pipeline.run_all(small_config)
    run = small_config.run_dir
    assert (run / "responses.jsonl").exists()
    assert (run / "run_config.json").exists()
    df = pd.read_csv(run / "features.csv")
    assert len(df) == 300 and 0.05 < df.is_wrong.mean() < 0.6
    for name in ["test_metrics.csv", "cv_metrics.csv", "ablation.csv",
                 "feature_importance.csv", "test_predictions.csv", "summary.json"]:
        assert (run / "results" / name).exists(), name
    figures = sorted(p.name for p in (run / "figures").glob("*.png"))
    assert len(figures) == 6
    text = report.read_text()
    assert "simulated data" in text
    for fig in figures:
        assert f"figures/{fig}" in text  # every figure is linked from the report
    summary = json.loads((run / "results" / "summary.json").read_text())
    assert summary["summary"]["n_questions"] == 300


def test_rerun_is_reproducible(small_config):
    r1, _ = pipeline.run_all(small_config)
    r2, _ = pipeline.run_all(small_config)  # responses are reused, not re-collected
    pd.testing.assert_frame_equal(r1.test_metrics, r2.test_metrics)


def test_cli_stages(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run_name": "cli",
        "output_dir": str(tmp_path / "runs"),
        "data": {"n_questions": 200},
        "llm": {"provider": "simulated"},
        "collection": {"n_samples": 2},
        "eval": {"n_bootstrap": 50, "n_permutations": 50, "cv_folds": 3},
    }))
    assert main(["estimate", "-c", str(cfg_path)]) == 0
    assert main(["collect", "-c", str(cfg_path)]) == 0
    assert main(["features", "-c", str(cfg_path)]) == 0
    assert main(["evaluate", "-c", str(cfg_path)]) == 0
    assert (tmp_path / "runs" / "cli" / "report.md").exists()


def test_cli_overrides_and_too_little_data(tmp_path, capsys):
    code = main(["run", "-c", "configs/demo.yaml",
                 "--set", f"output_dir={tmp_path}", "--set", "data.n_questions=12",
                 "--set", "collection.n_samples=1"])
    assert code == 2
    assert "need at least" in capsys.readouterr().err


def test_module_entry_point_runs():
    out = subprocess.run([sys.executable, "-m", "llm_reliability", "--help"],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    assert out.returncode == 0 and "collect" in out.stdout


@pytest.mark.parametrize("name", ["demo.yaml", "claude.yaml", "ollama.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(f"configs/{name}")
    assert cfg.data.n_questions >= 1000
    assert cfg.llm.provider in {"simulated", "anthropic", "openai_compatible"}


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("llm:\n  modle: typo\n")
    with pytest.raises(ValueError, match="modle"):
        load_config(p)
