"""The walkthrough notebook must run top to bottom without errors."""

import shutil

import pytest

from llm_reliability.config import REPO_ROOT

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

pytestmark = pytest.mark.slow


def test_walkthrough_notebook_executes(tmp_path):
    # Run a copy from inside a different directory to prove paths don't depend on the cwd.
    src = REPO_ROOT / "notebooks" / "walkthrough.ipynb"
    nb_path = tmp_path / "walkthrough.ipynb"
    shutil.copy(src, nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    # Use a temporary output folder so the test never touches the real demo run.
    nb.cells[1].source = nb.cells[1].source.replace(
        'cfg = load_config(CONFIG)',
        f'cfg = load_config(CONFIG, {{"output_dir": r"{tmp_path / "runs"}", "data.n_questions": 300, '
        '"eval.n_bootstrap": 100, "eval.n_permutations": 100})',
    )
    assert str(tmp_path) in nb.cells[1].source
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
    assert (tmp_path / "runs" / "demo" / "report.md").exists()
