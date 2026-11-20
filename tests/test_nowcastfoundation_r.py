"""The simulator engine's own tests (packages/nowcastfoundation/r)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")
def test_the_engine_s_testthat_suite_passes():
    out = subprocess.run(
        [
            "Rscript",
            "-e",
            'testthat::test_local("packages/nowcastfoundation/r", '
            'reporter = "summary", stop_on_failure = TRUE)',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]


def test_ci_runs_the_engine_s_testthat_suite():
    """CI's test workflow has a job that runs the engine's testthat suite."""
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    steps = workflow["jobs"]["engine"]["steps"]
    uses = " ".join(s.get("uses", "") for s in steps)
    runs = " ".join(s.get("run", "") for s in steps)
    assert "r-lib/actions/setup-r@" in uses
    assert "r-lib/actions/setup-r-dependencies@" in uses
    assert 'test_local("packages/nowcastfoundation/r"' in runs
    assert "stop_on_failure = TRUE" in runs
