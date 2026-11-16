"""ci/steps/archive_candidates.sh, with a CLI that stands in."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "steps" / "archive_candidates.sh"


def test_no_candidates_file_fails(tmp_path: Path) -> None:
    # A CLI that exits 0 but writes no file: nothing is archived, so the
    # step fails rather than report success.
    env = {
        **os.environ,
        "WORK": str(tmp_path / "work"),
        "FORECAST_CLI": "true",
        "SKIP_FETCH": "true",
        "REFERENCE_DATE": "2026-11-07",
        "AS_OF": "2026-10-31",
    }
    run = subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True,
        text=True,
    )  # fmt: skip
    assert run.returncode != 0
    assert "no candidates file" in run.stdout
