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


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


def test_a_partial_archive_does_not_say_every_candidate_was(
    tmp_path: Path,
) -> None:
    # `fit-store archive` writes the candidates it can and exits 1 for
    # the rest: those are pushed, but the step must not claim all (#322).
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    repo = tmp_path / "repo"
    _git("clone", "-q", str(origin), str(repo), cwd=tmp_path)
    _git("commit", "-q", "--allow-empty", "-m", "init", cwd=repo)
    _git("branch", "-q", "-M", "base", cwd=repo)
    _git("checkout", "-q", "--orphan", "submission-ledger", cwd=repo)
    _git("commit", "-q", "--allow-empty", "-m", "ledger", cwd=repo)
    _git("push", "-q", "origin", "submission-ledger", cwd=repo)
    _git("checkout", "-q", "base", cwd=repo)
    fake = tmp_path / "fake_cli.sh"
    fake.write_text(
        'while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done\n'
        'mkdir -p "$(dirname "$out")"; echo candidates > "$out"\n'
        'echo "Archived 6 of 12 models\' forecasts"\n'
        "exit 1\n"
    )
    env = {
        **os.environ,
        "WORK": str(tmp_path / "work"),
        "FORECAST_CLI": f"bash {fake}",
        "SKIP_FETCH": "true",
        "REFERENCE_DATE": "2026-11-21",
        "AS_OF": "2026-11-15",
        "GITHUB_RUN_ID": "1",
    }
    env.pop("GITHUB_STEP_SUMMARY", None)
    run = subprocess.run(
        ["bash", str(SCRIPT)], cwd=repo, env=env, capture_output=True,
        text=True,
    )  # fmt: skip
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Ledger: every candidate" not in run.stdout
    assert "not every candidate" in run.stdout
    ledger = tmp_path / "check"
    _git("clone", "-q", "-b", "submission-ledger", str(origin), str(ledger),
         cwd=tmp_path)  # fmt: skip
    assert (ledger / "archive/2026-11-21/1/candidates.csv.gz").exists()
