from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "results-release.sh"

# A stand-in gh. `release view --json isDraft,assets` prints $ASSETS, or
# fails when $ASSETS is `none` (no published release); every call is
# logged to $GH_LOG.
GH = """#!/bin/sh
echo "$*" >> "$GH_LOG"
if [ "$1 $2" = "release view" ]; then
  case "$*" in
    *isDraft,assets*)
      [ "$ASSETS" = none ] && exit 1
      echo "$ASSETS" ;;
    *body*) echo "Forecasts submitted for the round." ;;
  esac
fi
exit 0
"""


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo(tmp_path: Path, as_of: str) -> Path:
    """A clone whose origin has a ledger with round 2026-11-07."""
    _git("init", "-q", "--bare", "remote.git", cwd=tmp_path)
    ledger = tmp_path / "ledger"
    _git("init", "-q", "-b", "submission-ledger", str(ledger), cwd=tmp_path)
    state = {"rounds": {"2026-11-07": {"as_of": as_of}}}
    (ledger / "state.json").write_text(json.dumps(state))
    for cmd in (
        ("add", "state.json"),
        ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"),
        ("push", "-q", str(tmp_path / "remote.git"), "submission-ledger"),
    ):
        _git(*cmd, cwd=ledger)
    clone = tmp_path / "clone"
    _git("clone", "-q", str(tmp_path / "remote.git"), str(clone), cwd=tmp_path)
    return clone


def _build(tmp_path: Path, current: bool) -> Path:
    build = tmp_path / "build"
    results = build / "results"
    for d in (results / "hub", results / "hub-comparison", build / "data"):
        d.mkdir(parents=True)
    status = [
        {"model": "renewal", "origin": "2026-11-01", "current": current},
        {"model": "renewal", "origin": "2026-10-24", "current": False},
    ]
    (results / "fit-store.json").write_text(json.dumps(status))
    (results / "nowcast-scores.csv").write_text("a\n1\n")
    (results / "forecasts.csv.gz").write_bytes(b"")
    (results / "hub" / "renewal-samples.csv.gz").write_bytes(b"")
    (build / "data" / "target.csv").write_text("a\n1\n")
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "index.html").write_text("site")
    return build


def _attach(
    tmp_path: Path, as_of: str, current: bool, assets: str
) -> list[str]:
    clone = _repo(tmp_path, as_of)
    build = _build(tmp_path, current)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(GH)
    (bin_dir / "gh").chmod(0o755)
    log = tmp_path / "gh.log"
    log.write_text("")
    subprocess.run(
        ["bash", str(SCRIPT), "attach", str(build), str(tmp_path / "site")],
        cwd=clone,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GH_LOG": str(log),
            "ASSETS": assets,
            "GITHUB_SHA": "a" * 40,
        },
        check=True,
        capture_output=True,
    )
    return log.read_text().splitlines()


def test_results_go_on_the_round_release_with_the_site_last(
    tmp_path: Path,
) -> None:
    calls = _attach(tmp_path, "2026-11-01", True, "")
    verbs = [" ".join(c.split()[:2]) for c in calls]
    assert verbs == [
        "release view",
        "release upload",
        "release view",
        "release edit",
        "release upload",
    ]
    assert "round-2026-11-07" in calls[1]
    assert "hub.zip" in calls[1]
    assert "data.zip" in calls[1]
    assert "site.zip" not in calls[1]
    assert calls[-1].split()[3].endswith("site.zip")


@pytest.mark.parametrize(
    ("as_of", "current", "assets", "views"),
    [
        # Already complete.
        ("2026-11-01", True, "hub.zip site.zip", 1),
        # No published round release yet.
        ("2026-11-01", True, "none", 1),
        # The fits at the round's hub release are not all of this code.
        ("2026-11-01", False, "", 0),
        # The round used an earlier hub release than this build.
        ("2026-10-24", True, "", 0),
    ],
)
def test_nothing_is_attached_otherwise(
    tmp_path: Path, as_of: str, current: bool, assets: str, views: int
) -> None:
    calls = _attach(tmp_path, as_of, current, assets)
    assert len(calls) == views
    assert all(c.startswith("release view") for c in calls)
