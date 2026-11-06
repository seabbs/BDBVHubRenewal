from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "results-release.sh"

# A stand-in gh. `release view` answers as $STATE says (none, draft or
# published), and every call is logged to $GH_LOG.
GH = """#!/bin/sh
echo "$*" >> "$GH_LOG"
if [ "$1 $2" = "release view" ]; then
  case "$STATE" in
    none) exit 1 ;;
    draft) echo false ;;
    published) echo true ;;
  esac
fi
exit 0
"""


def _run(
    tmp_path: Path, args: list[str], state: str, env: dict[str, str]
) -> tuple[str, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(GH)
    gh.chmod(0o755)
    log = tmp_path / "gh.log"
    log.write_text("")
    out = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GH_LOG": str(log),
            "STATE": state,
            "GITHUB_SHA": "a" * 40,
            **env,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip(), log.read_text().splitlines()


def _status(tmp_path: Path, rows: list[tuple[str, bool]]) -> Path:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    status = [{"model": "renewal", "origin": o, "current": c} for o, c in rows]
    (results / "fit-store.json").write_text(json.dumps(status))
    return results


MAIN = {"GITHUB_EVENT_NAME": "push"}
READY = [("2026-11-01", True), ("2026-10-24", False)]


@pytest.mark.parametrize(
    ("rows", "state", "tag"),
    [
        (READY, "none", "tag=results-2026-11-01"),
        (READY, "draft", "tag=results-2026-11-01"),
        (READY, "published", "tag="),
        ([("2026-11-01", False), ("2026-10-24", True)], "none", "tag="),
    ],
)
def test_main_releases_each_hub_release_once_fitted(
    tmp_path: Path, rows: list[tuple[str, bool]], state: str, tag: str
) -> None:
    results = _status(tmp_path, rows)
    out, _ = _run(tmp_path, ["tag", str(results)], state, MAIN)
    assert out == tag


@pytest.mark.parametrize(
    ("ref", "state", "tag"),
    [
        ("v0.2.0", "none", "tag=results-v0.2.0"),
        ("v0.2.0", "published", "tag="),
        ("round-2026-11-07", "none", "tag="),
    ],
)
def test_code_releases_get_their_own_results(
    tmp_path: Path, ref: str, state: str, tag: str
) -> None:
    results = _status(tmp_path, READY)
    env = {"GITHUB_EVENT_NAME": "release", "GITHUB_REF_NAME": ref}
    out, _ = _run(tmp_path, ["tag", str(results)], state, env)
    assert out == tag


@pytest.mark.parametrize("state", ["none", "draft"])
def test_publish_uploads_a_draft_then_publishes(
    tmp_path: Path, state: str
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "fit-store.json").write_text(
        json.dumps([{"origin": "2026-11-01"}])
    )
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("site")
    args = ["publish", "results-2026-11-01", str(release), str(site)]
    _, calls = _run(tmp_path, args, state, MAIN)
    verbs = [" ".join(c.split()[:2]) for c in calls]
    # A draft left by a failed attempt is replaced, never published.
    expected = ["release view", "release create", "release edit"]
    if state == "draft":
        expected.insert(1, "release delete")
    assert verbs == expected
    assert "--draft" in calls[-2]
    assert "--draft=false" in calls[-1]
    assert (release / "site.zip").exists()
