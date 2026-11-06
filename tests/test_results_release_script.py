from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "results-release.sh"


def _tag(
    tmp_path: Path, status: list[dict], env: dict[str, str], exists: bool
) -> str:
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "fit-store.json").write_text(json.dumps(status))
    # A stand-in gh: `release view` finds the release only if `exists`.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(f"#!/bin/sh\nexit {0 if exists else 1}\n")
    gh.chmod(0o755)
    out = subprocess.run(
        ["bash", str(SCRIPT), "tag"],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _row(origin: str, current: bool) -> dict:
    return {"model": "renewal", "origin": origin, "current": current}


MAIN = {"GITHUB_EVENT_NAME": "push"}


@pytest.mark.parametrize(
    ("status", "exists", "tag"),
    [
        (
            [_row("2026-11-01", True), _row("2026-10-24", False)],
            False,
            "tag=results-2026-11-01",
        ),
        ([_row("2026-11-01", False), _row("2026-10-24", True)], False, "tag="),
        ([_row("2026-11-01", True)], True, "tag="),
    ],
)
def test_main_releases_each_hub_release_once_fitted(
    tmp_path: Path, status: list[dict], exists: bool, tag: str
) -> None:
    assert _tag(tmp_path, status, MAIN, exists) == tag


@pytest.mark.parametrize(
    ("ref", "tag"), [("v0.2.0", "tag=results-v0.2.0"), ("round-x", "tag=")]
)
def test_code_releases_get_their_own_results(
    tmp_path: Path, ref: str, tag: str
) -> None:
    env = {"GITHUB_EVENT_NAME": "release", "GITHUB_REF_NAME": ref}
    assert _tag(tmp_path, [_row("2026-11-01", True)], env, False) == tag
