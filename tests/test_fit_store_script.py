from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit-store.sh"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("event", "ref", "scratch", "write"),
    [
        ("push", "refs/heads/main", "", "true"),
        ("schedule", "refs/heads/main", "", "true"),
        ("workflow_dispatch", "refs/heads/branch", "", "false"),
        ("workflow_dispatch", "refs/heads/branch", "fit-store-scratch", "true"),
        ("workflow_dispatch", "refs/heads/main", "fit-store-scratch-a", "true"),
        ("pull_request", "refs/pull/1/merge", "", "false"),
    ],
)
def test_guard_writes_main_and_scratch_stores_only(
    event: str, ref: str, scratch: str, write: str
) -> None:
    out = _run("guard", event, ref, scratch)
    assert out.returncode == 0
    assert out.stdout.strip() == write


@pytest.mark.parametrize("scratch", ["main", "fit-store", "gh-pages", "x"])
def test_guard_refuses_other_store_branches(scratch: str) -> None:
    out = _run("guard", "workflow_dispatch", "refs/heads/branch", scratch)
    assert out.returncode != 0


def _fit(root: Path, name: str, status: str, fitted_at: str) -> None:
    path = root / f"{name}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"status": status, "fitted_at": fitted_at}
    path.write_bytes(gzip.compress(json.dumps({"meta": meta}).encode()))


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_push_copies_planned_fits_and_prunes(tmp_path: Path) -> None:
    _git("init", "-q", "--bare", "remote.git", cwd=tmp_path)
    _git("clone", "-q", "remote.git", "clone", cwd=tmp_path)
    clone = tmp_path / "clone"
    for key, value in (("user.name", "t"), ("user.email", "t@t")):
        _git("config", key, value, cwd=clone)
    _git("commit", "-q", "--allow-empty", "-m", "init", cwd=clone)
    _git("push", "-q", "origin", "HEAD:main", cwd=clone)

    def push(new: Path, keys: list[str], plan: list[str]) -> None:
        (tmp_path / "keys.json").write_text(json.dumps(keys))
        include = [
            dict(zip(("model", "origin", "key"), p.split("/"), strict=True))
            for p in plan
        ]
        (tmp_path / "plan.json").write_text(json.dumps({"include": include}))
        out = _run(
            "push",
            str(new),
            str(tmp_path / "keys.json"),
            str(tmp_path / "plan.json"),
            cwd=clone,
        )
        assert out.returncode == 0, out.stderr

    first = tmp_path / "first"
    _fit(first, "m/o/old1", "ok", "2026-11-01")
    _fit(first, "m/o/old2", "ok", "2026-11-02")
    _fit(first, "m/o/bad", "failed", "2026-11-03")
    _fit(first, "n/o/other", "ok", "2026-11-01")
    _fit(first, "m/o/stray", "ok", "2026-11-04")
    push(first, [], ["m/o/old1", "m/o/old2", "m/o/bad", "n/o/other"])
    second = tmp_path / "second"
    _fit(second, "m/o/cur", "ok", "2026-11-05")
    push(second, ["m/o/cur"], ["m/o/cur"])

    got = tmp_path / "got"
    assert _run("fetch", str(got), cwd=clone).returncode == 0
    files = sorted(p.relative_to(got).as_posix() for p in got.rglob("*.gz"))
    # The current key, the newest good older fit and another model's fit.
    assert files == ["m/o/cur.json.gz", "m/o/old2.json.gz", "n/o/other.json.gz"]
