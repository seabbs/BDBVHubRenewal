"""`ledger_worktree` and `ledger_push` in ci/steps/lib.sh, on local repos."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "steps" / "lib.sh"
WHO = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(
        ["git", *WHO, *args], cwd=cwd, check=True, capture_output=True,
        text=True,
    )  # fmt: skip
    return out.stdout


def _ledger(tmp_path: Path) -> tuple[Path, Path]:
    """A bare origin with a ledger branch, and the clone that seeded it.

    A clone to run steps in is made at `tmp_path / "repo"`.
    """
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    seed = tmp_path / "seed"
    _git("clone", "-q", str(origin), str(seed), cwd=tmp_path)
    _git("checkout", "-q", "--orphan", "submission-ledger", cwd=seed)
    (seed / "state.json").write_text(json.dumps({"rounds": []}))
    _git("add", "-A", cwd=seed)
    _git("commit", "-q", "-m", "init", cwd=seed)
    _git("push", "-q", "origin", "submission-ledger", cwd=seed)
    repo = tmp_path / "repo"
    _git("clone", "-q", str(origin), str(repo), cwd=tmp_path)
    return origin, seed


def _run(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "WORK": str(cwd / "work"), "LEDGER_PUSH_WAIT": "0"}
    return subprocess.run(
        ["bash", "-c", f"source {LIB}\n{script}"],
        cwd=cwd, env=env, capture_output=True, text=True,
    )  # fmt: skip


def test_a_rejected_push_writes_again_on_the_moved_branch(
    tmp_path: Path,
) -> None:
    origin, seed = _ledger(tmp_path)
    # Another run records a round after this one checked out the ledger:
    # the first push is rejected, and the writer runs again on its state.
    script = f"""
ledger_worktree "$WORK/ledger"
moved=false
add_round() {{
  if [ "$moved" = false ]; then
    git -C {seed} {" ".join(WHO)} commit -q --allow-empty -m other
    python3 -c "import json; p='{seed}/state.json'; \\
s=json.load(open(p)); s['rounds'].append('other'); \\
json.dump(s, open(p, 'w'))"
    git -C {seed} {" ".join(WHO)} commit -qam other
    git -C {seed} push -q origin submission-ledger
    moved=true
  fi
  python3 -c "import json; p='$1/state.json'; \\
s=json.load(open(p)); s['rounds'].append('mine'); \\
json.dump(s, open(p, 'w'))"
}}
ledger_push "$WORK/ledger" "Record mine" add_round "$WORK/ledger"
"""
    run = _run(script, tmp_path / "repo")
    assert run.returncode == 0, run.stderr
    assert "rejected" in run.stdout
    state = _git("show", "submission-ledger:state.json", cwd=origin)
    assert json.loads(state)["rounds"] == ["other", "mine"]


def test_nothing_to_write_pushes_nothing(tmp_path: Path) -> None:
    origin, _ = _ledger(tmp_path)
    before = _git("rev-parse", "submission-ledger", cwd=origin)
    run = _run(
        'ledger_worktree "$WORK/ledger"\nledger_push "$WORK/ledger" none true',
        tmp_path / "repo",
    )
    assert run.returncode == 0, run.stderr
    assert _git("rev-parse", "submission-ledger", cwd=origin) == before


def test_a_push_that_keeps_failing_fails(tmp_path: Path) -> None:
    origin, _ = _ledger(tmp_path)
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    script = """
ledger_worktree "$WORK/ledger"
write() { echo x > "$WORK/ledger/x"; }
LEDGER_PUSH_TRIES=2 ledger_push "$WORK/ledger" "x" write
"""
    run = _run(script, tmp_path / "repo")
    assert run.returncode != 0
    assert "failed after 2 tries" in run.stdout
