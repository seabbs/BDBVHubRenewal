"""The slots check runner (`scripts/slots_check.sh`) runs a label once.

checkpoint-scoring's slots_followups.sh calls it with an explicit label
after the auto watcher has usually checked that checkpoint, so such a
call must return at once, and wait rather than run alongside a check in
progress.
"""

import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "slots_check.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("flock") is None, reason="needs flock"
)


def env(tmp_path: Path) -> dict:
    return {
        **os.environ,
        "RUN_ROOT": str(tmp_path / "runs"),
        "PAGE_ROOT": str(tmp_path / "page"),
        "CKPTS": str(tmp_path / "none"),
        "MAILBOX": str(tmp_path / "mailbox.md"),
        "POLL": "1",
    }


def finish(tmp_path: Path, label: str) -> None:
    run = tmp_path / "runs" / label
    run.mkdir(parents=True, exist_ok=True)
    (run / "finished").touch()


def test_an_explicit_label_already_checked_exits_at_once(tmp_path):
    finish(tmp_path, "slots-abc123")
    out = subprocess.run(
        ["bash", str(SCRIPT), "slots-abc123"],
        env=env(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert out.returncode == 0
    assert "already checked" in out.stdout
    assert not (tmp_path / "mailbox.md").exists()


def test_an_explicit_label_waits_for_a_check_in_progress(tmp_path):
    locks = tmp_path / "runs" / "locks"
    locks.mkdir(parents=True)
    with (locks / "slots-abc123.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "slots-abc123"],
            env=env(tmp_path),
            stdout=subprocess.PIPE,
            text=True,
        )
        time.sleep(3)
        # Still waiting on the other caller's check.
        assert proc.poll() is None
        finish(tmp_path, "slots-abc123")
        fcntl.flock(held, fcntl.LOCK_UN)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 0
    assert "already checked" in out
