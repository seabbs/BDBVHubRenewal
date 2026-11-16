"""A batch starts only with memory and disk to spare."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "workflow" / "scripts" / "wait_for_room.sh"


def run(tmp_path, free, disk, timeout=5):
    return subprocess.run(
        ["bash", str(SCRIPT), str(free), str(disk), str(tmp_path)],
        env={"PATH": "/usr/bin:/bin", "ROOM_POLL": "1", "ROOM_TRIES": "2"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_goes_on_when_there_is_room(tmp_path):
    assert run(tmp_path, 0, 0).returncode == 0


def test_waits_then_fails_without_room(tmp_path):
    out = run(tmp_path, 0, 10**9)
    assert out.returncode == 1
    assert "disk" in out.stderr
