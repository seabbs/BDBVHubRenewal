"""A heavy job starts only with memory, disk and load to spare."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "workflow" / "scripts" / "wait_for_room.sh"


def run(tmp_path, free, disk, *ceiling, load="0.50", where=None, tries="2"):
    """Run the script with `uptime` reporting a 1-minute load of `load`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "uptime"
    stub.write_text(
        "#!/bin/sh\n"
        "echo ' 10:00:00 up 1 day,  1 user,  "
        f"load average: {load}, 1.00, 1.00'\n"
    )
    stub.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "ROOM_POLL": "1"}
    if tries:
        env["ROOM_TRIES"] = tries
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(free),
            str(disk),
            str(where or tmp_path),
            *ceiling,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_goes_on_when_there_is_room(tmp_path):
    assert run(tmp_path, 0, 0).returncode == 0


def test_waits_then_fails_without_room(tmp_path):
    out = run(tmp_path, 0, 10**9)
    assert out.returncode == 1
    assert "disk" in out.stderr


def test_fails_at_once_when_it_cannot_read_the_room(tmp_path):
    # A path df cannot read gives no number, which must not read as "no
    # room": without ROOM_TRIES that would wait forever.
    out = run(tmp_path, 0, 0, where=tmp_path / "missing", tries="")
    assert out.returncode == 2
    assert "cannot read" in out.stderr


def test_waits_then_fails_above_the_load_ceiling(tmp_path):
    out = run(tmp_path, 0, 0, "12", load="20.31")
    assert out.returncode == 1
    assert "load 20.31" in out.stderr


def test_goes_on_below_the_load_ceiling(tmp_path):
    assert run(tmp_path, 0, 0, "12", load="11.90").returncode == 0


def test_the_ceiling_is_the_core_count_by_default(tmp_path):
    cores = os.cpu_count()
    assert run(tmp_path, 0, 0, load=f"{cores - 0.5}").returncode == 0
    assert run(tmp_path, 0, 0, load=f"{cores + 1}").returncode == 1


def test_fails_at_once_when_it_cannot_read_the_load(tmp_path):
    out = run(tmp_path, 0, 0, "12", load="?", tries="")
    assert out.returncode == 2
    assert "cannot read" in out.stderr
