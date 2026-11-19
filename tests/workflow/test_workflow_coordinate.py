"""The persistent coordinator run: a tick, then the next, until stopped."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))

import coordinate

ROOT = Path(__file__).parents[2]


@pytest.fixture
def passes(tmp_path):
    codes = []
    slept = []
    stop = tmp_path / "coordinator.stop"

    def tick():
        codes.append(len(codes))
        if len(codes) == 3:
            stop.write_text("")
        return 1 if len(codes) == 2 else 0

    return tick, slept, stop, codes


def test_runs_ticks_until_the_stop_file(passes):
    tick, slept, stop, codes = passes
    rc = coordinate.run(tick, stop, pause=5, backoff=60, sleep=slept.append)
    assert rc == 0
    assert len(codes) == 3
    # A pause after a pass, a longer one after a failed pass, none after
    # the pass that found the stop file.
    assert slept == [5, 60]
    assert not stop.exists()


def test_a_stop_file_before_it_starts_runs_nothing(passes):
    tick, slept, stop, codes = passes
    stop.write_text("")
    rc = coordinate.run(tick, stop, pause=5, backoff=60, sleep=slept.append)
    assert rc == 0
    assert codes == []


def test_an_installed_timer_is_refused():
    got = subprocess.run(
        ["python3", "workflow/scripts/schedule.py", "install"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert got.returncode != 0
    assert "task run" in got.stdout + got.stderr


def test_the_taskfile_runs_the_coordinator():
    text = (ROOT / "Taskfile.yml").read_text()
    assert "workflow/scripts/coordinate.py" in text


def test_a_tick_that_raises_backs_off_and_goes_on(tmp_path):
    stop = tmp_path / "coordinator.stop"
    slept = []
    calls = []

    def tick():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("a worker went away")
        stop.write_text("")
        return 0

    rc = coordinate.run(tick, stop, pause=5, backoff=60, sleep=slept.append)
    assert rc == 0
    assert len(calls) == 2
    assert slept == [60]
