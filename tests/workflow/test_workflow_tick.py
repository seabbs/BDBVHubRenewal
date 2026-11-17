"""One scheduled tick: runs the targets in order, or skips."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))

import tick
from flowlib import schedule


@pytest.fixture
def ran(tmp_path, monkeypatch):
    calls = []
    codes = {}

    def run(cmd, **_):
        target = cmd[8]
        calls.append(target)
        return subprocess.CompletedProcess(cmd, codes.get(target, 0))

    monkeypatch.setattr(tick, "ROOT", tmp_path)
    monkeypatch.setattr(tick.subprocess, "run", run)
    monkeypatch.setattr(
        tick.localconfig,
        "read",
        lambda: {
            "SCHEDULE_TARGETS": "score land",
            "MIN_FREE_GB": "0",
            "MIN_DISK_GB": "0",
        },
    )
    return calls, codes


def test_targets_run_in_order(ran):
    calls, _ = ran
    assert tick.main() == 0
    assert calls == ["score", "land"]


def test_a_failed_target_stops_the_tick(ran):
    calls, codes = ran
    codes["score"] = 1
    assert tick.main() == 1
    assert calls == ["score"]


def test_a_running_tick_means_this_one_is_skipped(ran, tmp_path):
    calls, _ = ran
    with schedule.locked(tmp_path / "output" / "workflow-tick.lock"):
        assert tick.main() == 0
    assert calls == []


def test_no_room_means_the_tick_is_skipped(ran, monkeypatch):
    calls, _ = ran
    monkeypatch.setattr(tick.schedule, "room", lambda *_: (False, "full"))
    assert tick.main() == 0
    assert calls == []
