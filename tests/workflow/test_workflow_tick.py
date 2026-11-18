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
    targets = ["score land"]

    real = subprocess.run

    def run(cmd, **kw):
        # Only the targets' runs; the machine's budget is read for real.
        if cmd[0] != "nice":
            return real(cmd, **kw)
        start = cmd.index("snakemake") + 1
        target = " ".join(cmd[start : cmd.index("--workflow-profile")])
        calls.append(target)
        return subprocess.CompletedProcess(cmd, codes.get(target, 0))

    monkeypatch.setattr(tick, "ROOT", tmp_path)
    monkeypatch.setattr(tick.subprocess, "run", run)
    monkeypatch.setattr(
        tick.localconfig,
        "read",
        lambda: {
            "SCHEDULE_TARGETS": targets[0],
            "MIN_FREE_GB": "0",
            "MIN_DISK_GB": "0",
        },
    )
    return calls, codes, targets


def test_targets_run_in_order(ran):
    calls, _, _ = ran
    assert tick.main() == 0
    assert calls == ["score", "land"]


def test_a_failed_target_stops_the_tick(ran):
    calls, codes, _ = ran
    codes["score"] = 1
    assert tick.main() == 1
    assert calls == ["score"]


def test_a_running_tick_means_this_one_is_skipped(ran, tmp_path):
    calls, _, _ = ran
    with schedule.locked(tmp_path / "output" / "workflow-tick.lock"):
        assert tick.main() == 0
    assert calls == []


def test_no_room_means_the_tick_is_skipped(ran, monkeypatch):
    calls, _, _ = ran
    monkeypatch.setattr(tick.schedule, "room", lambda *_: (False, "full"))
    assert tick.main() == 0
    assert calls == []


def test_one_run_takes_every_pool_target(ran):
    calls, _, targets = ran
    targets[0] = "simulate finetune score land publish"
    assert tick.main() == 0
    assert calls == ["simulate finetune score", "land", "publish"]


def test_a_failed_pool_skips_land_but_still_publishes(ran):
    # A held-out scoring unit failing until its round is in must not hold
    # back a checkpoint's release, nor land half-built tables.
    calls, codes, targets = ran
    targets[0] = "finetune score land publish release"
    codes["finetune score"] = 1
    assert tick.main() == 1
    assert calls == ["finetune score", "publish"]
