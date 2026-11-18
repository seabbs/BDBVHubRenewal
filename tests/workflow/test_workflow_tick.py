"""One scheduled tick: runs the targets in order, or skips."""

import json
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

    def watch(cmd, step, _cfg):
        # The machine's budget in `cmd` is read for real.
        assert cmd[:3] == ["nice", "-n", "19"]
        calls.append(step)
        return codes.get(step, 0)

    monkeypatch.setattr(tick, "ROOT", tmp_path)
    monkeypatch.setattr(tick, "watch", watch)
    monkeypatch.setattr(
        tick.localconfig,
        "read",
        lambda: {
            "SCHEDULE_TARGETS": targets[0],
            "MIN_FREE_GB": "0",
            "MIN_DISK_GB": "0",
            "ROOT": str(tmp_path),
            "HOST_NAME": "test-host",
        },
    )
    return calls, codes, targets


def test_targets_run_in_order(ran):
    calls, _, _ = ran
    assert tick.main() == 0
    assert calls == ["refresh", "score", "land"]


def test_a_failed_target_stops_the_tick(ran):
    calls, codes, _ = ran
    codes["score"] = 1
    assert tick.main() == 1
    assert calls == ["refresh", "score"]


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
    assert calls == [
        "refresh",
        "simulate finetune score",
        "land",
        "publish",
    ]


def test_a_failed_pool_skips_land_but_still_publishes(ran):
    # A held-out scoring unit failing until its round is in must not hold
    # back a checkpoint's release, nor land half-built tables.
    calls, codes, targets = ran
    targets[0] = "finetune score land publish release"
    codes["finetune score"] = 1
    assert tick.main() == 1
    assert calls == ["refresh", "finetune score", "publish"]


def status_of(tmp_path):
    return json.loads((tmp_path / "status" / "test-host.json").read_text())


@pytest.mark.usefixtures("ran")
def test_a_tick_leaves_an_idle_status_when_it_ends(tmp_path):
    assert tick.main() == 0
    got = status_of(tmp_path)
    assert got["step"] == "idle"
    assert got["alive"] is False


@pytest.mark.usefixtures("ran")
def test_a_skipped_tick_says_so_in_its_status(tmp_path, monkeypatch):
    monkeypatch.setattr(tick.schedule, "room", lambda *_: (False, "full"))
    assert tick.main() == 0
    assert status_of(tmp_path)["step"] == "skipped: full"


def test_a_running_step_is_reported_while_it_runs(tmp_path):
    cfg = {"ROOT": str(tmp_path), "HOST_NAME": "test-host", "STATUS_EVERY": "1"}
    seen = tmp_path / "seen.json"
    # The step copies its own status, written while it runs, then fails.
    status = tmp_path / "status" / "test-host.json"
    cmd = ["sh", "-c", f"sleep 2; cp {status} {seen}; exit 3"]
    assert tick.watch(cmd, "simulate", cfg) == 3
    got = json.loads(seen.read_text())
    assert got["step"] == "simulate"
    assert got["alive"] is True


def test_a_failed_refresh_still_runs_the_pool(ran):
    # Unreachable releases leave the last lists; the pool still runs.
    calls, codes, _ = ran
    codes["refresh"] = 1
    assert tick.main() == 1
    assert calls == ["refresh", "score"]
