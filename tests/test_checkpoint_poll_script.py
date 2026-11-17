"""Tests of scripts/checkpoint_poll.py's pure parts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def poll():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "checkpoint_poll", SCRIPTS / "checkpoint_poll.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["checkpoint_poll"] = mod
    spec.loader.exec_module(mod)
    return mod


def _budget(verdict: str, max_parallel: int) -> dict:
    return {"verdict": verdict, "max_parallel": max_parallel}


def test_a_unit_starts_only_with_a_free_core_and_memory(poll) -> None:
    assert poll.can_launch(_budget("green", 2), mem_avail_gb=30)
    # The budget says green but no core is free (load near the cores):
    # one nice-19, one-thread unit still runs, so scoring never stalls
    # behind the simulation rounds, but no second one.
    assert poll.can_launch(_budget("green", 0), mem_avail_gb=30, running=0)
    assert not poll.can_launch(_budget("green", 0), mem_avail_gb=30, running=1)
    assert not poll.can_launch(_budget("red", 0), mem_avail_gb=30, running=0)
    assert not poll.can_launch(_budget("red", 3), mem_avail_gb=30)
    # 16 GB must stay free beside the unit's 6 GB scope.
    # The gate uses a unit's measured peak (about 5 GB), the 6 GB scope
    # staying its hard cap.
    assert not poll.can_launch(_budget("green", 2), mem_avail_gb=20.9)
    assert poll.can_launch(_budget("green", 2), mem_avail_gb=21)
    assert poll.can_launch(_budget("amber", 1), mem_avail_gb=22)


def test_a_checkpoint_needs_its_runs_the_reference_and_old_bdbv(poll) -> None:
    units = poll.units_for("bbbbbbbbbbbb", scored={"aaaaaaaaaaaa"})
    assert units[0] == ("heldout", "bbbbbbbbbbbb")
    assert set(units) == {
        ("heldout", "bbbbbbbbbbbb"),
        ("bdbv", "reference"),
        ("bdbv", "bbbbbbbbbbbb"),
        ("bdbv", "aaaaaaaaaaaa"),
        ("germany", "reference"),
        ("germany", "bbbbbbbbbbbb"),
    }
    # BDBV first: its few origins give rows within minutes, then the long
    # German runs, which write each date to the cache as it finishes.
    order = [u for u in units if u[0] != "heldout"]
    assert order[:2] == [("bdbv", "reference"), ("bdbv", "bbbbbbbbbbbb")]


def test_mailbox_lines_carry_the_label(poll) -> None:
    line = poll.mailbox_line("rows landed", when="2026-11-16 12:00")
    assert line == "2026-11-16 12:00 archie: ckpt-scores: rows landed\n"


def test_the_zero_shot_reference_lands_before_any_checkpoint(poll, tmp_path):
    # Nothing run yet: the reference's German and BDBV units come first.
    assert poll.reference_pending(tmp_path) == [
        ("germany", "reference"),
        ("bdbv", "reference"),
    ]
    # Once landed (its stamp written), the reference is not pending.
    (tmp_path / poll.REFERENCE_STAMP).write_text("abc\n")
    assert poll.reference_pending(tmp_path) == []


def test_only_slots_checkpoints_start_the_follow_up_jobs(poll) -> None:
    # The aggregation arms and the slots check run for the final run only.
    cmd = poll.follow_ups("slots-dddddddddddd")
    assert cmd is not None
    assert cmd[-1] == "slots-dddddddddddd"
    assert Path(cmd[0]).name == "slots_followups.sh"
    assert poll.follow_ups("aaaaaaaaaaaa") is None


def test_slots_first_then_the_complete_reference_then_no_slots(poll) -> None:
    assert poll.next_job(["slots-a", "b"], complete_pending=True) == "slots-a"
    assert poll.next_job(["b"], complete_pending=True) == poll.COMPLETE_JOB
    assert poll.next_job([], complete_pending=True) == poll.COMPLETE_JOB
    assert poll.next_job(["b"], complete_pending=False) == "b"
    assert poll.next_job([], complete_pending=False) is None


def test_slots_checkpoints_land_bdbv_before_germany(poll) -> None:
    units = poll.units_for("slots-a", scored=set())
    fast, slow = poll.split_fast(units)
    assert {u[0] for u in fast} == {"heldout", "bdbv"}
    assert {u[0] for u in slow} == {"germany"}


def test_a_new_slots_checkpoint_s_bdbv_goes_before_an_older_german_run(
    poll,
) -> None:
    # n500 publishes while n250's German run is going: n500's held-out and
    # BDBV run first (minutes), then n250's German run resumes.
    queue = ["slots-a", "slots-b", "c"]
    assert poll.next_job(queue, False, fresh={"slots-b"}) == "slots-b"
    assert poll.next_job(queue, False, fresh=set()) == "slots-a"
    assert poll.should_pause("slots-a", ["slots-a", "slots-b"], {"slots-b"})
    assert not poll.should_pause("slots-a", ["slots-a"], {"slots-a"})
    assert not poll.should_pause("slots-b", ["slots-a", "slots-b"], set())


def test_a_rebuild_with_no_change_tells_bdbv_paper_nothing(poll) -> None:
    assert poll.interim_note("slots-a", "abc", changed=False) is None
    note = poll.interim_note("slots-a", "abc", changed=True)
    assert note is not None
    assert "abc" in note


def test_the_new_slots_run_goes_first_and_zero_init_runs_last(poll) -> None:
    # The lead: the random-init slots run first; the zero-init run's German
    # runs only with spare capacity, after every other job.
    queue = ["slots-z1", "slots-z2", "slots-n1", "c"]
    zero = {"slots-z1", "slots-z2"}
    assert poll.next_job(queue, True, fresh=set(), zero_init=zero) == "slots-n1"
    assert poll.next_job(queue, True, fresh={"slots-z2"}, zero_init=zero) == (
        "slots-n1"
    )
    rest = ["slots-z1", "c"]
    assert poll.next_job(rest, True, set(), zero) == poll.COMPLETE_JOB
    assert poll.next_job(rest, False, set(), zero) == "c"
    assert poll.next_job(["slots-z1"], False, set(), zero) == "slots-z1"
    # A fresh new-run checkpoint pauses a zero-init German run, not the
    # other way round.
    assert poll.should_pause(
        "slots-z1", ["slots-z1", "slots-n1"], {"slots-n1"}, zero
    )
    assert not poll.should_pause(
        "slots-n1", ["slots-z1", "slots-n1"], {"slots-z1"}, zero
    )


def test_slots_run_checkpoints_are_held_while_the_hold_file_exists(
    poll, tmp_path
) -> None:
    # The new run's adapters carry parent_scale, which slots.py drops until
    # bdbv-covariates' fix is merged: hold them, not the zero-init ones.
    queue = ["slots-z", "slots-new", "c"]
    zero = {"slots-z"}
    hold = tmp_path / "hold"
    assert poll.unheld(queue, zero, hold) == queue
    hold.write_text("pending\n")
    assert poll.unheld(queue, zero, hold) == ["slots-z", "c"]


def test_only_current_run_slots_checkpoints_are_scored(poll) -> None:
    # This code loads the one-model run's adapters only; superseded slots
    # runs (zero-init, random slots, the attention-bias trials) are
    # skipped, and unclassified ones wait for their settings.
    runs = {
        "slots-a": "one-model",
        "slots-b": "attn-bias",
        "slots-z": "slots-zeroinit",
    }
    queue = ["slots-z", "slots-b", "slots-a", "slots-new", "c"]
    assert poll.scoreable(queue, runs) == ["slots-a", "c"]
    assert poll.unclassified(queue, runs) == ["slots-new"]


def test_a_queue_of_only_held_or_superseded_checkpoints_is_idle(
    poll, tmp_path
) -> None:
    # At 01:04 every queued checkpoint was held, and the loop crashed on
    # None; nothing to score must mean idle.
    hold = tmp_path / "hold"
    hold.write_text("x\n")
    runs = {"slots-a": "one-model", "slots-b": "attn-bias"}
    queue = ["slots-a", "slots-b"]
    assert poll.choose(queue, runs, set(), hold, False, set()) is None
    assert poll.choose([], {}, set(), hold, False, set()) is None
    assert poll.choose(queue, runs, set(), tmp_path / "none", False, set()) == (
        "slots-a"
    )


def test_new_bdbv_origins_skip_what_this_code_cannot_load(poll) -> None:
    runs = {
        "a": "no-slots",
        "slots-z": "slots-zeroinit",
        "slots-o": "one-model",
    }
    labels = ["reference", "a", "slots-z", "slots-o"]
    assert poll.extendable(labels, runs) == ["reference", "a", "slots-o"]


# The daily target --------------------------------------------------------


def _settings(cache: Path, label: str, run: str) -> None:
    import json

    held = cache / "heldout"
    held.mkdir(exist_ok=True)
    settings = {
        "one-model": {
            "series_bias": {"init": "zero"},
            "sources": ["daily", "daily+weekly"],
        },
        "attn-bias": {"series_bias": {"init": "zero"}},
        "no-slots": {},
    }[run]
    (held / f"{label}.json").write_text(json.dumps(settings))


def test_units_name_their_target_on_the_command_line(poll, tmp_path) -> None:
    # A 7-day unit must not pick up the label's default targets (the
    # reference's include the daily one), so the target is always given.
    cmd = poll.unit_command(tmp_path, ("germany", "reference"))
    assert cmd[-4:] == ["germany", str(tmp_path), "reference", "7day"]
    daily = poll.unit_command(tmp_path, ("bdbv", "slots-a", "daily"))
    assert daily[-4:] == ["bdbv", str(tmp_path), "slots-a", "daily"]


def test_a_checkpoint_s_units_are_7day_only(poll) -> None:
    # The 7-day rows never wait for the daily ones.
    units = poll.units_for("slots-a", {"slots-b"})
    assert all(len(u) == 2 or u[2] == "7day" for u in units)


def test_the_reference_daily_run_comes_first_when_idle(poll, tmp_path) -> None:
    _settings(tmp_path, "slots-a", "one-model")
    runs = poll.runs_of(tmp_path)
    jobs = poll.daily_units(tmp_path, ["slots-a"], runs)
    assert jobs == [
        ("bdbv", "reference", "daily"),
        ("germany", "reference", "daily"),
    ]


def test_checkpoints_daily_units_wait_for_the_reference_s(
    poll, monkeypatch, tmp_path
) -> None:
    _settings(tmp_path, "slots-a", "one-model")
    _settings(tmp_path, "slots-old", "attn-bias")
    _settings(tmp_path, "aaaa", "no-slots")
    _settings(tmp_path, "slots-done", "one-model")
    runs = poll.runs_of(tmp_path)
    done = {
        ("bdbv", "reference", "daily"),
        ("germany", "reference", "daily"),
        ("bdbv", "slots-done", "daily"),
        ("germany", "slots-done", "daily"),
    }
    monkeypatch.setattr(
        poll.cs,
        "unit_done",
        lambda _c, study, label, target="7day": (study, label, target) in done,
    )
    labels = ["aaaa", "slots-a", "slots-done", "slots-old"]
    jobs = poll.daily_units(tmp_path, labels, runs)
    # Only the one-model run takes the daily target by default; the
    # superseded runs and finished units are left out.
    assert jobs == [
        ("bdbv", "slots-a", "daily"),
        ("germany", "slots-a", "daily"),
    ]
    done |= set(jobs)
    assert poll.daily_units(tmp_path, labels, runs) == []
