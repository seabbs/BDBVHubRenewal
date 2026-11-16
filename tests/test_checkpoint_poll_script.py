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
    # The budget says green but no core is free (load near the cores).
    assert not poll.can_launch(_budget("green", 0), mem_avail_gb=30)
    assert not poll.can_launch(_budget("red", 3), mem_avail_gb=30)
    # 16 GB must stay free beside the unit's 6 GB scope.
    assert not poll.can_launch(_budget("green", 2), mem_avail_gb=21)
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
    # The long German runs start first, so they overlap the short ones.
    order = [u for u in units if u[0] != "heldout"]
    assert order[:2] == [("germany", "reference"), ("germany", "bbbbbbbbbbbb")]


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
