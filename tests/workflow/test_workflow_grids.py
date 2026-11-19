"""Each simulation grid has its own rounds, seeds and releases."""

from pathlib import Path

import pytest
import yaml
from flowlib import grids

ROOT = Path(__file__).parents[2]

CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
TABLE = CONFIG["grid_rounds"]


def test_r6_has_new_rounds_20_to_30():
    assert TABLE["r6"]["rounds"] == list(range(21, 31))
    assert TABLE["r6"]["held_round"] == 20
    assert TABLE["r5"]["rounds"] == list(range(10, 20))
    assert TABLE["r5"]["held_round"] == 9


def test_no_two_grids_share_a_round():
    seen = [r for g in TABLE.values() for r in (*g["rounds"], g["held_round"])]
    assert len(seen) == len(set(seen))


def test_a_run_takes_its_grid_s_rounds_unless_set():
    unset = {"rounds": None, "held_round": None, "sources": None}
    ft = grids.resolve({"grid": "r6", **unset}, TABLE)
    assert ft["rounds"] == list(range(21, 31))
    assert ft["held_round"] == 20
    ft = grids.resolve(
        {"grid": "r6", "rounds": [1], "held_round": 2, "sources": ["a"]},
        TABLE,
    )
    assert (ft["rounds"], ft["held_round"]) == ([1], 2)
    with pytest.raises(ValueError, match="r7"):
        grids.resolve({"grid": "r7", **unset}, TABLE)


def test_a_grid_may_not_release_another_grid_s_round():
    grids.check_release([20, 21, 4], "r6", TABLE)
    with pytest.raises(ValueError, match="round 9 is grid r5's"):
        grids.check_release([9], "r6", TABLE)


def test_a_release_s_grid_is_in_its_notes_and_unmarked_ones_are_r5():
    assert grids.release_grid("Fine-tuning simulations. Grid r6.") == "r6"
    assert grids.release_grid("Fine-tuning simulations of round 9.") == "r5"
    assert grids.release_grid("") == "r5"


def test_the_top_up_round_is_its_grid_s():
    # r6's top-up round (32) is r6's alone, so no other grid releases it.
    assert TABLE["r6"]["topup_round"] == 32
    assert grids.owner(32, TABLE) == "r6"
    with pytest.raises(ValueError, match="round 32 is grid r6's"):
        grids.check_release([32], "r5", TABLE)
