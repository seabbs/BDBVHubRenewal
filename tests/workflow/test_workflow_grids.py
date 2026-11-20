"""Each simulation grid has its own rounds, seeds and releases."""

from pathlib import Path

import pytest
import yaml
from flowlib import grids

ROOT = Path(__file__).parents[2]

CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
TABLE = CONFIG["grid_rounds"]
RETIRED = CONFIG["retired_rounds"]


def test_r7_has_rounds_33_to_38():
    assert TABLE["r7"]["rounds"] == [34, 35, 36, 37]
    assert TABLE["r7"]["validation_round"] == 33
    # Never trained on or monitored, and scored only for the final model.
    assert TABLE["r7"]["test_round"] == 38
    assert grids.owner(38, TABLE) == "r7"


def test_only_r7_is_listed_and_earlier_grids_rounds_are_retired():
    assert set(TABLE) == {"r7"}
    assert (RETIRED["first"], RETIRED["last"]) == (9, 32)
    for r in (9, 20, 32):
        assert grids.owner(r, TABLE, RETIRED) == grids.RETIRED
    assert grids.owner(33, TABLE, RETIRED) == "r7"
    assert grids.owner(8, TABLE, RETIRED) is None


def test_no_two_grids_share_a_round():
    seen = [
        r
        for g in TABLE.values()
        for r in (*g["rounds"], g["validation_round"], g["test_round"])
        if r is not None
    ]
    assert len(seen) == len(set(seen))
    assert not [r for r in seen if RETIRED["first"] <= r <= RETIRED["last"]]


def test_a_run_takes_its_grid_s_rounds_unless_set():
    unset = {"rounds": None, "validation_round": None, "sources": None}
    ft = grids.resolve({"grid": "r7", **unset}, TABLE)
    assert ft["rounds"] == [34, 35, 36, 37]
    assert ft["validation_round"] == 33
    ft = grids.resolve(
        {"grid": "r7", "rounds": [1], "validation_round": 2, "sources": ["a"]},
        TABLE,
    )
    assert (ft["rounds"], ft["validation_round"]) == ([1], 2)
    with pytest.raises(ValueError, match="r99"):
        grids.resolve({"grid": "r99", **unset}, TABLE)


def test_a_grid_may_not_release_another_grid_s_or_a_retired_round():
    grids.check_release([33, 34, 4], "r7", TABLE, RETIRED)
    with pytest.raises(ValueError, match="round 9 is grid retired's"):
        grids.check_release([9], "r7", TABLE, RETIRED)
    with pytest.raises(ValueError, match="round 33 is grid r7's"):
        grids.check_release([33], "r8", TABLE, RETIRED)


def test_a_release_s_grid_is_in_its_notes_and_unmarked_ones_no_grid_s():
    assert grids.release_grid("Fine-tuning simulations. Grid r7.") == "r7"
    assert grids.release_grid("Fine-tuning simulations of round 9.") == (
        grids.UNMARKED
    )
    assert grids.release_grid("") == grids.UNMARKED
    assert grids.UNMARKED not in TABLE
