"""The grid's rounds, and the rounds of its earlier versions, are never
simulated twice."""

from pathlib import Path

import pytest
import yaml
from flowlib import grids

ROOT = Path(__file__).parents[2]

CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
TABLE = CONFIG["grid_rounds"]
RETIRED = CONFIG["retired_rounds"]


def test_the_grid_has_rounds_33_to_38():
    assert TABLE["rounds"] == [34, 35, 36, 37]
    assert TABLE["validation_round"] == 33
    # Never trained on or monitored, and scored only for the final model.
    assert TABLE["test_round"] == 38
    assert grids.owner(38, TABLE) == grids.GRID


def test_earlier_versions_rounds_are_retired():
    assert (RETIRED["first"], RETIRED["last"]) == (9, 32)
    for r in (9, 20, 32):
        assert grids.owner(r, TABLE, RETIRED) == grids.RETIRED
    assert grids.owner(33, TABLE, RETIRED) == grids.GRID
    assert grids.owner(8, TABLE, RETIRED) is None


def test_the_grid_s_rounds_are_distinct_and_none_retired():
    seen = [
        r
        for r in (
            *TABLE["rounds"],
            TABLE["validation_round"],
            TABLE["test_round"],
            TABLE["topup_round"],
        )
        if r is not None
    ]
    assert len(seen) == len(set(seen))
    assert not [r for r in seen if RETIRED["first"] <= r <= RETIRED["last"]]


def test_a_run_takes_the_grid_s_rounds_unless_set():
    unset = {"rounds": None, "validation_round": None, "sources": None}
    ft = grids.resolve(dict(unset), TABLE)
    assert ft["rounds"] == [34, 35, 36, 37]
    assert ft["validation_round"] == 33
    assert ft["sources"] == TABLE["sources"]
    ft = grids.resolve(
        {"rounds": [1], "validation_round": 2, "sources": ["a"]}, TABLE
    )
    assert (ft["rounds"], ft["validation_round"]) == ([1], 2)


def test_a_retired_round_is_never_released():
    grids.check_release([33, 34, 38, 4], TABLE, RETIRED)
    with pytest.raises(ValueError, match="round 9 is an earlier grid's"):
        grids.check_release([33, 9], TABLE, RETIRED)


def test_the_config_names_no_grid_version():
    text = (ROOT / "workflow" / "config.yaml").read_text()
    assert "grids/" not in text
    assert "grid" not in CONFIG["finetune"]
    assert CONFIG["simulate"]["grid_file"] == "config/finetune/grid.yaml"
    assert "name" not in yaml.safe_load(
        (ROOT / CONFIG["simulate"]["grid_file"]).read_text()
    )
