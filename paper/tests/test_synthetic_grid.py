"""Tests of the synthetic scenario grid table (`paper.tables.synthetic_grid`)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from paper.tables import synthetic_grid

SIMULATOR = Path(__file__).parents[2] / "scripts" / "finetune_simulate.R"


def _drawn_settings() -> set[str]:
    """Return the names `draw_settings` sets in the simulator."""
    text = SIMULATOR.read_text()
    body = text[text.index("draw_settings <- function") :]
    body = body[: body.index("\n}\n")]
    return set(re.findall(r"^\s+(\w+) = ", body, flags=re.MULTILINE))


def test_the_grid_has_a_row_for_every_drawn_setting() -> None:
    drawn = _drawn_settings() - {"outbreak", "timestep"}
    assert drawn <= set(synthetic_grid.GRID["setting"])


def test_the_table_adds_realised_values_from_the_settings() -> None:
    settings = pd.DataFrame(
        {
            "outbreak": ["a", "b", "c"],
            "gt_mean": [6.0, 10.0, 20.0],
            "n_places": [1, 1, 5],
            "report_days": [7, 2, 7],
            "cases": [50, 100, 30000],
        }
    )
    table = synthetic_grid.grid_table(settings)
    assert list(table.columns) == list(synthetic_grid.COLUMNS)
    row = table.set_index("setting").loc["gt_mean"]
    assert row["realised"] == "10 [6, 20]"
    places = table.set_index("setting").loc["n_places"]
    assert places["realised"] == "1 [1, 5]; 67% one place"
    days = table.set_index("setting").loc["report_days"]
    assert days["realised"] == "67% every day"
    # A setting not in the settings files has no realised value.
    assert table.set_index("setting").loc["max_delay", "realised"] == ""


def test_markdown_has_one_row_per_setting() -> None:
    settings = pd.DataFrame({"outbreak": ["a"], "gt_mean": [6.0]})
    table = synthetic_grid.grid_table(settings)
    md = synthetic_grid.to_markdown(table)
    lines = md.strip().splitlines()
    assert len(lines) == len(table) + 2
    assert lines[0].startswith("| Factor |")
