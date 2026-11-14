"""Tests of the synthetic scenario grid (`papertables.synthetic_grid`)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from papertables import synthetic_grid

SIMULATOR = Path(__file__).parents[2] / "scripts" / "finetune_simulate.R"

needs_simulator = pytest.mark.skipif(
    not SIMULATOR.exists(),
    reason="needs scripts/finetune_simulate.R, which comes with #192",
)


def _drawn_settings() -> set[str]:
    """Return the names `draw_design` and `draw_values` set."""
    text = SIMULATOR.read_text()
    names: set[str] = set()
    for fn in ("draw_design", "draw_values"):
        body = text[text.index(f"{fn} <- function") :]
        body = body[: body.index("\n}\n")]
        names |= set(re.findall(r"^    (\w+) = ", body, flags=re.MULTILINE))
    return names


@needs_simulator
def test_the_grid_has_a_row_for_every_drawn_setting() -> None:
    drawn = _drawn_settings() - {"grid", "prob"}
    rows = {s for row in synthetic_grid.GRID["setting"] for s in row.split(";")}
    assert drawn <= rows


def test_the_table_adds_realised_values_from_the_settings() -> None:
    settings = pd.DataFrame(
        {
            "outbreak": ["a", "b", "c"],
            "delay_mean": [6.0, 10.0, 20.0],
            "n_places": [1, 1, 5],
            "report_days": ["all", "Monday", "all"],
            "family": ["negbin", "poisson", "negbin"],
            "stable": [True, False, False],
            "cases": [50, 100, 30000],
        }
    )
    table = synthetic_grid.grid_table(settings)
    assert list(table.columns) == list(synthetic_grid.COLUMNS)
    row = table.set_index("setting").loc["delay_mean"]
    assert row["realised"] == "10 [6, 20]"
    places = table.set_index("setting").loc["n_places"]
    assert places["realised"] == "1 [1, 5]; 67% one place"
    days = table.set_index("setting").loc["report_days"]
    assert days["realised"] == "67% every day"
    family = table.set_index("setting").loc["family"]
    assert family["realised"] == "negbin 67%, poisson 33%"
    stable = table.set_index("setting").loc["stable"]
    assert stable["realised"] == "33% yes"
    # A setting not in the settings files has no realised value.
    assert table.set_index("setting").loc["max_delay", "realised"] == ""


def test_markdown_has_one_row_per_setting() -> None:
    settings = pd.DataFrame(
        {"outbreak": ["a"], "delay_mean": [6.0], "rt_drift_sd": [1.3e-5]}
    )
    table = synthetic_grid.grid_table(settings)
    assert "e-" not in table.set_index("setting").loc["rt_drift_sd", "realised"]
    md = synthetic_grid.to_markdown(table)
    lines = md.strip().splitlines()
    assert len(lines) == len(table) + 2
    assert lines[0].startswith("| Factor |")
    # Pipes in formulas are escaped, so each row has its five cells.
    assert all(line.count(" | ") == 4 for line in lines[2:])


def _design_settings() -> set[str]:
    """Return the names `draw_design` sets."""
    text = SIMULATOR.read_text()
    body = text[text.index("draw_design <- function") :]
    body = body[: body.index("\n}\n")]
    return set(re.findall(r"^    (\w+) = ", body, flags=re.MULTILINE))


@needs_simulator
def test_the_main_table_has_one_row_per_design_branch() -> None:
    main = synthetic_grid.MAIN
    covered = {s for row in main["setting"] for s in row.split(";")}
    # Places and ages follow from the strata.
    assert (
        _design_settings() - {"grid", "prob", "n_places", "n_ages"} <= covered
    )
    assert len(main) < len(synthetic_grid.GRID) / 3
    md = synthetic_grid.to_markdown(
        synthetic_grid.main_table(pd.DataFrame({"outbreak": []})),
        synthetic_grid.MAIN_HEADERS,
    )
    assert md.splitlines()[0].startswith("| Design |")
