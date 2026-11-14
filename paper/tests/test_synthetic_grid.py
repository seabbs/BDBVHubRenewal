"""Tests of the synthetic scenario grid (`paper.tables.synthetic_grid`)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from paper.tables import synthetic_grid

ROOT = Path(__file__).parents[2]
SIMULATOR = ROOT / "scripts" / "finetune_simulate.R"
TEXT = SIMULATOR.read_text()


def test_factors_and_levels_are_read_from_the_simulator() -> None:
    factors = synthetic_grid.parse_factors(TEXT)
    assert "strata" in factors
    assert factors["observation"] == ["negbin", "negbin1d", "poisson"]
    assert all(len(levels) >= 2 for levels in factors.values())


def test_values_are_read_from_the_simulator() -> None:
    values = synthetic_grid.parse_values(TEXT)
    assert values["offset"] == "sample(0:6, 1)"
    # Multi-line expressions are kept whole.
    assert values["gt_mean"].startswith("if (renewal)")
    assert "max_delay" in values


def test_every_setting_has_a_label() -> None:
    names = [
        *synthetic_grid.parse_factors(TEXT),
        *synthetic_grid.DERIVED,
        *synthetic_grid.parse_values(TEXT),
    ]
    missing = [n for n in names if n not in synthetic_grid.LABELS]
    assert not missing


def test_the_main_table_has_one_row_per_factor_with_its_levels() -> None:
    factors = synthetic_grid.parse_factors(TEXT)
    table = synthetic_grid.main_table(TEXT, pd.DataFrame({"outbreak": []}))
    assert list(table["setting"]) == list(factors)
    for (_, row), levels in zip(
        table.iterrows(), factors.values(), strict=True
    ):
        for level in levels:
            assert level in row["levels"]
        assert f"1/{len(levels)} each" in row["levels"]


def test_the_si_table_has_every_setting() -> None:
    table = synthetic_grid.si_table(TEXT, pd.DataFrame({"outbreak": []}))
    names = [
        *synthetic_grid.parse_factors(TEXT),
        *synthetic_grid.DERIVED,
        *synthetic_grid.parse_values(TEXT),
    ]
    assert list(table["setting"]) == names


def test_realised_values_come_from_the_settings() -> None:
    settings = pd.DataFrame(
        {
            "outbreak": ["a", "b", "c"],
            "observation": ["negbin", "poisson", "negbin"],
            "delay_mean": [6.0, 10.0, 20.0],
            "n_places": [1, 1, 5],
            "report_days": ["all", "Monday", "all"],
        }
    )
    main = synthetic_grid.main_table(TEXT, settings).set_index("setting")
    assert main.loc["observation", "realised"] == "negbin 67%, poisson 33%"
    si = synthetic_grid.si_table(TEXT, settings).set_index("setting")
    assert si.loc["delay_mean", "realised"] == "10 [6, 20]"
    assert si.loc["n_places", "realised"] == "1 [1, 5]; 67% one place"
    assert si.loc["report_days", "realised"] == "67% every day"
    assert si.loc["offset", "realised"] == ""


def test_markdown_escapes_pipes() -> None:
    table = synthetic_grid.si_table(TEXT, pd.DataFrame({"outbreak": []}))
    md = synthetic_grid.to_markdown(table, synthetic_grid.SI_HEADERS)
    lines = md.strip().splitlines()
    assert len(lines) == len(table) + 2
    assert all(line.count(" | ") == 4 for line in lines[2:])


def test_the_si_notes_name_every_level_and_the_held_out_seeds() -> None:
    config = tomllib.loads((ROOT / "config/paper/synthetic.toml").read_text())
    model = synthetic_grid.model_note(TEXT)
    for levels in synthetic_grid.parse_factors(TEXT).values():
        for level in levels:
            assert f"`{level}`" in model
    assert "e084b080" in model
    heldout = synthetic_grid.heldout_note(config)
    seeds = config["heldout_seeds"]
    assert f"{seeds[0]} to {seeds[-1]}" in heldout
    assert all(str(b) in heldout for b in config["blocks"])
    # One sentence per line.
    assert all(line.count(". ") == 0 for line in heldout.splitlines() if line)
