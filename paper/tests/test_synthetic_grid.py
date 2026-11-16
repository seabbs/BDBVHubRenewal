"""Tests of the synthetic scenario grid (`papertables.synthetic_grid`)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from papertables import synthetic_grid

PAPER = Path(__file__).parents[1]
CONFIG = yaml.safe_load((PAPER / "workflow" / "config.yaml").read_text())


def _simulator() -> str:
    """The simulator at the commit the workflow pins, if git has it."""
    entry = CONFIG["evidence"][CONFIG["synthetic"]["simulator"]]
    done = subprocess.run(
        ["git", "show", f"{entry['ref']}:{entry['path']}"],
        cwd=PAPER,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        pytest.skip(
            "the pinned simulator is not fetched", allow_module_level=True
        )
    return done.stdout


TEXT = _simulator()


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
    # Five unescaped pipes bound the four cells of each row.
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 5 for line in lines[2:])


def test_the_si_notes_name_every_level_and_the_held_out_seeds() -> None:
    config = {
        **CONFIG["synthetic"],
        "heldout_seeds": CONFIG["heldout"]["synthetic_seeds"],
        "development_seeds": CONFIG["development"]["synthetic_seeds"],
    }
    model = synthetic_grid.model_note(TEXT)
    for levels in synthetic_grid.parse_factors(TEXT).values():
        for level in levels:
            assert f"`{level}`" in model
    assert "e084b080" in model
    heldout = synthetic_grid.heldout_note(config)
    seeds = config["heldout_seeds"]
    assert f"{seeds[0]} to {seeds[-1]}" in heldout
    assert all(str(b) in heldout for b in config.get("blocks") or [])
    # One sentence per line.
    assert all(line.count(". ") == 0 for line in heldout.splitlines() if line)
