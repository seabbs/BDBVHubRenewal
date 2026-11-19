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
    assert factors["observation"] == ["negbin1d", "poisson"]
    assert all(len(levels) >= 1 for levels in factors.values())


def test_repeated_levels_give_their_shares() -> None:
    shares = synthetic_grid.parse_shares(TEXT)
    assert shares["observation"] == {"negbin1d": 0.8, "poisson": 0.2}
    assert shares["reporting"] == {"daily": 0.5, "weekdays": 0.5}


def test_values_are_read_from_the_simulator() -> None:
    values = synthetic_grid.parse_values(TEXT)
    assert values["offset"] == "sample(0:6, 1)"
    # Multi-line expressions are kept whole.
    assert values["gt_mean"].startswith("if (renewal)")
    assert "max_delay" in values
    # Draws set after the list, one line each, are read too.
    assert values["shock_mult"] == "runif(n_shocks, 2, 5)"


def test_size_bounds_are_read_from_the_simulator() -> None:
    bounds = synthetic_grid.parse_bounds(TEXT)
    assert list(bounds) == ["min_size", "max_size", "max_independent"]
    assert bounds["min_size"] == "10"
    # The upper bound scales with the strata.
    assert "design$n_places" in bounds["max_size"]


def test_every_setting_has_a_layer_and_a_label() -> None:
    names = [
        *synthetic_grid.parse_factors(TEXT),
        *synthetic_grid.DERIVED,
        *synthetic_grid.parse_values(TEXT),
        *synthetic_grid.parse_bounds(TEXT),
    ]
    assert not [n for n in names if n not in synthetic_grid.FACTOR_OF]
    assert not [
        f
        for f in synthetic_grid.parse_factors(TEXT)
        if f not in synthetic_grid.LAYER
    ]
    assert not [n for n in names if n not in synthetic_grid.LABELS]


def test_the_layer_tables_split_the_factors_with_their_draws() -> None:
    tables = synthetic_grid.layer_tables(TEXT)
    assert list(tables) == ["epidemic", "reporting"]
    factors = synthetic_grid.parse_factors(TEXT)
    rows = pd.concat(tables.values())
    assert sorted(rows["setting"]) == sorted(factors)
    for table in tables.values():
        assert list(table.columns) == ["setting", "factor", "levels", "draws"]
    epi = tables["epidemic"].set_index("setting")
    rep = tables["reporting"].set_index("setting")
    assert "mixing" in epi.index
    assert "shocks" in rep.index
    assert "80%" in rep.loc["observation", "levels"]
    assert "log-uniform(0.01, 0.5)" in epi.loc["mixing", "draws"]
    assert "U(2, 5)" in rep.loc["shocks", "draws"]


def test_markdown_escapes_pipes() -> None:
    table = synthetic_grid.layer_tables(TEXT)["reporting"]
    md = synthetic_grid.to_markdown(table, synthetic_grid.HEADERS)
    lines = md.strip().splitlines()
    assert len(lines) == len(table) + 2
    # Four unescaped pipes bound the three cells of each row.
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 4 for line in lines[2:])


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


def test_the_model_note_gives_one_draw_and_no_nb2() -> None:
    model = synthetic_grid.model_note(TEXT)
    assert "with one observation draw." in model
    assert "NB2 or NB1" not in model


# The simulator in this tree, which may be ahead of the pinned one.
LOCAL = (PAPER.parent / "scripts" / "finetune_simulate.R").read_text()


def test_sampling_distributions_are_read_with_their_reasons() -> None:
    dists = synthetic_grid.parse_distributions(LOCAL)
    if not dists:
        pytest.skip("this simulator has no sampling distributions")
    assert dists["sqrt_phi"]["dist"] == "halfnormal"
    table = synthetic_grid.distribution_table(LOCAL).set_index("setting")
    assert table.loc["sqrt_phi", "distribution"] == "HalfNormal(0.25)"
    size = table.loc["size_per_stratum", "distribution"]
    assert size.startswith("LogNormal(median 3,000, sdlog 2.1) on [10, ")
    assert table["why"].str.len().gt(0).all()


def test_draws_from_distributions_appear_in_the_grid_tables() -> None:
    if not synthetic_grid.parse_distributions(LOCAL):
        pytest.skip("this simulator has no sampling distributions")
    tables = synthetic_grid.layer_tables(LOCAL)
    rep = tables["reporting"].set_index("setting")
    epi = tables["epidemic"].set_index("setting")
    assert "HalfNormal(0.25)" in rep.loc["observation", "draws"]
    assert "50%" in rep.loc["shocks", "levels"]
    # A summary's <name> is filled from that setting's distribution.
    assert "±LogNormal(median 0.2" in rep.loc["delay_change", "draws"]
    assert "<" not in " ".join(epi["draws"])
