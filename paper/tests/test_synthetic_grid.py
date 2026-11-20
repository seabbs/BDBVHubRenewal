"""Tests of the synthetic scenario grid (`papertables.synthetic_grid`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from papertables import synthetic_grid

PAPER = Path(__file__).parents[1]
CONFIG = yaml.safe_load((PAPER / "workflow" / "config.yaml").read_text())
GRID_FILE = PAPER / CONFIG["synthetic"]["grid_file"]
GRID = synthetic_grid.read_grid(GRID_FILE)


def test_the_paper_reads_the_main_workflow_s_grid() -> None:
    main = yaml.safe_load(
        (PAPER.parent / "workflow" / "config.yaml").read_text()
    )
    assert (
        GRID_FILE.resolve()
        == (PAPER.parent / main["simulate"]["grid_file"]).resolve()
    )


def test_factor_shares_are_the_weights_normalised() -> None:
    shares = synthetic_grid.factor_shares(GRID)
    assert set(shares) == set(GRID["factors"])
    for levels in shares.values():
        assert abs(sum(levels.values()) - 1) < 1e-12


def test_every_factor_and_distribution_is_in_one_table() -> None:
    tables = synthetic_grid.layer_tables(GRID)
    shown = {s.strip("`") for t in tables.values() for s in t["setting"]}
    assert shown == set(GRID["factors"]) | set(GRID["distributions"])
    layers = {layer for t in tables.values() for layer in t["layer"]}
    assert layers == set(synthetic_grid.LAYER_NAMES.values())


def test_a_factor_with_no_layer_fails() -> None:
    grid = {**GRID, "factor_layers": dict(GRID["factor_layers"])}
    grid["factor_layers"].pop("window")
    with pytest.raises(KeyError, match="window"):
        synthetic_grid.layer_tables(grid)


def test_distributions_read_as_text() -> None:
    d = synthetic_grid.distribution
    assert d({"dist": "lognormal", "median": 7, "sdlog": 0.55}) == (
        "LogNormal(median 7, sdlog 0.55)"
    )
    assert d({"dist": "halfnormal", "scale": 0.25}) == "HalfNormal(0.25)"
    assert d({"dist": "beta", "a": 2, "b": 2, "lower": 0, "upper": 1}) == (
        "Beta(2, 2)"
    )
    assert (
        d({"dist": "beta", "a": 1, "b": 2, "lower": 0, "upper": 0.9})
        == "Beta(1, 2) on [0, 0.9]"
    )
    assert d({"dist": "gamma", "mean": 120, "sd": 60}) == (
        "Gamma(mean 120, sd 60)"
    )


def test_the_distribution_table_gives_every_reason() -> None:
    table = synthetic_grid.distribution_table(GRID)
    assert len(table) == len(GRID["distributions"])
    assert table["why"].str.len().gt(0).all()
    gt = table.set_index("setting").loc["`gt_mean`", "distribution"]
    assert gt.startswith("LogNormal(median 7")


def test_uneven_levels_show_their_shares() -> None:
    epi = synthetic_grid.layer_tables(GRID)["epidemic"].set_index("setting")
    # window: early 5, endemic 7, established 8 of 20.
    assert "`early` (25%)" in epi.loc["`window`", "values"]
    assert "%" not in epi.loc["`gt_family`", "values"]


def test_markdown_escapes_pipes() -> None:
    table = synthetic_grid.layer_tables(GRID)["reporting"]
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
    model = synthetic_grid.model_note(GRID, "config/finetune/grid.yaml")
    for levels in GRID["factors"].values():
        for level in levels:
            assert f"`{level}`" in model
    assert GRID["epinowcast_sha"] in model
    assert "config/finetune/grid.yaml" in model
    assert "finetune_simulate" not in model
    heldout = synthetic_grid.heldout_note(config)
    seeds = config["heldout_seeds"]
    assert f"{seeds[0]} to {seeds[-1]}" in heldout
    # One sentence per line.
    for note in (model, heldout):
        assert all(line.count(". ") == 0 for line in note.splitlines() if line)
