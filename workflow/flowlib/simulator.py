"""The simulator's pins and engine files, read from the grid config."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
ENGINE = ("packages/nowcastfoundation/r", "workflow/scripts/simulate.R")
"""The simulator's code: the engine package and the rule's wrapper."""


def read_grid(grid_file: Path | str) -> dict:
    """Return the grid config (`config/finetune/grids/<grid>.yaml`)."""
    return yaml.safe_load((ROOT / grid_file).read_text())


def epinowcast_sha(grid_file: Path | str) -> str:
    """Return the epinowcast commit the grid's simulator is pinned to.

    Raises `ValueError` if the grid sets no `epinowcast_sha`.
    """
    sha = str(read_grid(grid_file).get("epinowcast_sha") or "")
    if not sha:
        raise ValueError(f"{grid_file} sets no epinowcast_sha")
    return sha


def grid_name(grid_file: Path | str) -> str:
    """Return the grid's name (its `name`)."""
    return str(read_grid(grid_file)["name"])
