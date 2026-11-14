"""The synthetic simulation grid: its factors and their levels.

`tables/synthetic-grid.csv` lists each crossed factor of the grid, its
levels separated by `; `, and the reason for them. Each simulation records
its cell, one level per factor, which the scores by factor read.
`factor_column` names a factor's column in a table of cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

__all__ = ["GRID", "factor_column", "grid_levels", "read_grid"]

GRID: Final = Path(__file__).parents[1] / "tables" / "synthetic-grid.csv"
"""The grid's factor table."""


def read_grid(path: Path = GRID) -> pd.DataFrame:
    """Read the grid's `factor`, `levels` and `reason` columns."""
    return pd.read_csv(path)


def factor_column(factor: str) -> str:
    """Return a factor's column name: lower case, spaces as underscores."""
    return factor.lower().replace(" ", "_")


def grid_levels(path: Path = GRID) -> dict[str, list[str]]:
    """Return each factor's column name and its levels."""
    grid = read_grid(path)
    return {
        factor_column(f): levels.split("; ")
        for f, levels in zip(grid["factor"], grid["levels"], strict=True)
    }
