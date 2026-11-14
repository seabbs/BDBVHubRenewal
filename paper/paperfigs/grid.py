"""The synthetic simulation grid: its factors and their levels.

`tables/synthetic-grid.csv` lists each crossed factor of the grid by the
column its simulations record it in (as `settings-<seed>.csv`), its levels
separated by `; `, and the reason for them. The scores by factor read
each simulation's recorded cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

__all__ = ["GRID", "grid_levels", "read_grid"]

GRID: Final = Path(__file__).parents[1] / "tables" / "synthetic-grid.csv"
"""The grid's factor table."""


def read_grid(path: Path = GRID) -> pd.DataFrame:
    """Read the grid's `factor`, `levels` and `reason` columns."""
    return pd.read_csv(path)


def grid_levels(path: Path = GRID) -> dict[str, list[str]]:
    """Return each factor's column and its levels."""
    grid = read_grid(path)
    return {
        f: levels.split("; ")
        for f, levels in zip(grid["factor"], grid["levels"], strict=True)
    }
