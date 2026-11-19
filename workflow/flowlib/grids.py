"""Each simulation grid's rounds, so no two grids share a round.

A round's number sets its seeds and its release (`sims-r<N>`), so a grid
that reused another's rounds would draw the same seeds and overwrite its
published batches. `grid_rounds` in workflow/config.yaml gives each grid
its training rounds, validation and test rounds and training sources.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

UNMARKED = "r5"
"""The grid of a release whose notes name none: r5's came before marks."""


def resolve(ft: dict[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a run's unset rounds, validation round and sources from its grid.

    Raises
    ------
    ValueError
        If the run's grid has no entry and a setting is unset.
    """
    keys = ("rounds", "validation_round", "sources")
    unset = [k for k in keys if ft.get(k) is None]
    if not unset:
        return ft
    if ft["grid"] not in table:
        raise ValueError(
            f"grid {ft['grid']} has no entry in grid_rounds; "
            f"set finetune {', '.join(unset)} or add it"
        )
    for key in unset:
        ft[key] = table[ft["grid"]][key]
    return ft


def owner(round_: int, table: Mapping[str, Any]) -> str | None:
    """Return the grid a round belongs to, or None for an unlisted round."""
    for grid, entry in table.items():
        own = (
            *entry["rounds"],
            entry["validation_round"],
            entry.get("test_round"),
            entry.get("topup_round"),
        )
        if round_ in own:
            return grid
    return None


def check_release(
    rounds: Iterable[int], grid: str, table: Mapping[str, Any]
) -> None:
    """Raise if `grid`'s simulator would release another grid's round."""
    for r in rounds:
        other = owner(int(r), table)
        if other is not None and other != grid:
            raise ValueError(
                f"round {r} is grid {other}'s; the simulator is grid "
                f"{grid}'s, so releasing it would overwrite sims-r{r}"
            )


def release_grid(notes: str) -> str:
    """Return the grid a release's notes name (`Grid r6.`), else r5."""
    found = re.search(r"\bGrid (r\d+)\.", notes or "")
    return found.group(1) if found else UNMARKED
