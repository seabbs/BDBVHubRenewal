"""The grid's rounds, so no round is simulated twice.

A round's number sets its seeds and its release (`sims-r<N>`), so
simulating a round again would draw the same seeds and overwrite its
published batches. `grid_rounds` in workflow/config.yaml gives the grid's
training rounds, validation and test rounds and training sources;
`retired_rounds` gives the rounds of the grid's earlier versions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

GRID = "grid"
"""The owner of one of the grid's rounds (`grid_rounds`)."""

RETIRED = "retired"
"""The owner of a round an earlier version of the grid used
(`retired_rounds`)."""


def resolve(ft: dict[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a run's unset rounds, validation round and sources."""
    for key in ("rounds", "validation_round", "sources"):
        if ft.get(key) is None:
            ft[key] = table[key]
    return ft


def owner(
    round_: int,
    table: Mapping[str, Any],
    retired: Mapping[str, int] | None = None,
) -> str | None:
    """Return who a round belongs to, or None for an unlisted round.

    A round in `retired` (`first` to `last`) is `RETIRED`'s, and one of
    the grid's rounds is `GRID`'s.
    """
    if retired and retired["first"] <= round_ <= retired["last"]:
        return RETIRED
    own = (
        *table["rounds"],
        table["validation_round"],
        table.get("test_round"),
        table.get("topup_round"),
    )
    return GRID if round_ in own else None


def check_release(
    rounds: Iterable[int],
    table: Mapping[str, Any],
    retired: Mapping[str, int] | None = None,
) -> None:
    """Raise if the simulator would release a retired round."""
    for r in rounds:
        if owner(int(r), table, retired) == RETIRED:
            raise ValueError(
                f"round {r} is an earlier grid's, so releasing it would "
                f"overwrite sims-r{r}"
            )
