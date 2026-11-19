"""Top up the training rounds' missing grid cells, so they are whole blocks.

A training round is one grid block of 840 cells (scripts/finetune_simulate.R
`block`), but some of its simulations are missing from training: the sim
QC excluded them, or the simulator failed all its tries. Sam's decision on
#408: the final model trains on ten whole blocks, so each missing cell is
simulated again with a fresh seed (same cell, so the same factor levels;
new continuous draws). The top-ups form their own round (the grid's
`topup_round` in workflow/config.yaml `grid_rounds`), simulated in tries:
each try's batches are checked by the sim QC, and the cells still missing
go to the next try, up to `tries`.

A cell is a grid index, `block * cell_block + cell_index`, as each
simulation's settings record it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

BLOCK = 840
"""Simulations a grid block holds (the simulator's `block`)."""

PER_TRY = 20
"""Batch seeds a try may use: try t takes seeds `first + PER_TRY * t` on."""


def kept_cells(round_dir: Path) -> set[int]:
    """Return the grid cells of a round's simulations the sim QC kept.

    Read from its batches' settings (`cell_block`, `cell_index`) less its
    `qc/excluded.csv`; a round with no QC yet keeps every simulation.
    """
    paths = sorted((round_dir / "sims").glob("settings-*.csv"))
    if not paths:
        return set()
    settings = pd.concat(
        pd.read_csv(p, usecols=["outbreak", "cell_block", "cell_index"])
        for p in paths
    )
    qc = round_dir / "qc" / "excluded.csv"
    excluded = (
        set(pd.read_csv(qc)["simulation"])
        if qc.exists() and qc.stat().st_size
        else set()
    )
    kept = settings.loc[~settings["outbreak"].isin(excluded)]
    return set(
        (kept["cell_block"] * BLOCK + kept["cell_index"]).astype(int).tolist()
    )


def missing_cells(
    rounds_dir: Path, rounds: Iterable[int], topup_round: int
) -> list[int]:
    """Return the training rounds' cells that no kept simulation fills.

    A training round `r` is block `r`; a kept simulation of the top-up
    round fills the cell it records.
    """
    rounds = list(rounds)
    want = {r * BLOCK + i for r in rounds for i in range(BLOCK)}
    have = set().union(
        *(kept_cells(rounds_dir / f"r{r}") for r in [*rounds, topup_round])
    )
    return sorted(want - have)


def plan(
    cells: Sequence[int], first_seed: int, attempt: int, batch: int
) -> list[tuple[int, list[int]]]:
    """Return a try's batches: (seed, cells), at most `batch` cells each.

    Raises
    ------
    ValueError
        If the cells need more than `PER_TRY` batches, or the try is past
        the seeds a top-up round has.
    """
    chunks = [list(cells[i : i + batch]) for i in range(0, len(cells), batch)]
    if len(chunks) > PER_TRY:
        raise ValueError(
            f"{len(cells)} missing cells need {len(chunks)} batches, more "
            f"than the {PER_TRY} a try has"
        )
    if PER_TRY * (attempt + 1) > 100:
        raise ValueError(f"try {attempt} is past the top-up round's seeds")
    start = first_seed + PER_TRY * attempt
    return [(start + k, chunk) for k, chunk in enumerate(chunks)]


def write_plan(
    batches: Sequence[tuple[int, Sequence[int]]], path: Path
) -> None:
    """Write a try's plan: one line a batch, its seed then its cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{s} {' '.join(map(str, c))}\n" for s, c in batches)
    )


def read_plan(path: Path) -> list[tuple[int, list[int]]]:
    """Return a try's plan (`write_plan`)."""
    out = []
    for line in Path(path).read_text().splitlines():
        seed, *cells = map(int, line.split())
        out.append((seed, cells))
    return out


def cells_of(plans: Iterable[Path], seed: int) -> list[int]:
    """Return the cells a top-up batch simulates, from the tries' plans."""
    for path in plans:
        for s, cells in read_plan(path) if Path(path).exists() else []:
            if s == seed:
                return cells
    return []
