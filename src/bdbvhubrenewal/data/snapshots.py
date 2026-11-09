# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Daily release snapshots as arrays on the onset clock.

The hub's `data/daily.csv` holds one table of daily onsets for each
release.
A release $r$ has an as-of date $A_r$ and a cut-off $c_r$, its last
report day (`bdbvhubrenewal.data.releases.release_cutoff`).
Its cell for onset day $t$ is observed when $t \le c_r$.
An observed day with no row counts 0.
The age of day $t$ at release $r$ is $c_r - t$, the onset clock.
Counts are observed plus imputed onsets, which sum to the hub's weekly
target.
Hub weeks run Sunday to Saturday and are labelled by the Saturday.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.releases import release_cutoff

__all__ = [
    "NATIONAL",
    "SnapshotArrays",
    "parent_code",
    "snapshot_arrays",
    "week_sum",
    "weekly_sums",
]

NATIONAL = "CD"
"""Location code of the whole country."""

_PROVINCE_LENGTH = 5


def parent_code(code: str) -> str | None:
    """Return the parent of a location code in the hierarchy.

    Zones (`CD-XX-NN`) sit in provinces (`CD-XX`), which sit in the
    country (`CD`).

    Parameters
    ----------
    code : str
        A location code.

    Returns
    -------
    str or None
        The parent, or None for the country.
    """
    if code == NATIONAL:
        return None
    if len(code) <= _PROVINCE_LENGTH:
        return NATIONAL
    return code[:_PROVINCE_LENGTH]


@dataclass(frozen=True)
class SnapshotArrays:
    """Daily release snapshots as arrays.

    Attributes
    ----------
    locations : tuple of str
        The $L$ nodes, parents before children.
    releases : tuple of str
        The $R$ release dates, oldest first.
    days : pandas.DatetimeIndex
        The $T$ onset days, from the first onset to the Saturday of the
        week holding the latest release's cut-off.
    counts : numpy.ndarray
        Observed plus imputed onsets by default, or the columns asked
        for (`snapshot_arrays`), `[L, R, T]`. Cells a release does
        not observe are 0.
    observed : numpy.ndarray
        Whether a release observes an onset day (`t <= cut`), `[R, T]`.
    cut : numpy.ndarray
        Day index of each release's cut-off, `[R]`.
    week : numpy.ndarray
        Hub week index of each day, `[T]`.
    week_ends : tuple of str
        The $W$ hub weeks, labelled by their Saturday.
    parent : numpy.ndarray
        Index of each node's parent, or -1, `[L]`.
    """

    locations: tuple[str, ...]
    releases: tuple[str, ...]
    days: pd.DatetimeIndex
    counts: np.ndarray
    observed: np.ndarray
    cut: np.ndarray
    week: np.ndarray
    week_ends: tuple[str, ...]
    parent: np.ndarray

    @property
    def n_weeks(self) -> int:
        """Number of hub weeks."""
        return len(self.week_ends)

    def cut_week(self, r: int) -> int:
        """Return the hub week index holding release `r`'s cut-off.

        Parameters
        ----------
        r : int
            Release index.

        Returns
        -------
        int
            Index into the hub weeks.
        """
        return int(self.week[self.cut[r]])


def _saturday(day: dt.date) -> dt.date:
    return day + dt.timedelta(days=(5 - day.weekday()) % 7)


def snapshot_arrays(
    daily: pd.DataFrame,
    vintage: str,
    locations: tuple[str, ...],
    columns: tuple[str, ...] = ("observed", "imputed"),
) -> SnapshotArrays:
    """Build the arrays of the releases on or before a vintage.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, with columns `vintage`, `location`,
        `onset_date`, `observed` and `imputed`.
    vintage : str
        The forecast vintage. Later releases are not used.
    locations : tuple of str
        Locations to include. Each location's ancestors are added, so
        the country is always the first node.
    columns : tuple of str
        The columns summed into the counts. The default, observed plus
        imputed onsets, is the hub's target. `("observed",)` gives the
        onsets known at each release.

    Returns
    -------
    SnapshotArrays
        The arrays.

    Raises
    ------
    ValueError
        If no release is on or before the vintage.
    """
    rows = daily.loc[daily["vintage"].astype(str) <= vintage]
    if rows.empty:
        raise ValueError(f"No daily releases on or before {vintage}.")
    releases = tuple(sorted(str(v) for v in rows["vintage"].unique()))
    nodes: dict[str, None] = {NATIONAL: None}
    for loc in locations:
        chain: list[str] = []
        code: str | None = loc
        while code is not None and code not in nodes:
            chain.append(code)
            code = parent_code(code)
        for c in reversed(chain):
            nodes[c] = None
    nodes_t = tuple(sorted(nodes, key=lambda c: (len(c), c)))
    first = as_date(str(rows["onset_date"].min()))
    cutoffs = [release_cutoff(rows, r) for r in releases]
    days = pd.date_range(first, _saturday(max(cutoffs)), freq="D")
    index = {d: i for i, d in enumerate(days.date)}
    cut = np.array([index[c] for c in cutoffs], dtype=np.int64)
    observed = np.arange(len(days))[None, :] <= cut[:, None]
    counts = np.zeros((len(nodes_t), len(releases), len(days)))
    take = rows.loc[rows["location"].isin(nodes_t)]
    li = take["location"].map({c: i for i, c in enumerate(nodes_t)})
    ri = take["vintage"].astype(str).map({r: i for i, r in enumerate(releases)})
    ti = pd.to_datetime(take["onset_date"]).dt.date.map(index)
    values = take[list(columns)].sum(axis=1).to_numpy(dtype=float)
    np.add.at(
        counts,
        (
            li.to_numpy(dtype=int),
            ri.to_numpy(dtype=int),
            ti.to_numpy(dtype=int),
        ),
        values,
    )
    counts = counts * observed[None]
    ends = [_saturday(d) for d in days.date]
    week_ends = tuple(sorted({e.isoformat() for e in ends}))
    where = {w: i for i, w in enumerate(week_ends)}
    week = np.array([where[e.isoformat()] for e in ends], dtype=np.int64)
    pos = {c: i for i, c in enumerate(nodes_t)}
    parent = np.array(
        [
            pos[p] if (p := parent_code(c)) is not None and p in pos else -1
            for c in nodes_t
        ],
        dtype=np.int64,
    )
    return SnapshotArrays(
        nodes_t, releases, days, counts, observed, cut, week,
        week_ends, parent,
    )  # fmt: skip


def week_sum(snap: SnapshotArrays, x: np.ndarray) -> np.ndarray:
    """Sum day values to hub weeks.

    Parameters
    ----------
    snap : SnapshotArrays
        The snapshots, whose `week` maps each day to its hub week.
    x : numpy.ndarray
        Day values, shape `[L, T]`.

    Returns
    -------
    numpy.ndarray
        Week sums, shape `[L, W]`.
    """
    onehot = np.zeros((snap.week.size, snap.n_weeks))
    onehot[np.arange(snap.week.size), snap.week] = 1.0
    return np.asarray(x @ onehot)


def weekly_sums(
    snap: SnapshotArrays, r: int, upto: int | None = None
) -> np.ndarray:
    """Return release `r`'s counts summed to hub weeks.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index.
    upto : int, optional
        Last day index to sum. Defaults to the release's cut-off.

    Returns
    -------
    numpy.ndarray
        Weekly counts, `[L, W]`.
    """
    last = snap.cut[r] if upto is None else upto
    days = np.arange(snap.week.size) <= last
    return week_sum(snap, snap.counts[:, r, :] * days)
