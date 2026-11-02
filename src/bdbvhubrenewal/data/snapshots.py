# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Daily release snapshots as arrays on the onset clock.

The hub's `data/daily.csv` holds one table of daily onsets for each
release.
A release $r$ has an as-of date $A_r$ and a cut-off $c_r$, the latest
onset day in the release over all locations.
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

__all__ = [
    "NATIONAL",
    "SnapshotArrays",
    "parent_code",
    "release_cutoff",
    "snapshot_arrays",
    "week_sum",
    "weekly_sums",
]

NATIONAL = "CD"
"""Location code of the whole country."""

_PROVINCE_LENGTH = 5


def release_cutoff(daily: pd.DataFrame, release: str) -> dt.date:
    """Return a release's data cut-off.

    The `cutoff` column where the data state it, else the release's latest
    onset day over all its locations.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release.
    release : str
        The release date.

    Returns
    -------
    datetime.date
        The cut-off.
    """
    rows = daily.loc[daily["vintage"] == release]
    if "cutoff" in rows.columns and rows["cutoff"].notna().any():
        return as_date(str(rows["cutoff"].dropna().iloc[0]))
    return as_date(str(rows["onset_date"].max()))


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
        week holding the latest release.
    counts : numpy.ndarray
        Observed plus imputed onsets, `[L, R, T]`. Cells a release does
        not observe are 0.
    observed : numpy.ndarray
        Whether a release observes an onset day (`t <= cut`), `[R, T]`.
    cut : numpy.ndarray
        Day index of each release's cut-off, `[R]`.
    asof : numpy.ndarray
        Day index of each release's as-of date, `[R]`.
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
    asof: np.ndarray
    week: np.ndarray
    week_ends: tuple[str, ...]
    parent: np.ndarray

    @property
    def n_weeks(self) -> int:
        """Number of hub weeks."""
        return len(self.week_ends)

    def asof_week(self, r: int) -> int:
        """Return the hub week index holding release `r`'s as-of date."""
        return int(self.week[self.asof[r]])

    def cut_week(self, r: int) -> int:
        """Return the hub week index holding release `r`'s cut-off."""
        return int(self.week[self.cut[r]])


def _saturday(day: dt.date) -> dt.date:
    return day + dt.timedelta(days=(5 - day.weekday()) % 7)


def snapshot_arrays(
    daily: pd.DataFrame, vintage: str, locations: tuple[str, ...]
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
    last = _saturday(as_date(releases[-1]))
    days = pd.date_range(first, last, freq="D")
    index = {d: i for i, d in enumerate(days.date)}
    cut = np.array(
        [index[release_cutoff(rows, r)] for r in releases], dtype=np.int64
    )
    asof = np.array([index[as_date(r)] for r in releases], dtype=np.int64)
    observed = np.arange(len(days))[None, :] <= cut[:, None]
    counts = np.zeros((len(nodes_t), len(releases), len(days)))
    take = rows.loc[rows["location"].isin(nodes_t)]
    li = take["location"].map({c: i for i, c in enumerate(nodes_t)})
    ri = take["vintage"].astype(str).map({r: i for i, r in enumerate(releases)})
    ti = pd.to_datetime(take["onset_date"]).dt.date.map(index)
    values = (take["observed"] + take["imputed"]).to_numpy(dtype=float)
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
        nodes_t, releases, days, counts, observed, cut, asof, week,
        week_ends, parent,
    )  # fmt: skip


def week_sum(snap: SnapshotArrays, x: np.ndarray) -> np.ndarray:
    """Sum `[L, T]` day values to hub weeks, `[L, W]`."""
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
