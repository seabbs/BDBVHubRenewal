# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Release labels, cut-offs and the renewal model's report to publication lag.

A hub release is labelled by its `as_of` date. From the 2026-11-01
release on, `as_of` is the extraction date, one day after the last
reported case. Earlier releases were labelled a day earlier, by the last
reported case (hub issue #90). `release_id` maps a label, as published or
as the hub may rewrite it, to the label we key releases by, so results,
the submission ledger and rounds keep their keys.
A release's data cut-off is its last report day. `last_report_days`
derives it from the data, whatever the label, and every tidy table
carries it as a `cutoff` column, which `release_cutoff` reads.
`cutoff_lags` gives the days from a release to its cut-off by weekday,
for simulated releases.
`publication_lag_pmf` estimates, from pairs of delay files, the lag from a
case's report to its first publication, which the renewal models mix into
their onset expectations.
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date

__all__ = [
    "MAX_PUBLICATION_LAG",
    "ORIGINAL_AS_OF",
    "cutoff_lags",
    "last_report_days",
    "publication_lag_pmf",
    "release_cutoff",
    "release_id",
    "with_cutoffs",
]

# Hub issue InsightNet-US/BDBV-Modeling-Hub#90: the hub may rewrite these.
ORIGINAL_AS_OF: frozenset[str] = frozenset(
    {"2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24"}
)
"""`as_of` labels of the releases published before 2026-11-01, which the
hub may rewrite a day later. They key results, the submission ledger and
rounds, so a rewritten label maps back to them."""


def release_id(as_of: dt.date | str) -> str:
    """Return the label we key a hub release by.

    A label in `ORIGINAL_AS_OF` is kept. If the hub rewrites one a day
    later, it maps back. Other labels are kept.

    Parameters
    ----------
    as_of : datetime.date or str
        The release's `as_of`, as published or as rewritten.

    Returns
    -------
    str
        The release's label, ISO formatted.

    Examples
    --------
    >>> release_id("2026-10-25"), release_id("2026-11-01")
    ('2026-10-24', '2026-11-01')
    """
    day = as_date(as_of)
    before = (day - dt.timedelta(days=1)).isoformat()
    return before if before in ORIGINAL_AS_OF else day.isoformat()


def last_report_days(daily: pd.DataFrame, delays: pd.DataFrame) -> pd.Series:
    r"""Return each release's last report day, as its data prove it.

    A case with onset on day $t$ was reported on or after $t$. A delay row
    with onset week ending $W$ and delay $d$ was reported on or after
    $W - 6 + d$. A release's last report day is the latest of these
    over its daily onsets and its delay rows.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, with `vintage` and `onset_date`.
    delays : pandas.DataFrame
        Delay rows by release, with `vintage`, `onset_week` and `delay`.
        A release may have none.

    Returns
    -------
    pandas.Series
        The last report day (ISO) by release label, for every release in
        `daily`.
    """
    onset = pd.to_datetime(daily["onset_date"]).groupby(daily["vintage"]).max()
    report = (
        pd.to_datetime(delays["onset_week"])
        - pd.Timedelta(days=6)
        + pd.to_timedelta(delays["delay"], unit="D")
    )
    report = report.groupby(delays["vintage"]).max()
    last = pd.concat([onset, report.reindex(onset.index)], axis=1).max(axis=1)
    out: pd.Series = last.dt.strftime("%Y-%m-%d").rename("cutoff")
    return out


def with_cutoffs(table: pd.DataFrame, cutoffs: pd.Series) -> pd.DataFrame:
    """Add each row's release cut-off as a `cutoff` column.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows with a `vintage` column.
    cutoffs : pandas.Series
        Cut-off by release label (`last_report_days`).

    Returns
    -------
    pandas.DataFrame
        A copy of `table` with `cutoff`.
    """
    return table.assign(cutoff=table["vintage"].map(cutoffs))


def release_cutoff(rows: pd.DataFrame, release: str) -> dt.date:
    """Return a release's data cut-off, its last report day.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows with `vintage` and `cutoff` columns (`with_cutoffs`).
    release : str
        The release label.

    Returns
    -------
    datetime.date
        The cut-off.
    """
    cut = rows.loc[rows["vintage"] == release, "cutoff"]
    return as_date(str(cut.iloc[0]))


def cutoff_lags(daily: pd.DataFrame) -> dict[int, int]:
    """Days from release to data cut-off, by release weekday.

    A weekday with no real release takes the median over the releases.

    Parameters
    ----------
    daily : pandas.DataFrame
        Real daily rows with `vintage` and `cutoff`.

    Returns
    -------
    dict of int to int
        Lag in days for every weekday, 0 (Monday) to 6.
    """
    cut = daily.groupby("vintage")["cutoff"].first().sort_index()
    label = pd.to_datetime(pd.Series(cut.index, index=cut.index))
    lag = (label - pd.to_datetime(cut)).dt.days
    seen = lag.groupby(label.dt.day_of_week).last()
    fill = int(np.median(lag.to_numpy()))
    return {w: int(seen.get(w, fill)) for w in range(7)}


MAX_PUBLICATION_LAG = 180
"""Longest report to publication lag kept in `publication_lag_pmf`, in days.
Between the real files 8-9% of new rows were over 60 days late and none
over 180."""


def publication_lag_pmf(
    delays: pd.DataFrame,
    as_of: dt.date | str,
    max_lag: int = MAX_PUBLICATION_LAG,
) -> np.ndarray | None:
    r"""Empirical pmf of the lag from report to first publication.

    For each pair of consecutive delay vintages $V_1 < V_2$ on or before
    `as_of`, with cut-offs $C_1 < C_2$ (`release_cutoff`), the rows of
    $V_2$ beyond those of $V_1$ (by location, onset week and delay) were
    first published in $(C_1, C_2]$. A row with onset week $W$ and delay
    $d$ was reported on $W - u + d$ for an onset $u$ days before $W$, so
    with $u$ uniform on $0..6$ and publication uniform over the days of
    $(C_1, C_2]$ its lag is spread over $C_1 + j - (W - u + d)$, the
    feasible (non-negative) ones only, and lags past `max_lag` go to the
    last cell. Pairs whose later vintage is after `as_of` are not used, so
    a backtest does not see later files.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay rows of several vintages, with `vintage` and `cutoff`.
    as_of : datetime.date or str
        The forecast date.
    max_lag : int
        Last lag cell, in days.

    Returns
    -------
    numpy.ndarray or None
        Pmf on `0..max_lag`, or None if no pair is available.
    """
    day = as_date(as_of).isoformat()
    vintages = sorted(str(v) for v in delays["vintage"].unique() if v <= day)
    key = ["location", "onset_week", "delay"]
    mass = np.zeros(max_lag + 1)
    for a, b in itertools.pairwise(vintages):
        before = delays.loc[delays["vintage"] == a].groupby(key).size()
        after = delays.loc[delays["vintage"] == b].groupby(key).size()
        change = after.sub(before, fill_value=0).clip(lower=0)
        added = change[change > 0].reset_index(name="n")
        if added.empty:
            continue
        v1 = pd.Timestamp(release_cutoff(delays, a))
        gap = (pd.Timestamp(release_cutoff(delays, b)) - v1).days
        onset = pd.to_datetime(pd.Series(added["onset_week"]))
        week = (onset - v1).dt.days.to_numpy()
        count = np.asarray(added["n"], dtype=float)
        delay = np.asarray(added["delay"], dtype=int)
        # Lag for publication day j in 1..gap and onset u days before W.
        lag = (
            np.arange(1, gap + 1)[None, :, None]
            - week[:, None, None]
            + np.arange(7)[None, None, :]
            - delay[:, None, None]
        )
        keep = lag >= 0
        # Each row's count is spread over its feasible lags.
        per_row = count / np.maximum(keep.sum(axis=(1, 2)), 1)
        weight = np.broadcast_to(per_row[:, None, None], lag.shape)
        mass += np.bincount(
            np.minimum(lag[keep], max_lag),
            weights=weight[keep],
            minlength=max_lag + 1,
        )
    if mass.sum() == 0:
        return None
    return mass / mass.sum()
