# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Release cut-offs and the renewal model's report to publication lag.

`cutoff_lags` gives the days from a release to its data cut-off by
weekday, for simulated releases.
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

__all__ = ["MAX_PUBLICATION_LAG", "cutoff_lags", "publication_lag_pmf"]


def cutoff_lags(daily: pd.DataFrame) -> dict[int, int]:
    """Days from release to data cut-off, by release weekday.

    The cut-off of a real release is its latest onset day. A weekday with
    no real release takes the median over the releases.

    Parameters
    ----------
    daily : pandas.DataFrame
        Real daily rows with `vintage` and `onset_date`
        (`bdbvhubrenewal.data.tidy_daily`).

    Returns
    -------
    dict of int to int
        Lag in days for every weekday, 0 (Monday) to 6.
    """
    last = pd.to_datetime(daily.groupby("vintage")["onset_date"].max())
    lag = (
        pd.to_datetime(pd.Series(last.index, index=last.index)) - last
    ).dt.days
    lag = lag.sort_index()
    weekday = pd.to_datetime(
        pd.Series(lag.index, index=lag.index)
    ).dt.day_of_week
    seen = lag.groupby(weekday).last()
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
    `as_of`, the rows of $V_2$ beyond those of $V_1$ (by location, onset
    week and delay) were first published in $(V_1, V_2]$. A row with
    onset week $W$ and delay $d$ was reported on $W - u + d$ for an onset
    $u$ days before $W$, so with $u$ uniform on $0..6$ and publication
    uniform over the days of $(V_1, V_2]$ its lag is spread over
    $V_1 + j - (W - u + d)$, the feasible (non-negative) ones only, and
    lags past `max_lag` go to the last cell. Pairs whose later vintage is after
    `as_of` are not used, so a backtest does not see later files.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay rows of several vintages, with a `vintage` column.
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
        v1, v2 = pd.Timestamp(a), pd.Timestamp(b)
        gap = (v2 - v1).days
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
