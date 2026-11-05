# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Release dates, cut-offs and the renewal model's report to publication lag.

A hub release is labelled by its `as_of` date. From the 2026-11-01
release on, `as_of` is the extraction date, one day after the last
reported case. Earlier releases were labelled one day early, by the last
reported case (hub issue #90). `release_id` maps a label, as published or
as the hub may rewrite it, to the label we key releases by, so results,
the submission ledger and rounds keep their keys.
`extraction_date` and `last_report` give a release's dates from either.
`cutoff_lags` gives the days from a release to its data cut-off by
weekday, for simulated releases.
`publication_lag_pmf` estimates, from pairs of delay files, the lag from a
case's report to its first publication, which the renewal models mix into
their onset expectations.
`release_pairs` tabulates, for each pair of consecutive delay files, which
rows of the later file were already in the earlier one, the data that
`renewallearntlag` fits its lag to.
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
    "extraction_date",
    "last_report",
    "publication_lag_pmf",
    "release_id",
    "release_pairs",
]

# Hub issue InsightNet-US/BDBV-Modeling-Hub#90: these labels are a day early.
ORIGINAL_AS_OF: frozenset[str] = frozenset(
    {"2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24"}
)
"""`as_of` labels of the releases published before 2026-11-01, which are
the last reported case's date, not the extraction date."""


def release_id(as_of: dt.date | str) -> str:
    """Return the label we key a hub release by.

    A label in `ORIGINAL_AS_OF` is kept. If the hub rewrites one to its
    extraction date, a day later, it maps back. Other labels are kept.

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


def extraction_date(as_of: dt.date | str) -> dt.date:
    """Return the date a hub release was extracted.

    Parameters
    ----------
    as_of : datetime.date or str
        The release's `as_of`, as published or as rewritten.

    Returns
    -------
    datetime.date
        A day after the label for a release in `ORIGINAL_AS_OF`, else the
        label.

    Examples
    --------
    >>> extraction_date("2026-10-24"), extraction_date("2026-11-01")
    (datetime.date(2026, 10, 25), datetime.date(2026, 11, 1))
    """
    label = release_id(as_of)
    late = label in ORIGINAL_AS_OF
    return as_date(label) + dt.timedelta(days=int(late))


def last_report(as_of: dt.date | str, kind: str = "real") -> dt.date:
    """Return the last report day in a release, its data cut-off.

    A simulated release's label is its last report day. A hub release's
    last report is the day before its extraction.

    Parameters
    ----------
    as_of : datetime.date or str
        The release's `as_of` label.
    kind : str, optional
        `"real"` for a hub release, else the label is the cut-off.

    Returns
    -------
    datetime.date
        The last report day.

    Examples
    --------
    >>> last_report("2026-10-24"), last_report("2026-11-01")
    (datetime.date(2026, 10, 24), datetime.date(2026, 10, 31))
    """
    if kind != "real":
        return as_date(as_of)
    return extraction_date(as_of) - dt.timedelta(days=1)


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


MAX_PAIR_AGE = 60
"""Oldest report age at the earlier release of a pair kept by
`release_pairs`, in days, the renewal models' maximum delay."""


def release_pairs(
    delays: pd.DataFrame,
    as_of: dt.date | str,
    kind: str = "real",
    max_age: int = MAX_PAIR_AGE,
) -> dict[str, np.ndarray] | None:
    r"""Rows of each delay file and whether the previous file had them.

    For each pair of consecutive delay files $k$ and $k + 1$ on or before
    `as_of`, every row cell (location, onset week, delay) of file $k + 1$
    is counted with how many of its rows file $k$ already had. A row with
    onset week ending $W$ and delay $d$ was reported on $W - u + d$ for
    an onset $u = 0..6$ days before $W$, so its report age at a release
    with last report day $T$ is $T - W + u - d$. Cells that no onset day
    could have reported by $T_k$, or that are older than `max_age` at
    $T_k$ for every onset day, are left out.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay rows of several releases, with `vintage`, `location`,
        `onset_week` and `delay`.
    as_of : datetime.date or str
        The forecast date. Later files are not used.
    kind : str, optional
        Release kind, for `last_report`.
    max_age : int, optional
        Oldest report age kept, in days.

    Returns
    -------
    dict or None
        `before` and `after`, report ages at $T_k$ and $T_{k + 1}$ by
        onset day, `(n, 7)`; `kept` and `total` rows of each cell,
        `(n,)`; and `pair`, the one-hot pair of each cell, `(n, K)`.
        None if no pair of files is available.
    """
    day = as_date(as_of).isoformat()
    vintages = sorted(str(v) for v in delays["vintage"].unique() if v <= day)
    key = ["location", "onset_week", "delay"]
    parts = []
    for a, b in itertools.pairwise(vintages):
        seen = delays.loc[delays["vintage"] == a].groupby(key).size()
        rows = (
            delays.loc[delays["vintage"] == b]
            .groupby(key)
            .size()
            .reset_index(name="total")
        )
        had = rows.set_index(key).index.map(seen).fillna(0).to_numpy()
        report = (
            pd.to_datetime(rows["onset_week"]).to_numpy()[:, None]
            - np.arange(7)[None, :] * np.timedelta64(1, "D")
            + rows["delay"].to_numpy()[:, None] * np.timedelta64(1, "D")
        )

        def age(release: str, report: np.ndarray = report) -> np.ndarray:
            cut = np.datetime64(last_report(release, kind), "D")
            return ((cut - report) / np.timedelta64(1, "D")).astype(int)

        before = age(a)
        keep = (before.max(axis=1) >= 0) & (before.min(axis=1) <= max_age)
        total = rows["total"].to_numpy()
        parts.append(
            {
                "before": before[keep],
                "after": age(b)[keep],
                "kept": np.minimum(had, total)[keep],
                "total": total[keep],
            }
        )
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["kept"] = out["kept"].astype(float)
    out["total"] = out["total"].astype(float)
    sizes = [len(p["total"]) for p in parts]
    out["pair"] = np.repeat(np.eye(len(parts)), sizes, axis=0)
    return out
