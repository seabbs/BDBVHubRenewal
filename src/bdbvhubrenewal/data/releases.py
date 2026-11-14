# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Release labels, cut-offs and the renewal model's report to publication lag.

A hub release is labelled by its `as_of` date. From the 2026-11-01
release on, `as_of` is the extraction date, one day after the last
reported case. The hub states that earlier releases were labelled by the
last reported case. `release_id` maps a label, as published or
as the hub may rewrite it, to the label we key releases by, so results,
the submission ledger and rounds keep their keys.
A release's data cut-off is its latest onset day over all places.
The last day or two before extraction are thin, as new reports take time
to reach the line list. `latest_onset_days` derives the cut-off from the
data, whatever the label, and every tidy table carries it as a `cutoff`
column, which `release_cutoff` reads.
`cutoff_lags` gives the days from a release to its cut-off by weekday,
for simulated releases.
`release_pairs` tabulates, for each pair of consecutive delay files, which
rows of the later file were already in the earlier one, the data the
renewal models learn their report to publication lag from.
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date

__all__ = [
    "MAX_PAIR_AGE",
    "ORIGINAL_AS_OF",
    "cutoff_lags",
    "latest_onset_days",
    "release_cutoff",
    "release_id",
    "release_pairs",
]

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


def latest_onset_days(daily: pd.DataFrame) -> pd.Series:
    """Return each release's data cut-off, its latest onset day.

    The cut-off is the latest onset day over all places in the release's
    daily onsets, one date per release.

    Parameters
    ----------
    daily : pandas.DataFrame
        Daily onsets by release, with `vintage` and `onset_date`.

    Returns
    -------
    pandas.Series
        The cut-off (ISO) by release label, for every release in `daily`.
    """
    onset = pd.to_datetime(daily["onset_date"]).groupby(daily["vintage"]).max()
    out: pd.Series = onset.dt.strftime("%Y-%m-%d").rename("cutoff")
    return out


def release_cutoff(rows: pd.DataFrame, release: str) -> dt.date:
    """Return a release's data cut-off, its latest onset day.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows with `vintage` and `cutoff` columns (`data.load_target`).
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


MAX_PAIR_AGE = 60
"""Oldest report age at the earlier release of a pair kept by
`release_pairs`, in days, the renewal models' maximum delay."""


def release_pairs(
    delays: pd.DataFrame,
    as_of: dt.date | str,
    max_age: int = MAX_PAIR_AGE,
) -> dict[str, np.ndarray] | None:
    r"""Rows of each delay file and whether the previous file had them.

    For each pair of consecutive delay files $k$ and $k + 1$ on or before
    `as_of`, every row cell (location, onset week, delay) of file $k + 1$
    is counted with how many of its rows file $k$ already had. A row with
    onset week ending $W$ and delay $d$ was reported on $W - u + d$ for
    an onset $u = 0..6$ days before $W$, so its report age at a release
    with cut-off $T$ is $T - W + u - d$. Cells that no onset day
    could have reported by $T_k$, or that are older than `max_age` at
    $T_k$ for every onset day, are left out.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay rows of several releases, with `vintage`, `cutoff`,
        `location`, `onset_week` and `delay`.
    as_of : datetime.date or str
        The forecast date. Later files are not used.
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
        had = seen.reindex(
            pd.MultiIndex.from_frame(rows[key]), fill_value=0
        ).to_numpy()
        # Report day of each row for an onset 0 to 6 days before the end
        # of its onset week, as days since the epoch.
        report = (
            pd.to_datetime(rows["onset_week"]).to_numpy().astype("datetime64[D]")
            .astype(int)[:, None]
            - np.arange(7)[None, :]
            + rows["delay"].to_numpy()[:, None]
        )  # fmt: skip

        def age(release: str, report: np.ndarray = report) -> np.ndarray:
            cut = np.datetime64(release_cutoff(delays, release), "D")
            return np.asarray(cut.astype(int) - report, dtype=int)

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
