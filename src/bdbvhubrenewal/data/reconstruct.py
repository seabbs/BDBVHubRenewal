# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Reconstructed pseudo-vintages from the latest real data.

The hub has published few vintages.
To test models at more forecast dates, `reconstruct_vintage` rebuilds what
a vintage at an earlier date $A$ would roughly have shown, using the
latest target and delay data.

**Known onsets.**
Each row of the latest delay data is a case with onset week $W$ and delay
$d$ days.
Its onset day is unknown within the week, so it counts as reported by $A$
with probability
$$\frac{1}{7}\sum_{u=0}^{6}\mathbb{1}\{W - u + d \le A\}.$$

**Imputed onsets.**
Report dates of cases without an onset are not published.
The hub gave each one an onset by subtracting a delay drawn from the
pooled empirical delays.
So a case imputed to week $W$ was reported roughly a pooled delay after
its imputed onset, and it counts as reported by $A$ with probability
$$\frac{1}{7}\sum_{u=0}^{6} F(A - W + u),$$
with $F$ the pooled empirical delay cdf of the latest delay data.

Cases reach a vintage some days after their report date.
So $A$ above is the vintage date minus `publication_lag_days`.
The default of 3 days makes the last four weeks of the reconstructed
national totals match the real 2026-08-18, 2026-10-09 and 2026-10-17
vintages with the smallest net difference (2 days gives the smallest
absolute difference).
Counts are expected values, rounded to whole cases per province and week.

**Limits.**
Pseudo-vintages see the latest onsets and imputations.
They miss real revisions: onsets filled in after the first report,
changes in the hub's imputation, and late data corrections.
So they are cleaner than real vintages and forecasts scored on them look
somewhat better.
`compare_reconstruction` measures the gap against real vintages.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY, empirical_pmf
from bdbvhubrenewal.forecasters.base import NATIONAL

__all__ = [
    "MAX_PUBLICATION_LAG",
    "PUBLICATION_LAG_DAYS",
    "VINTAGE_KINDS",
    "compare_reconstruction",
    "publication_lag_pmf",
    "reconstruct_delays",
    "reconstruct_vintage",
    "reconstruct_vintages",
]

PUBLICATION_LAG_DAYS = 3
"""Days from a case's report to its first vintage, used in reconstruction."""

VINTAGE_KINDS = ("real", "reconstructed", "simulated")
"""The kinds of vintage. Scores and plots always show the kind."""


def _onset_offsets() -> np.ndarray:
    return np.arange(7)


def _known_reported(delays: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
    """Count expected known-onset cases reported by `as_of`."""
    week = pd.to_datetime(delays["onset_week"])
    gap = (pd.Timestamp(as_of) - week).dt.days.to_numpy()
    d = delays["delay"].to_numpy()
    # Onset day is W - u for u in 0..6; reported by A if W - u + d <= A.
    frac = ((gap[:, None] + _onset_offsets()[None, :] - d[:, None]) >= 0).mean(
        axis=1
    )
    out = delays[["location", "onset_week"]].assign(observed=frac)
    summed = out.groupby(["location", "onset_week"])["observed"].sum()
    return summed.reset_index()


def _imputed_fraction(
    weeks: pd.Series, as_of: dt.date, cdf: np.ndarray
) -> np.ndarray:
    gap = (pd.Timestamp(as_of) - pd.to_datetime(weeks)).dt.days.to_numpy()
    lags = gap[:, None] + _onset_offsets()[None, :]
    idx = np.clip(lags, 0, cdf.size - 1)
    frac = np.where(lags < 0, 0.0, cdf[idx])
    return np.asarray(frac.mean(axis=1), dtype=np.float64)


def reconstruct_vintage(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    as_of: dt.date | str,
    max_delay: int = DEFAULT_MAX_DELAY,
    publication_lag_days: int = PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Rebuild an approximate vintage at an earlier date from later data.

    Parameters
    ----------
    target : pandas.DataFrame
        The target rows of one later real vintage (provinces and country).
    delays : pandas.DataFrame
        The delay line data of the same vintage.
    as_of : datetime.date or str
        The date to reconstruct.
    max_delay : int
        Longest delay used for the imputed-onset cdf.
    publication_lag_days : int
        Days from report to appearing in a vintage.

    Returns
    -------
    pandas.DataFrame
        Target rows like `bdbvhubrenewal.data.tidy_target` with
        `kind = "reconstructed"`, for weeks up to the epi week holding
        `as_of`. The country row is the sum of provinces.
    """
    day = as_date(as_of)
    last_week = epiweek_end(day).isoformat()
    cut = day - dt.timedelta(days=publication_lag_days)
    prov = target.loc[
        (target["location"] != NATIONAL)
        & (target["target_end_date"] <= last_week)
    ].copy()
    known = _known_reported(delays, cut).rename(
        columns={"onset_week": "target_end_date"}
    )
    prov = prov.drop(columns=["observed"]).merge(
        known, on=["location", "target_end_date"], how="left"
    )
    prov["observed"] = prov["observed"].fillna(0.0)
    cdf = np.cumsum(empirical_pmf(delays["delay"], max_delay))
    prov["imputed"] = prov["imputed"] * _imputed_fraction(
        prov["target_end_date"], cut, cdf
    )
    prov["observed"] = np.rint(prov["observed"]).astype("int64")
    prov["imputed"] = np.rint(prov["imputed"]).astype("int64")
    prov["total"] = prov["observed"] + prov["imputed"]
    nat = prov.groupby("target_end_date", as_index=False)[
        ["observed", "imputed", "total"]
    ].sum()
    nat_name = target.loc[target["location"] == NATIONAL, "location_name"]
    nat = nat.assign(
        location=NATIONAL,
        location_name=nat_name.iloc[0] if len(nat_name) else "DRC",
    )
    out = pd.concat([prov, nat], ignore_index=True)
    out["kind"] = "reconstructed"
    out["vintage"] = day.isoformat()
    cols = [
        "kind",
        "vintage",
        "location",
        "location_name",
        "target_end_date",
        "observed",
        "imputed",
        "total",
    ]
    return (
        out[cols]
        .sort_values(["location", "target_end_date"])
        .reset_index(drop=True)
    )


def reconstruct_vintages(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    dates: list[str] | list[dt.date],
    source_vintage: str | None = None,
) -> pd.DataFrame:
    """Reconstruct several vintages from one real vintage.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table (all vintages).
    delays : pandas.DataFrame
        The tidy delay table (all vintages).
    dates : list of str or datetime.date
        The `as_of` dates to reconstruct.
    source_vintage : str, optional
        The real vintage to reconstruct from. Defaults to the latest one
        with delay data.

    Returns
    -------
    pandas.DataFrame
        Reconstructed target rows for every date, stacked.
    """
    src = source_vintage or max(delays["vintage"])
    tgt = target.loc[(target["vintage"] == src) & (target["kind"] == "real")]
    dly = delays.loc[delays["vintage"] == src]
    frames = [reconstruct_vintage(tgt, dly, d) for d in dates]
    return pd.concat(frames, ignore_index=True)


def reconstruct_delays(
    delays: pd.DataFrame,
    as_of: dt.date | str,
    publication_lag_days: int = PUBLICATION_LAG_DAYS,
    lag_pmf: Sequence[float] | None = None,
) -> pd.DataFrame:
    r"""Keep the delay rows a vintage at an earlier date would have had.

    A case is kept if its report, taking onset at the middle of its onset
    week, falls by the vintage date minus `publication_lag_days`, and its
    onset week has started by then (a few delays are negative).
    With `lag_pmf`, each case is instead kept with the chance it was
    published by the vintage date $A$,
    $\frac{1}{7}\sum_u F_\ell(A - (W - u + d))$, so late entries in the
    later file are thinned out. The draws are seeded from the date.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one later vintage.
    as_of : datetime.date or str
        The date to reconstruct.
    publication_lag_days : int
        Days from report to appearing in a vintage.
    lag_pmf : sequence of float, optional
        Report to publication lag pmf from 0 days.

    Returns
    -------
    pandas.DataFrame
        The kept rows, with `vintage` set to `as_of`.
    """
    day = as_date(as_of)
    if lag_pmf is not None:
        return _thin_by_lag(delays, day, np.asarray(lag_pmf, dtype=float))
    cut = pd.Timestamp(day - dt.timedelta(days=publication_lag_days))
    report = pd.to_datetime(delays["onset_week"]) + pd.to_timedelta(
        delays["delay"] - 3, unit="D"
    )
    started = pd.to_datetime(delays["onset_week"]) - pd.Timedelta(days=6)
    out = delays.loc[(report <= cut) & (started <= cut)].copy()
    out["vintage"] = day.isoformat()
    return out.reset_index(drop=True)


def _thin_by_lag(
    delays: pd.DataFrame, day: dt.date, lag_pmf: np.ndarray
) -> pd.DataFrame:
    """Keep each row with the chance it was published by `day`."""
    cdf = np.cumsum(lag_pmf)
    gap = (pd.Timestamp(day) - pd.to_datetime(delays["onset_week"])).dt.days
    # Days from report to `day` for an onset u days before the week end.
    since = (
        gap.to_numpy()[:, None]
        + _onset_offsets()[None, :]
        - delays["delay"].to_numpy()[:, None]
    )
    seen = np.where(since < 0, 0.0, cdf[np.clip(since, 0, cdf.size - 1)])
    rng = np.random.default_rng(day.toordinal())
    keep = rng.uniform(size=len(delays)) < seen.mean(axis=1)
    out = delays.loc[keep].copy()
    out["vintage"] = day.isoformat()
    return out.reset_index(drop=True)


def compare_reconstruction(
    target: pd.DataFrame, reconstructed: pd.DataFrame
) -> pd.DataFrame:
    """Compare reconstructed vintages with real vintages of the same date.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table with real vintages.
    reconstructed : pandas.DataFrame
        Output of `reconstruct_vintages` for dates that are also real
        vintages.

    Returns
    -------
    pandas.DataFrame
        One row per vintage, location and week, with real and
        reconstructed `observed`, `imputed` and `total`, and
        `total_difference` (reconstructed minus real).
    """
    real = target.loc[target["kind"] == "real"]
    cols = ["vintage", "location", "target_end_date"]
    vals = ["observed", "imputed", "total"]
    out = real[cols + vals].merge(
        reconstructed[cols + vals],
        on=cols,
        how="inner",
        suffixes=("_real", "_reconstructed"),
    )
    out["total_difference"] = out["total_reconstructed"] - out["total_real"]
    return out


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
