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
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import optimize, stats

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY, empirical_pmf
from bdbvhubrenewal.forecasters.base import NATIONAL

__all__ = [
    "MAX_PUBLICATION_LAG",
    "PUBLICATION_LAG_DAYS",
    "VINTAGE_KINDS",
    "PublicationLagFit",
    "compare_reconstruction",
    "cutoff_lags",
    "fit_publication_lag",
    "lag_cdf",
    "publication_lag_pmf",
    "reconstruct_daily_vintage",
    "reconstruct_daily_vintages",
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


def lag_cdf(lag_pmf: ArrayLike) -> np.ndarray:
    """Return the cdf of a report to publication lag pmf from 0 days."""
    return np.cumsum(np.asarray(lag_pmf, dtype=float))


def _by(cdf: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Look up a cdf at integer days, 0 before day 0 and the last after."""
    return np.where(days < 0, 0.0, cdf[np.clip(days, 0, cdf.size - 1)])


def _daily_expected(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    day: dt.date,
    lag_pmf: ArrayLike,
    cutoff_lag: int,
    max_delay: int,
) -> pd.DataFrame:
    """Return expected counts by province, week and onset day.

    Returns one row per province and week, with the weekly imputed share
    and the 7 daily columns `o0..o6` and `i0..i6`, onset day `W - u`.
    """
    last_week = epiweek_end(day).isoformat()
    prov = target.loc[
        (target["location"] != NATIONAL)
        & (target["target_end_date"] <= last_week),
        ["location", "location_name", "target_end_date", "imputed"],
    ].reset_index(drop=True)
    cdf = lag_cdf(lag_pmf)
    last = pd.Timestamp(day - dt.timedelta(days=cutoff_lag))
    u = _onset_offsets()

    def open_days(week: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Days from onset day `W - u` to the release, and if cut off."""
        gap = (pd.Timestamp(day) - pd.to_datetime(week)).dt.days.to_numpy()
        onset = (
            pd.to_datetime(week).to_numpy()[:, None]
            - pd.to_timedelta(u, unit="D").to_numpy()[None, :]
        )
        return gap[:, None] + u[None, :], onset <= last.to_datetime64()

    gap, open_ = open_days(delays["onset_week"])
    d = delays["delay"].to_numpy()[:, None]
    known = pd.DataFrame(
        _by(cdf, gap - d) * open_ / 7, columns=[f"o{k}" for k in u]
    ).assign(
        location=delays["location"].to_numpy(),
        target_end_date=delays["onset_week"].to_numpy(),
    )
    known = known.groupby(["location", "target_end_date"], as_index=False).sum()
    pmf = empirical_pmf(delays["delay"], max_delay)
    both = np.cumsum(np.convolve(pmf, np.diff(cdf, prepend=0.0)))
    gap, open_ = open_days(prov["target_end_date"])
    imputed = prov["imputed"].to_numpy()[:, None] / 7 * _by(both, gap) * open_
    out = prov.merge(
        known, on=["location", "target_end_date"], how="left"
    ).fillna(0.0)
    out = out.drop(columns="imputed")
    for k in u:
        out[f"i{k}"] = imputed[:, k]
    return out


def _allocate(expected: np.ndarray) -> np.ndarray:
    """Round rows of expected counts so each row sums to its rounded sum."""
    total = np.rint(expected.sum(axis=1)).astype("int64")
    scale = np.divide(
        total,
        expected.sum(axis=1),
        out=np.zeros(len(total)),
        where=expected.sum(axis=1) > 0,
    )
    scaled = expected * scale[:, None]
    base = np.floor(scaled).astype("int64")
    short = total - base.sum(axis=1)
    rank = np.argsort(
        np.argsort(-(scaled - base), axis=1, kind="stable"), axis=1
    )
    return np.asarray(base + (rank < short[:, None]), dtype=np.int64)


def _daily_counts(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    day: dt.date,
    lag_pmf: ArrayLike,
    cutoff_lag: int,
    max_delay: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    exp = _daily_expected(target, delays, day, lag_pmf, cutoff_lag, max_delay)
    obs = _allocate(exp[[f"o{k}" for k in _onset_offsets()]].to_numpy())
    imp = _allocate(exp[[f"i{k}" for k in _onset_offsets()]].to_numpy())
    return exp[["location", "location_name", "target_end_date"]], obs, imp


def reconstruct_vintage(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    as_of: dt.date | str,
    lag_pmf: ArrayLike,
    max_delay: int = DEFAULT_MAX_DELAY,
    cutoff_lag: int = 0,
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
    lag_pmf : sequence of float
        Report to publication lag pmf from 0 days
        (`fit_publication_lag`).
    max_delay : int
        Longest delay used for the imputed-onset cdf.
    cutoff_lag : int
        Days from the release to the data cut-off. Onsets after the
        cut-off are not yet in the release.

    Returns
    -------
    pandas.DataFrame
        Target rows like `bdbvhubrenewal.data.tidy_target` with
        `kind = "reconstructed"`, for weeks up to the epi week holding
        `as_of`. The country row is the sum of provinces.
    """
    day = as_date(as_of)
    keys, obs, imp = _daily_counts(
        target, delays, day, lag_pmf, cutoff_lag, max_delay
    )
    prov = keys.assign(observed=obs.sum(axis=1), imputed=imp.sum(axis=1))
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


def reconstruct_daily_vintage(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    as_of: dt.date | str,
    lag_pmf: ArrayLike,
    max_delay: int = DEFAULT_MAX_DELAY,
    cutoff_lag: int = 0,
) -> pd.DataFrame:
    """Rebuild the daily release a vintage at an earlier date would show.

    The counts are those of `reconstruct_vintage` split by onset day, so
    each week's daily counts add up to its weekly total.

    Parameters
    ----------
    target, delays, as_of, lag_pmf, max_delay, cutoff_lag
        As for `reconstruct_vintage`.

    Returns
    -------
    pandas.DataFrame
        Columns `kind` (`"reconstructed"`), `vintage`, `location`,
        `onset_date`, `observed`, `imputed` and `cutoff` (the last onset
        day the release can hold, ISO format), as the hub's daily file
        with the country and provinces and only days with a case.
    """
    day = as_date(as_of)
    keys, obs, imp = _daily_counts(
        target, delays, day, lag_pmf, cutoff_lag, max_delay
    )
    week = pd.to_datetime(keys["target_end_date"]).to_numpy()
    onset = (
        week[:, None] - pd.to_timedelta(_onset_offsets(), unit="D").to_numpy()
    )
    prov = pd.DataFrame(
        {
            "location": np.repeat(keys["location"].to_numpy(), 7),
            "onset_date": onset.ravel(),
            "observed": obs.ravel(),
            "imputed": imp.ravel(),
        }
    )
    nat = prov.groupby("onset_date", as_index=False)[
        ["observed", "imputed"]
    ].sum()
    out = pd.concat([prov, nat.assign(location=NATIONAL)], ignore_index=True)
    out = out.loc[(out["observed"] + out["imputed"]) > 0].copy()
    out["onset_date"] = out["onset_date"].dt.strftime("%Y-%m-%d")
    out["kind"] = "reconstructed"
    out["vintage"] = day.isoformat()
    out["cutoff"] = (day - dt.timedelta(days=cutoff_lag)).isoformat()
    cols = [
        "kind",
        "vintage",
        "location",
        "onset_date",
        "observed",
        "imputed",
        "cutoff",
    ]
    return (
        out[cols].sort_values(["location", "onset_date"]).reset_index(drop=True)
    )


def reconstruct_vintages(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    dates: list[str] | list[dt.date],
    source_vintage: str | None = None,
    cutoff_lags: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Reconstruct several vintages from one real vintage.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table (all vintages).
    delays : pandas.DataFrame
        The tidy delay table (all vintages). The publication lag is
        fitted to the pairs of vintages up to the source vintage.
    dates : list of str or datetime.date
        The `as_of` dates to reconstruct.
    source_vintage : str, optional
        The real vintage to reconstruct from. Defaults to the latest one
        with delay data.
    cutoff_lags : mapping of int to int, optional
        Days from release to data cut-off by release weekday
        (`cutoff_lags`). None for no cut-off.

    Returns
    -------
    pandas.DataFrame
        Reconstructed target rows for every date, stacked.
    """
    _, tgt, dly, lag = _source(target, delays, source_vintage)
    frames = [
        reconstruct_vintage(
            tgt, dly, d, lag, cutoff_lag=_cutoff(d, cutoff_lags)
        )
        for d in dates
    ]
    return pd.concat(frames, ignore_index=True)


def reconstruct_daily_vintages(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    dates: list[str] | list[dt.date],
    source_vintage: str | None = None,
    cutoff_lags: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Reconstruct several daily releases from one real vintage.

    Parameters
    ----------
    target, delays, dates, source_vintage, cutoff_lags
        As for `reconstruct_vintages`.

    Returns
    -------
    pandas.DataFrame
        Daily rows (`reconstruct_daily_vintage`) for every date, stacked.
    """
    _, tgt, dly, lag = _source(target, delays, source_vintage)
    frames = [
        reconstruct_daily_vintage(
            tgt, dly, d, lag, cutoff_lag=_cutoff(d, cutoff_lags)
        )
        for d in dates
    ]
    return pd.concat(frames, ignore_index=True)


def _source(
    target: pd.DataFrame, delays: pd.DataFrame, source_vintage: str | None
) -> tuple[str, pd.DataFrame, pd.DataFrame, np.ndarray]:
    src = source_vintage or max(delays["vintage"])
    tgt = target.loc[(target["vintage"] == src) & (target["kind"] == "real")]
    dly = delays.loc[delays["vintage"] == src]
    fit = fit_publication_lag(target, delays, src)
    return src, tgt, dly, fit.pmf


def _cutoff(day: dt.date | str, lags: Mapping[int, int] | None) -> int:
    return 0 if lags is None else lags[as_date(day).weekday()]


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


class PublicationLagFit(NamedTuple):
    """A lognormal report to publication lag.

    Attributes
    ----------
    meanlog, sdlog : float
        Parameters of the lognormal for the lag in days.
    pmf : numpy.ndarray
        Pmf on `0..max_lag` of the lag rounded down to whole days, with
        the tail past `max_lag` in the last cell.
    """

    meanlog: float
    sdlog: float
    pmf: np.ndarray


def _lognormal_pmf(meanlog: float, sdlog: float, max_lag: int) -> np.ndarray:
    edges = np.arange(max_lag + 1, dtype=float)
    cdf = np.r_[0.0, stats.lognorm.cdf(edges[1:], sdlog, scale=np.exp(meanlog))]
    return np.diff(np.r_[cdf, 1.0])


def fit_publication_lag(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    source_vintage: str | None = None,
    max_lag: int = MAX_PUBLICATION_LAG,
) -> PublicationLagFit:
    """Fit a lognormal to the lag from report to first publication.

    Each earlier real release is reconstructed from the source vintage
    (`reconstruct_vintage`). The lognormal minimises the Pearson
    chi-squared distance between the reconstructed and real national
    weekly totals, so it is the lag by which the source vintage's reports
    reach earlier releases as the real releases added them.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table (all vintages).
    delays : pandas.DataFrame
        The tidy delay table (all vintages).
    source_vintage : str, optional
        The real vintage to reconstruct from. Defaults to the latest one
        with delay data.
    max_lag : int
        Last lag cell, in days.

    Returns
    -------
    PublicationLagFit
        The fitted lognormal and its pmf.

    Raises
    ------
    ValueError
        If no real release is earlier than the source vintage.
    """
    src = source_vintage or str(max(delays["vintage"]))
    tgt = target.loc[(target["vintage"] == src) & (target["kind"] == "real")]
    dly = delays.loc[delays["vintage"] == src]
    real = target.loc[
        (target["kind"] == "real")
        & (target["vintage"] < src)
        & (target["location"] == NATIONAL)
    ]
    dates = sorted(str(v) for v in real["vintage"].unique())
    if not dates:
        raise ValueError("Need a real release before the source vintage.")
    truth = {
        v: real.loc[real["vintage"] == v].set_index("target_end_date")["total"]
        for v in dates
    }

    def loss(par: np.ndarray) -> float:
        pmf = _lognormal_pmf(par[0], float(np.exp(par[1])), max_lag)
        out = 0.0
        for v in dates:
            full = reconstruct_vintage(tgt, dly, v, pmf)
            rec = full.loc[full["location"] == NATIONAL].set_index(
                "target_end_date"
            )["total"]
            both = truth[v].index.intersection(rec.index)
            want = truth[v][both].to_numpy(dtype=float)
            out += float(
                ((rec[both].to_numpy() - want) ** 2 / np.maximum(want, 1)).sum()
            )
        return out

    # Counts are rounded, so start from a simplex wide enough to see changes.
    fit = optimize.minimize(
        loss,
        np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"initial_simplex": [[1, 0], [2, 0], [1, 0.5]]},
    )
    meanlog, sdlog = float(fit.x[0]), float(np.exp(fit.x[1]))
    pmf = _lognormal_pmf(meanlog, sdlog, max_lag)
    return PublicationLagFit(meanlog, sdlog, pmf)
