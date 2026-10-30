# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Persistence baseline.

For each province, take the last week judged complete, $c$, and carry its
count $y_c$ forward on the log scale with random walk noise:

$$
\log(1 + \lambda_{c+h}) = \log(1 + y_c) + \sum_{j=1}^{h} \epsilon_j,
\quad \epsilon_j \sim \mathcal{N}(0, \sigma^2),
$$

$$
\tilde y_{c+h} \sim \text{Poisson}(\lambda_{c+h}), \quad h \ge 1.
$$

$\sigma$ is the standard deviation of the weekly changes in
$\log(1 + y)$ over the last `window` complete weeks, floored at
`min_sd`. A place with fewer than `low_count` cases a week over those
weeks takes $\sigma$ from the changes of the places above it, pooled.
Weeks up to $c$ are returned as published (a point mass).
The median is flat and the spread grows with the horizon.
Each sample is one joint trajectory across weeks.

A week is judged complete once `completeness` of its final total is
published, from how the hub's totals grow between releases a week apart
(`release_shares`). Late entries and re-imputation are in the totals, so
they are counted. Without two such releases, a week is complete if the
vintage date is at least `lag_days` after its midpoint, with `lag_days`
the `completeness` quantile of the pooled delay published with the
vintage, or `default_lag_days` if no delays were published.
Weeks from $c$ to the last forecast week are drawn by `random_walk`, with
$\sigma$ from `walk_spreads`.

For each later date $T$ in `task.report_by`, the counts reported by $T$
are predicted by thinning each sample of week $w$ with the share of a
week reported $T - w$ days after it ends
(`bdbvhubrenewal.data.completeness.reported_share`, from the delays
published with the vintage), and taking at least the count already
published for that week.
There are no such rows when the vintage has no delay data.

`PersistenceNow` (`persistencenow`) nowcasts the recent weeks with a
chain ladder on the hub's totals by release, then runs the same random
walk forward from each nowcast draw.

**Nowcast.** The totals of each onset week at each release form a
cumulative reporting triangle with onset week as reference and release
as report date. With few releases it is a band with no complete row, so
the chain ladder is in development-factor form: releases a week apart
give the growth $f_k$ of weeks $k$ weeks old (`release_growth`), own for
places with at least `own_per_week` cases a week and pooled otherwise.
A week $k$ weeks old gains a negative binomial count with mean
$Y (\prod_{j \ge k} f_j - 1)$ and size $\phi_k$ from nowcasts made at
earlier releases (`retrospective_release_errors`, `release_sizes`), as
baselinenowcast does.

**Forecast.** Each sample starts from its own nowcast of the latest week,
$\log\{1 + C_p(c)\}$, and follows the same random walk as `Persistence`
(`random_walk`), with $\sigma$ from the weekly changes of the place's
point nowcast (`walk_spreads`), pooled below `low_count` cases a week.
Later weeks are at least the known onsets.

Without two releases a week apart, it is `Persistence`.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from bdbvhubrenewal.data.completeness import reported_share
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.delays import EmpiricalDelayModel
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)

__all__ = [
    "Persistence",
    "PersistenceNow",
    "completeness_lag",
    "cutoff_week",
    "fit_skellam_spread",
    "last_complete_week",
    "nowcast_draws",
    "random_walk",
    "release_complete_weeks",
    "release_factors",
    "release_shares",
    "release_week",
    "retrospective_release_errors",
    "walk_spreads",
]


def completeness_lag(
    delays: pd.DataFrame | None,
    vintage: str,
    quantile: float = 0.9,
    default: int = 10,
) -> int:
    """Return the days after a week's midpoint at which it counts as complete.

    Parameters
    ----------
    delays : pandas.DataFrame or None
        Delay line data published with the vintage.
    vintage : str
        The vintage date.
    quantile : float
        Share of cases that must have been reported.
    default : int
        Lag used when no delays are available.

    Returns
    -------
    int
        The lag in days.
    """
    if delays is None or delays.empty:
        return default
    fit = EmpiricalDelayModel().fit(delays, vintage)
    return fit.quantile(quantile)


def last_complete_week(vintage: str, lag_days: int) -> dt.date:
    """Return the last week whose midpoint is at least `lag_days` old.

    Parameters
    ----------
    vintage : str
        The vintage date.
    lag_days : int
        Days from a week's midpoint (its Wednesday) to completeness.

    Returns
    -------
    datetime.date
        The Saturday ending that week.
    """
    # Midpoint of week ending s is s - 3; need s - 3 + lag <= vintage.
    latest = as_date(vintage) - dt.timedelta(days=lag_days - 3)
    return latest - dt.timedelta(days=(latest.weekday() - 5) % 7)


@dataclass
class Persistence:
    """Carry the last complete week forward with growing noise.

    Attributes
    ----------
    name : str
        Model name.
    window : int
        Complete weeks used to estimate the week-to-week spread.
    min_sd : float
        Floor on the log-scale weekly spread.
    completeness : float
        Share of a week's final total published for it to be complete,
        from the hub's releases (`release_complete_weeks`), or the delay
        quantile without two releases a week apart.
    default_lag_days : int
        Completeness lag with neither releases nor delay data.
    low_count : float
        Fewest cases a week, on average over the last `window` weeks, for a
        place to use its own weekly changes rather than the pooled ones.
    """

    name: str = "persistence"
    window: int = 8
    min_sd: float = 0.1
    completeness: float = 0.95
    default_lag_days: int = 10
    low_count: float = 10.0

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return persistence trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.
        """
        rng = task.rng()
        history = (
            task.target if task.target_history is None else task.target_history
        )
        weeks_old = release_complete_weeks(
            history, task.vintage, self.completeness
        )
        if weeks_old is None:
            lag = completeness_lag(
                task.delays,
                task.vintage,
                self.completeness,
                self.default_lag_days,
            )
            last = last_complete_week(task.vintage, lag)
        else:
            last = release_week(task.vintage) - dt.timedelta(weeks=weeks_old)
        weeks = list(task.target_end_dates)
        ahead = [
            (last + dt.timedelta(weeks=h)).isoformat()
            for h in range(1, _weeks_between(last, weeks[-1]) + 1)
        ]
        n = task.n_samples
        series = [
            task.target.loc[task.target["location"] == loc]
            .set_index("target_end_date")["total"]
            .sort_index()
            for loc in task.locations
        ]
        completes = [x.loc[x.index <= last.isoformat()] for x in series]
        sds = walk_spreads(
            [c.to_numpy(dtype=float) for c in completes],
            self.window,
            self.min_sd,
            self.low_count,
        )
        frames = []
        for loc, series_p, complete, sd in zip(
            task.locations, series, completes, sds, strict=True
        ):
            level = float(complete.iloc[-1]) if len(complete) else 0.0
            start = np.full(n, np.log1p(level))
            walk = dict(
                zip(
                    ahead,
                    random_walk(start, sd, len(ahead), rng).T,
                    strict=True,
                )
            )
            values = np.column_stack(
                [
                    walk[w]
                    if w in walk
                    else np.full(n, float(series_p.get(w, 0)))
                    for w in weeks
                ]
            )
            frames.append(_frame(loc, weeks, values))
        final = pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]
        if not task.report_by:
            return final
        return pd.concat(
            [final.assign(as_of=np.nan), *self._reported_by(task, final, rng)],
            ignore_index=True,
        )

    def _reported_by(
        self,
        task: ForecastTask,
        final: pd.DataFrame,
        rng: np.random.Generator,
    ) -> list[pd.DataFrame]:
        """Thin the final samples to the counts reported by each later date.

        The share of a week published by date $T$ is from the hub's
        releases (`release_shares`), or from the delays published with the
        vintage without two releases a week apart.
        """
        history = (
            task.target if task.target_history is None else task.target_history
        )
        shares = release_shares(history, task.vintage)
        if shares is None and (task.delays is None or task.delays.empty):
            return []
        seen = task.target.set_index(["location", "target_end_date"])["total"]
        floor = np.array(
            [
                seen.get((loc, week), 0)
                for loc, week in zip(
                    final["location"], final["target_end_date"], strict=True
                )
            ],
            dtype=float,
        )
        ends = pd.to_datetime(final["target_end_date"])
        out = []
        for date in task.report_by:
            days = (pd.Timestamp(date) - ends).dt.days.to_numpy(dtype=np.int64)
            due = days >= 0
            if shares is None:
                assert task.delays is not None
                vintage = str(task.delays["vintage"].max())
                share = reported_share(task.delays, vintage, days[due])
            else:
                end = pd.Timestamp(release_week(date))
                old = ((end - ends[due]).dt.days // 7).to_numpy()
                share = shares[np.clip(old, 0, shares.size - 1)]
                share = np.where(old >= shares.size, 1.0, share)
            counts = final["value"].to_numpy()[due].astype(np.int64)
            thinned = rng.binomial(counts, share)
            value = np.maximum(thinned, floor[due])
            out.append(final.loc[due].assign(value=value, as_of=date))
        return out


register_model(
    "persistence",
    Persistence,
    "Last complete week carried forward, with log-scale random walk noise.",
    baseline=True,
)


def random_walk(
    start: np.ndarray, sd: float, horizons: int, rng: np.random.Generator
) -> np.ndarray:
    r"""Draw the random walk forward on the log scale with Poisson counts.

    Each draw starts from its own $\log(1 + y)$ and adds independent
    $\mathcal{N}(0, \sigma^2)$ steps, one a week. Each week's count is
    Poisson with mean $\exp(\max(0, \text{level})) - 1$. Both baselines
    use this walk.

    Parameters
    ----------
    start : numpy.ndarray
        $\log(1 + y)$ for each draw.
    sd : float
        $\sigma$, the weekly step's standard deviation.
    horizons : int
        Weeks to draw.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Counts, one row per draw and one column per week ahead.
    """
    steps = rng.normal(0.0, sd, size=(start.size, horizons))
    level = start[:, None] + steps.cumsum(axis=1)
    return rng.poisson(np.expm1(np.maximum(level, 0.0))).astype(float)


def walk_spreads(
    series: list[np.ndarray], window: int, min_sd: float, low_count: float
) -> list[float]:
    r"""Return each place's walk spread $\sigma$, pooled at low counts.

    $\sigma$ is the standard deviation of the weekly changes in
    $\log(1 + y)$ over the last `window` weeks, floored at `min_sd`. A
    place with fewer than `low_count` cases a week over those weeks takes
    the spread of the changes of the places above it, pooled. With no
    place above it, each keeps its own.

    Parameters
    ----------
    series : list of numpy.ndarray
        Weekly counts of each place, oldest first.
    window : int
        Weeks used.
    min_sd : float
        Floor on $\sigma$.
    low_count : float
        Fewest cases a week for a place's own spread.

    Returns
    -------
    list of float
        One $\sigma$ per place.
    """

    def sd(diffs: np.ndarray) -> float:
        value = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0
        return max(value, min_sd)

    recent = [x[-window:] for x in series]
    diffs = [np.diff(np.log1p(r)) for r in recent]
    own = [r.size > 0 and r.mean() >= low_count for r in recent]
    if not any(own):
        return [sd(d) for d in diffs]
    pooled = sd(
        np.concatenate([d for d, o in zip(diffs, own, strict=True) if o])
    )
    return [sd(d) if o else pooled for d, o in zip(diffs, own, strict=True)]


def release_week(vintage: str) -> dt.date:
    """Return the Saturday ending the week a release covers."""
    day = as_date(vintage)
    return day + dt.timedelta(days=(5 - day.weekday()) % 7)


def cutoff_week(vintage: str) -> dt.date:
    """Return the Saturday ending the last full week on or before a date."""
    day = as_date(vintage)
    return day - dt.timedelta(days=(day.weekday() - 5) % 7)


def _weeks_between(start: dt.date, end: str) -> int:
    """Return whole weeks from `start` to the week ending `end`."""
    return max((as_date(end) - start).days // 7, 0)


def _age(release: str, week: str) -> int:
    """Return whole weeks from the end of `week` to `release`."""
    return (as_date(release) - as_date(week)).days // 7


def _release_fit(
    history: pd.DataFrame,
    vintage: str,
    max_weeks: int,
    location: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the development factors and which ages the data cover."""
    rows = history.loc[history["vintage"] <= vintage]
    rows = rows.loc[
        rows["location"] == location
        if location is not None
        else rows["location"] != NATIONAL
    ]
    totals = rows.groupby(["vintage", "target_end_date"])["total"].sum()
    releases = sorted(str(v) for v in rows["vintage"].unique())
    design, logs, weights = [], [], []
    for a, b in itertools.pairwise(releases):
        old, new = totals.loc[a], totals.loc[b]
        groups: dict[tuple[int, int], list[float]] = {}
        for week, count in old.items():
            ka, kb = _age(a, str(week)), _age(b, str(week))
            if ka < 0 or ka > max_weeks or count <= 0 or week not in new.index:
                continue
            pair = groups.setdefault((ka, kb), [0.0, 0.0])
            pair[0] += float(count)
            pair[1] += float(new[week])
        for (ka, kb), (before, after) in groups.items():
            if after <= 0:
                continue
            row = np.zeros(max_weeks + 1)
            row[ka : min(kb, max_weeks + 1)] = 1.0
            design.append(row)
            logs.append(np.log(after / before))
            weights.append(np.sqrt(before))
    if not design:
        return None
    x, w = np.asarray(design), np.asarray(weights)
    g = np.linalg.lstsq(x * w[:, None], np.asarray(logs) * w, rcond=None)[0]
    return np.exp(g), x.any(axis=0)


def release_factors(
    history: pd.DataFrame,
    vintage: str,
    max_weeks: int = 12,
    location: str | None = None,
) -> np.ndarray | None:
    r"""Return the chain ladder's development factors from the hub releases.

    The hub's totals by release form a cumulative reporting triangle with
    onset week $w$ as reference and release $r$ as report date. Week $w$
    is $k = \lfloor (r - w) / 7 \rfloor$ weeks old at $r$, and a week
    ending after $r$ is partial and left out. Over consecutive releases
    $a < b$ up to `vintage`, the weeks $k_a$ weeks old at $a$ and $k_b$ at
    $b$ grow by $R = \sum Y_b / \sum Y_a$, and
    $\log R = \sum_{j = k_a}^{k_b - 1} \log f_j$. The $\log f_j$ are
    fitted by least squares weighted by $\sum Y_a$; with releases a week
    apart this is $f_k = R$, the chain ladder. Ages no pair covers get
    $f_k = 1$, and weeks over `max_weeks` weeks old are final.

    Parameters
    ----------
    history : pandas.DataFrame
        Target rows of several releases.
    vintage : str
        The forecast vintage. Later releases are not used.
    max_weeks : int
        Oldest week, in weeks, that may still grow.
    location : str, optional
        One place. By default the provinces, pooled.

    Returns
    -------
    numpy.ndarray or None
        $f_k$ for $k = 0, \dots,$ `max_weeks`, or None with one release.
    """
    fit = _release_fit(history, vintage, max_weeks, location)
    return None if fit is None else fit[0]


def release_shares(
    history: pd.DataFrame, vintage: str, max_weeks: int = 12
) -> np.ndarray | None:
    r"""Return the share of a week's final total published, by weeks old.

    The hub's totals by release form a reporting triangle by onset week
    and release. Each pair of releases a week apart, up to `vintage`,
    gives, pooled over provinces, the growth $f_k$ of the totals of weeks
    $k$ weeks old at the first release to the next. The share published
    at $k$ weeks old is $1 / \prod_{j \ge k} f_j$, with $f_j = 1$ where no
    pair gives it.

    Parameters
    ----------
    history : pandas.DataFrame
        Target rows of several releases.
    vintage : str
        The forecast vintage. Later releases are not used.
    max_weeks : int
        Oldest week, in weeks, that may still grow.

    Returns
    -------
    numpy.ndarray or None
        Shares for 0 to `max_weeks` weeks old, or None without two
        releases a week apart.
    """
    rows = history.loc[
        (history["vintage"] <= vintage) & (history["location"] != NATIONAL)
    ]
    totals = rows.groupby(["vintage", "target_end_date"])["total"].sum()
    releases = sorted(str(v) for v in rows["vintage"].unique())
    num, den = np.zeros(max_weeks + 1), np.zeros(max_weeks + 1)
    for a, b in itertools.pairwise(releases):
        end = release_week(a)
        if (release_week(b) - end).days != 7:
            continue
        old, new = totals.loc[a], totals.loc[b]
        for week, count in old.items():
            k = (end - as_date(str(week))).days // 7
            if 0 <= k <= max_weeks and week in new.index and count > 0:
                num[k] += new[week]
                den[k] += count
    if not den.any():
        return None
    growth = np.where(den > 0, num / np.maximum(den, 1.0), 1.0)
    return np.asarray(1.0 / np.cumprod(growth[::-1])[::-1], dtype=np.float64)


def _to_final(factors: np.ndarray) -> np.ndarray:
    """Return each age's growth to final: the later factors multiplied."""
    return np.asarray(np.cumprod(factors[::-1])[::-1], dtype=np.float64)


def nowcast_draws(
    observed: np.ndarray,
    mean: np.ndarray,
    spread: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Draw final totals from published totals and Skellam increments.

    The change still to come in place $p$ and week $w$ is
    $A - B$, $A \sim \text{Poisson}(\mu^+ + \nu)$,
    $B \sim \text{Poisson}(\mu^- + \nu)$, with $\mu$ = `mean`, $\mu^+$ and
    $\mu^-$ its positive and negative parts and $\nu$ = `spread`, so its
    mean is $\mu$ and its variance $|\mu| + 2\nu$. Totals are floored at
    zero.

    Parameters
    ----------
    observed : numpy.ndarray
        Published totals $Y_p(w)$, one row per place, one column per week.
    mean : numpy.ndarray
        Expected change still to come, shaped as `observed`.
    spread : numpy.ndarray
        $\nu$, shaped as `observed`.
    n : int
        Number of draws.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Final totals, shaped `(n, places, weeks)`.
    """
    obs = np.asarray(observed, dtype=float)
    shape = (n, *obs.shape)
    up = rng.poisson(np.maximum(mean, 0.0) + spread, size=shape)
    down = rng.poisson(np.maximum(-mean, 0.0) + spread, size=shape)
    return np.asarray(np.maximum(obs + up - down, 0.0), dtype=np.float64)


def retrospective_release_errors(
    history: pd.DataFrame,
    vintage: str,
    locations: Sequence[str],
    max_weeks: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compare nowcasts from earlier releases with what changed since.

    From each release $r$ before `vintage` with factors (`release_factors`,
    pooled), a week $k$ weeks old at $r$ and $k'$ at `vintage` is expected
    to change by $Y_r(w) (\prod_{j=k}^{k'-1} f_j - 1)$ by `vintage`. The
    change seen is $Y_{\text{vintage}}(w) - Y_r(w)$, which can be
    negative. One row per release and place, one column per $k$, as
    baselinenowcast's retrospective errors.

    Parameters
    ----------
    history : pandas.DataFrame
        Target rows of several releases.
    vintage : str
        The forecast vintage.
    locations : sequence of str
        Places.
    max_weeks : int
        Oldest week, in weeks, that may still grow.

    Returns
    -------
    obs, mean, scale : numpy.ndarray
        Change seen, change expected and $Y_r(w)$, shaped
        `(rows, max_weeks + 1)`, missing where the week is not in both.
    """
    rows = history.loc[history["vintage"] <= vintage]
    now = rows.loc[rows["vintage"] == vintage].set_index(
        ["location", "target_end_date"]
    )["total"]
    out: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = (
        [],
        [],
        [],
    )
    for r in sorted(str(v) for v in rows["vintage"].unique() if v < vintage):
        f = release_factors(history, r, max_weeks)
        if f is None:
            continue
        then = rows.loc[rows["vintage"] == r].set_index(
            ["location", "target_end_date"]
        )["total"]
        start = cutoff_week(r)
        for loc in locations:
            o, m, s = (np.full(max_weeks + 1, np.nan) for _ in range(3))
            for k in range(max_weeks + 1):
                week = (start - dt.timedelta(weeks=k)).isoformat()
                if (loc, week) not in then.index or (
                    loc,
                    week,
                ) not in now.index:
                    continue
                y = float(then[(loc, week)])
                later = min(_age(vintage, week), max_weeks + 1)
                o[k] = float(now[(loc, week)]) - y
                m[k] = y * (np.prod(f[_age(r, week) : later]) - 1)
                s[k] = y
            for acc, value in zip(out, (o, m, s), strict=True):
                acc.append(value)
    shape = (len(out[0]), max_weeks + 1)
    obs, mean, scale = (
        np.asarray(a, dtype=np.float64).reshape(shape) for a in out
    )
    return obs, mean, scale


def fit_skellam_spread(
    obs: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    min_points: int = 3,
) -> np.ndarray:
    r"""Fit the Skellam spread rate for each weeks-old horizon.

    With $\nu = \rho Y$, the changes seen are Skellam as in
    `nowcast_draws`, and $\rho$ is fitted by maximum likelihood on 0 to
    10. Each horizon with at least `min_points` points with $Y > 0$ gets
    its own $\rho$; the rest share one fitted to every point. With no
    points, $\rho = 0$.

    Parameters
    ----------
    obs, mean, scale : numpy.ndarray
        From `retrospective_release_errors`.
    min_points : int
        Fewest points for a horizon's own rate.

    Returns
    -------
    numpy.ndarray
        One $\rho$ per horizon.
    """

    def fit(o: np.ndarray, m: np.ndarray, s: np.ndarray) -> float:
        keep = np.isfinite(o) & (s > 0)
        if not keep.any():
            return 0.0
        o, m, s = o[keep], m[keep], s[keep]

        def nll(rho: float) -> float:
            mu1 = np.maximum(m, 0.0) + rho * s + 1e-9
            mu2 = np.maximum(-m, 0.0) + rho * s + 1e-9
            return float(-stats.skellam.logpmf(np.round(o), mu1, mu2).sum())

        return float(
            minimize_scalar(nll, bounds=(0.0, 10.0), method="bounded").x
        )

    pooled = fit(obs.ravel(), mean.ravel(), scale.ravel())
    rates = np.full(obs.shape[1], pooled)
    for k in range(obs.shape[1]):
        points = np.isfinite(obs[:, k]) & (scale[:, k] > 0)
        if points.sum() >= min_points:
            rates[k] = fit(obs[:, k], mean[:, k], scale[:, k])
    return rates


@dataclass
class _Nowcast:
    """The point nowcast of a task and what the walk needs from it."""

    weeks: list[str]
    grid: list[str]
    cols: list[int]
    last: int
    seen: np.ndarray
    mean: np.ndarray
    spread: np.ndarray
    to_final: list[np.ndarray]
    point: np.ndarray


@dataclass
class PersistenceNow(Persistence):
    """The random walk run forward from a nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    max_weeks : int
        Oldest week, in weeks, whose total may still change.
    own_per_week : float
        Fewest cases a week, on average over the last `window` weeks, for
        a place to use its own development factors where it has them.

    See `Persistence` for `window`, `min_sd` and `low_count`, which set the
    walk's spread from the point nowcast.
    """

    name: str = "persistencenow"
    max_weeks: int = 12
    own_per_week: float = 50.0

    def _prepare(self, task: ForecastTask) -> _Nowcast | None:
        """Return the point nowcast and its inputs, or None if impossible."""
        history = (
            task.target if task.target_history is None else task.target_history
        )
        pooled = _release_fit(history, task.vintage, self.max_weeks)
        if pooled is None:
            return None
        latest = cutoff_week(task.vintage).isoformat()
        weeks = list(task.target_end_dates)
        ends = pd.date_range(
            min(task.target["target_end_date"].min(), weeks[0]),
            weeks[-1],
            freq="7D",
        )
        grid = [d.date().isoformat() for d in ends]
        cols = [grid.index(w) for w in weeks]
        last = grid.index(latest)
        seen = np.array(
            [
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .reindex(grid, fill_value=0)
                .to_numpy(dtype=float)
                for loc in task.locations
            ]
        )
        obs, mean_r, scale = retrospective_release_errors(
            history, task.vintage, task.locations, self.max_weeks
        )
        rates = fit_skellam_spread(obs, mean_r, scale)
        age = np.array([_age(task.vintage, w) for w in grid[: last + 1]])
        young = age <= self.max_weeks
        mean = np.zeros(seen.shape)
        spread = np.zeros(seen.shape)
        to_final = []
        for p, loc in enumerate(task.locations):
            f = pooled[0]
            if seen[p, last + 1 - self.window : last + 1].mean() >= (
                self.own_per_week
            ):
                own = _release_fit(history, task.vintage, self.max_weeks, loc)
                if own is not None:
                    f = np.where(own[1], own[0], f)
            g = np.r_[_to_final(f), 1.0]
            to_final.append(g)
            growth = g[np.minimum(age, self.max_weeks + 1)]
            mean[p, : last + 1] = seen[p, : last + 1] * (growth - 1)
            spread[p, : last + 1][young] = (
                rates[age[young]] * seen[p, : last + 1][young]
            )
        point = np.maximum(seen[:, : last + 1] + mean[:, : last + 1], 0.0)
        return _Nowcast(
            weeks, grid, cols, last, seen, mean, spread, to_final, point
        )

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return random walk trajectories from a nowcast for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`,
            and `as_of` if the task asks for `report_by`.
        """
        prep = self._prepare(task)
        if prep is None:
            return super().forecast(task)
        weeks, grid, cols, last = prep.weeks, prep.grid, prep.cols, prep.last
        seen = prep.seen
        rng = task.rng()
        n = task.n_samples
        totals = nowcast_draws(seen, prep.mean, prep.spread, n, rng)
        ahead = len(grid) - 1 - last
        # The walk's spread comes from the point nowcast; each draw starts
        # from its own nowcast of the latest week.
        sds = walk_spreads(
            list(prep.point), self.window, self.min_sd, self.low_count
        )
        for p, sd in enumerate(sds):
            walk = random_walk(np.log1p(totals[:, p, last]), sd, ahead, rng)
            totals[:, p, last + 1 :] = walk
        final, as_of = [], []
        cap = self.max_weeks + 1
        for p, loc in enumerate(task.locations):
            final.append(_frame(loc, weeks, totals[:, p, cols]))
            diff = totals[:, p] - seen[p]
            g = prep.to_final[p]
            for date in task.report_by:
                # The share of the change still to come that arrives by
                # `date`, from the same development factors.
                k = np.array(
                    [min(max(_age(task.vintage, w), 0), cap) for w in grid]
                )
                k2 = np.array([min(max(_age(date, w), 0), cap) for w in grid])
                total = g[k] - 1
                q = np.where(
                    np.abs(total) > 1e-12,
                    (g[k] / g[k2] - 1) / np.where(total == 0, 1.0, total),
                    1.0,
                )
                q = np.clip(q, 0, 1)
                up = rng.binomial(np.maximum(diff, 0).astype(np.int64), q)
                counts = np.maximum(seen[p] + up + np.minimum(diff, 0), 0.0)
                due = [c for c in cols if grid[c] <= date]
                as_of.append(
                    _frame(loc, [grid[c] for c in due], counts[:, due]).assign(
                        as_of=date
                    )
                )
        out = pd.concat(final, ignore_index=True)[list(SAMPLE_COLUMNS)]
        if not task.report_by:
            return out
        return pd.concat([out.assign(as_of=np.nan), *as_of], ignore_index=True)


def _frame(loc: str, weeks: list[str], values: np.ndarray) -> pd.DataFrame:
    """Return sample rows for one location, one column of `values` a week."""
    n = values.shape[0]
    return pd.DataFrame(
        {
            "location": loc,
            "target_end_date": np.tile(weeks, n),
            "sample": np.repeat(np.arange(1, n + 1), len(weeks)),
            "value": values.reshape(-1),
        }
    )


register_model(
    "persistencenow",
    PersistenceNow,
    "Persistence carried forward from a nowcast of the recent weeks.",
)


def release_complete_weeks(
    history: pd.DataFrame, vintage: str, share: float = 0.95
) -> int | None:
    """Return the weeks old at which a week counts as complete.

    The first age at which `release_shares` reaches `share`.

    Parameters
    ----------
    history : pandas.DataFrame
        Target rows of several releases.
    vintage : str
        The forecast vintage.
    share : float
        Share of the final total that must be published.

    Returns
    -------
    int or None
        Weeks old, or None without two releases a week apart.
    """
    shares = release_shares(history, vintage)
    return None if shares is None else int(np.argmax(shares >= share))
