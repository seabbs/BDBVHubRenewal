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

`PersistenceNow` (`persistencenow`) nowcasts the recent weeks with the
baselinenowcast method (`bdbvhubrenewal.data.triangle`), then runs the
hub baseline forward from each nowcast draw.

**Nowcast.** A chain ladder on each place's reporting triangle from the
hub's delay file, over the last `n_history` onset weeks, gives a delay
pmf, and past nowcasts of the triangle give a negative binomial size
$\phi_h$ for each horizon $h$, as in baselinenowcast. Places with at
least `own_per_week` delays a week use their own; the rest borrow those
of the triangle pooled over places. Each onset week of the triangle, the
latest included, has a share $s_p(w)$ of its point nowcast already in
the delay file (`bdbvhubrenewal.data.triangle.week_shares`), and its
published total
$Y_p(w)$, known plus imputed, gains $U_p(w) \sim \text{NegBin}$ with
mean $Y_p(w) \{1 - s_p(w)\} / s_p(w)$ (`nowcast_means`) and size
$\phi_h$. The hub re-imputes missing onsets at each vintage, so each
place also adds a revision $E_p(w) = A - B$, with $A, B \sim
\text{Poisson}(\kappa I_p(w) / 2)$, $I_p(w)$ its imputed count and
$\kappa$ from past revisions (`revision_variance`), and

$$
C_p(w) = \max\{Y_p(w) + U_p(w) + E_p(w),\ K_p(w)\},
$$

with $K_p(w)$ the known onsets.

**Forecast.** Each sample starts from its own nowcast of the latest week,
$\log\{1 + C_p(c)\}$, and follows the same random walk as `Persistence`
(`random_walk`), with $\sigma$ from the weekly changes of the place's
point nowcast (`walk_spreads`), pooled below `low_count` cases a week.
Later weeks are at least the known onsets.

If the delay data have no complete onset week or fewer than `min_delays`
cases in the last `n_history` weeks, or give fewer than `min_retro` past
nowcasts, it is `Persistence`.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.completeness import reported_share
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.delays import EmpiricalDelayModel
from bdbvhubrenewal.data.triangle import (
    delay_pmf,
    fit_nb_size,
    reporting_triangle,
    retrospective_errors,
    week_shares,
)
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
    "last_complete_week",
    "nowcast_draws",
    "nowcast_means",
    "random_walk",
    "release_complete_weeks",
    "release_shares",
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


def _weeks_between(start: dt.date, end: str) -> int:
    """Return whole weeks from `start` to the week ending `end`."""
    return max((as_date(end) - start).days // 7, 0)


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


def nowcast_draws(
    observed: np.ndarray,
    added: np.ndarray,
    size: np.ndarray,
    n: int,
    rng: np.random.Generator,
    known: np.ndarray | None = None,
    revision: float = 0.0,
) -> np.ndarray:
    r"""Draw final totals from published totals and point nowcasts.

    The count still to come in place $p$ and week $w$ is negative
    binomial with mean `added` and size $\phi_p(w)$, or Poisson where
    $\phi_p$ is infinite, independently. With `known` and `revision`
    $\kappa > 0$, each place also adds the difference of two Poisson
    draws with mean $\kappa (Y_p - K_p) / 2$, and is kept at least $K_p$.

    Parameters
    ----------
    observed : numpy.ndarray
        Published totals $Y_p(w)$, one row per place, one column per week.
    added : numpy.ndarray
        Point nowcast of the cases still to come, shaped as `observed`.
    size : numpy.ndarray
        Negative binomial size $\phi_p(w)$, shaped as `observed`.
    n : int
        Number of draws.
    rng : numpy.random.Generator
        Random generator.
    known : numpy.ndarray, optional
        Known onsets $K_p(w)$, shaped as `observed`.
    revision : float
        $\kappa$, the revision variance per imputed case.

    Returns
    -------
    numpy.ndarray
        Final totals, shaped `(n, places, weeks)`.
    """
    obs = np.asarray(observed, dtype=float)
    out = np.broadcast_to(obs, (n, *obs.shape)).copy()
    for p, j in zip(*np.nonzero(added > 0), strict=True):
        mean = added[p, j]
        if np.isfinite(size[p, j]):
            phi = size[p, j]
            out[:, p, j] += rng.negative_binomial(phi, phi / (phi + mean), n)
        else:
            out[:, p, j] += rng.poisson(mean, size=n)
    if known is not None and revision > 0:
        lam = revision * np.clip(obs - known, 0, None) / 2
        shape = out.shape
        out += rng.poisson(lam, size=shape) - rng.poisson(lam, size=shape)
        out = np.maximum(out, known)
    return out


def nowcast_means(totals: np.ndarray, share: np.ndarray) -> np.ndarray:
    r"""Return the point nowcast of the cases still to come in each week.

    A week with published total $Y$ and share reported $s$ is expected to
    reach $Y / s$, so $Y (1 - s) / s$ are still to come. A week with no
    cases adds none.

    Parameters
    ----------
    totals : numpy.ndarray
        Published totals $Y$, known plus imputed.
    share : numpy.ndarray
        Shares reported $s$, from `week_shares`.

    Returns
    -------
    numpy.ndarray
        Cases still to come, one per week.
    """
    s = np.clip(share, 1e-12, 1.0)
    return np.asarray(totals * (1 - s) / s, dtype=np.float64)


def revision_variance(
    history: pd.DataFrame, vintage: str, min_age: int = 63
) -> float:
    r"""Return $\kappa$, the variance of total revisions per imputed case.

    Over each pair of consecutive vintages up to `vintage`, provinces
    and weeks that ended at least `min_age` days before the earlier
    vintage, whose cases are all reported, give
    $\kappa = \sum \Delta^2 / \sum I$, with $\Delta$ the change in the
    total and $I$ the imputed count at the earlier vintage.

    Parameters
    ----------
    history : pandas.DataFrame
        Target rows of several vintages.
    vintage : str
        The forecast vintage. Later vintages are not used.
    min_age : int
        Days from the end of a week to the earlier vintage.

    Returns
    -------
    float
        $\kappa$, or 0 with no pair of vintages.
    """
    rows = history.loc[
        (history["vintage"] <= vintage) & (history["location"] != NATIONAL)
    ]
    key = ["location", "target_end_date"]
    num = den = 0.0
    vintages = sorted(str(v) for v in rows["vintage"].unique())
    for a, b in itertools.pairwise(vintages):
        old = rows.loc[rows["vintage"] == a].set_index(key)
        new = rows.loc[rows["vintage"] == b].set_index(key)
        both = old.join(new[["total"]].add_suffix("_new"), how="inner")
        end = pd.to_datetime(both.index.get_level_values("target_end_date"))
        settled = np.asarray((pd.Timestamp(a) - end).days >= min_age)
        both = both.loc[settled]
        num += float(((both["total_new"] - both["total"]) ** 2).sum())
        den += float(both["imputed"].sum())
    return num / den if den > 0 else 0.0


@dataclass
class _Nowcast:
    """The point nowcast of a task and what the walk needs from it."""

    weeks: list[str]
    ends: pd.DatetimeIndex
    grid: list[str]
    cols: list[int]
    last: int
    seen: np.ndarray
    known: np.ndarray
    added: np.ndarray
    size: np.ndarray
    share: np.ndarray
    own: list[bool]
    cdfs: list[np.ndarray]
    point: np.ndarray


@dataclass
class PersistenceNow(Persistence):
    """The hub baseline run forward from a nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    n_history : int
        Onset weeks of the reporting triangle the delay is estimated on.
        The longest delay, $D$, is `7 * (n_history - 1)` days, so the
        oldest week is complete. Longer delays are left out.
    n_retro : int
        Past nowcasts to fit the negative binomial sizes to.
    min_retro : int
        Fewest past nowcasts to nowcast at all.
    min_delays : int
        Fewest delays in the triangle's onset weeks to nowcast at all.
    own_per_week : float
        Fewest delays a week, on average over those onset weeks, for a
        place to use its own delay rather than the pooled one.

    See `Persistence` for `window`, `min_sd` and `low_count`, which set the
    walk's spread from the point nowcast.
    """

    name: str = "persistencenow"
    n_history: int = 13
    n_retro: int = 8
    min_retro: int = 3
    min_delays: int = 50
    own_per_week: float = 50.0

    @property
    def max_delay(self) -> int:
        """Longest delay of the triangle, in days."""
        return 7 * (self.n_history - 1)

    def _triangle(
        self, task: ForecastTask, location: str | None = None
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return the triangle's weeks, counts and cases in the file.

        Delays are as reported in the task's delay file. Onsets the hub
        fills in later for cases it had imputed keep their reported delay,
        as they were already in the totals.
        """
        assert task.delays is not None
        vintage = str(task.delays["vintage"].max())
        rows = task.delays.loc[task.delays["vintage"] == vintage]
        rows = rows.loc[rows["delay"] <= self.max_delay]
        if location is not None:
            rows = rows.loc[rows["location"] == location]
        weeks, tri = reporting_triangle(rows, vintage, self.max_delay)
        published = rows.groupby("onset_week").size().reindex(weeks)
        return weeks, tri, published.fillna(0).to_numpy(dtype=float)

    def _delay(
        self, task: ForecastTask, location: str | None = None
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the latest weeks, delay pmf, sizes and shares, or None.

        Pooled over places, or for `location` alone with at least
        `own_per_week` delays a week. None means too little delay data.
        """
        if task.delays is None or task.delays.empty:
            return None
        need = float(self.min_delays)
        if location is not None:
            need = max(need, self.own_per_week * self.n_history)
        weeks, tri, published = self._triangle(task, location)
        hist = tri[-self.n_history :]
        complete = (~np.isnan(hist)).all(axis=1).any()
        if len(hist) < self.n_history or not complete or np.nansum(hist) < need:
            return None
        obs, pred = retrospective_errors(tri, self.n_retro, self.n_history)
        if obs.shape[0] < self.min_retro:
            return None
        pmf = delay_pmf(hist)
        shares = week_shares(hist, pmf, published[-self.n_history :])
        return weeks[-self.n_history :], pmf, fit_nb_size(obs, pred), shares

    def delay_sources(self, task: ForecastTask) -> dict[str, str]:
        """Return `"own"` or `"pooled"`, the delay each place nowcasts with.

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        dict
            One entry per place in `task.locations`.
        """
        return {
            loc: "own" if self._delay(task, loc) is not None else "pooled"
            for loc in task.locations
        }

    def _prepare(self, task: ForecastTask) -> _Nowcast | None:
        """Return the point nowcast and its inputs, or None without delays."""
        pooled = self._delay(task)
        if pooled is None:
            return None
        assert task.delays is not None
        recent = pooled[0]
        weeks = list(task.target_end_dates)
        ends = pd.date_range(
            min(task.target["target_end_date"].min(), weeks[0]),
            weeks[-1],
            freq="7D",
        )
        grid = [d.date().isoformat() for d in ends]
        cols = [grid.index(w) for w in weeks]
        rows = [grid.index(w) for w in recent]

        def wide(column: str) -> np.ndarray:
            return np.array(
                [
                    task.target.loc[task.target["location"] == loc]
                    .set_index("target_end_date")[column]
                    .reindex(grid, fill_value=0)
                    .to_numpy(dtype=float)
                    for loc in task.locations
                ]
            )

        seen, known = wide("total"), wide("observed")
        last = rows[-1]
        added = np.zeros(seen.shape)
        size = np.full(seen.shape, np.inf)
        share = np.ones(seen.shape)
        own, cdfs = [], []
        for p, loc in enumerate(task.locations):
            fit = self._delay(task, loc)
            own.append(fit is not None)
            _, pmf, sizes, shares = fit or pooled
            cdfs.append(np.cumsum(pmf))
            share[p, rows] = shares
            added[p, rows] = nowcast_means(seen[p, rows], share[p, rows])
            # Horizon 0 is the latest week; horizons with no fit are Poisson.
            h = np.arange(self.n_history)[::-1]
            fitted = np.r_[sizes, np.full(self.n_history, np.inf)]
            size[p, rows] = fitted[h]
        point = seen[:, : last + 1] + added[:, : last + 1]
        return _Nowcast(
            weeks, ends, grid, cols, last, seen, known, added, size, share,
            own, cdfs, point,
        )  # fmt: skip

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return baseline trajectories from a nowcast for a task.

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
        assert task.delays is not None
        weeks, ends, grid, cols, last = (
            prep.weeks,
            prep.ends,
            prep.grid,
            prep.cols,
            prep.last,
        )
        seen, known, added, size, share = (
            prep.seen,
            prep.known,
            prep.added,
            prep.size,
            prep.share,
        )
        cdfs = prep.cdfs
        history = (
            task.target if task.target_history is None else task.target_history
        )
        rng = task.rng()
        n = task.n_samples
        kappa = revision_variance(history, task.vintage)
        totals = nowcast_draws(seen, added, size, n, rng, known, kappa)
        ahead = len(grid) - 1 - last
        # The walk's spread comes from the point nowcast; each draw starts
        # from its own nowcast of the latest week.
        sds = walk_spreads(
            list(prep.point), self.window, self.min_sd, self.low_count
        )
        for p, sd in enumerate(sds):
            walk = random_walk(np.log1p(totals[:, p, last]), sd, ahead, rng)
            totals[:, p, last + 1 :] = np.maximum(walk, known[p, last + 1 :])
        final, as_of = [], []
        delay_end = pd.Timestamp(str(task.delays["vintage"].max()))
        age = (delay_end - ends).days.to_numpy()
        for p, loc in enumerate(task.locations):
            final.append(_frame(loc, weeks, totals[:, p, cols]))
            diff = totals[:, p] - seen[p]
            cdf = cdfs[p]
            for date in task.report_by:
                # The cases still to come arrive by delay, in the triangle's
                # days from the end of the week.
                later = (pd.Timestamp(date) - ends).days.to_numpy()
                now = np.where(age < 0, 0.0, cdf[np.clip(age, 0, cdf.size - 1)])
                then = np.where(
                    later < 0, 0.0, cdf[np.clip(later, 0, cdf.size - 1)]
                )
                q = np.clip((then - now) / np.maximum(1 - now, 1e-12), 0, 1)
                q = np.where((now >= 1.0) | (share[p] >= 1.0), 1.0, q)
                up = rng.binomial(np.maximum(diff, 0).astype(np.int64), q)
                counts = seen[p] + up + np.minimum(diff, 0)
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
