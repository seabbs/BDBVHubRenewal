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
`min_sd`.
Weeks up to $c$ are returned as published (a point mass).
The median is flat and the spread grows with the horizon.
Each sample is one joint trajectory across weeks.

A week is judged complete if the vintage date is at least `lag_days`
after its midpoint.
`lag_days` is the `completeness` quantile of the pooled delay published
with the vintage, or `default_lag_days` if no delays were published.
Recent onset weeks only show the delays reported so far, so this lag is
biased short.

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
of the triangle pooled over places. Each week, the latest included, has
a share reported $s_p(w)$ (`week_share`), and its published total
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

**Forecast.** Each sample is centred on the mean of $\log\{1 + C_p(w)\}$
over the latest `centre_weeks` weeks, weighted by $s_p(w)$, so a week
with few cases reported yet counts little. From there it follows the
COVID-19 Forecast Hub baseline on the log scale (`baseline_walk`): its
median stays at the centre, and each weekly step is drawn from the past
weekly changes of the place's point nowcast, and their negatives, as
BVDOutbreakSize does. Places without their own delay borrow the steps of
those with one. A place with fewer than `low_count` cases a week over the
last `window` weeks walks the log of that mean as a Poisson rate instead,
so a few cases a week stay plausible. Later weeks are at least the known
onsets.

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
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY, EmpiricalDelayModel
from bdbvhubrenewal.data.reconstruct import PUBLICATION_LAG_DAYS
from bdbvhubrenewal.data.triangle import (
    delay_pmf,
    first_publication_delays,
    fit_nb_size,
    reporting_triangle,
    retrospective_errors,
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
    "baseline_walk",
    "completeness_lag",
    "last_complete_week",
    "nowcast_draws",
    "nowcast_means",
    "week_share",
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
        Delay quantile defining a complete week.
    default_lag_days : int
        Completeness lag when a vintage has no delay data.
    """

    name: str = "persistence"
    window: int = 8
    min_sd: float = 0.1
    completeness: float = 0.9
    default_lag_days: int = 10

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
        lag = completeness_lag(
            task.delays,
            task.vintage,
            self.completeness,
            self.default_lag_days,
        )
        last = last_complete_week(task.vintage, lag).isoformat()
        weeks = list(task.target_end_dates)
        n = task.n_samples
        frames = []
        for loc in task.locations:
            series = (
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .sort_index()
            )
            complete = series.loc[series.index <= last]
            level = float(complete.iloc[-1]) if len(complete) else 0.0
            sd = self._spread(complete.to_numpy(dtype=float))
            values = np.empty((n, len(weeks)))
            log_level = np.full(n, np.log1p(level))
            for j, week in enumerate(weeks):
                if week <= last:
                    values[:, j] = float(series.get(week, 0))
                    continue
                log_level, values[:, j] = _step(log_level, sd, rng)
            frames.append(
                pd.DataFrame(
                    {
                        "location": loc,
                        "target_end_date": np.tile(weeks, n),
                        "sample": np.repeat(np.arange(1, n + 1), len(weeks)),
                        "value": values.reshape(-1),
                    }
                )
            )
        final = pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]
        if not task.report_by:
            return final
        return pd.concat(
            [final.assign(as_of=np.nan), *self._reported_by(task, final, rng)],
            ignore_index=True,
        )

    def _spread(self, complete: np.ndarray) -> float:
        """Return the SD of the last weekly log changes, floored."""
        recent = np.log1p(complete[-self.window :])
        diffs = np.diff(recent)
        sd = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0
        return max(sd, self.min_sd)

    def _reported_by(
        self,
        task: ForecastTask,
        final: pd.DataFrame,
        rng: np.random.Generator,
    ) -> list[pd.DataFrame]:
        """Thin the final samples to the counts reported by each later date."""
        if task.delays is None or task.delays.empty:
            return []
        delay_vintage = str(task.delays["vintage"].max())
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
            share = reported_share(task.delays, delay_vintage, days[due])
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


def _step(
    log_level: np.ndarray, sd: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Take one random walk step on the log scale and draw Poisson counts."""
    log_level = log_level + rng.normal(0.0, sd, size=log_level.size)
    return log_level, rng.poisson(np.expm1(np.maximum(log_level, 0)))


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


def week_share(
    cdf: np.ndarray, age: np.ndarray, lag: int = PUBLICATION_LAG_DAYS
) -> np.ndarray:
    r"""Return the share of a week's cases in a vintage `age` days after it.

    A case with onset on day $W - u$ of the week ending $W$, delay $d$
    and publication lag $\ell$ is in a vintage $t$ days after $W$ if
    $d + \ell - u \le t$. With onsets spread evenly over the week and
    $\ell$ uniform on $0, \dots, L$, as in the renewal model,

    $$
    s(t) = \frac{1}{7 (L + 1)} \sum_{u=0}^{6} \sum_{\ell=0}^{L}
    F(t + u - \ell).
    $$

    Parameters
    ----------
    cdf : numpy.ndarray
        Delay cdf $F$ on days $0, 1, \dots$.
    age : numpy.ndarray
        Days $t$ from the end of each week to the vintage.
    lag : int
        Longest publication lag $L$ in days.

    Returns
    -------
    numpy.ndarray
        $s(t)$ for each week.
    """
    shift = (np.arange(7)[:, None] - np.arange(lag + 1)[None, :]).ravel()
    day = np.asarray(age)[:, None] + shift[None, :]
    covered = np.where(day < 0, 0.0, cdf[np.clip(day, 0, cdf.size - 1)])
    return np.asarray(covered.mean(axis=1), dtype=np.float64)


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
        Shares reported $s$, from `week_share`.

    Returns
    -------
    numpy.ndarray
        Cases still to come, one per week.
    """
    s = np.clip(share, 1e-12, 1.0)
    return np.asarray(totals * (1 - s) / s, dtype=np.float64)


def baseline_walk(
    centre: np.ndarray,
    steps: np.ndarray,
    horizons: int,
    rng: np.random.Generator,
    rate: bool = False,
) -> np.ndarray:
    r"""Draw the COVID-19 Forecast Hub baseline forward on the log scale.

    Each draw adds `horizons` weekly steps sampled with replacement from
    `steps` and their negatives to its centre, so the median stays at the
    centre and the spread grows with the horizon. With fewer than three
    past steps it is Poisson around the centre. With `rate`, the walk is
    on the log of a Poisson rate and each week is a Poisson draw from it,
    so a place with a few cases keeps a few cases plausible.

    Parameters
    ----------
    centre : numpy.ndarray
        $\log(1 + C)$ for each draw, or the log rate with `rate`.
    steps : numpy.ndarray
        Past weekly changes in $\log(1 + y)$.
    horizons : int
        Weeks to draw.
    rng : numpy.random.Generator
        Random generator.
    rate : bool
        Walk on the log of a Poisson rate.

    Returns
    -------
    numpy.ndarray
        Counts, one row per draw and one column per week ahead.
    """
    shape = (centre.size, horizons)
    level = np.exp(centre) if rate else np.expm1(centre)
    if steps.size < 3:
        return rng.poisson(level[:, None], size=shape).astype(float)
    pool = np.r_[steps, -steps]
    walk = centre[:, None] + rng.choice(pool, size=shape).cumsum(axis=1)
    if rate:
        return rng.poisson(np.exp(walk)).astype(float)
    return np.round(np.maximum(np.expm1(walk), 0.0))


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
class PersistenceNow(Persistence):
    """The hub baseline run forward from a nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    n_history : int
        Onset weeks of the reporting triangle the delay is estimated on.
        The longest delay is `7 * (n_history - 1)` days, at most
        `DEFAULT_MAX_DELAY`, so the oldest week is complete.
    n_retro : int
        Past nowcasts to fit the negative binomial sizes to.
    min_retro : int
        Fewest past nowcasts to nowcast at all.
    min_delays : int
        Fewest delays in the triangle's onset weeks to nowcast at all.
    own_per_week : float
        Fewest delays a week, on average over those onset weeks, for a
        place to use its own delay rather than the pooled one.
    low_count : float
        Below this mean count a week over the last `window` weeks, a place
        walks the log of its mean as a Poisson rate.
    centre_weeks : int
        Latest nowcast weeks the walk is centred on, weighted by the share
        of each reported, so a week with few cases reported yet counts
        little.

    See `Persistence` for `window`, the weeks of past changes.
    """

    name: str = "persistencenow"
    n_history: int = 6
    n_retro: int = 8
    min_retro: int = 3
    min_delays: int = 50
    own_per_week: float = 50.0
    centre_weeks: int = 3
    low_count: float = 10.0

    @property
    def max_delay(self) -> int:
        """Longest delay of the triangle, in days."""
        return min(7 * (self.n_history - 1), DEFAULT_MAX_DELAY)

    def _triangle(
        self, task: ForecastTask, location: str | None = None
    ) -> tuple[list[str], np.ndarray]:
        """Return the triangle's weeks and counts, pooled or for a place.

        Delays run to each case's first publication, over every delay file
        up to the task's (`first_publication_delays`).
        """
        assert task.delays is not None
        vintage = str(task.delays["vintage"].max())
        files = task.delays
        if task.delay_history is not None:
            past = task.delay_history["vintage"] < vintage
            files = pd.concat([task.delay_history.loc[past], task.delays])
        rows = first_publication_delays(files, vintage, self.max_delay)
        if location is not None:
            rows = rows.loc[rows["location"] == location]
        return reporting_triangle(rows, vintage, self.max_delay)

    def _delay(
        self, task: ForecastTask, location: str | None = None
    ) -> tuple[list[str], np.ndarray, np.ndarray] | None:
        """Return the latest weeks, delay pmf and sizes, or None.

        Pooled over places, or for `location` alone with at least
        `own_per_week` delays a week. None means too little delay data.
        """
        if task.delays is None or task.delays.empty:
            return None
        need = float(self.min_delays)
        if location is not None:
            need = max(need, self.own_per_week * self.n_history)
        weeks, tri = self._triangle(task, location)
        hist = tri[-self.n_history :]
        complete = (~np.isnan(hist)).all(axis=1).any()
        if len(hist) < self.n_history or not complete or np.nansum(hist) < need:
            return None
        obs, pred = retrospective_errors(tri, self.n_retro, self.n_history)
        if obs.shape[0] < self.min_retro:
            return None
        pmf = delay_pmf(hist)
        return weeks[-self.n_history :], pmf, fit_nb_size(obs, pred)

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
        pooled = self._delay(task)
        if pooled is None:
            return super().forecast(task)
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
        delay_end = pd.Timestamp(str(task.delays["vintage"].max()))
        age = np.asarray((delay_end - ends).days)
        last = rows[-1]
        added = np.zeros(seen.shape)
        size = np.full(seen.shape, np.inf)
        share = np.ones(seen.shape)
        own, cdfs = [], []
        for p, loc in enumerate(task.locations):
            fit = self._delay(task, loc)
            own.append(fit is not None)
            _, pmf, sizes = fit or pooled
            cdfs.append(np.cumsum(pmf))
            share[p] = week_share(cdfs[p], age)
            added[p, rows] = nowcast_means(seen[p, rows], share[p, rows])
            # Horizon 0 is the latest week; horizons with no fit are Poisson.
            h = np.arange(self.n_history)[::-1]
            fitted = np.r_[sizes, np.full(self.n_history, np.inf)]
            size[p, rows] = fitted[h]
        history = (
            task.target if task.target_history is None else task.target_history
        )
        rng = task.rng()
        n = task.n_samples
        kappa = revision_variance(history, task.vintage)
        totals = nowcast_draws(seen, added, size, n, rng, known, kappa)
        ahead = len(grid) - 1 - last
        # Steps come from the weeks at least half reported, so the noise of
        # a barely reported week does not widen the walk.
        point = np.log1p(seen + added)
        steps = [
            np.diff(row[: last + 1][sh[: last + 1] >= 0.5][-self.window :])
            for row, sh in zip(point, share, strict=True)
        ]
        # Places with their own delay have enough data for their own steps;
        # the rest borrow the steps of those places, as they borrow a delay.
        shared = [st for st, o in zip(steps, own, strict=True) if o]
        pooled_steps = np.concatenate(shared) if shared else None
        span = slice(last + 1 - self.centre_weeks, last + 1)
        past = slice(last + 1 - self.window, last + 1)
        for p in range(len(task.locations)):
            mine = steps[p] if own[p] or pooled_steps is None else pooled_steps
            mean = totals[:, p, past].mean(axis=1)
            if np.median(mean) < self.low_count:
                # A place with few cases walks the log of its recent mean as
                # a Poisson rate, so a few cases a week stay plausible.
                centre = np.log(np.maximum(mean, 1e-9))
                walk = baseline_walk(centre, mine, ahead, rng, rate=True)
            else:
                weight = share[p, span]
                centre = np.log1p(totals[:, p, span]) @ weight / weight.sum()
                walk = baseline_walk(centre, mine, ahead, rng)
            totals[:, p, last + 1 :] = np.maximum(walk, known[p, last + 1 :])
        final, as_of = [], []
        for p, loc in enumerate(task.locations):
            s_v = share[p]
            final.append(_frame(loc, weeks, totals[:, p, cols]))
            diff = totals[:, p] - seen[p]
            for date in task.report_by:
                later = (pd.Timestamp(date) - ends).days.to_numpy()
                s_t = week_share(cdfs[p], later)
                q = np.clip((s_t - s_v) / np.maximum(1 - s_v, 1e-12), 0, 1)
                q = np.where(s_v >= 1.0, 1.0, q)
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
