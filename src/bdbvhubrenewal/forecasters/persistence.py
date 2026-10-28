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

`PersistenceNow` (`persistencenow`) first nowcasts the recent weeks, with
the baselinenowcast method (`bdbvhubrenewal.data.triangle`).
A chain ladder on the reporting triangle of the hub's delay file, pooled
over provinces, gives for each open onset week $w$ the ratio $r(w)$ of
the cases still to come to those reported.
Past nowcasts of the same triangle give a negative binomial size
$\phi_h$ for each horizon $h$.
Each week's published total $Y(w)$, known plus imputed, then gets

$$
C(w) = Y(w) + U(w), \quad
U(w) \sim \text{NegBin}\big(\text{mean } r(w) Y(w),\ \text{size } \phi_h\big),
$$

the pooled delay applied to each place.
The last week with a share reported $1 / \{1 + r(w)\} \ge$ `min_share`
takes the place of the last complete week, and each sample carries its
own nowcast forward as above.
Later weeks are at least the published total.
If the delay data have no complete onset week or fewer than `min_delays`
cases in the last `n_history` weeks, or give fewer than `min_retro` past
nowcasts, it is `Persistence`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.completeness import reported_share
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.delays import EmpiricalDelayModel
from bdbvhubrenewal.data.triangle import (
    chainladder_fill,
    delay_pmf,
    fit_nb_size,
    reporting_triangle,
    retrospective_errors,
)
from bdbvhubrenewal.forecasters.base import (
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
    ratio: np.ndarray,
    size: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Draw final totals from published totals and point nowcast ratios.

    Each draw is $Y + U$ with $U$ negative binomial with mean $r Y$ and
    size $\phi$, or Poisson where $\phi$ is infinite.

    Parameters
    ----------
    observed : numpy.ndarray
        Published totals $Y$, one per week.
    ratio : numpy.ndarray
        Cases still to come per case reported, $r$, one per week.
    size : numpy.ndarray
        Negative binomial size $\phi$, one per week.
    n : int
        Number of draws.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Final totals, one row per draw and one column per week.
    """
    mean = np.broadcast_to(observed * ratio, (n, observed.size))
    phi = np.broadcast_to(size, (n, observed.size))
    added = np.zeros((n, observed.size))
    pois = (mean > 0) & ~np.isfinite(phi)
    added[pois] = rng.poisson(mean[pois])
    nb = (mean > 0) & np.isfinite(phi)
    added[nb] = rng.negative_binomial(phi[nb], phi[nb] / (phi[nb] + mean[nb]))
    return observed + added


@dataclass
class PersistenceNow(Persistence):
    """Persistence carried forward from a nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    min_share : float
        Smallest share reported for a week's nowcast to be carried
        forward. The last such week is.
    n_history : int
        Onset weeks of the reporting triangle each chain ladder uses.
    n_retro : int
        Past nowcasts to fit the negative binomial sizes to.
    min_retro : int
        Fewest past nowcasts to nowcast at all.
    min_delays : int
        Fewest delays in the chain ladder's onset weeks to nowcast at all.

    See `Persistence` for the other attributes.
    """

    name: str = "persistencenow"
    min_share: float = 0.5
    n_history: int = 14
    n_retro: int = 8
    min_retro: int = 3
    min_delays: int = 50

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return persistence trajectories from a nowcast for a task.

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
        if task.delays is None or task.delays.empty:
            return super().forecast(task)
        delay_vintage = str(task.delays["vintage"].max())
        tri_weeks, tri = reporting_triangle(task.delays, delay_vintage)
        hist = tri[-self.n_history :]
        obs, pred = retrospective_errors(tri, self.n_retro, self.n_history)
        complete = (~np.isnan(hist)).all(axis=1).any()
        enough = np.nansum(hist) >= self.min_delays
        if len(hist) < self.n_history or not complete or not enough:
            return super().forecast(task)
        if obs.shape[0] < self.min_retro:
            return super().forecast(task)
        filled = chainladder_fill(hist)
        reported = np.nansum(hist, axis=1)
        to_come = filled.sum(axis=1) - reported
        ratio_tri = np.where(reported > 0, to_come / np.maximum(reported, 1), 0)
        sizes = fit_nb_size(obs, pred)
        cdf = np.cumsum(delay_pmf(hist))
        weeks = list(task.target_end_dates)
        ends = pd.date_range(
            min(task.target["target_end_date"].min(), weeks[0]),
            weeks[-1],
            freq="7D",
        )
        grid = [d.date().isoformat() for d in ends]
        # Ratio and size of each grid week; weeks before the history are
        # complete, weeks after the delay vintage are not nowcast.
        ratio, size = np.zeros(len(grid)), np.full(len(grid), np.inf)
        share = np.where(np.array(grid) <= tri_weeks[0], 1.0, 0.0)
        for i, week in enumerate(tri_weeks[-self.n_history :]):
            if week in grid:
                j = grid.index(week)
                h = self.n_history - 1 - i
                ratio[j] = ratio_tri[i]
                size[j] = sizes[h] if h < sizes.size else np.inf
                share[j] = 1.0 / (1.0 + ratio_tri[i])
        share[np.array(grid) < tri_weeks[-self.n_history]] = 1.0
        anchor = np.flatnonzero(share >= self.min_share)
        if anchor.size == 0:
            return super().forecast(task)
        last = int(anchor[-1])
        ratio[last + 1 :] = 0.0
        rng = task.rng()
        n = task.n_samples
        cols = [grid.index(w) for w in weeks]
        end = pd.Timestamp(delay_vintage)
        age = (end - ends).days.to_numpy()
        final, as_of = [], []
        for loc in task.locations:
            seen = (
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .reindex(grid, fill_value=0)
                .to_numpy(dtype=float)
            )
            totals = nowcast_draws(seen, ratio, size, n, rng)
            sd = self._spread(seen[: last + 1] * (1 + ratio[: last + 1]))
            log_level = np.log1p(totals[:, last])
            for j in range(last + 1, len(grid)):
                log_level, draw = _step(log_level, sd, rng)
                totals[:, j] = np.maximum(draw, seen[j])
            final.append(_frame(loc, weeks, totals[:, cols]))
            for date in task.report_by:
                later = (pd.Timestamp(date) - ends).days.to_numpy()
                s_v = _share(cdf, age)
                s_t = _share(cdf, later)
                q = np.clip((s_t - s_v) / np.maximum(1 - s_v, 1e-12), 0, 1)
                q = np.where(s_v >= 1.0, 1.0, q)
                extra = rng.binomial((totals - seen).astype(np.int64), q)
                due = [c for c in cols if grid[c] <= date]
                as_of.append(
                    _frame(
                        loc, [grid[c] for c in due], (seen + extra)[:, due]
                    ).assign(as_of=date)
                )
        out = pd.concat(final, ignore_index=True)[list(SAMPLE_COLUMNS)]
        if not task.report_by:
            return out
        return pd.concat([out.assign(as_of=np.nan), *as_of], ignore_index=True)


def _share(cdf: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Share of a week's cases published `age` days after it ends."""
    return np.where(age < 0, 0.0, cdf[np.clip(age, 0, cdf.size - 1)])


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
