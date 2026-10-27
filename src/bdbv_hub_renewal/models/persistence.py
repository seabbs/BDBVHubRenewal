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
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbv_hub_renewal.dates import as_date
from bdbv_hub_renewal.delays import EmpiricalDelayModel
from bdbv_hub_renewal.models.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)

__all__ = ["Persistence", "completeness_lag", "last_complete_week"]


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
            task.delays, task.vintage, self.completeness, self.default_lag_days
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
            recent = np.log1p(complete.to_numpy(dtype=float)[-self.window :])
            diffs = np.diff(recent)
            sd = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0
            sd = max(sd, self.min_sd)
            values = np.empty((n, len(weeks)))
            log_level = np.full(n, np.log1p(level))
            for j, week in enumerate(weeks):
                if week <= last:
                    values[:, j] = float(series.get(week, 0))
                    continue
                log_level = log_level + rng.normal(0.0, sd, size=n)
                values[:, j] = rng.poisson(np.expm1(np.maximum(log_level, 0)))
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
        return pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]


register_model(
    "persistence",
    Persistence,
    "Last complete week carried forward, with log-scale random walk noise.",
    baseline=True,
)
