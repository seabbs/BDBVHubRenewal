# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The random walk baselines.

Both models run on the hub's daily release snapshots (`chain`), at
province and zone level, from one random walk (`walk.random_walk`).

`RandomWalk` (`randomwalk`) starts from the reported count of the latest
week judged complete, $\pi \ge 0.95$, where

$$
\pi(w) = \frac{1}{7} \sum_{t \in w} \frac{\mathbb{1}\{t \le c\}}{G(a(t))}
$$

is the share of the week's final count reported, under the delay
estimate of `chain`.
Weeks up to the start are returned as reported.
Later weeks are walk draws, floored at the count reported.

`RandomWalkNowcast` (`randomwalknowcast`) nowcasts every day with data and
sums to hub weeks, with negative binomial new reports on the remaining
count minus Poisson removals of the count reported (`obs`).
Each sample then starts the walk from its own nowcast of the latest week
whose seven days are all on or before the cut-off.
If the week holding the cut-off is partial, its days after the cut-off
add Poisson draws from that week's walk rate.

The spread of the walk is the standard deviation of the weekly changes of
$\log(1 + y)$ in reported counts, over the weeks ending at the start of
`randomwalk`.
The delay, the observation model and the walk spread each use their own
estimate where a place has at least 10 cases a week, and otherwise the
pooled estimate of the nearest parent that does (`pooling`).
National is the sum of provinces by sample.
Each model raises if the task has no snapshots or no pair of releases
within the delay window.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.snapshots import snapshot_arrays
from bdbvhubrenewal.forecasters.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)
from bdbvhubrenewal.forecasters.chain import (
    MissingSnapshots,
    NoCompleteWeek,
    Origin,
    Settings,
    build_origin,
)
from bdbvhubrenewal.forecasters.obs import sample_count
from bdbvhubrenewal.forecasters.walk import random_walk

__all__ = ["RandomWalk", "RandomWalkNowcast"]


def _key(task: ForecastTask) -> jax.Array:
    return jax.random.key(int(task.rng().integers(0, 2**31 - 1)))


def _timeline(origin: Origin, task: ForecastTask) -> list[str]:
    """Return every hub week from the first data week to the last target."""
    weeks = list(origin.snap.week_ends)
    last = as_date(max(task.target_end_dates))
    while as_date(weeks[-1]) < last:
        weeks.append((as_date(weeks[-1]) + dt.timedelta(days=7)).isoformat())
    return weeks


def _label(origin: Origin, i: int) -> str:
    n = len(origin.snap.locations)
    kind = "own" if i < n else "pooled"
    return f"{kind}:{origin.snap.locations[i % n]}"


def _delay_label(origin: Origin, i: int) -> str:
    own = origin.fit.own[i]
    return f"own at {int(own.sum())} of {own.size} ages"


def _walks(
    key: jax.Array, x0: np.ndarray, sigma: np.ndarray, h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run the walk for every place, `x0` of shape `[L, S]`."""
    keys = jax.random.split(key, x0.shape[0])
    lam, y = jax.vmap(lambda k, x, s: random_walk(k, x, s, h))(
        keys, jnp.asarray(x0, jnp.float32), jnp.asarray(sigma, jnp.float32)
    )
    return np.asarray(lam, dtype=np.float64), np.asarray(y, dtype=np.float64)


def frame(
    task: ForecastTask, weeks: list[str], values: np.ndarray, rows: list[int]
) -> pd.DataFrame:
    """Return sample rows from `values` of shape `[S, L, H]`."""
    col = {w: i for i, w in enumerate(weeks)}
    targets = list(task.target_end_dates)
    n = values.shape[0]
    frames = []
    for loc, row in zip(task.locations, rows, strict=True):
        picked = np.column_stack(
            [
                values[:, row, col[w]] if w in col else np.zeros(n)
                for w in targets
            ]
        )
        frames.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "target_end_date": np.tile(targets, n),
                    "sample": np.repeat(np.arange(1, n + 1), len(targets)),
                    "value": np.maximum(np.round(picked), 0.0).reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]


@dataclass
class RandomWalk:
    """Carry the latest complete week forward with growing noise.

    Attributes
    ----------
    name : str
        Model name.
    settings : Settings
        Delay window, thresholds and the walk's window.
    fit_diagnostics : list of dict
        One row per place from the last forecast: the start week, the
        errors behind the spread and the source of the delay, spread and
        walk.
    """

    name: str = "randomwalk"
    settings: Settings = field(default_factory=Settings)
    fit_diagnostics: list[dict[str, object]] = field(default_factory=list)
    needs_snapshots: ClassVar[bool] = True

    def origin_at(self, task: ForecastTask, errors: bool) -> Origin:
        """Fit the delays and the walk at the task's vintage.

        Parameters
        ----------
        task : ForecastTask
            The task. Its `daily_history` holds the release snapshots.
        errors : bool
            Also fit the nowcast errors, which `RandomWalkNowcast` needs.

        Returns
        -------
        Origin
            The fit at the task's vintage.

        Raises
        ------
        MissingSnapshots
            If the task has no release snapshots.
        """
        daily = task.daily_history
        if daily is None or daily.empty:
            raise MissingSnapshots(
                f"{self.name} needs daily release snapshots, and the task "
                f"for {task.vintage} has none."
            )
        snap = snapshot_arrays(daily, task.vintage, task.locations)
        return build_origin(snap, self.settings, errors)

    def rows(self, origin: Origin, task: ForecastTask) -> list[int]:
        """Return the origin's row of each of the task's places.

        Parameters
        ----------
        origin : Origin
            The fit, from `origin_at`.
        task : ForecastTask
            The task.

        Returns
        -------
        list of int
            Row indices, in the order of `task.locations`.
        """
        pos = {c: i for i, c in enumerate(origin.snap.locations)}
        return [pos[c] for c in task.locations]

    def _diagnostics(
        self, origin: Origin, task: ForecastTask, start: np.ndarray
    ) -> None:
        rows: list[dict[str, object]] = []
        for loc, i in zip(task.locations, self.rows(origin, task), strict=True):
            n_err = None if origin.obs_n is None else int(origin.obs_n[i, 0])
            obs_src = (
                None
                if origin.obs_source is None
                else _label(origin, int(origin.obs_source[i, 0]))
            )
            rows.append(
                {
                    "location": loc,
                    "start_week": origin.snap.week_ends[int(start[i])],
                    "n_err": n_err,
                    "delay_source": _delay_label(origin, i),
                    "obs_source": obs_src,
                    "sigma_source": _label(origin, int(origin.sigma_source[i])),
                }
            )
        self.fit_diagnostics = rows

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return random walk trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `daily_history` holds the
            release snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no snapshots, no pair of releases within
            the delay window, or a place with no week that is 95 per cent
            complete (`NoCompleteWeek`).

        Examples
        --------
        >>> from bdbvhubrenewal.forecasters import get_model
        >>> samples = get_model("randomwalk").forecast(task)
        >>> samples.groupby(["location", "target_end_date"])["value"].median()
        """
        origin = self.origin_at(task, errors=False)
        weeks = _timeline(origin, task)
        n_data, n_all, n = origin.snap.n_weeks, len(weeks), task.n_samples
        n_loc = len(origin.snap.locations)
        good = origin.complete >= origin.settings.completeness
        if not good.any(axis=1).all():
            has = good.any(axis=1)
            none = [
                x
                for x, ok in zip(origin.snap.locations, has, strict=True)
                if not ok
            ]
            raise NoCompleteWeek(
                f"No week reaches {origin.settings.completeness:.0%} "
                f"complete at {origin.snap.releases[origin.o]} for "
                f"{', '.join(none)}, so the walk has no start week."
            )
        start = origin.start
        x0 = origin.reported[np.arange(n_loc), start]
        _, y = _walks(
            _key(task), np.repeat(x0[:, None], n, axis=1), origin.sigma, n_all
        )
        values = np.zeros((n, n_loc, n_all))
        for loc in range(n_loc):
            for u in range(n_all):
                if u <= start[loc]:
                    values[:, loc, u] = origin.reported[loc, u]
                else:
                    draw = y[loc, :, u - start[loc] - 1]
                    floor = origin.reported[loc, u] if u < n_data else 0.0
                    values[:, loc, u] = np.maximum(draw, floor)
        self._diagnostics(origin, task, start)
        return frame(task, weeks, values, self.rows(origin, task))


@dataclass
class RandomWalkNowcast(RandomWalk):
    """The random walk run forward from a daily nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    """

    name: str = "randomwalknowcast"

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return random walk trajectories from a nowcast for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `daily_history` holds the
            release snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no snapshots, or no pair of releases within
            the delay window.
        """
        origin = self.origin_at(task, errors=True)
        values, weeks = self.trajectories(origin, task, origin.point)
        self._diagnostics(
            origin,
            task,
            np.full(len(origin.snap.locations), origin.start_nowcast),
        )
        return frame(task, weeks, values, self.rows(origin, task))

    def trajectories(
        self, origin: Origin, task: ForecastTask, point: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Return trajectories from a nowcast with mean `point`.

        Parameters
        ----------
        origin : Origin
            The fit at the task's vintage, with its observation model.
        task : ForecastTask
            The task.
        point : numpy.ndarray
            Mean weekly nowcast, `[L, W]`. The model's own is
            `origin.point`. A later release's expected count gives the
            prediction scored in the backtest.

        Returns
        -------
        tuple
            Samples `[S, L, H]` for every hub week from the first data
            week to the last target, and those weeks.
        """
        snap = origin.snap
        assert origin.size is not None
        assert origin.removal is not None
        weeks = _timeline(origin, task)
        n_data, n_all, n = snap.n_weeks, len(weeks), task.n_samples
        n_loc = len(snap.locations)
        k_n = origin.size.shape[1]
        back = snap.cut_week(origin.o) - np.arange(n_data)
        k = np.clip(back, 0, k_n - 1)
        # Weeks older than the fitted ones are final, with no removals.
        b = np.where(back < k_n, origin.removal[:, k], 0.0)
        key_x, key_walk, key_part = jax.random.split(_key(task), 3)
        shape = (n, *point.shape)
        reported = jnp.broadcast_to(
            jnp.asarray(origin.reported, jnp.float32), shape
        )
        nowcast = np.asarray(
            sample_count(
                key_x,
                reported,
                jnp.maximum(jnp.asarray(point, jnp.float32) - reported, 0.0),
                jnp.broadcast_to(
                    jnp.asarray(origin.size[:, k], jnp.float32), shape
                ),
                jnp.broadcast_to(jnp.asarray(b, jnp.float32), shape),
            ),
            np.float64,
        )
        w0 = origin.start_nowcast
        lam, y = _walks(key_walk, nowcast[:, :, w0].T, origin.sigma, n_all)
        cut = int(snap.cut[origin.o])
        c_week = int(snap.week[cut])
        miss = int(np.flatnonzero(snap.week == c_week).max()) - cut
        values = np.zeros((n, n_loc, n_all))
        part = np.asarray(
            jax.random.poisson(
                key_part, jnp.asarray(lam[:, :, 0] * miss / 7, jnp.float32)
            ),
            dtype=np.float64,
        )  # [L, S]
        for u in range(n_all):
            if u <= w0:
                values[:, :, u] = nowcast[:, :, u]
            else:
                values[:, :, u] = y[:, :, u - w0 - 1].T
                if u == w0 + 1 and miss > 0 and u < n_data:
                    values[:, :, u] = nowcast[:, :, u] + part.T
        return values, weeks


register_model(
    "randomwalk",
    RandomWalk,
    "Latest complete week carried forward, with log-scale random walk noise.",
    baseline=True,
)
register_model(
    "randomwalknowcast",
    RandomWalkNowcast,
    "Random walk carried forward from a daily nowcast of the recent weeks.",
)
