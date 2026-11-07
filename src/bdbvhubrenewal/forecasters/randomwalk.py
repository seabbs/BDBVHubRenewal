# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The random walk baselines.

Both models run at province and zone level from one random walk
(`walk.random_walk`).

`RandomWalk` (`randomwalk`) is the baseline with no nowcast. It starts
from the count reported for the latest week, the one holding the
cut-off, with the spread of its own reported changes, and returns weeks
with data as reported.

`RandomWalkNowcast` (`randomwalknowcast`) nowcasts every day with data and
sums to hub weeks, with negative binomial new reports on the remaining
count minus Poisson removals of the count reported (`obs`).
Each sample then starts the walk from its own nowcast of the latest week
whose seven days are all on or before the cut-off.
If the week holding the cut-off is partial, its days after the cut-off
add Poisson draws from that week's walk rate.

The spread of the walk is the standard deviation of the weekly changes of
$\log(1 + y)$ in reported counts, over the weeks ending at the start.
In `randomwalknowcast`, which runs on the hub's daily release snapshots
(`chain`), the delay, the observation model and the walk spread each use
their own estimate where a place has at least 10 cases a week, and
otherwise the pooled estimate of the nearest parent that does
(`pooling`).
A place with no case in the latest `walk_weeks` weeks is 0 until its first
introduction, a geometric number of weeks in with its pooled weekly
probability. A silent place's walk has spread 0, so from its
introduction its rate stays at 1 case a week.
In `randomwalknowcast` this runs from the week after the latest week at
least 95% reported, so a silent place's recent weeks are not nowcast.
National is the sum of provinces by sample.
`randomwalknowcast` raises if the task has no snapshots or no pair of
releases within the delay window.
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
    Origin,
    Settings,
    build_origin,
    delay_summary,
    walk_changes,
)
from bdbvhubrenewal.forecasters.obs import sample_count
from bdbvhubrenewal.forecasters.walk import random_walk

__all__ = ["RandomWalk", "RandomWalkNowcast"]


def _key(task: ForecastTask) -> jax.Array:
    return jax.random.key(int(task.rng().integers(0, 2**31 - 1)))


def _timeline(data: list[str], task: ForecastTask) -> list[str]:
    """Return every hub week from the first data week to the last target."""
    weeks = list(data)
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
    key: jax.Array, x0: np.ndarray, origin: Origin, h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run the walk for every place, `x0` of shape `[L, S]`.

    A silent place starts its walk from 1 case at its first
    introduction, a geometric number of weeks in with the place's
    weekly probability, and is 0 before it.
    The introduction week has at least that case, as `introductions`
    counts a week with a case.
    A silent place's walk has spread 0, so from its introduction its
    rate stays at 1 case a week.
    """
    k_walk, k_intro = jax.random.split(key)
    x0 = np.where(origin.silent[:, None], 1.0, x0)
    sigma = np.where(origin.silent, 0.0, origin.sigma)
    keys = jax.random.split(k_walk, x0.shape[0])
    lam, y = jax.vmap(lambda k, x, s: random_walk(k, x, s, h))(
        keys,
        jnp.asarray(x0, jnp.float32),
        jnp.asarray(sigma, jnp.float32),
    )
    u = np.asarray(jax.random.uniform(k_intro, x0.shape), dtype=np.float64)
    q = origin.intro[:, None]
    # A probability of 0 gives an introduction that never comes.
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(q > 0, np.floor(np.log(u) / np.log1p(-q)), np.inf)
    first = np.where(origin.silent[:, None], first, 0.0)
    lag = np.arange(h)[None, None, :] - first[:, :, None]
    at = np.clip(lag, 0, h - 1).astype(int)

    def shift(x: jax.Array) -> np.ndarray:
        moved = np.take_along_axis(np.asarray(x, dtype=np.float64), at, axis=2)
        return np.where(lag >= 0, moved, 0.0)

    first_case = origin.silent[:, None, None] & (lag == 0)
    return shift(lam), np.where(first_case, np.maximum(shift(y), 1.0), shift(y))


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
    """Carry the latest reported week forward with growing noise.

    Attributes
    ----------
    name : str
        Model name.
    settings : Settings
        Settings; the walk's spread uses the latest `walk_weeks` weeks,
        and `randomwalknowcast` its delay window and thresholds.
    fit_diagnostics : list of dict
        One row per place from the last forecast: the start week and the
        spread, and for `randomwalknowcast` the errors behind the spread
        and the source of the delay, spread and walk.
    """

    name: str = "randomwalk"
    settings: Settings = field(default_factory=Settings)
    fit_diagnostics: list[dict[str, object]] = field(default_factory=list)
    needs_snapshots: ClassVar[bool] = False

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return random walk trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Examples
        --------
        >>> from bdbvhubrenewal.forecasters import get_model
        >>> samples = get_model("randomwalk").forecast(task)
        >>> samples.groupby(["location", "target_end_date"])["value"].median()
        """
        weekly = task.target.pivot_table(
            index="location", columns="target_end_date", values="total"
        )
        weekly = weekly.reindex(list(task.locations)).fillna(0.0)
        reported = weekly.to_numpy(dtype=float)
        weeks = _timeline(list(weekly.columns), task)
        n_loc, n_data = reported.shape
        n_all, n = len(weeks), task.n_samples
        start = np.full(n_loc, n_data - 1)
        sigma = [
            float(np.std(d, ddof=1)) if d.size >= 2 else 0.0
            for d in walk_changes(reported, start, self.settings)
        ]
        keys = jax.random.split(_key(task), n_loc)
        _, y = jax.vmap(lambda k, x, s: random_walk(k, x, s, n_all - n_data))(
            keys,
            jnp.asarray(np.repeat(reported[:, -1:], n, axis=1), jnp.float32),
            jnp.asarray(sigma, jnp.float32),
        )
        values = np.zeros((n, n_loc, n_all))
        for u in range(n_all):
            if u < n_data:
                values[:, :, u] = reported[:, u]
            else:
                values[:, :, u] = np.asarray(y)[:, :, u - n_data].T
        self.fit_diagnostics = [
            {"location": loc, "start_week": weeks[n_data - 1], "sigma": sd}
            for loc, sd in zip(task.locations, sigma, strict=True)
        ]
        return frame(task, weeks, values, list(range(n_loc)))


@dataclass
class RandomWalkNowcast(RandomWalk):
    """The random walk run forward from a daily nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    delay_summary : pandas.DataFrame or None
        The delay of the last forecast (`chain.delay_summary`), which maps
        its final counts to what later releases show.
    """

    name: str = "randomwalknowcast"
    delay_summary: pd.DataFrame | None = field(
        init=False, default=None, repr=False
    )
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
        # Each origin compiles its own triangle shapes, so cached executables
        # from earlier origins never help and exhaust LLVM memory over a run.
        jax.clear_caches()  # type: ignore[no-untyped-call]
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
        self.delay_summary = delay_summary(origin, self.name, task.vintage)
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
        weeks = _timeline(list(origin.snap.week_ends), task)
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
        lam, y = _walks(key_walk, nowcast[:, :, w0].T, origin, n_all)
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
        # A silent place follows its introductions from the week after its
        # start week, as in `randomwalk`, so its nowcast weeks too.
        for loc in np.flatnonzero(origin.silent):
            s = int(origin.start[loc])
            values[:, loc, s + 1 :] = y[loc, :, : n_all - s - 1]
        return values, weeks


register_model(
    "randomwalk",
    RandomWalk,
    "Latest reported week carried forward, with log-scale random walk noise.",
    baseline=True,
)
register_model(
    "randomwalknowcast",
    RandomWalkNowcast,
    "Random walk carried forward from a daily nowcast of the recent weeks.",
)
