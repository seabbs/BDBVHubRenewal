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

`RandomWalkNowcast` (`randomwalknowcast`) nowcasts each hub week, with
negative binomial new reports on the remaining count (`obs`).
Weeks with data keep their nowcast draws. If the week holding the cut-off
is partial, its days after the cut-off add a Poisson draw from that week's
walk rate, walked by each sample from its own nowcast of the latest full
week at least `Settings.start_share` reported with at least
`Settings.start_count` cases (`_start_weeks`). Each sample's completed
cut-off week, its nowcast plus that draw, is the base the walk goes on
from for the weeks after the cut-off.

The spread of the walk is the standard deviation of the weekly changes of
$\log(1 + y)$ in reported counts, over the weeks ending at the start of
`randomwalk`, net of their delta-method Poisson noise
(`walk.walk_variance`).
A place whose net variance is 0 or below has no own estimate.
The delay, the observation model and the walk spread each use their own
estimate where a place has at least 10 cases a week, and otherwise the
pooled estimate of the nearest parent that does (`pooling`).
A zone takes its province's delay, whatever its level (`chain.fit_delay`).
A place with no case in the latest `walk_weeks` weeks is silent. It is
exactly 0 until its first introduction, which comes each week with its
pooled probability times the share of the week not yet reported, and then
walks from a count drawn from Poisson(1) given at least 1
(`introductions.silent_trajectories`).
In `randomwalknowcast` a place with no week to start from is quiet like
a silent one: after its latest complete week it draws introductions,
floored at the counts reported (`_quiet`).
National is the sum of provinces by sample.
Each model raises if the task has no snapshots or no pair of releases
within the delay window.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import ClassVar, Final

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
    delay_summary,
)
from bdbvhubrenewal.forecasters.introductions import silent_trajectories
from bdbvhubrenewal.forecasters.obs import sample_count
from bdbvhubrenewal.forecasters.walk import random_walks

__all__ = ["RandomWalk", "RandomWalkNowcast"]


_KEPT_ORIGINS: Final = 4
"""Origins whose compiled programs are kept in memory at once."""

_ORIGINS: set[tuple[object, ...]] = set()


def _clear_caches(task: ForecastTask) -> None:
    """Drop JAX's compiled programs when a task brings a new origin.

    Each origin compiles its own shapes, and keeping every origin's
    programs over a run fills `vm.max_map_count` (an LLVM "Cannot allocate
    memory" error). The programs of the last `_KEPT_ORIGINS` origins are
    kept, so the models forecasting one task, and the same task forecast
    again (the release backtest after the development one), reuse them.
    """
    origin = (
        task.kind,
        task.vintage,
        task.reference_date,
        tuple(task.locations),
        task.n_samples,
    )
    if origin in _ORIGINS:
        return
    if len(_ORIGINS) >= _KEPT_ORIGINS:
        jax.clear_caches()  # type: ignore[no-untyped-call]
        _ORIGINS.clear()
    _ORIGINS.add(origin)


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


def _walks(
    key: jax.Array,
    x0: np.ndarray,
    origin: Origin,
    start: np.ndarray,
    h: int,
    quiet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `h` weeks of walk rates and counts after `start` from `x0`.

    Places in `quiet` `[L]` draw their introductions
    (`silent_trajectories`), and the others walk from `x0` `[L, S]`. The
    rates `[L, S, h]` are the walk's, also for quiet places, whose counts
    do not use them.
    """
    k_walk, k_silent = jax.random.split(key)
    keys = jax.random.split(k_walk, x0.shape[0])
    lam, drawn = random_walks(
        keys,
        jnp.asarray(x0, jnp.float32),
        jnp.asarray(origin.sigma, jnp.float32),
        h,
    )
    silent = silent_trajectories(
        k_silent,
        quiet,
        origin.intro,
        1.0 - np.clip(origin.complete, 0.0, 1.0),
        start,
        origin.sigma,
        h,
        x0.shape[1],
    )
    counts = np.where(
        quiet[:, None, None], silent, np.asarray(drawn, np.float64)
    )
    return np.asarray(lam, np.float64), counts


def start_window(settings: Settings) -> int:
    """Return the weeks back from the cut-off week a start week may be.

    It is `Settings.history_delay`, the onset days the nowcast's delay is
    fitted on (1.5 D), in whole weeks.

    Parameters
    ----------
    settings : Settings
        Settings.

    Returns
    -------
    int
        Weeks, counting the cut-off week.
    """
    return -(-settings.history_delay // 7)


def _start_ok(origin: Origin) -> np.ndarray:
    """Return the full weeks a walk can start from, `[L, W]`.

    A week can when its share reported is at least `Settings.start_share`,
    its reported count is at least `Settings.start_count`, and it is within
    `start_window` weeks of the cut-off week.
    """
    settings, full = origin.settings, origin.start_nowcast
    ok = (origin.complete[:, : full + 1] >= settings.start_share) & (
        origin.reported[:, : full + 1] >= settings.start_count
    )
    first = origin.snap.cut_week(origin.o) - start_window(settings) + 1
    ok[:, : max(first, 0)] = False
    return ok


def _quiet(origin: Origin) -> np.ndarray:
    """Return the places that draw introductions instead of a walk, `[L]`.

    These are the silent places and those with no week to start from in
    the `start_window`.
    """
    return origin.silent | ~_start_ok(origin).any(axis=1)


def _start_weeks(origin: Origin) -> np.ndarray:
    """Return the week each place's walk starts from, `[L]`.

    It is the latest week a walk can start from (`_start_ok`). A quiet
    place (`_quiet`) starts at its latest complete week, after which its
    introductions are drawn.
    """
    ok = _start_ok(origin)
    latest = origin.start_nowcast - np.argmax(ok[:, ::-1], 1)
    return np.where(_quiet(origin), origin.start, latest)


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
        _clear_caches(task)
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
                    "delay_source": _label(origin, int(origin.fit.source[i])),
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
            _key(task),
            np.repeat(x0[:, None], n, axis=1),
            origin,
            start,
            n_all,
            origin.silent,
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
    delay_summary : pandas.DataFrame or None
        The delay of the last forecast (`chain.delay_summary`), which maps
        its final counts to what later releases show.
    """

    name: str = "randomwalknowcast"
    delay_summary: pd.DataFrame | None = field(
        init=False, default=None, repr=False
    )

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
            _start_weeks(origin),
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
        weeks = _timeline(origin, task)
        n_data, n_all, n = snap.n_weeks, len(weeks), task.n_samples
        n_loc = len(snap.locations)
        k_n = origin.size.shape[1]
        back = snap.cut_week(origin.o) - np.arange(n_data)
        # Weeks older than the fitted ones have nothing to come.
        k = np.clip(back, 0, k_n - 1)
        key_x, key_walk, key_part, key_ahead = jax.random.split(_key(task), 4)
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
            ),
            np.float64,
        )
        # Each sample walks on from its own nowcast of the start week.
        cw = snap.cut_week(origin.o)
        w0 = _start_weeks(origin)
        quiet = _quiet(origin)
        rows = np.arange(n_loc)
        lam, y = _walks(
            key_walk, nowcast[:, rows, w0].T, origin, w0, n_all, quiet
        )
        # Days of the cut-off week after the cut-off are drawn from the
        # walk's rate that week. A partial cut-off week's start week is the
        # week before it when that week meets the start rule, so the rate
        # is one step from each sample's nowcast of the week before, and
        # otherwise steps on from the latest earlier week that meets it.
        cut = int(snap.cut[origin.o])
        miss = int(np.flatnonzero(snap.week == cw).max()) - cut
        at_cw = np.clip(cw - w0 - 1, 0, n_all - 1)
        part = np.asarray(
            jax.random.poisson(
                key_part,
                jnp.asarray(lam[rows, :, at_cw] * miss / 7, jnp.float32),
            ),
            dtype=np.float64,
        )  # [L, S]
        # Each sample's completed cut-off week is the base for the weeks
        # after it, from which the walk goes on.
        done = nowcast[:, rows, cw].T + np.where(miss > 0, part, 0.0)
        h = n_all - cw - 1
        _, ahead = random_walks(
            jax.random.split(key_ahead, n_loc),
            jnp.asarray(done, jnp.float32),
            jnp.asarray(origin.sigma, jnp.float32),
            max(h, 1),
        )
        ahead = np.asarray(ahead, np.float64)[:, :, :h]
        values = np.zeros((n, n_loc, n_all))
        for loc in range(n_loc):
            s = int(w0[loc])
            if quiet[loc]:
                # A quiet place is its reported counts, then its
                # introductions, floored at the counts reported.
                values[:, loc, : s + 1] = origin.reported[loc, : s + 1]
                values[:, loc, s + 1 :] = y[loc, :, : n_all - s - 1]
                values[:, loc, :n_data] = np.maximum(
                    values[:, loc, :n_data], origin.reported[loc]
                )
                continue
            values[:, loc, :cw] = nowcast[:, loc, :cw]
            values[:, loc, cw] = done[loc]
            values[:, loc, cw + 1 :] = ahead[loc]
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
