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
negative binomial new reports on the remaining count minus Poisson
removals of the count reported (`obs`).
Its walk is a local level on $\log(1 + y)$. Each sample starts from its
nowcast of the latest full week at least `Settings.start_share` reported
with at least `Settings.start_count` cases (`_start_weeks`). Each later
week up to the one holding the cut-off is not nowcast: its reported count
updates the level as an observation weighted by how complete the week is
(`_filter`), and its value is a Poisson draw from a level path drawn back
from the cut-off week (`_draw_path`), floored at the count reported.
The walk forecasts on from the cut-off week's level.

The spread of the walk is the standard deviation of the weekly changes of
$\log(1 + y)$ in reported counts, over the weeks ending at the start of
`randomwalk`.
The delay, the observation model and the walk spread each use their own
estimate where a place has at least 10 cases a week, and otherwise the
pooled estimate of the nearest parent that does (`pooling`).
A place with no case in the latest `walk_weeks` weeks is silent. It is
exactly 0 until its first introduction, which comes each week with its
pooled probability times the share of the week not yet reported
(`_intro_from`), and then walks from a count drawn from Poisson(1) given
at least 1.
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
from scipy import stats

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
    key: jax.Array,
    x0: np.ndarray,
    sigma: np.ndarray,
    silent: np.ndarray,
    intro: np.ndarray,
) -> np.ndarray:
    """Return walk counts for every place from `x0` of shape `[L, S]`.

    Each place walks with spread `sigma` `[L]`. A `silent` `[L]` place is
    0 until its first introduction, drawn week by week with probability
    `intro` `[L, H]`. The introduction brings a count drawn from
    Poisson(1) given at least 1, and the place walks from it.
    """
    h = intro.shape[1]
    k_walk, k_intro, k_count = jax.random.split(key, 3)
    u = np.asarray(jax.random.uniform(k_count, x0.shape), np.float64)
    first_count = stats.poisson.ppf(np.exp(-1.0) + u * -np.expm1(-1.0), 1.0)
    x0 = np.where(silent[:, None], first_count, x0)
    keys = jax.random.split(k_walk, x0.shape[0])
    _, drawn = jax.vmap(lambda k, x, s: random_walk(k, x, s, h))(
        keys,
        jnp.asarray(x0, jnp.float32),
        jnp.asarray(sigma, jnp.float32),
    )
    y: np.ndarray = np.asarray(drawn, dtype=np.float64)
    hit = (
        np.asarray(jax.random.uniform(k_intro, (*x0.shape, h)))
        < (intro[:, None, :])
    )
    first = np.where(hit.any(axis=2), np.argmax(hit, axis=2), h)
    first = np.where(silent[:, None], first, 0)
    lag = np.arange(h)[None, None, :] - first[:, :, None]
    # A silent place has its introduced count in its first week and walks
    # from it after.
    step = np.where(silent[:, None, None], lag - 1, lag)
    walked = np.take_along_axis(y, np.clip(step, 0, h - 1), axis=2)
    out = np.where(step >= 0, walked, 0.0)
    return np.where(silent[:, None, None] & (lag == 0), x0[:, :, None], out)


def _start_weeks(origin: Origin) -> np.ndarray:
    """Return the week each place's walk starts from, `[L]`.

    It is the latest full week whose share reported is at least
    `Settings.start_share` and whose reported count is at least
    `Settings.start_count`, else the first week. A silent place starts at
    its latest complete week, after which its introductions are drawn.
    """
    settings, full = origin.settings, origin.start_nowcast
    ok = (origin.complete[:, : full + 1] >= settings.start_share) & (
        origin.reported[:, : full + 1] >= settings.start_count
    )
    latest = np.where(ok.any(axis=1), full - np.argmax(ok[:, ::-1], 1), 0)
    return np.where(origin.silent, origin.start, latest)


def _filter(
    start: np.ndarray, origin: Origin, w0: np.ndarray, cw: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Filter the walk's level on log(1 + y) from week `w0` to week `cw`.

    The level starts at `start` `[L, S]` in week `w0` with no variance,
    and steps with variance $\sigma^2$ a week. Week $u$ after `w0` with
    share reported $F_u > 0$ observes it as $\log(1 + y_u / F_u)$ with
    variance $1 / (1 + y_u) + (1 - F_u) / F_u$, Poisson noise plus the
    share still to come.

    Returns
    -------
    tuple of numpy.ndarray
        The filtered level `[L, S, W]` and its variance `[L, W]`, held
        at the start from `w0` back.
    """
    n_loc, n_data = origin.reported.shape
    level = np.repeat(start[:, :, None], n_data, axis=2)
    var = np.zeros((n_loc, n_data))
    for u in range(n_data):
        after = (u > w0) & (u <= cw)
        share, y = origin.complete[:, u], origin.reported[:, u]
        prior = var[:, u - 1] + origin.sigma**2 if u else var[:, 0]
        noise = 1.0 / (1.0 + y) + (1.0 - share) / np.maximum(share, 1e-12)
        gain = np.where(after & (share > 0), prior / (prior + noise), 0.0)
        seen = np.log1p(y / np.maximum(share, 1e-12))
        prev = level[:, :, u - 1] if u else level[:, :, 0]
        step = prev + gain[:, None] * (seen[:, None] - prev)
        level[:, :, u] = np.where(after[:, None], step, level[:, :, u])
        var[:, u] = np.where(after, (1.0 - gain) * prior, var[:, u])
    return level, var


def _draw_path(
    key: jax.Array,
    level: np.ndarray,
    var: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray,
    cw: int,
) -> np.ndarray:
    """Draw one level path per sample from week `cw` back to week `w0`.

    Each sample's level at `cw` is drawn from the filter, and each earlier
    week given the week after it (backward sampling), so the weeks after
    `w0` and the walk on from `cw` form one trajectory.
    """
    drawn = level.copy()
    z = np.asarray(jax.random.normal(key, level.shape), np.float64)
    drawn[:, :, cw] = (
        level[:, :, cw] + np.sqrt(var[:, cw])[:, None] * z[:, :, cw]
    )
    for u in range(cw - 1, -1, -1):
        inside = (u > w0)[:, None]
        pull = var[:, u] / (var[:, u] + sigma**2)
        mean = level[:, :, u] + pull[:, None] * (
            drawn[:, :, u + 1] - level[:, :, u]
        )
        sd = np.sqrt(np.maximum(var[:, u] * (1.0 - pull), 0.0))[:, None]
        drawn[:, :, u] = np.where(
            inside, mean + sd * z[:, :, u], level[:, :, u]
        )
    return drawn


def _intro_from(origin: Origin, start: np.ndarray, h: int) -> np.ndarray:
    """Return each place's weekly introduction probability after `start`.

    Week `start + 1 + i` of a silent place has probability $q$ times the
    share of the week not yet reported, so an introduction is likelier in
    recent weeks, whose cases may still be reported, and has probability
    $q$ in weeks without data.
    """
    n_data = origin.complete.shape[1]
    u = start[:, None] + 1 + np.arange(h)[None, :]
    seen = np.take_along_axis(
        np.pad(origin.complete, ((0, 0), (0, h + 1))), u, axis=1
    )
    seen = np.where(u < n_data, np.clip(seen, 0.0, 1.0), 0.0)
    return origin.intro[:, None] * (1.0 - seen)


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
        y = _walks(
            _key(task),
            np.repeat(x0[:, None], n, axis=1),
            origin.sigma,
            origin.silent,
            _intro_from(origin, start, n_all),
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
        assert origin.removal is not None
        weeks = _timeline(origin, task)
        n_data, n_all, n = snap.n_weeks, len(weeks), task.n_samples
        n_loc = len(snap.locations)
        k_n = origin.size.shape[1]
        back = snap.cut_week(origin.o) - np.arange(n_data)
        k = np.clip(back, 0, k_n - 1)
        # Weeks older than the fitted ones are final, with no removals.
        b = np.where(back < k_n, origin.removal[:, k], 0.0)
        key_x, key_walk, key_filter = jax.random.split(_key(task), 3)
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
        # The walk is a local level on log(1 + y). It starts from the
        # nowcast of the latest full week with enough reported, and each
        # later week updates it with its reported count, weighted by how
        # complete the week is. The walk forecasts from the last update.
        cw = snap.cut_week(origin.o)
        w0 = _start_weeks(origin)
        level, var = _filter(
            np.log1p(nowcast[:, np.arange(n_loc), w0]).T,
            origin,
            w0,
            cw,
        )
        key_path, key_count = jax.random.split(key_filter)
        drawn = _draw_path(key_path, level, var, origin.sigma, w0, cw)
        rate = np.expm1(np.maximum(drawn, 0.0))
        filled = np.asarray(
            jax.random.poisson(key_count, jnp.asarray(rate, jnp.float32)),
            dtype=np.float64,
        )  # [L, S, W]
        y = _walks(
            key_walk,
            rate[:, :, cw],
            origin.sigma,
            origin.silent,
            _intro_from(origin, w0, n_all),
        )
        values = np.zeros((n, n_loc, n_all))
        for loc in range(n_loc):
            s = int(w0[loc])
            if origin.silent[loc]:
                # A silent place is its reported 0s, then its introductions.
                values[:, loc, : s + 1] = origin.reported[loc, : s + 1]
                values[:, loc, s + 1 :] = y[loc, :, : n_all - s - 1]
                continue
            values[:, loc, : s + 1] = nowcast[:, loc, : s + 1]
            for u in range(s + 1, cw + 1):
                values[:, loc, u] = np.maximum(
                    filled[loc, :, u], origin.reported[loc, u]
                )
            values[:, loc, cw + 1 :] = y[loc, :, : n_all - cw - 1]
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
