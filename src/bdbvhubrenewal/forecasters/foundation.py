# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The pretrained Chronos-2 foundation model, on three layouts of the data.

`FoundationModel` gives the vintage's data to Chronos-2
(`amazon/chronos-2`) and reads the nowcast and the forecast off the same
sample trajectories.
Each layout is one registered model.

**Weeks and delay classes.** $R_{p,w}$ is the reported total (known and
hub-imputed onsets) of province $p$ and onset week $w$ at the vintage
$T$.
Each case of the delay line data has an onset week $w$ and a delay of
$d$ days, in class $k = \lfloor d / 7 \rfloor$ for $k < K$, with $K$ the
first class past `data.delays.DEFAULT_MAX_DELAY` days.
Negative delays go to class 0, and delays of $7K$ days or more are left
out.
$n_{p,w,k}$ counts the cases of cell $(w, k)$.
A cell is complete when its last possible report day, $w + 7k + 6$, is on
or before $T$, that is when $w + k \le s^*$, with $s^*$ the latest week
whose Saturday is at least six days before $T$.
Week $c = s^* - K + 1$ is the latest week with every cell complete.

**Layouts.** Each province is one group of series, which Chronos-2's group
attention reads together.

- `complete` (`fm_complete`): the target is $\log(1 + R_{p,w})$ for
  $w \le c$, forecast for every later week.
- `delay_cov` (`fm_delay_cov`): as `complete`, with $\log(1 + n_{p,w,k})$
  for each class $k$ as a covariate.
  The covariates are complete up to $c$ and, past it, known for the cells
  already complete and missing for the rest, so Chronos-2 masks the cells
  still to be reported.
- `delay_series` (`fm_delay_series`): each class on the report clock, as
  $K$ target series: series $k$ at report week $s$ holds
  $\log(1 + n_{p, s - k, k})$, complete up to $s^*$ and forecast past it.
  A class is told apart from the others by its own history only, as
  Chronos-2 scales each series to zero mean and unit variance, which
  maps a constant label to zero.

**Trajectories.** Chronos-2 gives quantiles for every week ahead from one
call.
Each trajectory draws one uniform $u$ for each series and reads every
week's quantile function at that $u$ (linear between the model's 21
levels, flat beyond them).
So a trajectory keeps its quantile level across weeks, and is
independent of the other series' trajectories.
A draw $z$ is the count $\max(\mathrm{round}(e^{z} - 1), 0)$.
For `complete` and `delay_cov` sample $i$ of week $w > c$ is
$\max(\hat y^{(i)}_{p,w}, R_{p,w})$; weeks up to $c$ are $R_{p,w}$.
For `delay_series` it is the reported total plus the new reports of the
incomplete cells,

$$
y^{(i)}_{p,w} = R_{p,w}
+ \sum_{k : w + k > s^*} \max\bigl(\hat n^{(i)}_{p,w,k} - n_{p,w,k}, 0\bigr),
$$

with $\hat n^{(i)}_{p,w,k}$ the draw of series $k$ at report week
$w + k$.
So no sample is below the count already reported.
National is the sum of provinces by sample, and health zones are split
from the provinces by `forecasters.zones`.

**Reported counts.** The release backtest maps a model's final counts to
what later releases show through the model's delay.
These models take the daily chain ladder delay fitted at the vintage
(`forecasters.chain`), as `randomwalknowcast` does.

Chronos-2 is an optional dependency (`uv sync --extra chronos`).
Without it the models raise `ModelUnavailableError` and the harness
skips them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY
from bdbvhubrenewal.forecasters.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from bdbvhubrenewal.forecasters.chain import delay_summary
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk

__all__ = [
    "CHRONOS_MODEL",
    "CHRONOS_REVISION",
    "LAYOUTS",
    "N_DELAYS",
    "DelayTriangle",
    "FoundationModel",
    "Sampler",
    "chronos_sampler",
    "delay_triangle",
    "quantile_draws",
    "report_series",
]

N_DELAYS: Final = DEFAULT_MAX_DELAY // 7 + 1
"""Weekly delay classes $K$, the first past `DEFAULT_MAX_DELAY` days."""

LAYOUTS: Final = ("complete", "delay_series", "delay_cov")
"""The ways the data are given to Chronos-2."""

CHRONOS_MODEL: Final = "amazon/chronos-2"
"""The Hugging Face checkpoint."""

CHRONOS_REVISION: Final = "29ec3766d36d6f73f0696f85560a422f50e8498c"
"""The checkpoint's commit, pinned so runs are reproducible."""

Series = Mapping[str, Any]
"""One group for Chronos-2: `target` `[V, S]` on the log1p scale, and
optionally `past_covariates` (name to `[S]`) and `future_covariates`
(name to `[H]`, NaN where unknown)."""

Sampler = Callable[[Sequence[Series], int, int, int], list[np.ndarray]]
"""Draws trajectories: `(groups, horizon, n, seed)` to one
`[n, V, horizon]` array per group, on the log1p scale."""

_WEEK = dt.timedelta(days=7)


@dataclass(frozen=True)
class DelayTriangle:
    """Cases by province, onset week and weekly delay class.

    Attributes
    ----------
    counts : numpy.ndarray
        Cases reported by the vintage, `[P, W, K]`.
    last : int
        Index of $s^*$, the latest report week whose cells are complete.
    """

    counts: np.ndarray
    last: int


def _last_complete(vintage: str, weeks: Sequence[str]) -> int:
    """Index of $s^*$ in `weeks`."""
    complete = epiweek_end(as_date(vintage) - dt.timedelta(days=12))
    return (complete - as_date(weeks[0])).days // 7


def delay_triangle(
    delays: pd.DataFrame,
    vintage: str,
    locations: Sequence[str],
    weeks: Sequence[str],
    n_delays: int = N_DELAYS,
) -> DelayTriangle:
    """Count the vintage's cases by onset week and weekly delay class.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one vintage, with columns `location`,
        `onset_week` and `delay` (days).
    vintage : str
        The vintage date $T$.
    locations : sequence of str
        Provinces, in the order of the output.
    weeks : sequence of str
        Consecutive onset weeks, ISO Saturdays. Cases in other weeks are
        left out.
    n_delays : int
        Delay classes $K$.

    Returns
    -------
    DelayTriangle
        The counts and the latest complete report week.
    """
    first = pd.Timestamp(weeks[0])
    place = delays["location"].map({c: i for i, c in enumerate(locations)})
    week = (pd.to_datetime(delays["onset_week"]) - first).dt.days.to_numpy()
    week = week // 7
    k = np.maximum(delays["delay"].to_numpy(), 0) // 7
    keep = (
        place.notna().to_numpy()
        & (k < n_delays)
        & (week >= 0)
        & (week < len(weeks))
    )
    counts = np.zeros((len(locations), len(weeks), n_delays))
    np.add.at(
        counts,
        (place.to_numpy()[keep].astype(int), week[keep], k[keep]),
        1.0,
    )
    return DelayTriangle(counts, _last_complete(vintage, weeks))


def report_series(triangle: DelayTriangle) -> np.ndarray:
    """Put each delay class on the report clock, up to $s^*$.

    Parameters
    ----------
    triangle : DelayTriangle
        The counts.

    Returns
    -------
    numpy.ndarray
        `[P, K, s^* + 1]`: class $k$ at report week $s$ holds
        $n_{p, s - k, k}$, and 0 for $s < k$.
    """
    n_loc, _, n_delays = triangle.counts.shape
    length = triangle.last + 1
    out = np.zeros((n_loc, n_delays, length))
    for k in range(n_delays):
        out[:, k, k:] = triangle.counts[:, : length - k, k]
    return out


def quantile_draws(
    levels: np.ndarray, values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Read quantile functions at uniform draws.

    Linear between the levels and flat beyond them.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    values : numpy.ndarray
        Quantiles at those levels, `[..., Q]`. They are sorted first, so
        crossed quantiles give a valid function.
    u : numpy.ndarray
        Uniform draws, the shape of `values` less its last axis.

    Returns
    -------
    numpy.ndarray
        The draws, the shape of `u`.
    """
    values = np.sort(values, axis=-1)
    hi = np.clip(np.searchsorted(levels, u), 1, len(levels) - 1)
    lo = hi - 1
    weight = np.clip((u - levels[lo]) / (levels[hi] - levels[lo]), 0.0, 1.0)
    at_lo = np.take_along_axis(values, lo[..., None], axis=-1)[..., 0]
    at_hi = np.take_along_axis(values, hi[..., None], axis=-1)[..., 0]
    return np.asarray(at_lo + weight * (at_hi - at_lo))


@cache
def _chronos() -> Any:
    import torch
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )


def chronos_sampler(
    groups: Sequence[Series], horizon: int, n_samples: int, seed: int
) -> list[np.ndarray]:
    """Draw trajectories from Chronos-2's quantiles, a level per series.

    Runs on the GPU when torch sees one.

    Parameters
    ----------
    groups : sequence of Series
        One group of series per province.
    horizon : int
        Weeks to draw.
    n_samples : int
        Trajectories.
    seed : int
        Seed for the uniform draws.

    Returns
    -------
    list of numpy.ndarray
        Draws, `[n_samples, V, horizon]` per group.

    Raises
    ------
    ModelUnavailableError
        If Chronos-2 is not installed.
    """
    try:
        pipeline = _chronos()
    except ImportError as err:
        raise ModelUnavailableError(
            "The Chronos-2 models need the extra chronos: "
            "uv sync --extra chronos."
        ) from err
    levels = np.asarray(pipeline.quantiles, dtype=float)
    rng = np.random.default_rng(seed)
    quantiles, _ = pipeline.predict_quantiles(
        [dict(g) for g in groups],
        prediction_length=horizon,
        quantile_levels=levels.tolist(),
    )
    out = []
    for q in quantiles:
        q = q.float().cpu().numpy()  # [V, horizon, Q]
        u = rng.uniform(size=(n_samples, q.shape[0], 1))
        out.append(
            quantile_draws(
                levels,
                np.broadcast_to(q, (n_samples, *q.shape)),
                np.broadcast_to(u, (n_samples, *q.shape[:2])),
            )
        )
    return out


def _counts(draws: np.ndarray) -> np.ndarray:
    """Return the counts of draws on the log1p scale."""
    return np.maximum(np.round(np.expm1(draws)), 0.0)


@dataclass(frozen=True)
class View:
    """What a vintage shows, on a grid of onset weeks.

    Attributes
    ----------
    weeks : list of str
        Consecutive onset weeks, ISO Saturdays.
    reported : numpy.ndarray
        Reported totals $R_{p,w}$, `[P, W]`.
    counts : numpy.ndarray or None
        Cases by delay class $n_{p,w,k}$, `[P, W, K]`, or None if the
        vintage has no delay line data.
    last : int
        Index of $s^*$, the latest report week whose cells are complete.
    """

    weeks: list[str]
    reported: np.ndarray
    counts: np.ndarray | None
    last: int

    @property
    def n_context(self) -> int:
        """Weeks up to and including $c$, the last with every cell complete."""
        return self.last - N_DELAYS + 2

    def cells(self) -> np.ndarray:
        r"""Return $\log(1 + n_{p,w,k})$, NaN where not yet complete.

        Returns
        -------
        numpy.ndarray
            `[P, W, K]`.
        """
        assert self.counts is not None
        ages = np.arange(len(self.weeks))[:, None] + np.arange(N_DELAYS)
        return np.where(ages <= self.last, np.log1p(self.counts), np.nan)


def vintage_view(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    vintage: str,
    locations: Sequence[str],
    last_week: str,
) -> View:
    """Lay out a vintage's totals and delay classes by onset week.

    Parameters
    ----------
    target : pandas.DataFrame
        The vintage's target rows, with `location`, `target_end_date` and
        `total`.
    delays : pandas.DataFrame or None
        The vintage's delay line data.
    vintage : str
        The vintage date $T$.
    locations : sequence of str
        Provinces, in the order of the output.
    last_week : str
        Last onset week of the grid, an ISO Saturday.

    Returns
    -------
    View
        Weeks from the first week of `target` to `last_week`.
    """
    provinces = target.loc[target["location"].isin(locations)]
    first = as_date(str(provinces["target_end_date"].min()))
    n_weeks = (as_date(last_week) - first).days // 7 + 1
    weeks = [(first + i * _WEEK).isoformat() for i in range(n_weeks)]
    reported = (
        provinces.pivot_table(
            index="location",
            columns="target_end_date",
            values="total",
            aggfunc="sum",
        )
        .reindex(index=list(locations), columns=weeks)
        .fillna(0.0)
        .to_numpy()
    )
    counts = None
    if delays is not None and not delays.empty:
        counts = delay_triangle(delays, vintage, locations, weeks).counts
    return View(weeks, reported, counts, _last_complete(vintage, weeks))


def cov_groups(view: View) -> list[Series]:
    """Return the `delay_cov` groups of a view, split at $c$.

    Parameters
    ----------
    view : View
        A vintage with delay line data.

    Returns
    -------
    list of Series
        One group per province: the totals to $c$ as target, and each
        delay class as a covariate, past to $c$ and future after it.
    """
    start = view.n_context
    cells = view.cells()
    return [
        {
            "target": np.log1p(r[None, :start]),
            "past_covariates": {
                f"delay_{k}": c[:start, k] for k in range(N_DELAYS)
            },
            "future_covariates": {
                f"delay_{k}": c[start:, k] for k in range(N_DELAYS)
            },
        }
        for r, c in zip(view.reported, cells, strict=True)
    ]


@dataclass
class FoundationModel:
    """Chronos-2 on one layout of the data.

    Attributes
    ----------
    name : str
        Model name.
    layout : str
        One of `LAYOUTS`.
    sample : Sampler
        Draws the trajectories, Chronos-2 by default.
    delay_summary : pandas.DataFrame or None
        The chain ladder delay of the last forecast, which maps its final
        counts to what later releases show.
    """

    name: str
    layout: str
    sample: Sampler = field(default=chronos_sampler, repr=False)
    delay_summary: pd.DataFrame | None = field(
        init=False, default=None, repr=False
    )
    needs_snapshots: ClassVar[bool] = True

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return joint trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `delays` hold the line
            data of the vintage and its `daily_history` the release
            snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If a delay layout's task has no delay line data, the task has
            no release snapshots, there are fewer than two complete weeks,
            or Chronos-2 is not installed.
        """
        if self.layout != "complete" and (
            task.delays is None or task.delays.empty
        ):
            raise ModelUnavailableError(
                f"{self.name} needs delay line data, and the task for "
                f"{task.vintage} has none."
            )
        origin = RandomWalk().origin_at(task, errors=False)
        self.delay_summary = delay_summary(origin, self.name, task.vintage)
        view = vintage_view(
            task.target,
            task.delays,
            task.vintage,
            task.locations,
            max(task.target_end_dates),
        )
        seed = int(task.rng(self.name).integers(0, 2**31 - 1))
        if self.layout == "delay_series":
            values = self._series(view, task.n_samples, seed)
        else:
            values = self._totals(view, task, seed)
        col = {w: i for i, w in enumerate(view.weeks)}
        targets = list(task.target_end_dates)
        picked = values[:, :, [col[w] for w in targets]]  # [S, P, T]
        n, n_loc = task.n_samples, len(task.locations)
        return pd.DataFrame(
            {
                "location": np.repeat(list(task.locations), n * len(targets)),
                "target_end_date": np.tile(targets, n * n_loc),
                "sample": np.tile(
                    np.repeat(np.arange(1, n + 1), len(targets)), n_loc
                ),
                "value": picked.transpose(1, 0, 2).reshape(-1),
            }
        )[list(SAMPLE_COLUMNS)]

    def _totals(self, view: View, task: ForecastTask, seed: int) -> np.ndarray:
        """Trajectories of the `complete` and `delay_cov` layouts."""
        start = view.n_context
        if start < 2:
            raise ModelUnavailableError(
                f"{self.name} needs two complete weeks, and the task for "
                f"{task.vintage} has {max(start, 0)}."
            )
        horizon = len(view.weeks) - start
        groups: list[Series] = (
            cov_groups(view)
            if self.layout == "delay_cov"
            else [{"target": np.log1p(r[None, :start])} for r in view.reported]
        )
        draws = self.sample(groups, horizon, task.n_samples, seed)
        drawn = _counts(np.stack([d[:, 0] for d in draws], axis=1))
        reported = view.reported
        values = np.broadcast_to(reported, (task.n_samples, *reported.shape))
        values = values.copy()
        values[:, :, start:] = np.maximum(drawn, reported[:, start:])
        return values

    def _series(self, view: View, n_samples: int, seed: int) -> np.ndarray:
        """Trajectories of the `delay_series` layout."""
        assert view.counts is not None
        tri = DelayTriangle(view.counts, view.last)
        n_weeks = len(view.weeks)
        horizon = n_weeks + N_DELAYS - 2 - tri.last
        groups: list[Series] = [
            {"target": np.log1p(s)} for s in report_series(tri)
        ]
        draws = self.sample(groups, horizon, n_samples, seed)
        drawn = _counts(np.stack(draws, axis=1))  # [S, P, K, H]
        reported = view.reported
        values = np.broadcast_to(reported, (n_samples, *reported.shape))
        values = values.copy()
        for k in range(N_DELAYS):
            for w in range(max(tri.last - k + 1, 0), n_weeks):
                new = drawn[:, :, k, w + k - tri.last - 1]
                values[:, :, w] += np.maximum(new - tri.counts[:, w, k], 0.0)
        return values


for _name, _layout, _text in (
    ("fm_complete", "complete", "the reported totals of complete weeks"),
    (
        "fm_delay_series",
        "delay_series",
        "each delay class on the report clock",
    ),
    (
        "fm_delay_cov",
        "delay_cov",
        "the totals, with each delay class as a masked covariate",
    ),
):
    register_model(
        _name,
        partial(FoundationModel, _name, _layout),
        f"Chronos-2 on {_text}, nowcast and forecast from the same "
        "trajectories.",
        fitted=True,
    )
