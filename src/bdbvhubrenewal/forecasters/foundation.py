# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The pretrained Chronos-2 foundation model on the hub target's releases.

See the Foundation models methods page for the model.
Each pair of a source (`SOURCES`) and a layout (`LAYOUTS`) is one
registered model, `fm_chronos_<source>_<layout>`.
`hub_triangle` lays out the hub target by place, time and release lag,
`FoundationModel` gives it to Chronos-2 as one group of series, and
`chronos_sampler` draws trajectories from Chronos-2's quantiles.
Chronos-2 is an optional dependency (`uv sync --extra chronos`).
Without it the models raise `ModelUnavailableError` and the harness skips
them.
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
from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from bdbvhubrenewal.forecasters.chain import delay_summary
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalk,
    _quiet,
    _start_weeks,
)

__all__ = [
    "CHRONOS_MODEL",
    "CHRONOS_REVISION",
    "LAYOUTS",
    "MAX_LAG",
    "SOURCES",
    "FoundationModel",
    "HubTriangle",
    "Sampler",
    "chronos_sampler",
    "hub_triangle",
    "quantile_draws",
]

SOURCES: Final = ("weekly", "daily")
"""The hub target's releases by onset week (`target.csv`) or onset day
(`daily.csv`)."""

LAYOUTS: Final = ("complete", "ragged")
"""The ways the releases are given to Chronos-2."""

MAX_LAG: Final = {"weekly": 4, "daily": 28}
"""Release lags $, in the source's steps, after which a count is taken as
final."""

CHRONOS_MODEL: Final = "amazon/chronos-2"
"""The Hugging Face checkpoint."""

CHRONOS_REVISION: Final = "29ec3766d36d6f73f0696f85560a422f50e8498c"
"""The checkpoint's commit, pinned so runs are reproducible."""

Series = Mapping[str, Any]
"""One group for Chronos-2: `target` `[V, S]`, and optionally
`past_covariates` (name to `[S]`) and `future_covariates` (name to
`[H]`, NaN where unknown)."""

Sampler = Callable[[Sequence[Series], int, int, int], list[np.ndarray]]
"""Draws trajectories: `(groups, horizon, n, seed)` to one
`[n, V, horizon]` array per group."""


@dataclass(frozen=True)
class HubTriangle:
    """The hub target by place, time step and release lag.

    Attributes
    ----------
    times : list of str
        Consecutive weeks (ISO Saturdays) or days.
    final : numpy.ndarray
        The count of each place and step at the vintage, `[L, T]`.
    partial : numpy.ndarray
        The count of each place and step at release lag  < D$, from the
        release whose cut-off is $ steps after it, NaN where no release
        is, `[L, T, D]`.
    last : int
        Index of the step holding the vintage's cut-off.
    """

    times: list[str]
    final: np.ndarray
    partial: np.ndarray
    last: int

    @property
    def n_context(self) -> int:
        """Steps whose count at the vintage is at lag $ or more."""
        return self.last - int(self.partial.shape[2]) + 1


def _step(source: str, day: str) -> dt.date:
    """Return the step holding a day: its week's Saturday, or the day."""
    return epiweek_end(day) if source == "weekly" else as_date(day)


def hub_triangle(
    rows: pd.DataFrame,
    source: str,
    vintage: str,
    places: Sequence[str],
    last_time: str,
) -> HubTriangle:
    """Lay out the hub target of every release by place, step and lag.

    Parameters
    ----------
    rows : pandas.DataFrame
        The releases on or before `vintage`. Weekly rows have
        `target_end_date` and `total`; daily rows have `onset_date`,
        `observed` and `imputed`. Both have `vintage`, `location` and
        `cutoff`.
    source : str
        One of `SOURCES`.
    vintage : str
        The vintage.
    places : sequence of str
        Places, in the order of the output.
    last_time : str
        Last step of the grid.

    Returns
    -------
    HubTriangle
        Steps from the first in `rows` to `last_time`.
    """
    width = 7 if source == "weekly" else 1
    time_col = "target_end_date" if source == "weekly" else "onset_date"
    rows = rows.loc[rows["location"].isin(places)]
    value = (
        rows["total"]
        if source == "weekly"
        else rows["observed"] + rows["imputed"]
    )
    first = _step(source, str(rows[time_col].min()))
    n_times = (as_date(last_time) - first).days // width + 1
    times = [
        (first + dt.timedelta(days=width * i)).isoformat()
        for i in range(n_times)
    ]
    place = rows["location"].map({c: i for i, c in enumerate(places)})
    days = (pd.to_datetime(rows[time_col]) - pd.Timestamp(first)).dt.days
    step = days.to_numpy() // width
    lags = MAX_LAG[source]
    partial = np.full((len(places), n_times, lags), np.nan)
    final = np.zeros((len(places), n_times))
    last = 0
    for release in sorted(set(rows["vintage"])):
        mine = (rows["vintage"] == release).to_numpy()
        cutoff = str(rows.loc[mine, "cutoff"].iloc[0])
        at = (_step(source, cutoff) - first).days // width
        counts = np.zeros((len(places), n_times))
        keep = mine & (step >= 0) & (step < n_times)
        np.add.at(
            counts,
            (place.to_numpy()[keep].astype(int), step[keep]),
            value.to_numpy()[keep],
        )
        for d in range(lags):
            if 0 <= at - d < n_times:
                partial[:, at - d, d] = counts[:, at - d]
        if release == vintage:
            final, last = counts, at
    return HubTriangle(times, final, partial, last)


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
    """Draw trajectories from Chronos-2's quantiles.

    Every series and step draws its own uniform level, so draws are
    independent across series and steps. Runs on the GPU when torch sees
    one.

    Parameters
    ----------
    groups : sequence of Series
        The groups of series.
    horizon : int
        Steps to draw.
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
        u = rng.uniform(size=(n_samples, *q.shape[:2]))
        out.append(
            quantile_draws(levels, np.broadcast_to(q, (n_samples, *q.shape)), u)
        )
    return out


def _group(
    tri: HubTriangle, layout: str, places: Sequence[str], start: int
) -> Series:
    """Return the one block of every place's series, split at `start`."""
    target = tri.final[:, :start]
    if layout == "complete":
        return {"target": target}
    lags = tri.partial.shape[2]
    names = [f"{p}_lag_{d}" for p in places for d in range(lags)]
    cells = tri.partial.transpose(0, 2, 1).reshape(len(names), -1)
    return {
        "target": target,
        "past_covariates": dict(zip(names, cells[:, :start], strict=True)),
        "future_covariates": dict(zip(names, cells[:, start:], strict=True)),
    }


def _walk_start(origin: Any) -> str:
    """Return the earliest week `randomwalk` starts a walk from."""
    start = _start_weeks(origin)
    walked = ~_quiet(origin)
    first = start[walked].min() if walked.any() else start.min()
    return str(origin.snap.week_ends[int(first)])


def _rows(*tables: pd.DataFrame | None) -> pd.DataFrame | None:
    """Stack the tables that are there, None if none is."""
    there = [t for t in tables if t is not None and not t.empty]
    return pd.concat(there, ignore_index=True) if there else None


@dataclass
class FoundationModel:
    """Chronos-2 on one source and layout of the hub target's releases.

    Attributes
    ----------
    name : str
        Model name.
    source : str
        One of `SOURCES`.
    layout : str
        One of `LAYOUTS`.
    sample : Sampler
        Draws the trajectories, Chronos-2 by default.
    delay_summary : pandas.DataFrame or None
        The chain ladder delay of the last forecast, which maps its final
        counts to what later releases show.
    """

    name: str
    source: str
    layout: str
    sample: Sampler = field(default=chronos_sampler, repr=False)
    delay_summary: pd.DataFrame | None = field(
        init=False, default=None, repr=False
    )
    needs_snapshots: ClassVar[bool] = True
    reads_zones: ClassVar[bool] = True

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return trajectories of every hub target in the task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `target_history` and
            `zone_history` hold the weekly releases, its `daily_history`
            and `zone_daily_history` the daily ones, and its `zone_codes`
            the zones.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`, for
            the country, the provinces and their zones, each its own
            series.

        Raises
        ------
        ModelUnavailableError
            If the task has no releases of the source, no release
            snapshots, fewer than two steps of context, or Chronos-2 is
            not installed.
        """
        weekly = self.source == "weekly"
        base = task.target_history if weekly else task.daily_history
        zone = task.zone_history if weekly else task.zone_daily_history
        rows = _rows(base, zone)
        if base is None or base.empty or rows is None:
            raise ModelUnavailableError(
                f"{self.name} needs the {self.source} releases, and the task "
                f"for {task.vintage} has none."
            )
        origin = RandomWalk().origin_at(task, errors=False)
        self.delay_summary = delay_summary(origin, self.name, task.vintage)
        zones = tuple(
            z for z in task.zone_codes if province_of(z) in task.locations
        )
        places = (NATIONAL, *task.locations, *zones)
        weeks = list(task.target_end_dates)
        tri = hub_triangle(rows, self.source, task.vintage, places, weeks[-1])
        start = tri.n_context
        if self.layout == "complete":
            start = tri.times.index(_walk_start(origin)) + 1
        if start < 2:
            raise ModelUnavailableError(
                f"{self.name} needs two steps of context, and the task for "
                f"{task.vintage} has {max(start, 0)}."
            )
        horizon = len(tri.times) - start
        seed = int(task.rng(self.name).integers(0, 2**31 - 1))
        group = _group(tri, self.layout, places, start)
        draws = self.sample([group], horizon, task.n_samples, seed)[0]
        final = tri.final
        steps = np.broadcast_to(final, (task.n_samples, *final.shape)).copy()
        drawn = np.maximum(np.round(draws), 0.0)
        steps[:, :, start:] = np.maximum(drawn, final[:, start:])
        # The hub's target is weekly, so the daily source is summed to weeks.
        week_of = np.array([_step("weekly", t).isoformat() for t in tri.times])
        values = np.stack(
            [steps[:, :, week_of == w].sum(axis=2) for w in weeks], axis=2
        )
        weekly_rows = _rows(task.target_history, task.zone_history)
        assert weekly_rows is not None
        reported = hub_triangle(
            weekly_rows, "weekly", task.vintage, places, weeks[-1]
        )
        at = [reported.times.index(w) for w in weeks]
        values = np.maximum(values, reported.final[:, at])  # [S, L, W]
        n, n_loc = task.n_samples, len(places)
        return pd.DataFrame(
            {
                "location": np.repeat(list(places), n * len(weeks)),
                "target_end_date": np.tile(weeks, n * n_loc),
                "sample": np.tile(
                    np.repeat(np.arange(1, n + 1), len(weeks)), n_loc
                ),
                "value": values.transpose(1, 0, 2).reshape(-1),
            }
        )[list(SAMPLE_COLUMNS)]


for _source in SOURCES:
    for _layout in LAYOUTS:
        _name = f"fm_chronos_{_source}_{_layout}"
        register_model(
            _name,
            partial(FoundationModel, _name, _source, _layout),
            f"Chronos-2 on the hub target's {_source} releases, {_layout} "
            "layout, every target in one block.",
            fitted=True,
        )
