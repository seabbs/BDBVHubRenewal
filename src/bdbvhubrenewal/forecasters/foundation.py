# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The pretrained Chronos-2 foundation model on the hub target's releases.

See the Foundation models methods page for the model.
Each pair of a source (`SOURCES`) and a layout (`LAYOUTS`) is one
registered model, `fm_chronos_<source>_<layout>`.
`hub_triangle` lays out the hub target by place, time and release lag,
`FoundationModel` gives it to Chronos-2 as one group of series, and
`chronos_sampler` draws trajectories from Chronos-2's quantiles.
Chronos-2 is an optional dependency group (`uv sync --group chronos`).
Without it the models raise `ModelUnavailableError` and the harness skips
them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any, Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk

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
    "lag_shares",
    "quantile_draws",
]

SOURCES: Final = ("weekly", "daily")
"""The hub target's releases by onset week (`target.csv`) or onset day
(`daily.csv`)."""

LAYOUTS: Final = ("complete", "ragged")
"""The ways the releases are given to Chronos-2."""

MAX_LAG: Final = {"weekly": 4, "daily": 28}
"""Release lag $D$, in the source's steps, from which a count is taken as
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
        The count of each place and step at the latest release, `[L, T]`.
    partial : numpy.ndarray
        The count of each place and step at release lag $d < D$, from the
        release whose cut-off is $d$ steps after it, NaN where no release
        is, `[L, T, D]`.
    last : int
        Index of the step holding the latest release's cut-off.
    """

    times: list[str]
    final: np.ndarray
    partial: np.ndarray
    last: int

    @property
    def n_context(self) -> int:
        """Steps whose count at the latest release is at lag $D$ or more."""
        return self.last - int(self.partial.shape[2]) + 1


def _step(source: str, day: str) -> dt.date:
    """Return the step holding a day: its week's Saturday, or the day."""
    return epiweek_end(day) if source == "weekly" else as_date(day)


def hub_triangle(
    rows: pd.DataFrame,
    source: str,
    places: Sequence[str],
    last_time: str,
) -> HubTriangle:
    """Lay out the hub target of every release by place, step and lag.

    Parameters
    ----------
    rows : pandas.DataFrame
        The releases. The latest gives the final counts. Weekly rows have
        `target_end_date` and `total`; daily rows have `onset_date`,
        `observed` and `imputed`. Both have `vintage`, `location` and
        `cutoff`.
    source : str
        One of `SOURCES`.
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
    # The latest release gives the final counts and the last step.
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
        final, last = counts, at
    return HubTriangle(times, final, partial, last)


def lag_shares(
    tri: HubTriangle,
    places: Sequence[str],
    source: str,
    model: str,
    vintage: str,
) -> pd.DataFrame:
    """Return the share of a final count reported by each release lag.

    At lag $d$ a place's share is the sum of its counts at lag $d$ over the
    sum of their final counts, over the steps final at the latest release
    with a count at that lag. Shares are capped at 1 and made
    non-decreasing, and are 1 from lag $D$. `pooled` sums over places, for
    places with none of their own. A weekly lag $d$ holds its share over
    ages $7d$ to $7d + 6$ days, so the mean over a week's days, as
    `evaluation.nowcast.week_shares` takes it, is the share.

    Parameters
    ----------
    tri : HubTriangle
        The releases.
    places : sequence of str
        The places of `tri`, in order.
    source : str
        One of `SOURCES`, for the age in days of a lag.
    model, vintage : str
        Labels for the rows.

    Returns
    -------
    pandas.DataFrame
        As `forecasters.chain.delay_summary`: columns `model`, `vintage`,
        `location`, `kind`, `x` (age in days), `quantity`
        (`"reported_share"`) and `q50`.
    """
    start = tri.n_context
    lags = tri.partial.shape[2]
    width = 7 if source == "weekly" else 1
    cells = tri.partial[:, :start]
    seen = ~np.isnan(cells)
    final = np.broadcast_to(tri.final[:, :start, None], cells.shape)
    num = np.where(seen, cells, 0.0).sum(axis=1)  # [L, D]
    den = np.where(seen, final, 0.0).sum(axis=1)
    rows = [*zip(places, num, den, strict=True)]
    rows.append(("pooled", num.sum(axis=0), den.sum(axis=0)))
    frames = []
    for place, n, d in rows:
        ok = d > 0
        if not ok.any():
            continue
        share = np.maximum.accumulate(np.minimum(n[ok] / d[ok], 1.0))
        first = np.arange(lags)[ok] * width
        ages = np.unique(np.stack([first, first + width - 1], axis=1))
        frames.append(
            pd.DataFrame(
                {
                    "location": place,
                    "x": [*ages, lags * width],
                    "q50": [*np.repeat(share, len(ages) // len(share)), 1.0],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "model", model)
    out.insert(1, "vintage", vintage)
    out.insert(3, "kind", "cumulative")
    out.insert(5, "quantity", "reported_share")
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )


def chronos_sampler(
    groups: Sequence[Series], horizon: int, n_samples: int, seed: int
) -> list[np.ndarray]:
    """Draw trajectories from Chronos-2's quantiles.

    Every series and step draws its own uniform level, so draws are
    independent across series and steps. Runs on a CUDA or Apple GPU when
    torch sees one.

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
    ImportError
        If Chronos-2 is not installed (`uv sync --group chronos`).
    """
    pipeline = _chronos()
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
    """Return the earliest week `randomwalk` starts its walk from."""
    return str(origin.snap.week_ends[int(origin.start.min())])


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
        The `ragged` models' share reported by release lag at the last
        forecast (`lag_shares`), which maps their final counts to what
        later releases show. None for `complete`, which is scored as
        reported, as `randomwalk` is (`scored_as_reported`).
    """

    name: str
    source: str
    layout: str
    sample: Sampler = field(default=chronos_sampler, repr=False)
    delay_summary: pd.DataFrame | None = field(
        init=False, default=None, repr=False
    )

    @property
    def needs_snapshots(self) -> bool:
        """Whether the model reads release snapshots: `complete` does."""
        return self.layout == "complete"

    @property
    def scored_as_reported(self) -> bool:
        """Whether its samples are scored as they are: `complete`'s are."""
        return self.layout == "complete"

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return trajectories of the country and the task's provinces.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `target_history` holds the
            weekly releases and its `daily_history` the daily ones.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`, for
            the country and the provinces, each its own series.

        Raises
        ------
        ModelUnavailableError
            If the task has no releases of the source or fewer than two
            steps of context, or, for `complete`, no week `randomwalk`
            can start from.
        ImportError
            If Chronos-2 is not installed.
        """
        weekly = self.source == "weekly"
        rows = task.target_history if weekly else task.daily_history
        if rows is None or rows.empty:
            raise ModelUnavailableError(
                f"{self.name} needs the {self.source} releases, and the task "
                f"for {task.vintage} has none."
            )
        places = (NATIONAL, *task.locations)
        weeks = list(task.target_end_dates)
        tri = hub_triangle(rows, self.source, places, weeks[-1])
        start = tri.n_context
        self.delay_summary = None
        if self.layout == "complete":
            origin = RandomWalk().origin_at(task, errors=False)
            start = tri.times.index(_walk_start(origin)) + 1
        else:
            self.delay_summary = lag_shares(
                tri, places, self.source, self.name, task.vintage
            )
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
        reported = (
            task.target.pivot_table(
                index="location",
                columns="target_end_date",
                values="total",
                aggfunc="sum",
            )
            .reindex(index=list(places), columns=weeks)
            .fillna(0.0)
            .to_numpy()
        )
        values = np.maximum(values, reported)  # [S, L, W]
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
            "layout, the country and provinces in one block.",
            fitted=True,
        )
