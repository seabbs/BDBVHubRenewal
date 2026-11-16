# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Forecasts as quantiles, with no needless sampling noise.

A model that gives its own quantiles for a task (a `quantiles(task,
levels)` method that returns a table) is scored from them, so its scores
do not depend on a seed. Chronos-2 on the weekly source does
(`bdbvhubrenewal.forecasters.foundation.FoundationModel.quantiles`).
Every other forecast, such as a week summed from daily steps or a sample
model's, is the sample quantile of `n_draws` trajectories drawn with one
seed for every model. Models with a `stream` (Chronos-2) also draw from
one shared random stream, `COMMON_STREAM`, so they share their random
numbers (common random numbers).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.forecasters.base import (
    ForecastModel,
    ForecastTask,
    ModelUnavailableError,
)

__all__ = [
    "COMMON_STREAM",
    "N_DRAWS",
    "forecast_draws",
    "forecast_quantiles",
    "own_quantiles",
]

N_DRAWS: Final = 1000
"""Draws per forecast where the quantiles come from draws."""

log = logging.getLogger(__name__)

COMMON_STREAM: Final = "common"
"""The task's random stream that models with a `stream` share."""


def forecast_quantiles(
    model: ForecastModel,
    task: ForecastTask,
    levels: Sequence[float],
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one model's forecasts of one task as quantiles at `levels`.

    Parameters
    ----------
    model : ForecastModel
        The model. Its own quantiles are used where it gives them.
    task : ForecastTask
        The task. Its `n_samples` and `seed` are replaced.
    levels : sequence of float
        Increasing quantile levels.
    n_draws : int
        Draws where the quantiles come from draws.
    seed : int
        The task's seed for those draws, the same for every model.

    Returns
    -------
    pandas.DataFrame
        `run_forecasts` columns with `sample` replaced by `quantile`: one
        row per location, week and level, the country included.
    """
    at = np.asarray(levels, dtype=float)
    own = own_quantiles(model, task, at)
    if own is not None:
        return own
    drawn = forecast_draws(model, task, n_draws, seed)
    keys = [c for c in drawn.columns if c not in ("sample", "value")]
    return (
        drawn.groupby(keys, dropna=False)["value"]
        .quantile(at)
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )


def forecast_draws(
    model: ForecastModel,
    task: ForecastTask,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one model's draws of one task, as `forecast_quantiles` takes them.

    Parameters
    ----------
    model : ForecastModel
        The model. One with a `stream` draws from `COMMON_STREAM`.
    task : ForecastTask
        The task. Its `n_samples` and `seed` are replaced.
    n_draws : int
        Draws per forecast.
    seed : int
        The task's seed, the same for every model.

    Returns
    -------
    pandas.DataFrame
        `run_forecasts` rows, the country included.
    """
    if hasattr(model, "stream"):
        model = _with_stream(model)
    return run_forecasts(
        [model], [dataclasses.replace(task, n_samples=n_draws, seed=seed)]
    )


def _with_stream(model: Any) -> Any:
    """Return the model drawing from the shared stream."""
    return dataclasses.replace(model, stream=COMMON_STREAM)


def own_quantiles(
    model: Any, task: ForecastTask, levels: Sequence[float] | np.ndarray
) -> pd.DataFrame | None:
    """Return the model's own quantiles as `run_forecasts` rows, if any.

    A quantile of a sum is not the sum of quantiles, so the national row
    is the model's own and is not added from the provinces.
    """
    quantiles = getattr(model, "quantiles", None)
    if not callable(quantiles):
        return None
    try:
        own: pd.DataFrame | None = quantiles(task, levels)
    except ModelUnavailableError as err:
        # As `run_forecasts`: an unavailable model has no rows.
        log.warning("%s skipped %s: %s", model.name, task.vintage, err)
        own = pd.DataFrame(
            columns=["location", "target_end_date", "quantile", "value"]
        )
    if own is None:
        return None
    # `run_forecasts`'s columns, with the quantile in the sample's place.
    by_level = own.rename(columns={"quantile": "sample"})
    if "as_of" not in by_level:
        by_level["as_of"] = np.nan
    ref = pd.Timestamp(task.reference_date)
    by_level["horizon"] = (
        pd.to_datetime(by_level["target_end_date"]) - ref
    ).dt.days // 7
    by_level.insert(0, "model", model.name)
    by_level.insert(1, "kind", task.kind)
    by_level.insert(2, "vintage", task.vintage)
    by_level.insert(3, "reference_date", task.reference_date.isoformat())
    out = by_level.rename(columns={"sample": "quantile"})
    keys = [c for c in out.columns if c not in ("quantile", "value")]
    return out[[*keys, "quantile", "value"]].reset_index(drop=True)
