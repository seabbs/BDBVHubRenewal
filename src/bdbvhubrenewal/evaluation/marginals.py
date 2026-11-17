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

`daily_quantiles` does the same for the daily target, each day's count:
a model's own daily quantiles (`quantiles(task, levels, target="daily")`)
where it gives them, else the quantiles of its daily draws (`daily(task)`).
A model with neither has no daily target.
"""

from __future__ import annotations

import dataclasses
import inspect
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
    add_national,
)

__all__ = ["COMMON_STREAM", "N_DRAWS", "daily_quantiles", "forecast_quantiles"]

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
    own = _own_quantiles(model, task, at)
    if own is not None:
        return own
    if hasattr(model, "stream"):
        model = _with_stream(model)
    drawn = run_forecasts(
        [model], [dataclasses.replace(task, n_samples=n_draws, seed=seed)]
    )
    keys = [c for c in drawn.columns if c not in ("sample", "value")]
    return (
        drawn.groupby(keys, dropna=False)["value"]
        .quantile(at)
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )


DAILY_COLUMNS: Final = (
    "model",
    "location",
    "target_end_date",
    "quantile",
    "value",
    "as_of",
)
"""Columns of `daily_quantiles`: `target_end_date` is the day."""


def daily_quantiles(
    model: Any,
    task: ForecastTask,
    levels: Sequence[float],
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> pd.DataFrame | None:
    """Return one model's daily forecasts of one task as quantiles.

    The model's own daily quantiles are used where it gives them
    (`quantiles(task, levels, target="daily")`), else the quantiles of
    `n_draws` daily draws (`daily(task)`), the country the sum of the
    provinces by draw.

    Parameters
    ----------
    model : object
        The model.
    task : ForecastTask
        The task. Its `n_samples` and `seed` are replaced for draws.
    levels : sequence of float
        Increasing quantile levels.
    n_draws : int
        Draws where the quantiles come from draws.
    seed : int
        The task's seed for those draws, the same for every model.

    Returns
    -------
    pandas.DataFrame or None
        `DAILY_COLUMNS`, one row per location, day and level, the country
        included; empty where the model is unavailable for the task; None
        where the model has no daily target.
    """
    at = np.asarray(levels, dtype=float)
    try:
        own = _own_daily(model, task, at)
        if own is not None:
            return own.assign(model=model.name, as_of=np.nan)[
                list(DAILY_COLUMNS)
            ]
        daily = getattr(model, "daily", None)
        if not callable(daily):
            return None
        if hasattr(model, "stream"):
            daily = _with_stream(model).daily
        drawn = daily(dataclasses.replace(task, n_samples=n_draws, seed=seed))
    except ModelUnavailableError as err:
        log.warning("%s skipped %s: %s", model.name, task.vintage, err)
        return pd.DataFrame(columns=list(DAILY_COLUMNS))
    if drawn is None:
        return None
    keys = ["location", "target_end_date"]
    out = (
        add_national(drawn)
        .groupby(keys)["value"]
        .quantile(at)
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )
    return out.assign(model=model.name, as_of=np.nan)[list(DAILY_COLUMNS)]


def _own_daily(
    model: Any, task: ForecastTask, levels: np.ndarray
) -> pd.DataFrame | None:
    """Return the model's own daily quantiles, if it gives any."""
    quantiles = getattr(model, "quantiles", None)
    if not callable(quantiles):
        return None
    if "target" not in inspect.signature(quantiles).parameters:
        return None
    out: pd.DataFrame | None = quantiles(task, levels, target="daily")
    return out


def _with_stream(model: Any) -> Any:
    """Return the model drawing from the shared stream."""
    return dataclasses.replace(model, stream=COMMON_STREAM)


def _own_quantiles(
    model: Any, task: ForecastTask, levels: np.ndarray
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
