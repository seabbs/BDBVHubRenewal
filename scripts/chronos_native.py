"""Chronos-2 forecasts as quantiles with no needless sampling noise.

Where a hub target is one of Chronos-2's own outputs (every step of the
weekly models), its quantiles are Chronos-2's native quantiles, read at
the scored levels by linear interpolation between the native levels
(0.01, 0.05, 0.1, ..., 0.95, 0.99): 0.025 lies between 0.01 and 0.05
and 0.975 between 0.95 and 0.99; every other scored level is native.
They pass through the model's own forecast code (`FoundationModel`) as
`K` "trajectories", one per level; every step after the draws (rounding,
the floor at the reported counts) is monotone per value, so trajectory
`k` comes out as the quantile at level `k`.

Where a target is a sum of Chronos-2's steps (the daily models' weeks),
it is the sample quantile of `N_DRAWS` trajectories, drawn with the same
random numbers for every model (common random numbers: the stream is
keyed on the task, not on the model).

`chronos_quantiles` returns one model's quantiles for one task.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.forecasters import foundation as fm
from bdbvhubrenewal.forecasters.base import ForecastTask

N_DRAWS = 1000
STREAM = "chronos-common"


def _native(levels: np.ndarray, checkpoint: fm.Checkpoint | None):
    """Return a sampler giving Chronos-2's quantiles at `levels` as paths."""

    def sample(groups, horizon, n, seed):  # noqa: ARG001
        pipeline = fm._chronos(checkpoint)
        native = np.asarray(pipeline.quantiles, dtype=float)
        quantiles, _ = pipeline.predict_quantiles(
            [dict(g) for g in groups],
            prediction_length=horizon,
            quantile_levels=native.tolist(),
        )
        out = []
        for q in quantiles:
            q = np.sort(q.float().cpu().numpy(), axis=-1)  # [V, H, Q]
            at = np.apply_along_axis(
                lambda v: np.interp(levels, native, v), -1, q
            )  # [V, H, K]
            out.append(np.moveaxis(at, -1, 0))  # [K, V, H]
        return out

    return sample


def _common(task: ForecastTask, checkpoint: fm.Checkpoint | None):
    """Return a sampler drawing from one stream per task, for every model."""

    def sample(groups, horizon, n, seed):  # noqa: ARG001
        pipeline = fm._chronos(checkpoint)
        levels = np.asarray(pipeline.quantiles, dtype=float)
        quantiles, _ = pipeline.predict_quantiles(
            [dict(g) for g in groups],
            prediction_length=horizon,
            quantile_levels=levels.tolist(),
        )
        rng = task.rng(STREAM)
        out = []
        for q in quantiles:
            q = q.float().cpu().numpy()
            u = rng.uniform(size=(n, *q.shape[:2]))
            out.append(
                fm.quantile_draws(levels, np.broadcast_to(q, (n, *q.shape)), u)
            )
        return out

    return sample


def chronos_quantiles(
    model: fm.FoundationModel,
    task: ForecastTask,
    levels: Sequence[float],
) -> pd.DataFrame:
    """Return one model's forecasts of one task as quantiles at `levels`.

    Returns `run_forecasts` columns with `sample` replaced by `quantile`.
    """
    levels = np.asarray(levels, dtype=float)
    if model.source == "weekly":
        run = dataclasses.replace(
            model, sample=_native(levels, model.checkpoint)
        )
        t = dataclasses.replace(task, n_samples=len(levels))
        out = run_forecasts([run], [t])
        out["quantile"] = levels[out["sample"].to_numpy() - 1]
        return out.drop(columns="sample")
    run = dataclasses.replace(model, sample=_common(task, model.checkpoint))
    t = dataclasses.replace(task, n_samples=N_DRAWS)
    out = run_forecasts([run], [t])
    keys = [c for c in out.columns if c not in ("sample", "value")]
    return (
        out.groupby(keys, dropna=False)["value"]
        .quantile(levels)
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )
