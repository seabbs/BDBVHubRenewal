# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The pretrained TimesFM 3 foundation model on the hub target's releases.

The models are `fm_timesfm_<source>_<layout>`: the Chronos-2 models of
`foundation` (the same `FoundationModel`, triangle and layouts) with
TimesFM 3 in place of Chronos-2. One block holds every place's target,
and for `ragged` each place's lag series as known past and future
covariates.
TimesFM 3's own `predict` fills missing covariate cells by interpolation
and carries the last value forward, so `timesfm_sampler` calls its
`decode` with the missing cells masked instead: they are zero and
flagged, and the model does not read them.
TimesFM 3 gives the deciles 0.1 to 0.9 only. `extend_quantiles` reads
levels beyond them linearly in normal scores from the two outer
deciles, so the tails are those of a normal through them.
TimesFM 3 is an optional dependency group (`uv sync --group timesfm`).
Its weights are under the TimesFM Non-Commercial License v1.0, for
non-commercial, non-production use only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, partial
from statistics import NormalDist
from typing import Any, Final

import numpy as np

from bdbvhubrenewal.forecasters.base import register_model
from bdbvhubrenewal.forecasters.foundation import (
    LAYOUTS,
    SOURCES,
    FoundationModel,
    Series,
    quantile_draws,
)

__all__ = [
    "EXTENDED_LEVELS",
    "TIMESFM_LEVELS",
    "TIMESFM_MODEL",
    "TIMESFM_REVISION",
    "TimesFMArrays",
    "extend_quantiles",
    "timesfm_arrays",
    "timesfm_quantiles",
    "timesfm_sampler",
]

TIMESFM_MODEL: Final = "google/timesfm-3.0-pytorch"
"""The Hugging Face checkpoint."""

TIMESFM_REVISION: Final = "43046b85ec22d584a13f8098c2ed39c889e129c2"
"""The checkpoint's commit, pinned so runs are reproducible."""

TIMESFM_LEVELS: Final = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""The quantile levels TimesFM 3 gives."""

EXTENDED_LEVELS: Final = (0.01, 0.05, *TIMESFM_LEVELS, 0.95, 0.99)
"""The levels draws are read from: TimesFM 3's, and Chronos-2's outer
levels from `extend_quantiles`."""


@dataclass(frozen=True)
class TimesFMArrays:
    """One block as TimesFM 3's `decode` takes it.

    Attributes
    ----------
    target : numpy.ndarray
        The targets' context, `[V, S]`.
    covariates : numpy.ndarray or None
        Known covariates over context and horizon, `[C, S + H]`, zero
        where missing. None for no covariates.
    mask : numpy.ndarray or None
        True where a covariate is missing, the shape of `covariates`.
    """

    target: np.ndarray
    covariates: np.ndarray | None
    mask: np.ndarray | None


def timesfm_arrays(group: Series) -> TimesFMArrays:
    """Return a block's arrays, with missing covariates masked.

    Parameters
    ----------
    group : Series
        One block, as `foundation.layout_block` gives it. Each covariate
        has its context in `past_covariates` and its horizon in
        `future_covariates`.

    Returns
    -------
    TimesFMArrays
        The target and the covariates of context and horizon joined.
    """
    target = np.atleast_2d(np.asarray(group["target"], dtype=np.float32))
    past = group.get("past_covariates") or {}
    if not past:
        return TimesFMArrays(target, None, None)
    future = group["future_covariates"]
    cells = np.stack(
        [np.concatenate([past[name], future[name]]) for name in past]
    ).astype(np.float32)
    mask = np.isnan(cells)
    return TimesFMArrays(target, np.where(mask, 0.0, cells), mask)


def extend_quantiles(
    levels: np.ndarray, values: np.ndarray, at: np.ndarray
) -> np.ndarray:
    """Read quantile functions at other levels, beyond their own too.

    Linear in the level between `levels`; beyond them linear in normal
    scores through the two outermost levels at that end.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    values : numpy.ndarray
        Quantiles at those levels, `[..., Q]`. They are sorted first.
    at : numpy.ndarray
        Levels to read, in (0, 1), `[K]`.

    Returns
    -------
    numpy.ndarray
        Quantiles at `at`, `[..., K]`, non-decreasing in the level.
    """
    levels = np.asarray(levels, dtype=float)
    at = np.asarray(at, dtype=float)
    values = np.sort(np.asarray(values, dtype=float), axis=-1)
    z = np.array([NormalDist().inv_cdf(p) for p in levels])
    out = np.empty((*values.shape[:-1], len(at)))
    for k, p in enumerate(at):
        if p < levels[0] or p > levels[-1]:
            lo, hi = (0, 1) if p < levels[0] else (-2, -1)
            slope = (values[..., hi] - values[..., lo]) / (z[hi] - z[lo])
            out[..., k] = values[..., lo] + slope * (
                NormalDist().inv_cdf(p) - z[lo]
            )
        else:
            hi = int(np.clip(np.searchsorted(levels, p), 1, len(levels) - 1))
            w = (p - levels[hi - 1]) / (levels[hi] - levels[hi - 1])
            out[..., k] = values[..., hi - 1] + w * (
                values[..., hi] - values[..., hi - 1]
            )
    return out


@cache
def _timesfm() -> Any:
    import torch
    from timesfm3 import TimesFM3Forecaster

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TimesFM3Forecaster.from_pretrained(
        TIMESFM_MODEL, device=device, revision=TIMESFM_REVISION
    ).model


def timesfm_quantiles(
    groups: Sequence[Series], horizon: int
) -> list[np.ndarray]:
    """Return TimesFM 3's deciles of each block's targets.

    Parameters
    ----------
    groups : sequence of Series
        The blocks.
    horizon : int
        Steps to forecast.

    Returns
    -------
    list of numpy.ndarray
        Sorted quantiles at `TIMESFM_LEVELS`, `[V, horizon, 9]` per block.

    Raises
    ------
    ImportError
        If TimesFM 3 is not installed (`uv sync --group timesfm`).
    """
    model = _timesfm()
    import torch

    out = []
    for group in groups:
        arrays = timesfm_arrays(group)
        n_target = arrays.target.shape[0]
        kwargs: dict[str, Any] = {"horizon": horizon}
        if arrays.covariates is not None and arrays.mask is not None:
            kwargs["past_future_covariates"] = torch.from_numpy(
                arrays.covariates[None]
            )
            kwargs["past_future_mask"] = torch.from_numpy(arrays.mask[None])
        with torch.inference_mode():
            q = model.decode(
                target=torch.from_numpy(arrays.target[None]), **kwargs
            )
        q = np.asarray(q[0, :n_target].float().cpu().numpy())
        out.append(np.sort(q, axis=-1))
    return out


def timesfm_sampler(
    groups: Sequence[Series], horizon: int, n_samples: int, seed: int
) -> list[np.ndarray]:
    """Draw trajectories from TimesFM 3's quantiles.

    Each block's deciles are extended to `EXTENDED_LEVELS`
    (`extend_quantiles`) and read at uniform draws as `chronos_sampler`
    reads Chronos-2's: each series and step draws its own level.

    Parameters
    ----------
    groups : sequence of Series
        The blocks.
    horizon : int
        Steps to draw.
    n_samples : int
        Trajectories.
    seed : int
        Seed for the uniform draws.

    Returns
    -------
    list of numpy.ndarray
        Draws, `[n_samples, V, horizon]` per block.

    Raises
    ------
    ImportError
        If TimesFM 3 is not installed (`uv sync --group timesfm`).
    """
    levels = np.asarray(EXTENDED_LEVELS)
    rng = np.random.default_rng(seed)
    out = []
    for q in timesfm_quantiles(groups, horizon):
        q = extend_quantiles(np.asarray(TIMESFM_LEVELS), q, levels)
        u = rng.uniform(size=(n_samples, *q.shape[:2]))
        out.append(
            quantile_draws(levels, np.broadcast_to(q, (n_samples, *q.shape)), u)
        )
    return out


for _source in SOURCES:
    for _layout in LAYOUTS:
        _name = f"fm_timesfm_{_source}_{_layout}"
        register_model(
            _name,
            partial(
                FoundationModel,
                _name,
                _source,
                _layout,
                sample=timesfm_sampler,
            ),
            f"TimesFM 3 on the hub target's {_source} releases, {_layout} "
            "layout, the country and provinces in one block.",
            fitted=True,
        )
