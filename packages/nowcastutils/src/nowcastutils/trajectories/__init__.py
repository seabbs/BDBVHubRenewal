# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Coherent sample trajectories from any model's quantile forecasts.

A quantile forecast gives each series and step its own marginal and
nothing about how they move together. `couple` draws trajectories whose
cells each follow the cell's quantile function, piecewise linear between
the levels and flat beyond them (so the end quantiles carry the tail
mass), and joins the cells by a copula:

- `independent`: every cell draws its own level;
- `comonotone`: one level per sample and series across its steps, the
  quantile-only analogue of ensemble copula coupling;
- `gaussian`: levels from a Gaussian copula with AR(1) correlation `rho`
  between steps and exchangeable correlation `s` between series, given
  in `fit`.

The parameters come from a `fit`; nothing is hard-coded. This module
depends on numpy and scipy only, so any model can use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
from scipy.special import ndtr

__all__ = ["METHODS", "couple", "read_quantiles"]

METHODS: Final = ("independent", "comonotone", "gaussian")
"""How `couple` joins the cells."""


def read_quantiles(
    levels: np.ndarray, values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Read quantile functions at uniform draws.

    Linear between the levels and flat beyond them, the same reading as
    `forecasters.foundation.quantile_draws`. That copy is part of the
    foundation model's stored-fit keys, so it is not imported here.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    values : numpy.ndarray
        Quantiles at those levels, `[..., Q]`, sorted first so crossed
        quantiles give a valid function.
    u : numpy.ndarray
        Uniform draws, the shape of `values` less its last axis.

    Returns
    -------
    numpy.ndarray
        The draws, the shape of `u`.
    """
    return _read_sorted(levels, np.sort(values, axis=-1), u)


def _read_sorted(
    levels: np.ndarray, values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Read sorted quantiles `values` at uniform draws `u`."""
    hi = np.clip(np.searchsorted(levels, u), 1, len(levels) - 1)
    lo = hi - 1
    weight = np.clip((u - levels[lo]) / (levels[hi] - levels[lo]), 0.0, 1.0)
    at_lo = np.take_along_axis(values, lo[..., None], axis=-1)[..., 0]
    at_hi = np.take_along_axis(values, hi[..., None], axis=-1)[..., 0]
    return np.asarray(at_lo + weight * (at_hi - at_lo))


def _gaussian_levels(
    n_samples: int,
    series: int,
    steps: int,
    rho: float,
    s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return uniform levels from the AR(1) x exchangeable Gaussian copula.

    Each series' steps follow a stationary AR(1) with correlation `rho`;
    a shared factor gives correlation `s` between any two series at the
    same step. Both pieces have unit variance, so the levels are uniform.
    """
    shared = rng.standard_normal((n_samples, 1, steps))
    own = rng.standard_normal((n_samples, series, steps))
    z = np.sqrt(s) * shared + np.sqrt(1 - s) * own
    path = np.empty_like(z)
    path[..., 0] = z[..., 0]
    scale = np.sqrt(1 - rho**2)
    for t in range(1, steps):
        path[..., t] = rho * path[..., t - 1] + scale * z[..., t]
    return np.asarray(ndtr(path))


def couple(
    quantiles: np.ndarray,
    levels: np.ndarray,
    method: str,
    fit: Mapping[str, Any] | None,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Return coherent sample trajectories from quantile forecasts.

    Parameters
    ----------
    quantiles : numpy.ndarray
        Quantiles `[series, steps, Q]` at `levels`.
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    method : str
        One of `METHODS`.
    fit : mapping or None
        The method's parameters: `rho` and `s` for `gaussian`.
    n_samples : int
        Trajectories.
    seed : int
        Seed for the levels.

    Returns
    -------
    numpy.ndarray
        Draws `[n_samples, series, steps]`, each cell's marginal its
        piecewise-linear quantile function.

    Raises
    ------
    ValueError
        For an unknown method, or a `gaussian` fit without `rho` and `s`
        or with them out of range.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    series, steps = quantiles.shape[:2]
    rng = np.random.default_rng(seed)
    if method == "independent":
        u = rng.random((n_samples, series, steps))
    elif method == "comonotone":
        u = np.repeat(rng.random((n_samples, series, 1)), steps, axis=2)
    elif method == "gaussian":
        if not fit or "rho" not in fit or "s" not in fit:
            raise ValueError(
                "a gaussian coupling needs fit['rho'] and fit['s']"
            )
        rho, s = float(fit["rho"]), float(fit["s"])
        if not (-1 < rho < 1 and 0 <= s <= 1):
            raise ValueError(
                "a gaussian coupling needs -1 < rho < 1 and 0 <= s <= 1, "
                f"got rho={rho}, s={s}"
            )
        u = _gaussian_levels(n_samples, series, steps, rho, s, rng)
    else:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    values = np.sort(quantiles, axis=-1)
    values = np.broadcast_to(values, (n_samples, *values.shape))
    return _read_sorted(levels, values, u)
