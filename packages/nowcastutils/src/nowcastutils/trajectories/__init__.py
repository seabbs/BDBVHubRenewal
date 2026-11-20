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

The parameters come from a `fit`; nothing is hard-coded. `fit_coupling`
estimates them from normal scores of the model's own matured forecasts
(`pit_scores`). This module depends on numpy and scipy only, so any model
can use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
from scipy.special import ndtr, ndtri

__all__ = ["METHODS", "couple", "fit_coupling", "pit_scores", "read_quantiles"]

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


def _knots(
    levels: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the quantile function's knots: levels `[K]`, values `[..., K]`.

    The function `read_quantiles` reads: flat from 0 to the first level
    and from the last level to 1, linear between.
    """
    u = np.concatenate([[0.0], levels, [1.0]])
    v = np.concatenate([values[..., :1], values, values[..., -1:]], axis=-1)
    return u, v


def pit_scores(
    quantiles: np.ndarray,
    levels: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Return normal scores of observed values in quantile forecasts.

    The score is $\Phi^{-1}$ of the randomised PIT,
    $F(y^-) + v\,(F(y) - F(y^-))$ with $v$ uniform, in the quantile
    function `read_quantiles` reads. So a value on a flat stretch (zero
    counts with every quantile zero, or the mass beyond the end levels)
    draws its level rather than taking a fixed one.

    Parameters
    ----------
    quantiles : numpy.ndarray
        Quantiles `[..., Q]` at `levels`.
    levels : numpy.ndarray
        Increasing quantile levels, `[Q]`.
    observed : numpy.ndarray
        Observed values, the shape of `quantiles` less its last axis; NaN
        where none is observed yet.
    rng : numpy.random.Generator
        Draws the uniforms.

    Returns
    -------
    numpy.ndarray
        Scores, the shape of `observed`, NaN where it is.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.sort(np.asarray(quantiles, dtype=float), axis=-1)
    y = np.asarray(observed, dtype=float)
    u, v = _knots(levels, values)
    k = len(u)
    filled = np.where(np.isnan(y), 0.0, y)[..., None]

    def between(j: np.ndarray) -> np.ndarray:
        """Return the level at `filled` on the segment from knot j to j + 1."""
        j = np.clip(j, 0, k - 2)
        v0 = np.take_along_axis(v, j[..., None], axis=-1)[..., 0]
        v1 = np.take_along_axis(v, j[..., None] + 1, axis=-1)[..., 0]
        width = np.where(v1 > v0, v1 - v0, 1.0)
        t = np.clip((filled[..., 0] - v0) / width, 0.0, 1.0)
        return np.asarray(u[j] + (u[j + 1] - u[j]) * t)

    # F(y): the last knot at or below y; F(y-): the first at or above it.
    last = (v <= filled).sum(axis=-1) - 1
    upper = np.where(last >= k - 1, 1.0, np.where(last < 0, 0.0, between(last)))
    first = (v < filled).sum(axis=-1)
    lower = np.where(
        first >= k, 1.0, np.where(first <= 0, 0.0, between(first - 1))
    )
    pit = lower + rng.random(y.shape) * (upper - lower)
    tiny = np.finfo(float).eps
    z = ndtri(np.clip(pit, tiny, 1 - tiny))
    return np.where(np.isnan(y), np.nan, z)


def _shrunk_correlation(
    a: np.ndarray, b: np.ndarray, origin: np.ndarray
) -> tuple[float, float, int]:
    """Return the pairs' correlation shrunk to zero, its shrinkage and n.

    The Ledoit-Wolf intensity for a target of zero, var(r) / r^2 capped at
    1, with var(r) from the pairs clustered by origin (pairs of one
    forecast are not independent). Fewer than two origins give no evidence:
    zero, shrinkage 1.
    """
    n = len(a)
    origins = np.unique(origin)
    if n < 2 or len(origins) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0, 1.0, n
    prod = (a - a.mean()) * (b - b.mean()) / (a.std() * b.std())
    r = float(prod.mean())
    sums = np.bincount(np.searchsorted(origins, origin), weights=prod - r)
    var = float((sums**2).sum() / n**2)
    shrink = 1.0 if r == 0 else min(1.0, var / r**2)
    return (1 - shrink) * r, shrink, n


def fit_coupling(scores: np.ndarray, method: str) -> dict[str, Any]:
    r"""Return a coupling's fit from normal scores of matured forecasts.

    For `gaussian`, $\rho$ is the correlation of a series' scores at
    consecutive steps and $s$ that of two series at the same step, each
    over every pair in the scores, shrunk towards zero (independence) by
    the Ledoit-Wolf intensity estimated from the pairs themselves
    (`_shrunk_correlation`). $s$ is floored at zero, as `couple`'s
    exchangeable copula needs, and both stay below 1. No parameter is
    fixed in advance.

    Parameters
    ----------
    scores : numpy.ndarray
        `pit_scores` of the model's earlier forecasts against what was
        later reported, `[origins, series, steps]`, NaN where a cell has
        not matured.
    method : str
        One of `METHODS`.

    Returns
    -------
    dict
        `method`; for `gaussian` also `rho`, `s`, their `shrinkage_rho`
        and `shrinkage_s`, `n_origins` and the pairs behind each,
        `n_step_pairs` and `n_series_pairs`.

    Raises
    ------
    ValueError
        For an unknown method.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    if method != "gaussian":
        return {"method": method}
    z = np.asarray(scores, dtype=float)
    n_origins, series = z.shape[:2]
    at = np.broadcast_to(np.arange(n_origins)[:, None, None], z.shape)
    a, b = z[:, :, :-1], z[:, :, 1:]
    ok = np.isfinite(a) & np.isfinite(b)
    rho, shrink_rho, n_steps = _shrunk_correlation(
        a[ok], b[ok], at[:, :, :-1][ok]
    )
    i, j = np.triu_indices(series, k=1)
    a, b = z[:, i, :], z[:, j, :]
    ok = np.isfinite(a) & np.isfinite(b)
    where = np.broadcast_to(np.arange(n_origins)[:, None, None], a.shape)
    s, shrink_s, n_series = _shrunk_correlation(a[ok], b[ok], where[ok])
    bound = np.nextafter(1.0, 0.0)
    return {
        "method": method,
        "rho": float(np.clip(rho, -bound, bound)),
        "s": float(np.clip(s, 0.0, bound)),
        "shrinkage_rho": shrink_rho,
        "shrinkage_s": shrink_s,
        "n_origins": int(n_origins),
        "n_step_pairs": int(n_steps),
        "n_series_pairs": int(n_series),
    }
