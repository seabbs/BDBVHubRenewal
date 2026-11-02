# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Skellam observation model with a spread fitted by moments.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
remaining change is $\mathrm{rem} = \mu - Y$.
The count at the later release is

$$
X = \max\big(0, Y + \mathrm{Pois}(\mathrm{rem}^+ + \rho\mu)
- \mathrm{Pois}(\mathrm{rem}^- + \rho\mu)\big),
$$

so a revision can go down.
$\rho \ge 0$ is the spread, fitted for each number of weeks back by
moments from nowcasts made at earlier releases.
With $\hat y$ the earlier nowcast, $y$ the later count, and errors
$e = y - \hat y$,

$$
\rho = \max\Big(0, \frac{\sum (e^2 - |\mathrm{rem}|)}{\sum 2\hat y}\Big).
$$

Baselinenowcast 0.2.2 has no Skellam model.
This one lives outside the port.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "fit_rho_moments",
    "rho_from_sums",
    "rho_sums",
    "sample_skellam_moments",
]


def rho_sums(
    obs: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the sums behind the spread.

    Parameters
    ----------
    obs : numpy.ndarray
        Count reported at the earlier release, $Y$.
    pred : numpy.ndarray
        Nowcast from the earlier release, $\hat y$.
    target : numpy.ndarray
        Count at the later release, $y$.
    valid : numpy.ndarray, optional
        Which errors to keep. Defaults to all.
    axis : int
        Axis to sum over.

    Returns
    -------
    tuple of numpy.ndarray
        The sums of $e^2 - |\mathrm{rem}|$ and of $2\hat y^+$, and the
        number of errors.
    """
    keep = np.ones(np.shape(obs), dtype=bool) if valid is None else valid
    rem = pred - obs
    err = target - pred
    num = np.where(keep, err**2 - np.abs(rem), 0.0).sum(axis=axis)
    den = np.where(keep, 2.0 * np.maximum(pred, 0.0), 0.0).sum(axis=axis)
    return num, den, keep.sum(axis=axis)


def rho_from_sums(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    r"""Return $\rho$ from the sums, 0 where the denominator is 0."""
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, np.maximum(0.0, num / safe), 0.0)


def fit_rho_moments(
    obs: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    axis: int = -1,
) -> np.ndarray:
    r"""Fit the Skellam spread by moments.

    Parameters
    ----------
    obs, pred, target, valid, axis
        As for `rho_sums`.

    Returns
    -------
    numpy.ndarray
        $\rho$, 0 with no errors.
    """
    num, den, _ = rho_sums(obs, pred, target, valid, axis)
    return rho_from_sums(num, den)


def sample_skellam_moments(
    key: Array, mean: Array, obs: Array, rho: Array, n: int
) -> Array:
    r"""Draw the count at the later release.

    Parameters
    ----------
    key : jax.Array
        Random key.
    mean : jax.Array
        Nowcast mean $\mu$ of each week.
    obs : jax.Array
        Count reported so far, $Y$.
    rho : jax.Array
        Spread $\rho$ of each week.
    n : int
        Number of samples.

    Returns
    -------
    jax.Array
        Samples, shape `(n, *mean.shape)`, floored at 0.
    """
    rem = mean - obs
    extra = rho * jnp.maximum(mean, 0.0)
    up, down = jax.random.split(key)
    shape = (n, *jnp.shape(mean))
    rise = jax.random.poisson(up, jnp.maximum(rem, 0.0) + extra, shape)
    fall = jax.random.poisson(down, jnp.maximum(-rem, 0.0) + extra, shape)
    return jnp.maximum(obs + rise - fall, 0).astype(mean.dtype)
