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

Two variants are scored against it in the release backtest
(`OBS_MODELS`).
`skellam_rem` has the extra spread $\rho\,|\mathrm{rem}|$ in place of
$\rho\mu$, so a week with nothing left to report gets no extra noise.
`nb` is baselinenowcast's own observation model, a negative binomial on
the remaining count, $X = Y + \mathrm{NB}(\mathrm{rem}^+, \phi)$ with
variance $\mathrm{rem}^+ + (\mathrm{rem}^+)^2 / \phi$.
Its $1/\phi$ is fitted by moments,
$\sum (e^2 - \mathrm{rem}^+) / \sum (\mathrm{rem}^+)^2$, floored at 0,
pooled over every number of weeks back.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "OBS_MODELS",
    "fit_rho_moments",
    "rho_from_sums",
    "rho_sums",
    "sample_nowcast",
    "sample_skellam_moments",
    "spread_sums",
]

OBS_MODELS = ("skellam", "skellam_rem", "nb")
"""Observation models: the Skellam, the Skellam on the remainder, the NB."""


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


def spread_sums(
    obs: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    axis: int = -1,
    model: str = "skellam",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the sums behind the spread of an observation model.

    Parameters
    ----------
    obs, pred, target, valid, axis
        As for `rho_sums`.
    model : str
        One of `OBS_MODELS`.

    Returns
    -------
    tuple of numpy.ndarray
        The moment numerator and denominator, and the number of errors.
        The spread is their ratio, floored at 0 (`rho_from_sums`).

    Raises
    ------
    ValueError
        If `model` is not one of `OBS_MODELS`.
    """
    if model == "skellam":
        return rho_sums(obs, pred, target, valid, axis)
    if model not in OBS_MODELS:
        raise ValueError(f"Unknown observation model {model!r}.")
    keep = np.ones(np.shape(obs), dtype=bool) if valid is None else valid
    rem = pred - obs
    err = target - pred
    if model == "skellam_rem":
        num = np.where(keep, err**2 - np.abs(rem), 0.0)
        den = np.where(keep, 2.0 * np.abs(rem), 0.0)
    else:
        pos = np.maximum(rem, 0.0)
        num = np.where(keep, err**2 - pos, 0.0)
        den = np.where(keep, pos**2, 0.0)
    return num.sum(axis=axis), den.sum(axis=axis), keep.sum(axis=axis)


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


def sample_nowcast(
    key: Array, mean: Array, obs: Array, spread: Array, n: int, model: str
) -> Array:
    r"""Draw the count at the later release under an observation model.

    Parameters
    ----------
    key : jax.Array
        Random key.
    mean : jax.Array
        Nowcast mean $\mu$ of each week.
    obs : jax.Array
        Count reported so far, $Y$.
    spread : jax.Array
        $\rho$ for the Skellam models, $1/\phi$ for the NB.
    n : int
        Number of samples.
    model : str
        One of `OBS_MODELS`.

    Returns
    -------
    jax.Array
        Samples, shape `(n, *mean.shape)`, floored at 0.

    Raises
    ------
    ValueError
        If `model` is not one of `OBS_MODELS`.
    """
    if model == "skellam":
        return sample_skellam_moments(key, mean, obs, spread, n)
    if model not in OBS_MODELS:
        raise ValueError(f"Unknown observation model {model!r}.")
    rem = mean - obs
    shape = (n, *jnp.shape(mean))
    if model == "skellam_rem":
        extra = spread * jnp.abs(rem)
        up, down = jax.random.split(key)
        rise = jax.random.poisson(up, jnp.maximum(rem, 0.0) + extra, shape)
        fall = jax.random.poisson(down, jnp.maximum(-rem, 0.0) + extra, shape)
        return jnp.maximum(obs + rise - fall, 0).astype(mean.dtype)
    pos = jnp.maximum(rem, 0.0)
    k_gamma, k_count = jax.random.split(key)
    inv = jnp.broadcast_to(spread, jnp.shape(mean))
    size = jnp.where(inv > 0, 1.0 / jnp.where(inv > 0, inv, 1.0), 1.0)
    mix = jax.random.gamma(k_gamma, size, shape) / size
    rate = pos * jnp.where(inv > 0, mix, 1.0)
    add = jax.random.poisson(k_count, rate, shape)
    return jnp.maximum(obs + add, 0).astype(mean.dtype)
