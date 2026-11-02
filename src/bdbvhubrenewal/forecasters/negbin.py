# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Negative binomial observation model on the remaining count.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
remaining count is $\mathrm{rem}^+ = \max(\mu - Y, 0)$.
The count at the later release is

$$
X = Y + \mathrm{NB}(\mathrm{rem}^+, \phi),
$$

the observation model of baselinenowcast, applied to the remaining count.
With $\hat y$ the earlier nowcast, $y$ the later count, and errors
$e = y - \hat y$, the dispersion $1/\phi$ is fitted by moments,
pooled over the weeks back,

$$
\frac{1}{\phi} = \max\Big(0, \frac{\sum (e^2 - \mathrm{rem}^+)}
{\sum (\mathrm{rem}^+)^2}\Big).
$$

With $1/\phi = 0$ the count is Poisson.
The sampler is the port's `sample_nb`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bdbvhubrenewal.baselinenowcast import sample_nb

__all__ = [
    "inv_phi_from_sums",
    "inv_phi_sums",
    "sample_nowcast",
]


def inv_phi_sums(
    obs: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the sums behind the dispersion.

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
        The sums of $e^2 - \mathrm{rem}^+$ and of $(\mathrm{rem}^+)^2$,
        and the number of errors.
    """
    keep = np.ones(np.shape(obs), dtype=bool) if valid is None else valid
    rem = np.maximum(pred - obs, 0.0)
    err = target - pred
    num = np.where(keep, err**2 - rem, 0.0).sum(axis=axis)
    den = np.where(keep, rem**2, 0.0).sum(axis=axis)
    return num, den, keep.sum(axis=axis)


def inv_phi_from_sums(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    r"""Return $1/\phi$ from the sums, 0 where the denominator is 0."""
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, np.maximum(0.0, num / safe), 0.0)


def sample_nowcast(
    key: Array, mean: Array, obs: Array, inv_phi: Array, n: int
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
    inv_phi : jax.Array
        Dispersion $1/\phi$ of each week. 0 gives a Poisson count.
    n : int
        Number of samples.

    Returns
    -------
    jax.Array
        Samples, shape `(n, *mean.shape)`, at least the count reported.
    """
    shape = (n, *jnp.shape(mean))
    rem = jnp.broadcast_to(jnp.maximum(mean - obs, 0.0), shape)
    inv = jnp.broadcast_to(inv_phi, shape)
    size = 1.0 / jnp.where(inv > 0, inv, 1.0)
    k_nb, k_pois = jax.random.split(key)
    nb = sample_nb(k_nb, rem, size)
    pois = jax.random.poisson(k_pois, rem).astype(rem.dtype)
    return (obs + jnp.where(inv > 0, nb, pois)).astype(jnp.asarray(mean).dtype)
