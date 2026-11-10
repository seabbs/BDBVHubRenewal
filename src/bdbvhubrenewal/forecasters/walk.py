# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The random walk, one function for both models.

From start value $x_i$ per sample, for $h \ge 1$,

$$
\log(1 + \lambda_i(h)) = \max\Big(0, \log(1 + x_i)
+ \sum_{j=1}^{h} \epsilon_{i,j}\Big),
\quad \epsilon \sim \mathcal{N}(0, \sigma^2),
$$

$$
y_i(h) \sim \mathrm{Poisson}(\lambda_i(h)).
$$

Steps are cumulative within a sample, so each sample is one trajectory.

The spread $\sigma$ comes from the weekly changes $\Delta_t$ of
$\log(1 + y_t)$ in reported counts, net of their Poisson noise,

$$
\sigma^2 = \widehat{\mathrm{Var}}(\Delta_t)
- \overline{v(y_{t-1}) + v(y_t)},
\quad v(y) = \frac{y}{(1 + y)^2},
$$

where $v$ is the delta-method variance of $\log(1 + Y)$ for
$Y \sim \mathrm{Poisson}(\lambda)$, $\lambda / (1 + \lambda)^2$, with the
observed count for $\lambda$ (`walk_variance`). The walk draws its Poisson
counts on top, so the noise is not counted twice.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["random_walk", "random_walks", "walk_noise", "walk_variance"]


def walk_noise(y: np.ndarray) -> np.ndarray:
    r"""Return the Poisson noise variance of each change of log(1 + y).

    Parameters
    ----------
    y : numpy.ndarray
        Weekly counts, `[W]`.

    Returns
    -------
    numpy.ndarray
        $v(y_{t-1}) + v(y_t)$ for each change, `[W - 1]`, with
        $v(y) = y / (1 + y)^2$ the delta-method variance of
        $\log(1 + Y)$ for $Y \sim \mathrm{Poisson}(y)$.
    """
    y = np.asarray(y, dtype=float)
    v = y / (1.0 + y) ** 2
    return v[1:] + v[:-1]


def walk_variance(changes: np.ndarray, noise: np.ndarray) -> float:
    """Return the variance of the walk's steps net of Poisson noise.

    Parameters
    ----------
    changes : numpy.ndarray
        Weekly changes of log(1 + y), `[N]`.
    noise : numpy.ndarray
        Their Poisson noise variances (`walk_noise`), `[N]`.

    Returns
    -------
    float
        The sample variance of `changes` less the mean of `noise`. It
        may be 0 or below, which says the changes hold no more than
        Poisson noise. NaN with fewer than 2 changes.
    """
    if changes.size < 2:
        return float("nan")
    return float(np.var(changes, ddof=1)) - float(np.mean(noise))


def random_walk(
    key: Array, x0: Array, sigma: Array | float, h: int
) -> tuple[Array, Array]:
    r"""Draw the walk forward on the log scale with Poisson counts.

    Parameters
    ----------
    key : jax.Array
        Random key.
    x0 : jax.Array
        Start value of each sample, shape `[S]`.
    sigma : float or jax.Array
        Standard deviation of a weekly step.
    h : int
        Weeks to draw.

    Returns
    -------
    tuple of jax.Array
        The rate $\lambda$ and the Poisson count, each `[S, h]`.
    """
    k_step, k_count = jax.random.split(key)
    steps = sigma * jax.random.normal(k_step, (x0.shape[0], h), x0.dtype)
    level = jnp.log1p(x0)[:, None] + jnp.cumsum(steps, axis=1)
    lam = jnp.expm1(jnp.maximum(level, 0.0))
    return lam, jax.random.poisson(k_count, lam).astype(x0.dtype)


@partial(jax.jit, static_argnums=3)
def random_walks(
    keys: Array, x0: Array, sigma: Array, h: int
) -> tuple[Array, Array]:
    """Draw `random_walk` for each place, compiled as one program.

    Parameters
    ----------
    keys : jax.Array
        One random key per place, `[L]`.
    x0 : jax.Array
        Start values, `[L, S]`.
    sigma : jax.Array
        Step standard deviation of each place, `[L]`.
    h : int
        Weeks to draw.

    Returns
    -------
    tuple of jax.Array
        The rates and the Poisson counts, each `[L, S, h]`.
    """
    return jax.vmap(lambda k, x, s: random_walk(k, x, s, h))(keys, x0, sigma)
