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
\sigma^2 = \max\Big(\widehat{\mathrm{Var}}(\Delta_t)
- \overline{2 v(m_t)},\ \sigma_{\min}^2\Big),
\quad m_t = \tfrac{1}{2}(y_{t-1} + y_t),
$$

where $v(\lambda)$ is the variance of $\log(1 + Y)$ for
$Y \sim \mathrm{Poisson}(\lambda)$ (`walk_spread`). The walk draws its
Poisson counts on top, so the noise is not counted twice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import stats

__all__ = ["log1p_poisson_var", "random_walk", "walk_noise", "walk_spread"]


def log1p_poisson_var(lam: np.ndarray) -> np.ndarray:
    """Return the variance of log(1 + Y) for Y ~ Poisson(`lam`).

    Parameters
    ----------
    lam : numpy.ndarray
        Poisson means, any shape.

    Returns
    -------
    numpy.ndarray
        The variances, the shape of `lam`, summed exactly over the
        Poisson probabilities.
    """
    lam = np.asarray(lam, dtype=float)
    if lam.size == 0:
        return np.zeros(lam.shape)
    top = float(lam.max())
    k = np.arange(int(top + 12.0 * np.sqrt(top) + 30.0))
    p = stats.poisson.pmf(k, lam.reshape(-1, 1))
    f = np.log1p(k)
    mean = p @ f
    var = np.asarray(p @ f**2 - mean**2, dtype=float)
    return np.clip(var, 0.0, None).reshape(lam.shape)


def walk_noise(y: np.ndarray) -> np.ndarray:
    """Return the Poisson noise variance of each change of log(1 + y).

    Parameters
    ----------
    y : numpy.ndarray
        Weekly counts, `[W]`.

    Returns
    -------
    numpy.ndarray
        $2 v(m_t)$ for each change, `[W - 1]`, with $m_t$ the mean of
        the two weeks.
    """
    y = np.asarray(y, dtype=float)
    return 2.0 * log1p_poisson_var(0.5 * (y[1:] + y[:-1]))


def walk_spread(changes: np.ndarray, noise: np.ndarray, floor: float) -> float:
    r"""Return the walk spread net of Poisson noise.

    Parameters
    ----------
    changes : numpy.ndarray
        Weekly changes of log(1 + y), `[N]`.
    noise : numpy.ndarray
        Their Poisson noise variances (`walk_noise`), `[N]`.
    floor : float
        Least spread, also used with fewer than 2 changes.

    Returns
    -------
    float
        $\sigma$, at least `floor`.
    """
    if changes.size < 2:
        return floor
    net = float(np.var(changes, ddof=1)) - float(np.mean(noise))
    return float(np.sqrt(max(net, floor**2)))


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
