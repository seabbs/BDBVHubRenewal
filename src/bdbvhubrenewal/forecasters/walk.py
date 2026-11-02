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
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["random_walk"]


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
