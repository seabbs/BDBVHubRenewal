# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Two-part Skellam observation model on the remaining count.

It plugs into baselinenowcast's uncertainty interface.
`fit_skellam` takes the place of the port's `fit_nb`, so that
`fit_by_horizon(obs, pred, fit_skellam)` fits one pair of rates a
horizon.
`sample_skellam` takes the place of `sample_nb`, with the same
`(key, pred, uncertainty_params)` arguments.
As in the port, `obs` and `pred` come from retrospective nowcasts, and
the draw is the remaining count, which the caller adds to the count
reported.
Here their last axis has two entries.
`obs` holds the realised additions and removals, and `pred` the nowcast
remaining count and the count reported.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
nowcast remaining count is $\hat r = \mu - Y$.
The count at the later release is

$$
X = \max\big(0, Y + A - B\big), \quad
A \sim \mathrm{Pois}(a_k \hat r), \quad
B \sim \mathrm{Pois}(b_k Y),
$$

so a revision can go down.
$A$ are the cases added to the week's onset days, and $B$ the cases
removed from them.
Each rate is fitted for each number of weeks back $k$ by Poisson
maximum likelihood, from nowcasts made at earlier releases.
With $\hat r$ and $Y$ of an earlier nowcast, and $u$ and $d$ the sums
over the week's days of the rises and the falls of each day's count up
to the origin,

$$
a_k = \frac{\sum u}{\sum \hat r}, \qquad b_k = \frac{\sum d}{\sum Y}.
$$

So the downward side is set by the observed removals alone.
With no errors, or nothing to scale by, $a_k = 1$ and $b_k = 0$.

Baselinenowcast 0.2.2 has no Skellam model, so this one lives outside
the port.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["NO_FIT", "fit_skellam", "n_errors", "sample_skellam"]

NO_FIT = (1.0, 0.0)
"""Rates $(a_k, b_k)$ with no errors: a Poisson nowcast, no removals."""


def n_errors(obs: np.ndarray) -> np.ndarray:
    """Return the number of retrospective nowcasts behind a fit.

    Parameters
    ----------
    obs : numpy.ndarray
        Realised additions and removals, `[N, ..., 2]`, NaN where missing.

    Returns
    -------
    numpy.ndarray
        Rows with finite values, `[...]`.
    """
    return np.isfinite(np.asarray(obs)).all(axis=-1).sum(axis=0)


def fit_skellam(obs: np.ndarray, pred: np.ndarray) -> Array:
    r"""Fit the rates of the additions and removals of one horizon.

    Same interface as the port's `fit_nb`, so it can be passed to
    `fit_by_horizon`.

    Parameters
    ----------
    obs : numpy.ndarray
        Realised additions $u$ and removals $d$ of each retrospective
        nowcast, `[N, 2]`. Rows with a NaN are left out.
    pred : numpy.ndarray
        Nowcast remaining count $\hat r$ and count reported $Y$, `[N, 2]`.

    Returns
    -------
    jax.Array
        $(a_k, b_k)$, `NO_FIT` where a sum to scale by is 0.
    """
    o = np.asarray(obs, dtype=float).reshape(-1, 2)
    p = np.asarray(pred, dtype=float).reshape(-1, 2)
    keep = np.isfinite(o).all(axis=1) & np.isfinite(p).all(axis=1)
    got = o[keep].sum(axis=0)
    by = np.maximum(p[keep], 0.0).sum(axis=0)
    safe = np.where(by > 0, by, 1.0)
    return jnp.asarray(np.where(by > 0, got / safe, np.asarray(NO_FIT)))


def sample_skellam(key: Array, pred: Array, uncertainty_params: Array) -> Array:
    r"""Draw the remaining count, $A - B$.

    Same interface as the port's `sample_nb`.

    Parameters
    ----------
    key : jax.Array
        Random key.
    pred : jax.Array
        Nowcast remaining count $\hat r$ and count reported $Y$,
        `[..., 2]`.
    uncertainty_params : jax.Array
        $(a_k, b_k)$, `[2]` or `[..., 2]`.

    Returns
    -------
    jax.Array
        Draws, `[...]`. They can be negative, so the caller adds the count
        reported and floors at 0.
    """
    p = jnp.asarray(pred)
    rates = jnp.asarray(uncertainty_params, dtype=p.dtype)
    lam = jnp.maximum(p, 0.0) * rates
    up, down = jax.random.split(key)
    shape = p.shape[:-1]
    rise = jax.random.poisson(up, jnp.broadcast_to(lam[..., 0], shape))
    fall = jax.random.poisson(down, jnp.broadcast_to(lam[..., 1], shape))
    return (rise - fall).astype(p.dtype)
