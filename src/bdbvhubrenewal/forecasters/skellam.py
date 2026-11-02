# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Skellam observation model on the remaining count.

It plugs into baselinenowcast's uncertainty interface.
`fit_skellam` takes the place of the port's `fit_nb`, so that
`fit_by_horizon(obs, pred, fit_skellam)` fits one spread a horizon.
`sample_skellam` takes the place of `sample_nb`, with the same
`(key, pred, uncertainty_params)` arguments.
As in the port, `obs` and `pred` are the realised and the nowcast
remaining count of a retrospective nowcast, and the draw is the
remaining count, which the caller adds to the count reported.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
remaining count is $\mathrm{rem} = \mu - Y$.
The count at the later release is

$$
X = \max\big(0, Y + \mathrm{Pois}(\mathrm{rem}^+ + \rho|\mathrm{rem}|)
- \mathrm{Pois}(\mathrm{rem}^- + \rho|\mathrm{rem}|)\big),
$$

so a revision can go down.
The variance is $|\mathrm{rem}| + 2\rho|\mathrm{rem}|$, so a finished
week has no spread.
$\rho \ge 0$ is fitted for each number of weeks back by moments from
nowcasts made at earlier releases.
With $\hat r$ the earlier nowcast of the remaining count, $r$ the
realised one and $e = r - \hat r$,

$$
\rho = \max\Big(0, \frac{\sum (e^2 - |\hat r|)}{\sum 2|\hat r|}\Big).
$$

Baselinenowcast 0.2.2 has no Skellam model, so this one lives outside
the port.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "fit_skellam",
    "rho_from_sums",
    "rho_sums",
    "sample_skellam",
]


def rho_sums(
    obs: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the sums behind the spread.

    Parameters
    ----------
    obs : numpy.ndarray
        Realised remaining count, $r$.
    pred : numpy.ndarray
        Nowcast of the remaining count, $\hat r$.
    valid : numpy.ndarray, optional
        Which errors to keep. Defaults to all.
    axis : int
        Axis to sum over.

    Returns
    -------
    tuple of numpy.ndarray
        The sums of $e^2 - |\hat r|$ and of $2|\hat r|$, and the number of
        errors.
    """
    keep = np.ones(np.shape(obs), dtype=bool) if valid is None else valid
    err = obs - pred
    num = np.where(keep, err**2 - np.abs(pred), 0.0).sum(axis=axis)
    den = np.where(keep, 2.0 * np.abs(pred), 0.0).sum(axis=axis)
    return num, den, keep.sum(axis=axis)


def rho_from_sums(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    r"""Return $\rho$ from the sums, 0 where the denominator is 0."""
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, np.maximum(0.0, num / safe), 0.0)


def fit_skellam(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    r"""Fit the Skellam spread of one horizon by moments.

    Same interface as the port's `fit_nb`, so it can be passed to
    `fit_by_horizon`.

    Parameters
    ----------
    obs, pred : numpy.ndarray
        Realised and nowcast remaining count of each retrospective
        nowcast.

    Returns
    -------
    numpy.ndarray
        $\rho$, 0 with no errors.
    """
    num, den, _ = rho_sums(np.asarray(obs), np.asarray(pred))
    return rho_from_sums(num, den)


def sample_skellam(key: Array, pred: Array, uncertainty_params: Array) -> Array:
    r"""Draw the remaining count.

    Same interface as the port's `sample_nb`.

    Parameters
    ----------
    key : jax.Array
        Random key.
    pred : jax.Array
        Nowcast of the remaining count, $\mathrm{rem}$, of any shape.
    uncertainty_params : jax.Array
        $\rho$, of length 1 or the shape of `pred`.

    Returns
    -------
    jax.Array
        Draws of the shape of `pred`. They can be negative, so the caller
        adds the count reported and floors at 0.
    """
    rem = jnp.asarray(pred)
    rho = jnp.asarray(uncertainty_params, dtype=rem.dtype)
    extra = rho * jnp.abs(rem)
    up, down = jax.random.split(key)
    rise = jax.random.poisson(up, jnp.maximum(rem, 0.0) + extra, rem.shape)
    fall = jax.random.poisson(down, jnp.maximum(-rem, 0.0) + extra, rem.shape)
    return (rise - fall).astype(rem.dtype)
