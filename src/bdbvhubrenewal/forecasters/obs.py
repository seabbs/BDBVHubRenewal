# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Observation model: new reports minus removals.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
count still to come is $\mathrm{rem} = \mu - Y \ge 0$.
The count at a later release is

$$
X = \max\big(0,\ Y + A - B\big),
\qquad
A \sim \mathrm{NB}(\mathrm{rem}, \phi_k),
\qquad
B \sim \mathrm{Poisson}(b_k Y),
$$

as new reports minus removals in epinowcast.
$A$ is baselinenowcast's negative binomial on the remaining count, fitted
and drawn with the port's `fit_nb` and `sample_nb` through
`fit_by_horizon`, one size $\phi_k$ for each week back $k$.
$b_k$ is the share of the reported count later removed, the Poisson
maximum likelihood estimate $\sum B / \sum Y$ from the falls of weekly
counts between earlier releases and the origin.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bdbvhubrenewal.baselinenowcast import fit_by_horizon, sample_nb

__all__ = ["MAX_SIZE", "fit_sizes", "rate", "sample_count"]

MAX_SIZE = 1000.0
"""Upper bound of `fit_nb`'s size, the Poisson end of its range."""


def fit_sizes(
    added: np.ndarray, remaining: np.ndarray, min_errors: int
) -> np.ndarray:
    r"""Fit the size $\phi_k$ of each week back by maximum likelihood.

    Parameters
    ----------
    added : numpy.ndarray
        New reports after each retrospective nowcast, `[N, K]`.
    remaining : numpy.ndarray
        Nowcast of the count still to come, `[N, K]`. A cell with 0 and
        no new reports adds nothing to the likelihood, so it pads.
    min_errors : int
        Cells with something to come a fit needs. With fewer, as before
        any earlier release chains to the origin, the size is `MAX_SIZE`.

    Returns
    -------
    numpy.ndarray
        $\phi_k$, `[K]`.
    """
    fitted = np.asarray(fit_by_horizon(added, remaining))
    return np.where((remaining > 0).sum(axis=0) >= min_errors, fitted, MAX_SIZE)


def rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return `num / den`, 0 where `den` is 0.

    Parameters
    ----------
    num, den : numpy.ndarray
        Numerator and denominator, broadcast together.

    Returns
    -------
    numpy.ndarray
        The ratio.
    """
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def sample_count(
    key: Array, reported: Array, remaining: Array, size: Array, b: Array
) -> Array:
    r"""Draw the count at the later release.

    Parameters
    ----------
    key : jax.Array
        Random key.
    reported : jax.Array
        $Y$, of any shape.
    remaining : jax.Array
        $\mathrm{rem}$, the shape of `reported`.
    size : jax.Array
        $\phi_k$, the shape of `reported`.
    b : jax.Array
        $b_k$, the shape of `reported`.

    Returns
    -------
    jax.Array
        $X$, the shape of `reported`.
    """
    k_add, k_rm = jax.random.split(key)
    added = sample_nb(k_add, remaining, size)
    removed = jax.random.poisson(k_rm, b * reported, reported.shape)
    return jnp.maximum(reported + added - removed, 0.0)
