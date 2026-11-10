# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Observation model: new reports on the count still to come.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
count still to come is $\mathrm{rem} = \mu - Y \ge 0$.
The count at a later release is

$$
X = Y + A,
\qquad
A \sim \mathrm{NB}(\mathrm{rem}, \phi_k),
$$

baselinenowcast's negative binomial on the remaining count, fitted and
drawn with the port's `fit_nb` and `sample_nb` through `fit_by_horizon`,
one size $\phi_k$ for each week back $k$.
As in baselinenowcast, downward revisions are moved onto earlier releases
before the fit (`chain._delay_counts`), so new reports only add.
"""

from __future__ import annotations

import numpy as np
from jax import Array

from bdbvhubrenewal.baselinenowcast import fit_by_horizon, sample_nb

__all__ = ["MAX_SIZE", "PRIOR_SIZE", "fit_sizes", "sample_count"]

MAX_SIZE = 1000.0
"""Upper bound of `fit_nb`'s size, the Poisson end of its range."""

PRIOR_SIZE = 1.0 / 0.6745**2
r"""Size with no errors to fit one, about 2.2.

It is the median of epinowcast's prior $1 / \sqrt{\phi} \sim
\mathcal{N}^+(0, 1)$.
"""


def fit_sizes(added: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    r"""Fit the size $\phi_k$ of each week back by maximum likelihood.

    Parameters
    ----------
    added : numpy.ndarray
        New reports after each retrospective nowcast, `[N, K]`.
    remaining : numpy.ndarray
        Nowcast of the count still to come, `[N, K]`. A cell with 0 and
        no new reports adds nothing to the likelihood, so it pads.

    Returns
    -------
    numpy.ndarray
        $\phi_k$, `[K]`.
    """
    fitted = np.asarray(fit_by_horizon(added, remaining))
    return np.where((remaining > 0).any(axis=0), fitted, PRIOR_SIZE)


def sample_count(
    key: Array, reported: Array, remaining: Array, size: Array
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

    Returns
    -------
    jax.Array
        $X$, the shape of `reported`.
    """
    return reported + sample_nb(key, remaining, size)
