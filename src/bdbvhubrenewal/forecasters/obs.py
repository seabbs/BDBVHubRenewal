# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Observation model: new reports on the count still to come, less removals.

For a week with count $Y$ reported so far and a nowcast mean $\mu$, the
count still to come is $\mathrm{rem} = \mu - Y \ge 0$.
The count at a later release is

$$
X = Y + A - B,
\qquad
A \sim \mathrm{NB}(\mathrm{rem}, \phi_k),
\qquad
B \sim \mathrm{Binomial}(Y, p_k).
$$

$A$ is baselinenowcast's negative binomial on the remaining count, fitted
and drawn with the port's `fit_nb` and `sample_nb` through
`fit_by_horizon`, one size $\phi_k$ for each week back $k$.
As in baselinenowcast, downward revisions are moved onto earlier releases
before the size is fitted (`chain._delay_counts`), so the removals never
enter it.
$B$ thins the count reported at $p_k$, the net fall of the raw count as a
share of the count reported in the retrospective nowcasts at $k$, pooled
over the places at one depth of the tree (`removal_rates`).
"""

from __future__ import annotations

import jax
import numpy as np
from jax import Array

from bdbvhubrenewal.baselinenowcast import fit_by_horizon, sample_nb

__all__ = [
    "MAX_SIZE",
    "PRIOR_SIZE",
    "fit_sizes",
    "removal_rates",
    "sample_count",
]

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


def depths(parent: np.ndarray) -> np.ndarray:
    """Return each node's depth in the tree, 0 for a root.

    Parameters
    ----------
    parent : numpy.ndarray
        Parent of each node, -1 for a root, `[L]`.

    Returns
    -------
    numpy.ndarray
        Depth of each node, `[L]`.
    """
    depth = np.zeros(parent.size, dtype=int)
    for n, p in enumerate(parent):
        while p >= 0:
            depth[n] += 1
            p = parent[p]
    return depth


def removal_rates(
    fell: np.ndarray,
    reported: np.ndarray,
    valid: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    r"""Return the removal rate $p_k$ of each place, pooled by depth.

    $p_k = \sum B / \sum Y$ over the errors at $k$ of every place at the
    place's depth in the tree (country, province, zone), the Binomial
    maximum likelihood estimate.

    Parameters
    ----------
    fell : numpy.ndarray
        Net fall of each week's count between each release and the
        origin, `[R, L, K]`.
    reported : numpy.ndarray
        Count reported at each release, `[R, L, K]`.
    valid : numpy.ndarray
        Whether each cell is an error, `[R, L, K]`.
    parent : numpy.ndarray
        Parent of each place, -1 for the country, `[L]`.

    Returns
    -------
    numpy.ndarray
        $p_k$, `[L, K]`, 0 where a depth has nothing reported in its errors.
    """
    fell_sum = np.where(valid, fell, 0.0).sum(axis=0)
    rep_sum = np.where(valid, reported, 0.0).sum(axis=0)
    depth = depths(np.asarray(parent))
    rate = np.zeros(fell_sum.shape)
    for d in np.unique(depth):
        m = depth == d
        num, den = fell_sum[m].sum(axis=0), rep_sum[m].sum(axis=0)
        rate[m] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return rate


def sample_count(
    key: Array,
    reported: Array,
    remaining: Array,
    size: Array,
    removal: Array,
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
    removal : jax.Array
        $p_k$, the shape of `reported`.

    Returns
    -------
    jax.Array
        $X$, the shape of `reported`, never below 0.
    """
    removed = jax.random.binomial(
        jax.random.fold_in(key, 1), reported, removal, reported.shape
    )
    return reported + sample_nb(key, remaining, size) - removed
