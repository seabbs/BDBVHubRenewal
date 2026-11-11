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

A place's own sizes are pooled over its siblings by empirical Bayes
(`pooled_sizes`): with $x_{p,k} = \log \phi_{p,k}$,

$$
x_{p,k} \sim \mathcal{N}(\mu_k, s_k^2),
$$

$\mu_k$ the pooled maximum likelihood estimate and $s_k$ the marginal
maximum likelihood estimate over the siblings, each place's NB likelihood
integrated on a grid of $x$. Each place takes its posterior mean, so a
place with few errors moves towards its siblings' size and one with many
keeps its own.
"""

from __future__ import annotations

import numpy as np
from jax import Array
from scipy.special import gammaln, logsumexp

from bdbvhubrenewal.baselinenowcast import fit_by_horizon, sample_nb

__all__ = [
    "MAX_SIZE",
    "PRIOR_SIZE",
    "fit_sizes",
    "pooled_sizes",
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


MIN_SIZE = 0.1
"""Lower bound of `fit_nb`'s size."""

_GRID = np.linspace(np.log(MIN_SIZE), np.log(MAX_SIZE), 241)
r"""Grid of $\log \phi$ on `fit_nb`'s range."""

_SPREADS = np.r_[0.0, np.geomspace(0.02, 5.0, 60)]
r"""Grid of the spread $s_k$ of $\log \phi$ between siblings."""


def _nb_loglik(added: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    """Return the NB log likelihood of each column on `_GRID`, `[K, G]`.

    Cells with nothing nowcast to come pad.
    """
    size = np.exp(_GRID)[None, None, :]
    mu = remaining[..., None]
    x = added[..., None]
    use = (remaining > 0)[..., None]
    safe = np.where(use, mu, 1.0)
    ll = (
        gammaln(x + size)
        - gammaln(size)
        - gammaln(x + 1.0)
        + size * np.log(size / (size + safe))
        + x * np.log(safe / (size + safe))
    )
    out: np.ndarray = np.where(use, ll, 0.0).sum(axis=0)
    return out


def pooled_sizes(added: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    r"""Pool sibling places' sizes $\phi_k$ by empirical Bayes.

    For each week back $k$, $\log \phi_{p,k} \sim \mathcal{N}(\mu_k,
    s_k^2)$ over the places $p$. $\mu_k$ maximises the pooled likelihood,
    and $s_k$ the marginal likelihood
    $\prod_p \int L_{p,k}(x) \, \mathcal{N}(x; \mu_k, s_k^2) \, dx$, both
    on a grid of $x = \log \phi$ over `fit_nb`'s range. Each place takes
    $\exp \mathbb{E}[x_{p,k}]$ under its posterior. $s_k = 0$ is the
    pooled size for all, and a large $s_k$ each place's own.

    Parameters
    ----------
    added : numpy.ndarray
        New reports after each retrospective nowcast, `[N, P, K]`.
    remaining : numpy.ndarray
        Nowcast of the count still to come, `[N, P, K]`. A cell with 0
        pads.

    Returns
    -------
    numpy.ndarray
        $\phi_{p,k}$, `[P, K]`. `PRIOR_SIZE` where no place has an error.
    """
    added = np.asarray(added, dtype=float)
    remaining = np.asarray(remaining, dtype=float)
    _, n_p, n_k = remaining.shape
    has = (remaining > 0).any(axis=0)  # [P, K]
    ll = np.stack(
        [_nb_loglik(added[:, p], remaining[:, p]) for p in range(n_p)]
    )  # [P, K, G]
    out = np.full((n_p, n_k), PRIOR_SIZE)
    for k in range(n_k):
        if not has[:, k].any():
            continue
        mine = ll[has[:, k], k]  # [P', G]
        mu = int(np.argmax(mine.sum(axis=0)))
        best, prior = -np.inf, np.zeros(_GRID.size)
        for s in _SPREADS:
            if s == 0.0:
                lp = np.full(_GRID.size, -np.inf)
                lp[mu] = 0.0
            else:
                lp = -0.5 * ((_GRID - _GRID[mu]) / s) ** 2
                lp = lp - logsumexp(lp)
            marg = float(logsumexp(mine + lp[None, :], axis=1).sum())
            if marg > best + 1e-9:
                best, prior = marg, lp
        post = ll[:, k] + prior[None, :]
        w = np.exp(post - logsumexp(post, axis=1, keepdims=True))
        out[:, k] = np.exp(w @ _GRID)
    return out


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
