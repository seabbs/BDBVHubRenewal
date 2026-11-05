# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The release-pair likelihood that the learnt publication lag is fitted to.

A delay-file row with report day $t$ is in the release with last report
day $T$ with probability $F_\ell(T - t)$, with $F_\ell$ the cdf of the
report to publication lag. Given that a row is in release $k + 1$, it was
already in release $k$ with probability

$$q = (1 - b_k) \frac{\sum_u F_\ell(T_k - t_u)}{\sum_u F_\ell(T_{k+1} - t_u)},$$

averaged over the onset days $u$ of the row's onset week, with $t_u$ the
report day for onset $u$. $b_k$ is the share of rows that appear late
whatever their report age, such as onsets recorded later for cases first
counted as imputed. The kept rows of each cell are beta-binomial with
mean $q$, so that rows arriving in batches do not make the lag
overconfident. The term conditions on the later release, so it adds
information on $F_\ell$ and does not count the latest release twice.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


def kept_share(
    pmf: ArrayLike, background: ArrayLike, pairs: dict[str, Array]
) -> Array:
    r"""Probability that each row cell was already in the earlier release.

    Parameters
    ----------
    pmf : ArrayLike
        Publication lag pmf on $0..L$.
    background : ArrayLike
        $b_k$ for each pair, `(K,)`.
    pairs : dict
        `before` and `after` report ages, `(n, 7)`, and the one-hot
        `pair` of each cell, `(n, K)` (`data.releases.release_pairs`).

    Returns
    -------
    Array
        $q$ for each cell, `(n,)`.
    """
    cdf = jnp.cumsum(jnp.asarray(pmf))
    last = cdf.shape[0] - 1

    def published(ages: Array) -> Array:
        ages = jnp.asarray(ages)
        return jnp.where(ages < 0, 0.0, cdf[jnp.clip(ages, 0, last)]).sum(-1)

    b = jnp.asarray(pairs["pair"]) @ jnp.asarray(background)
    later = jnp.maximum(published(pairs["after"]), 1e-12)
    return (1 - b) * published(pairs["before"]) / later


class ReleasePairLikelihood:
    r"""Beta-binomial likelihood of the rows each release kept of the last.

    The backgrounds $b_k$ are `{name}_background`, one per pair from
    `background`. The beta-binomial has concentration $1 / \phi^2$, with
    $\phi$ from `overdispersion_rv`.
    """

    def __init__(
        self,
        name: str,
        background: dist.Distribution,
        overdispersion_rv: RandomVariable,
    ) -> None:
        r"""Initialize the likelihood.

        Parameters
        ----------
        name
            Prefix for the sample sites.
        background
            Prior of each pair's background share $b_k$.
        overdispersion_rv
            $\phi$, the beta-binomial's overdispersion.

        Raises
        ------
        TypeError
            If `overdispersion_rv` is not a `RandomVariable`.
        """
        if not isinstance(overdispersion_rv, RandomVariable):
            raise TypeError("overdispersion_rv must be a RandomVariable.")
        self.name = name
        self.background = background
        self.overdispersion_rv = overdispersion_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ReleasePairLikelihood(name={self.name!r}, "
            f"background={self.background!r}, "
            f"overdispersion_rv={self.overdispersion_rv!r})"
        )

    def score(self, pmf: ArrayLike, pairs: dict[str, Array]) -> None:
        """Score the kept rows of each cell.

        Parameters
        ----------
        pmf : ArrayLike
            Publication lag pmf on $0..L$.
        pairs : dict
            As `kept_share`, with `kept` and `total` rows, `(n,)`.
        """
        n_pairs = jnp.shape(pairs["pair"])[1]
        background = numpyro.sample(
            f"{self.name}_background", self.background.expand([n_pairs])
        )
        q = jnp.clip(kept_share(pmf, background, pairs), 1e-6, 1 - 1e-6)
        concentration = 1 / jnp.square(self.overdispersion_rv())
        numpyro.sample(
            f"{self.name}_kept",
            dist.BetaBinomial(
                q * concentration,
                (1 - q) * concentration,
                total_count=jnp.asarray(pairs["total"]),
            ),
            obs=jnp.asarray(pairs["kept"]),
        )
