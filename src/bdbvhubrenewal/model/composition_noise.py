# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""
Noise models for counts split into categories with known totals.

Written as PyRenew's count noise models (`pyrenew.observation.noise`),
for the share of each total that falls in each category, such as cases
by reporting delay given cases by onset week.

- ``MultinomialNoise``: multinomial counts. No parameters.
- ``DirichletMultinomialNoise``: overdispersed relative to the
  multinomial. Takes ``concentration_rv`` (higher concentration is less
  overdispersed, more multinomial-like).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

from .distributions import ObservedCellsDirichletMultinomial


class CompositionNoise(ABC):
    """
    Abstract base for noise models of counts split into categories.

    Defines how counts in each category are distributed given the
    category probabilities and the total of each row.
    """

    @abstractmethod
    def validate(self) -> None:
        """
        Check the noise parameters.

        Raises
        ------
        ValueError
            If a parameter is out of range.
        """
        pass  # pragma: no cover

    @abstractmethod
    def sample(
        self,
        name: str,
        probs: ArrayLike,
        total_count: ArrayLike,
        obs: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample category counts given probabilities and totals.

        Parameters
        ----------
        name
            Numpyro sample site name.
        probs
            Category probabilities, the last axis summing to one.
        total_count
            Total count of each row, the shape of ``probs`` without its
            last axis.
        obs
            Observed counts for conditioning, or None for prior sampling.
        mask
            Boolean mask of the rows to include in the likelihood. If
            None, all rows are included.

        Returns
        -------
        ArrayLike
            Counts with the shape of ``probs``.
        """
        pass  # pragma: no cover


class MultinomialNoise(CompositionNoise):
    """Multinomial noise for counts split into categories."""

    def __repr__(self) -> str:
        """Return string representation."""
        return "MultinomialNoise()"

    @staticmethod
    def validate() -> None:
        """MultinomialNoise always passes validation."""

    def sample(
        self,
        name: str,
        probs: ArrayLike,
        total_count: ArrayLike,
        obs: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample from the multinomial distribution.

        Parameters
        ----------
        name
            Numpyro sample site name.
        probs
            Category probabilities, the last axis summing to one.
        total_count
            Total count of each row.
        obs
            Observed counts for conditioning.
        mask
            Boolean mask of the rows to include in the likelihood. If
            None, all rows are included.

        Returns
        -------
        ArrayLike
            Multinomial counts.
        """
        with numpyro.handlers.mask(mask=True if mask is None else mask):
            counts: ArrayLike = numpyro.sample(
                name,
                dist.Multinomial(total_count=total_count, probs=probs),
                obs=obs,
            )
        return counts


class DirichletMultinomialNoise(CompositionNoise):
    """
    Dirichlet-multinomial noise for overdispersed category counts.

    The category probabilities are Dirichlet with concentration
    ``concentration * probs``, so the mean share of each category is
    ``probs`` and the variance of the counts exceeds the multinomial's.
    The log density is evaluated on the non-zero cells of the observed
    counts only (``ObservedCellsDirichletMultinomial``), which gives the
    same value at a fraction of the cost for sparse tables.

    Parameters
    ----------
    concentration_rv
        Concentration (must be > 0). Higher values reduce
        overdispersion; as it grows the noise approaches the multinomial.
    """

    def __init__(self, concentration_rv: RandomVariable) -> None:
        """
        Initialize Dirichlet-multinomial noise.

        Parameters
        ----------
        concentration_rv
            Concentration (must be > 0).

        Raises
        ------
        TypeError
            If ``concentration_rv`` is not a ``RandomVariable``.
        """
        if not isinstance(concentration_rv, RandomVariable):
            raise TypeError(
                "concentration_rv must be a RandomVariable, got "
                f"{type(concentration_rv).__name__}."
            )
        self.concentration_rv = concentration_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            "DirichletMultinomialNoise("
            f"concentration_rv={self.concentration_rv!r})"
        )

    def validate(self) -> None:
        """
        Check that the concentration is positive.

        Raises
        ------
        ValueError
            If the concentration is not positive.
        """
        concentration = jnp.asarray(self.concentration_rv())
        if jnp.any(concentration <= 0):
            raise ValueError(
                "DirichletMultinomialNoise: concentration must be positive, "
                f"got {concentration}"
            )

    def sample(
        self,
        name: str,
        probs: ArrayLike,
        total_count: ArrayLike,
        obs: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Sample from the Dirichlet-multinomial distribution.

        Parameters
        ----------
        name
            Numpyro sample site name.
        probs
            Category probabilities, the last axis summing to one.
        total_count
            Total count of each row.
        obs
            Observed counts for conditioning.
        mask
            Boolean mask of the rows to include in the likelihood. If
            None, all rows are included.

        Returns
        -------
        ArrayLike
            Dirichlet-multinomial counts.
        """
        concentration = self.concentration_rv()
        with numpyro.handlers.mask(mask=True if mask is None else mask):
            counts: ArrayLike = numpyro.sample(
                name,
                ObservedCellsDirichletMultinomial(
                    concentration=concentration * probs,
                    total_count=total_count,
                ),
                obs=obs,
            )
        return counts
