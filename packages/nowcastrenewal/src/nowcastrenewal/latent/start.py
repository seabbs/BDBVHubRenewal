# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Each stratum's infections over the generation time's span before day 1.

`InitializeStrata` is a PyRenew `InfectionInitializationMethod` for
strata: an outbreak that starts in the series has its seed on the last
day before day 1 and none before (PyRenew's zero padding, one column per
stratum); an established or endemic one runs at its level over the whole
span. It is used through PyRenew's `InfectionInitializationProcess`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent.infection_initialization_method import (
    InfectionInitializationMethod,
)

__all__ = ["InitializeStrata"]


class InitializeStrata(InfectionInitializationMethod):
    """A seed on the last day, or a constant level, for each stratum.

    Parameters
    ----------
    n_timepoints : int
        Days before day 1 (the generation time's span).
    early : bool
        Whether the outbreak starts in the series (a seed of at least 1
        on the last day, in strata with a positive value) or runs at its
        level before it.
    """

    def __init__(self, n_timepoints: int, early: bool) -> None:
        super().__init__(n_timepoints)
        self.early = early

    def initialize_infections(self, I_pre_init: ArrayLike) -> Array:
        """Return the starting infections.

        Parameters
        ----------
        I_pre_init : array_like
            Each stratum's seed or level, 0 for strata that start
            without infections.

        Returns
        -------
        Array
            `(n_timepoints, k)`.
        """
        x = jnp.atleast_1d(jnp.asarray(I_pre_init, dtype=float))
        if not self.early:
            return jnp.broadcast_to(x, (self.n_timepoints, x.shape[0]))
        seed = jnp.where(x > 0, jnp.maximum(1.0, x), 0.0)
        return jnp.zeros((self.n_timepoints, x.shape[0])).at[-1].set(seed)
