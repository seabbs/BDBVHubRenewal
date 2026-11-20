# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Expected reference events from infections, one delay per stratum.

`ReferenceEvents` spreads each stratum's infections over its own
infection-to-reference delay with PyRenew's
`compute_delay_ascertained_incidence` and scales them by its
ascertainment. A stratum may have run at a constant level before the
first day, so its first days also hold the delayed part of that history.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.metaclass import RandomVariable

__all__ = ["ReferenceEvents"]


class ReferenceEvents(RandomVariable):
    r"""Each stratum's expected reference events.

    With infections $I_k(t)$, delay pmf $f_k$ and ascertainment $\alpha_k$,

    $$E_k(t) = \alpha_k \sum_{d \ge 0} f_k(d) I_k(t - d),$$

    where $I_k(t) = b_k$ on the days before the first, $b_k$ the
    stratum's level before the span (0 for an outbreak that starts in
    it).

    Parameters
    ----------
    name : str
        Name of the component. The events are recorded as the
        deterministic site `"{name}::expected"`.
    delay_rvs : sequence of RandomVariable
        One delay pmf per stratum on 0, 1, ... days, such as PyRenew's
        `DeterministicPMF`. The pmfs may differ in length.
    ascertainment_rv : RandomVariable
        Returns $\alpha$, a scalar or one value per stratum.
    """

    def __init__(
        self,
        name: str,
        delay_rvs: Sequence[RandomVariable],
        ascertainment_rv: RandomVariable,
    ) -> None:
        super().__init__(name=name)
        self.delay_rvs = list(delay_rvs)
        self.ascertainment_rv = ascertainment_rv

    def sample(
        self,
        infections: ArrayLike,
        before: ArrayLike | None = None,
        **kwargs: object,
    ) -> Array:
        """Return the expected reference events.

        Parameters
        ----------
        infections : array_like
            Days by strata.
        before : array_like, optional
            Each stratum's constant infections before the first day.
            `None` is none.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Days by strata, the shape of `infections`.

        Raises
        ------
        ValueError
            If there is not one delay per stratum.
        """
        x = jnp.asarray(infections)
        k = x.shape[1]
        if len(self.delay_rvs) != k:
            raise ValueError("there must be one delay per stratum")
        b = jnp.zeros(k) if before is None else jnp.asarray(before)
        asc = jnp.broadcast_to(jnp.asarray(self.ascertainment_rv()), (k,))
        cols = []
        for j, rv in enumerate(self.delay_rvs):
            pmf = jnp.asarray(rv())
            # The history before the first day, long enough that the
            # "valid" convolution returns one value per day.
            past = jnp.full(pmf.shape[0] - 1, b[j])
            col, _ = compute_delay_ascertained_incidence(
                jnp.concatenate([past, x[:, j]]), pmf, asc[j]
            )
            cols.append(col)
        out = jnp.stack(cols, axis=1)
        numpyro.deterministic(f"{self.name}::expected", out)
        return out
