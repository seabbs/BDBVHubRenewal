# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Delay pmfs: the censored parametric pmf and a fixed time-varying input.

`FixedTimeVarying` returns a matrix $f_s(d)$ of shape
`(n_timepoints, dmax + 1)`, the delay pmf for cases with onset on day $s$.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class FixedTimeVarying(RandomVariable):
    """A fixed time-varying input: the hub's delay in the total-stream steps.

    Stands in for `HazardDelay` with a matrix fixed beforehand. It is
    data, so it adds no sites.
    Rows past the end are filled with the last row, so the same input
    covers the forecast horizon.

    Parameters
    ----------
    name : str
        Name of the input.
    value : array_like
        Shape `(n_rows, k)`.
    """

    def __init__(self, name: str, value: ArrayLike) -> None:
        super().__init__(name=name)
        self.value = jnp.asarray(value)

    def sample(self, n_timepoints: int, **kwargs: object) -> Array:
        """Return the input padded or cut to `n_timepoints` rows.

        Parameters
        ----------
        n_timepoints : int
            Rows to return.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, k)`.
        """
        idx = jnp.minimum(jnp.arange(n_timepoints), self.value.shape[0] - 1)
        return self.value[idx]
