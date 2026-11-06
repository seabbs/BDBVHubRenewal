# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Group values that pool around a shared value with sum-to-zero offsets.

The offsets of $P$ groups are sampled as $P - 1$ coordinates on an
orthonormal basis of the sum-to-zero subspace, so they sum to zero with
no direction the data cannot identify. PyRenew's
`SubpopulationInfections` instead subtracts the mean of $P$ processes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class PooledInitialValue(RandomVariable):
    r"""One value per group, drawn around a shared value.

    $$v_p = \mu + \sum_j Q_{pj} \eta_j, \qquad \eta_j \sim N(0, \sigma),$$

    with $Q$ from `sum_to_zero_basis`, so the offsets sum to zero and
    $\mu$ is their mean. Centred, the $P - 1$ coordinates $\eta_j$ are
    the site `{name}_offsets`; non-centred, $\eta_j = \sigma z_j$ with
    $z_j \sim N(0, 1)$ at `{name}_z`.
    """

    def __init__(
        self,
        name: str,
        n_groups: int,
        mean_rv: RandomVariable,
        sd_rv: RandomVariable,
        centred: bool = True,
    ) -> None:
        r"""Initialize the pooled values.

        Parameters
        ----------
        name
            Prefix for the offset site.
        n_groups
            Number of groups $P$.
        mean_rv
            Shared value $\mu$.
        sd_rv
            Spread $\sigma$ of the offsets. Not sampled for one group.
        centred
            Sample the offsets on the basis directly (centred), rather
            than standardised.

        Raises
        ------
        TypeError
            If `mean_rv` or `sd_rv` is not a `RandomVariable`.
        """
        for label, rv in (("mean_rv", mean_rv), ("sd_rv", sd_rv)):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        super().__init__(name=name)
        self.n_groups = n_groups
        self.mean_rv = mean_rv
        self.sd_rv = sd_rv
        self.centred = centred

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PooledInitialValue(name={self.name!r}, n_groups={self.n_groups},"
            f" mean_rv={self.mean_rv!r}, sd_rv={self.sd_rv!r}, "
            f"centred={self.centred})"
        )

    def sample(self, **kwargs: object) -> Array:
        """Sample one value per group.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(n_groups,)`.
        """
        mu = self.mean_rv()
        if self.n_groups == 1:
            return jnp.full((1,), mu)
        sd = self.sd_rv()
        basis = jnp.asarray(sum_to_zero_basis(self.n_groups))
        if self.centred:
            offsets = numpyro.sample(
                f"{self.name}_offsets",
                dist.Normal(0.0, sd).expand([self.n_groups - 1]),
            )
            return jnp.asarray(mu + basis @ offsets)
        z = numpyro.sample(
            f"{self.name}_z",
            dist.Normal(0.0, 1.0).expand([self.n_groups - 1]),
        )
        return jnp.asarray(mu + sd * (basis @ z))


def split_initial_value(
    initial: ArrayLike, n_groups: int
) -> tuple[ArrayLike, ArrayLike]:
    """Split starting values into a national value and province deviations.

    Mirrors `SubpopulationInfections._split_initial_log_rt`: a scalar gives
    zero deviations, a vector gives its mean and the centred deviations.

    Parameters
    ----------
    initial
        Scalar or shape `(n_groups,)`.
    n_groups
        Number of provinces.

    Returns
    -------
    tuple of ArrayLike
        National start (scalar) and deviation starts, shape `(n_groups,)`.
    """
    initial = jnp.asarray(initial)
    if initial.ndim == 0:
        return initial, jnp.zeros(n_groups)
    start = jnp.mean(initial)
    return start, initial - start


def sum_to_zero_basis(n: int) -> np.ndarray:
    r"""Orthonormal basis of $\{\delta \in R^n : \sum_p \delta_p = 0\}$.

    Parameters
    ----------
    n
        Dimension.

    Returns
    -------
    numpy.ndarray
        Shape `(n, n - 1)`, orthonormal columns that each sum to zero.
    """
    _, _, vt = np.linalg.svd(np.ones((1, n)))
    return np.asarray(vt[1:].T)


def pooled_walk(
    baseline_process: TemporalProcess,
    deviation_process: TemporalProcess,
    initial: ArrayLike,
    n_timepoints: int,
    n_groups: int,
    name_prefix: str,
    first_day_dow: int | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    r"""Sample a national process plus sum-to-zero province deviations.

    $$x_p(t) = b(t) + \sum_j Q_{pj} \eta_j(t),$$

    where $b$ follows `baseline_process`, $Q$ is an orthonormal basis of
    the sum-to-zero subspace (`sum_to_zero_basis`) and the $P - 1$
    coordinates $\eta_j$ follow `deviation_process`, which shares one
    innovation SD (partial pooling). The deviations sum to zero exactly
    with no redundant direction to sample (BVDOutbreakSize #855). Each
    deviation has the prior of $P$ independent walks centred on their
    mean.

    Parameters
    ----------
    baseline_process, deviation_process
        PyRenew temporal processes, for example a `WeeklyTemporalProcess`
        wrapping a `RandomWalk`.
    initial
        Starting value(s), scalar or shape `(n_groups,)`.
    n_timepoints, n_groups
        Output shape.
    name_prefix
        Prefix for the sample sites.
    first_day_dow
        Day of week of element 0, needed by calendar-aligned processes.

    Returns
    -------
    tuple of ArrayLike
        National trajectory `(n_timepoints, 1)` and province trajectories
        `(n_timepoints, n_groups)`.
    """
    base_start, dev_start = split_initial_value(initial, n_groups)
    baseline = baseline_process.sample(
        n_timepoints=n_timepoints,
        initial_value=base_start,
        name_prefix=f"{name_prefix}_baseline",
        first_day_dow=first_day_dow,
    )
    if n_groups == 1:
        return baseline, baseline
    basis = jnp.asarray(sum_to_zero_basis(n_groups))
    coords = deviation_process.sample(
        n_timepoints=n_timepoints,
        n_processes=n_groups - 1,
        initial_value=basis.T @ dev_start,
        name_prefix=f"{name_prefix}_deviations",
        first_day_dow=first_day_dow,
    )
    return baseline, baseline + coords @ basis.T
