# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Group values that pool around a shared value with sum-to-zero offsets.

The offsets of $P$ groups are sampled as $P - 1$ coordinates on an
orthonormal basis of the sum-to-zero subspace, so they sum to zero with
no direction the data cannot identify. PyRenew's
`SubpopulationInfections` instead subtracts the mean of $P$ processes.
Groups outside an `active` set take the shared value with no offset, so
nothing is sampled for groups no data can inform.
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
    $\mu$ is their mean. Centred, the coordinates $\eta_j$ are the site
    `{name}_offsets`; non-centred, $\eta_j = \sigma z_j$ with
    $z_j \sim N(0, 1)$ at `{name}_z`. Groups outside `active` take $\mu$.
    """

    def __init__(
        self,
        name: str,
        n_groups: int,
        mean_rv: RandomVariable,
        sd_rv: RandomVariable,
        centred: bool = True,
        active: np.ndarray | None = None,
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
        active
            Boolean `(n_groups,)`: groups with an offset. `None` is all.

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
        self.active = active

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PooledInitialValue(name={self.name!r}, n_groups={self.n_groups},"
            f" mean_rv={self.mean_rv!r}, sd_rv={self.sd_rv!r}, "
            f"centred={self.centred}, active={self.active!r})"
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
        basis = jnp.asarray(sum_to_zero_basis(self.n_groups, self.active))
        k = basis.shape[1]
        if k == 0:
            return jnp.full((self.n_groups,), mu)
        sd = self.sd_rv()
        if self.centred:
            offsets = numpyro.sample(
                f"{self.name}_offsets", dist.Normal(0.0, sd).expand([k])
            )
            return jnp.asarray(mu + basis @ offsets)
        z = numpyro.sample(f"{self.name}_z", dist.Normal(0.0, 1.0).expand([k]))
        return jnp.asarray(mu + sd * (basis @ z))

    def priors(self) -> tuple[dist.Normal, dist.Normal | None]:
        r"""Return the priors of $\mu$ and of the coordinates $\eta_j$.

        For a walk that samples its start through an anchor
        (`NonCentredRandomWalk`). Samples $\sigma$ when there are offsets.

        Returns
        -------
        tuple
            The normal prior of $\mu$, and $N(0, \sigma)$ for each
            $\eta_j$, or None when no group has an offset.

        Raises
        ------
        TypeError
            If the prior of $\mu$ is not normal.
        """
        mean = getattr(self.mean_rv, "distribution", None)
        if not isinstance(mean, dist.Normal):
            raise TypeError("mean_rv must hold a normal distribution.")
        if sum_to_zero_basis(self.n_groups, self.active).shape[1] == 0:
            return mean, None
        return mean, dist.Normal(0.0, self.sd_rv())


class ActiveValues(RandomVariable):
    """Values sampled for the active groups, a fixed value for the rest."""

    def __init__(
        self,
        name: str,
        base_rv: RandomVariable,
        active: np.ndarray,
        fill: float,
    ) -> None:
        """Initialize the values.

        Parameters
        ----------
        name
            Name of the values.
        base_rv
            Returns one value per active group.
        active
            Boolean, one per group.
        fill
            Value of the inactive groups.

        Raises
        ------
        TypeError
            If `base_rv` is not a `RandomVariable`.
        """
        if not isinstance(base_rv, RandomVariable):
            raise TypeError("base_rv must be a RandomVariable.")
        super().__init__(name=name)
        self.base_rv = base_rv
        self.active = np.asarray(active)
        self.fill = fill

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ActiveValues(name={self.name!r}, base_rv={self.base_rv!r}, "
            f"active={self.active!r}, fill={self.fill})"
        )

    def sample(self, **kwargs: object) -> Array:
        """Sample the active values and fill the rest.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            One value per group.
        """
        out = jnp.full(self.active.shape, self.fill)
        return out.at[np.flatnonzero(self.active)].set(self.base_rv())


def split_initial_value(
    initial: ArrayLike, n_groups: int
) -> tuple[ArrayLike, ArrayLike]:
    """Split starting values into a national value and province deviations.

    Mirrors `BaseSubpopulationInfections._split_initial_log_rt`: a scalar gives
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


def sum_to_zero_basis(n: int, active: np.ndarray | None = None) -> np.ndarray:
    r"""Orthonormal basis of $\{\delta \in R^n : \sum_p \delta_p = 0\}$.

    With `active`, the basis spans the deviations that sum to zero over
    the active groups and are zero elsewhere.

    Parameters
    ----------
    n
        Dimension.
    active
        Boolean `(n,)`, or `None` for every group.

    Returns
    -------
    numpy.ndarray
        Shape `(n, k - 1)` with `k` the active groups, orthonormal
        columns that each sum to zero.
    """
    keep = np.ones(n, dtype=bool) if active is None else np.asarray(active)
    k = int(keep.sum())
    _, _, vt = np.linalg.svd(np.ones((1, k)))
    basis = np.zeros((n, max(k - 1, 0)))
    basis[keep] = vt[1:].T
    return basis


def pooled_walk(
    baseline_process: TemporalProcess,
    deviation_process: TemporalProcess,
    initial: ArrayLike | PooledInitialValue,
    n_timepoints: int,
    n_groups: int,
    name_prefix: str,
    first_day_dow: int | None = None,
    active: np.ndarray | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    r"""Sample a national process plus sum-to-zero province deviations.

    $$x_p(t) = b(t) + \sum_j Q_{pj} \eta_j(t),$$

    where $b$ follows `baseline_process`, $Q$ is an orthonormal basis of
    the sum-to-zero subspace (`sum_to_zero_basis`) and the $P - 1$
    coordinates $\eta_j$ follow `deviation_process`, which shares one
    innovation SD (partial pooling). The deviations sum to zero exactly
    with no redundant direction to sample, as in BVDOutbreakSize
    (https://github.com/epiforecasts/BVDOutbreakSize/issues/855). Each
    deviation has the prior of $P$ independent walks centred on their
    mean.

    Parameters
    ----------
    baseline_process, deviation_process
        PyRenew temporal processes, for example a `WeeklyTemporalProcess`
        wrapping a `RandomWalk`.
    initial
        Starting value(s), scalar or shape `(n_groups,)`, or the pooled
        starting values, whose priors the walks sample through their
        anchors (`NonCentredRandomWalk`).
    n_timepoints, n_groups
        Output shape.
    name_prefix
        Prefix for the sample sites.
    first_day_dow
        Day of week of element 0, needed by calendar-aligned processes.
    active
        Boolean `(n_groups,)`: provinces with a deviation, the rest
        following the pooled trend. `None` is all.

    Returns
    -------
    tuple of ArrayLike
        National trajectory `(n_timepoints, 1)` and province trajectories
        `(n_timepoints, n_groups)`.
    """
    basis = jnp.asarray(sum_to_zero_basis(n_groups, active))
    if isinstance(initial, PooledInitialValue):
        base_start, dev_start = initial.priors()
    else:
        base_start, dev = split_initial_value(initial, n_groups)
        dev_start = basis.T @ dev
    baseline = baseline_process.sample(
        n_timepoints=n_timepoints,
        initial_value=base_start,
        name_prefix=f"{name_prefix}_baseline",
        first_day_dow=first_day_dow,
    )
    if basis.shape[1] == 0:
        return baseline, jnp.broadcast_to(baseline, (n_timepoints, n_groups))
    coords = deviation_process.sample(
        n_timepoints=n_timepoints,
        n_processes=basis.shape[1],
        initial_value=dev_start,
        name_prefix=f"{name_prefix}_deviations",
        first_day_dow=first_day_dow,
    )
    return baseline, baseline + coords @ basis.T
