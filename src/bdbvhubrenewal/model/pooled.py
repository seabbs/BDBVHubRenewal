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


def parent_active(parents: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return the parents with at least one active child.

    Parameters
    ----------
    parents
        Index of each place's parent, `(n,)`.
    active
        Boolean `(n,)`.

    Returns
    -------
    numpy.ndarray
        Boolean, one per parent.
    """
    parents = np.asarray(parents, dtype=int)
    out = np.zeros(int(parents.max()) + 1, dtype=bool)
    np.logical_or.at(out, parents, np.asarray(active, dtype=bool))
    return out


def nested_basis(
    parents: np.ndarray, active: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Bases of parent and within-parent deviations of nested places.

    For places $z$ in parents $p(z)$ (zones in provinces), a deviation is
    $\delta_z = \sum_j B_{zj} \xi_j + \sum_k Q_{zk} \eta_k$.
    `between` $B$ lifts `sum_to_zero_basis` over the parents with an
    active place to their places, so a parent's places share its
    deviation and the parent deviations sum to zero. `within` $Q$ is
    block diagonal: each parent's block is `sum_to_zero_basis` over its
    active places, so the place deviations sum to zero within each
    parent. A parent with one active place has no block. An inactive
    place takes its parent's deviation and has no own. The columns of
    $B$ and $Q$ are orthogonal, and with one place per parent $B$ is
    `sum_to_zero_basis` and $Q$ is empty.

    Parameters
    ----------
    parents
        Index of each place's parent, `(n,)`.
    active
        Boolean `(n,)`, or `None` for every place.

    Returns
    -------
    tuple of numpy.ndarray
        `between`, `(n, k_p - 1)` with `k_p` the parents with an active
        place, and `within`, `(n, sum_p (k_z(p) - 1))` with `k_z(p)` the
        active places of parent `p`.
    """
    parents = np.asarray(parents, dtype=int)
    n = parents.size
    keep = np.ones(n, dtype=bool) if active is None else np.asarray(active)
    par = parent_active(parents, keep)
    between = sum_to_zero_basis(par.size, par)[parents]
    blocks = []
    for p in range(par.size):
        mine = keep & (parents == p)
        if mine.sum() < 2:
            continue
        block = np.zeros((n, int(mine.sum()) - 1))
        block[mine] = sum_to_zero_basis(int(mine.sum()))
        blocks.append(block)
    within = np.hstack(blocks) if blocks else np.zeros((n, 0))
    return between, within


def pooled_walk(
    baseline_process: TemporalProcess,
    deviation_process: TemporalProcess,
    initial: ArrayLike,
    n_timepoints: int,
    n_groups: int,
    name_prefix: str,
    first_day_dow: int | None = None,
    active: np.ndarray | None = None,
    parents: np.ndarray | None = None,
    place_process: TemporalProcess | None = None,
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
        Starting value(s), scalar or shape `(n_groups,)`.
    n_timepoints, n_groups
        Output shape.
    name_prefix
        Prefix for the sample sites.
    first_day_dow
        Day of week of element 0, needed by calendar-aligned processes.
    active
        Boolean `(n_groups,)`: provinces with a deviation, the rest
        following the pooled trend. `None` is all.
    parents
        Index of each group's parent, `(n_groups,)`, for nested groups
        (zones in provinces). The deviations are then a parent deviation
        that follows `deviation_process`, on the `between` basis of
        `nested_basis`, plus a deviation within the parent that follows
        `place_process`, on its `within` basis. `None` is one level.
    place_process
        Temporal process of the deviations within parents. Needed with
        `parents`.

    Returns
    -------
    tuple of ArrayLike
        National trajectory `(n_timepoints, 1)` and province trajectories
        `(n_timepoints, n_groups)`.
    """
    base_start, dev_start = split_initial_value(initial, n_groups)
    if parents is not None:
        return _nested_walk(
            baseline_process,
            deviation_process,
            place_process,
            base_start,
            dev_start,
            n_timepoints,
            name_prefix,
            first_day_dow,
            nested_basis(parents, active),
        )
    baseline = baseline_process.sample(
        n_timepoints=n_timepoints,
        initial_value=base_start,
        name_prefix=f"{name_prefix}_baseline",
        first_day_dow=first_day_dow,
    )
    basis = jnp.asarray(sum_to_zero_basis(n_groups, active))
    if basis.shape[1] == 0:
        return baseline, jnp.broadcast_to(baseline, (n_timepoints, n_groups))
    coords = deviation_process.sample(
        n_timepoints=n_timepoints,
        n_processes=basis.shape[1],
        initial_value=basis.T @ dev_start,
        name_prefix=f"{name_prefix}_deviations",
        first_day_dow=first_day_dow,
    )
    return baseline, baseline + coords @ basis.T


def _nested_walk(
    baseline_process: TemporalProcess,
    parent_process: TemporalProcess,
    place_process: TemporalProcess | None,
    base_start: ArrayLike,
    dev_start: ArrayLike,
    n_timepoints: int,
    name_prefix: str,
    first_day_dow: int | None,
    bases: tuple[np.ndarray, np.ndarray],
) -> tuple[ArrayLike, ArrayLike]:
    """`pooled_walk` with parent and within-parent deviations.

    The starting deviations are projected onto each basis by least
    squares, which for an orthonormal basis is `basis.T @ start`.
    """
    baseline = baseline_process.sample(
        n_timepoints=n_timepoints,
        initial_value=base_start,
        name_prefix=f"{name_prefix}_baseline",
        first_day_dow=first_day_dow,
    )
    out = baseline
    for basis, process, label in (
        (bases[0], parent_process, "deviations"),
        (bases[1], place_process, "zone_deviations"),
    ):
        if basis.shape[1] == 0:
            continue
        if process is None:
            raise ValueError("Nested deviations need a place process.")
        coords = process.sample(
            n_timepoints=n_timepoints,
            n_processes=basis.shape[1],
            initial_value=jnp.asarray(np.linalg.pinv(basis)) @ dev_start,
            name_prefix=f"{name_prefix}_{label}",
            first_day_dow=first_day_dow,
        )
        out = out + coords @ jnp.asarray(basis).T
    return baseline, jnp.broadcast_to(out, (n_timepoints, bases[0].shape[0]))


def pooled_groups(active: np.ndarray | None, n_groups: int) -> np.ndarray:
    """Return the groups that `pooled_walk` gives the baseline exactly.

    Groups outside `active` have zero rows in the sum-to-zero basis, so
    their trajectory is the baseline plus zero, whatever the starting
    values, and the same for all of them.

    Parameters
    ----------
    active
        As `pooled_walk`.
    n_groups
        Number of groups.

    Returns
    -------
    numpy.ndarray
        Boolean `(n_groups,)`.
    """
    if active is None:
        return np.zeros(n_groups, dtype=bool)
    return ~np.asarray(active, dtype=bool)


class ByParent(RandomVariable):
    """A per-group random variable sampled for parents, given to places.

    Wraps a random variable that takes `n_groups` (such as `HazardDelay`
    or `MissingOnsetFraction`): it is sampled once per parent (province)
    and each place (zone) takes its parent's value along `axis`.
    """

    def __init__(
        self, name: str, inner: RandomVariable, parents: np.ndarray, axis: int
    ) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        name
            Name of the variable.
        inner
            Takes `n_groups` and returns one value per group on `axis`.
        parents
            Index of each place's parent, `(n_places,)`.
        axis
            Group axis of `inner`'s value.

        Raises
        ------
        TypeError
            If `inner` is not a `RandomVariable`.
        """
        if not isinstance(inner, RandomVariable):
            raise TypeError("inner must be a RandomVariable.")
        super().__init__(name=name)
        self.inner = inner
        self.parents = np.asarray(parents, dtype=int)
        self.n_parents = int(self.parents.max()) + 1
        self.axis = axis

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ByParent(name={self.name!r}, inner={self.inner!r}, "
            f"parents={self.parents.tolist()!r}, axis={self.axis})"
        )

    def shared_groups(self, n_groups: int) -> np.ndarray:
        """Return the places whose parent takes the shared value.

        Parameters
        ----------
        n_groups
            Number of places.

        Returns
        -------
        numpy.ndarray
            Boolean `(n_groups,)`.
        """
        inner = getattr(self.inner, "shared_groups", None)
        if inner is None:
            return np.zeros(n_groups, dtype=bool)
        shared = np.asarray(inner(self.n_parents), dtype=bool)
        out: np.ndarray = shared[self.parents]
        return out

    def sample(self, n_groups: int | None = None, **kwargs: object) -> Array:
        """Sample the parents' values and give each place its parent's.

        Parameters
        ----------
        n_groups
            Number of places. Ignored beyond a check.
        **kwargs
            Passed to `inner`.

        Returns
        -------
        Array
            `inner`'s value with the group axis indexed by place.

        Raises
        ------
        ValueError
            If `n_groups` is not the number of places.
        """
        if n_groups is not None and n_groups != self.parents.size:
            raise ValueError(
                f"Expected {self.parents.size} places, got {n_groups}."
            )
        value = self.inner(n_groups=self.n_parents, **kwargs)
        return jnp.take(jnp.asarray(value), self.parents, axis=self.axis)
