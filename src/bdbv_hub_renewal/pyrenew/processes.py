# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
# Synced from the prototype by scripts/sync_pyrenew.py. Edit there.
r"""Pooled time-varying effects shared by the latent and observation parts.

Rt and the missing-onset fraction use the same structure as PyRenew's
`SubpopulationInfections`: a national temporal process plus province
deviations with a sum-to-zero constraint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class NonCentredRandomWalk:
    r"""Random walk with vectorised steps, frozen before the data begin.

    $$x(t) = x_0 \ (t \le k), \qquad x(t) = x(t - 1) + \sigma z_t,
    \qquad z_t \sim N(0, 1) \ (t > k).$$

    The same prior as PyRenew's `RandomWalk` after the first `frozen`
    steps, which PyRenew samples one site per step in a scan as
    $N(0, \sigma)$. Here the steps are one vectorised site.
    `centred=False` samples the standardised steps $z$ (no funnel between
    $\sigma$ and the steps). `centred=True` samples the path itself
    (`GaussianRandomWalk`), which suits walks the data pin down. The
    first `frozen` steps are zero, so nothing moves before the data begin
    (the seeding period). It satisfies the `TemporalProcess` protocol, so
    it can be wrapped in `WeeklyTemporalProcess`.

    Parameters
    ----------
    innovation_sd_rv : RandomVariable
        Step SD $\sigma$.
    frozen : int, optional
        Number of leading steps fixed at zero.
    centred : bool, optional
        Sample the path rather than the standardised steps.
    """

    step_size: int = 1

    def __init__(
        self,
        innovation_sd_rv: RandomVariable,
        frozen: int = 0,
        centred: bool = False,
    ) -> None:
        self.innovation_sd_rv = innovation_sd_rv
        self.frozen = frozen
        self.centred = centred

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "rw",
        *,
        first_day_dow: int | None = None,
    ) -> ArrayLike:
        """Sample trajectories.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, n_processes)`.
        """
        x0 = jnp.broadcast_to(
            jnp.asarray(0.0 if initial_value is None else initial_value),
            (n_processes,),
        )
        frozen = min(self.frozen, n_timepoints - 1)
        n_free = n_timepoints - 1 - frozen
        sd = self.innovation_sd_rv()
        if self.centred:
            path = numpyro.sample(
                f"{name_prefix}_path",
                dist.GaussianRandomWalk(
                    jnp.broadcast_to(sd, (n_processes,)), num_steps=n_free
                ),
            ).T
            walk = jnp.vstack([jnp.zeros((frozen + 1, n_processes)), path])
            return x0[None, :] + walk
        z = numpyro.sample(
            f"{name_prefix}_z",
            dist.Normal(0.0, 1.0).expand([n_free, n_processes]),
        )
        # Cumulative sum as a lower-triangular product: faster than
        # cumsum on CPU, forward and backward.
        lower = jnp.tril(jnp.ones((n_timepoints, n_free)), -1 - frozen)
        return x0[None, :] + lower @ (sd * z)


class PooledInitialValue(RandomVariable):
    r"""Starting values for each province drawn around a national value.

    $$v_p = \mu + \sigma z_p, \qquad z_p \sim N(0, 1),$$

    with $\mu \sim N(m, s)$ and $\sigma \sim N^+(0, \tau)$.

    Parameters
    ----------
    name : str
        Prefix for the sample sites `{name}_mean`, `{name}_sd` and
        `{name}_z`.
    n_groups : int
        Number of provinces.
    loc, scale : float
        Mean $m$ and SD $s$ of the national value.
    pooling_sd : float
        Scale $\tau$ of the half-normal prior on the province spread.
    """

    def __init__(
        self,
        name: str,
        n_groups: int,
        loc: float,
        scale: float,
        pooling_sd: float,
    ) -> None:
        super().__init__(name=name)
        self.n_groups = n_groups
        self.loc = loc
        self.scale = scale
        self.pooling_sd = pooling_sd

    def sample(self, **kwargs: object) -> ArrayLike:
        """Sample one starting value per province.

        Returns
        -------
        ArrayLike
            Shape `(n_groups,)`.
        """
        mu = numpyro.sample(
            f"{self.name}_mean", dist.Normal(self.loc, self.scale)
        )
        sd = numpyro.sample(f"{self.name}_sd", dist.HalfNormal(self.pooling_sd))
        z = numpyro.sample(
            f"{self.name}_z", dist.Normal(0.0, 1.0).expand([self.n_groups])
        )
        return mu + sd * z


def split_initial_value(
    initial: ArrayLike, n_groups: int
) -> tuple[ArrayLike, ArrayLike]:
    """Split starting values into a national value and province deviations.

    Mirrors `SubpopulationInfections._split_initial_log_rt`: a scalar gives
    zero deviations, a vector gives its mean and the centred deviations.

    Parameters
    ----------
    initial : ArrayLike
        Scalar or shape `(n_groups,)`.
    n_groups : int
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

    $$x_p(t) = b(t) + \delta_p(t) - \frac{1}{P} \sum_q \delta_q(t),$$

    where $b$ follows `baseline_process` and each $\delta_p$ follows
    `deviation_process`, which shares one innovation SD across
    provinces (partial pooling of the deviations).

    Parameters
    ----------
    baseline_process, deviation_process : TemporalProcess
        PyRenew temporal processes, for example a `WeeklyTemporalProcess`
        wrapping a `RandomWalk`.
    initial : ArrayLike
        Starting value(s), scalar or shape `(n_groups,)`.
    n_timepoints, n_groups : int
        Output shape.
    name_prefix : str
        Prefix for the sample sites.
    first_day_dow : int, optional
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
    deviations = deviation_process.sample(
        n_timepoints=n_timepoints,
        n_processes=n_groups,
        initial_value=dev_start,
        name_prefix=f"{name_prefix}_deviations",
        first_day_dow=first_day_dow,
    )
    deviations = deviations - jnp.mean(deviations, axis=1, keepdims=True)
    return baseline, baseline + deviations


class MissingOnsetFraction(RandomVariable):
    r"""Time-varying probability that a case is reported without an onset.

    $$\operatorname{logit} \pi_p(t) = b(t) + \delta_p(t),$$

    from `pooled_walk`, indexed by report day $t$. With small innovation
    SDs it stays near its starting values, so it reduces to a constant
    $\pi_p$ when the data show no change.

    Parameters
    ----------
    name : str
        Prefix for the sample sites.
    initial_rv : RandomVariable
        Starting logit values, scalar or one per province.
    baseline_process, deviation_process : TemporalProcess
        Temporal processes for the national and province parts.
    """

    def __init__(
        self,
        name: str,
        initial_rv: RandomVariable,
        baseline_process: TemporalProcess,
        deviation_process: TemporalProcess,
    ) -> None:
        super().__init__(name=name)
        self.initial_rv = initial_rv
        self.baseline_process = baseline_process
        self.deviation_process = deviation_process

    def sample(
        self,
        n_timepoints: int,
        n_groups: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> ArrayLike:
        r"""Sample $\pi_p(t)$.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, n_groups)`, values in (0, 1).
        """
        _, logit_pi = pooled_walk(
            self.baseline_process,
            self.deviation_process,
            self.initial_rv(),
            n_timepoints,
            n_groups,
            self.name,
            first_day_dow,
        )
        pi = jax.nn.sigmoid(logit_pi)
        numpyro.deterministic(self.name, pi)
        return pi
