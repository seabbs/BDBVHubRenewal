# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A delay pmf that varies with the reference day, through a logit hazard.

`HazardDelay` returns $f_{p,s}(d)$ of shape
`(n_groups, n_timepoints, dmax + 1)`, the delay pmf of group $p$ for
events on day $s$.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from pyrenew.latent.temporal_processes import (
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.metaclass import RandomVariable

from .pooled import pooled_walk


def hazard_to_pmf(logit_hazard: Array) -> Array:
    r"""Convert discrete-time hazards to a pmf on the last axis.

    $$f(d) = h(d) \prod_{j < d} \{1 - h(j)\}, \qquad h(D) = 1,$$

    so each row sums to one. The cumulative sum of $\log(1 - h)$ is a
    product with a triangular matrix, which is faster than `cumsum` on
    CPU, forward and backward.

    Parameters
    ----------
    logit_hazard
        Logit hazards for delays 0..D-1, shape `(..., D)`.

    Returns
    -------
    Array
        Pmf on 0..D, shape `(..., D + 1)`.
    """
    k = logit_hazard.shape[-1]
    log_h = jax.nn.log_sigmoid(logit_hazard)
    log_1mh = jax.nn.log_sigmoid(-logit_hazard)
    # Column d sums the first d terms, so the result is the log survival
    # before each delay, with zero at d = 0.
    upper = jnp.triu(jnp.ones((k, k + 1), logit_hazard.dtype), 1)
    log_surv_before = log_1mh @ upper
    zero = jnp.zeros((*log_h.shape[:-1], 1), log_h.dtype)
    log_h = jnp.concatenate([log_h, zero], axis=-1)
    return jnp.exp(log_h + log_surv_before)


class HazardDelay(RandomVariable):
    r"""Nonparametric delay with a smooth baseline and pooled time shifts.

    $$\operatorname{logit} h_{p,s}(d) = \beta(d) + \gamma(s) + \delta_p
    + \eta_p(s),$$

    where $\beta$ is a temporal process over delay $d$ starting at
    $\beta(0)$ (a smoothness prior), $\gamma$ is the shared temporal
    process over the reference day $s$, starting at 0, $\delta_p$ is
    group $p$'s offset and $\eta_p$ its own walk around $\gamma$, starting
    at 0. The offsets and walks sum to zero over groups and share a
    learnt spread each (`pooled_walk`), so each group's delay is pooled
    towards the shared delay and $\gamma$ stays identified. All hazards
    of a group and day shift together (proportional odds). The pmf
    follows from `hazard_to_pmf`. With one group, $\delta$ and $\eta$
    are zero. The shape $\beta$, the shared shift $\gamma$, each
    group's shift $\gamma + \delta_p + \eta_p$ and each group's mean
    delay by day are recorded as `{name}_beta`, `{name}_shared_shift`,
    `{name}_shift` and `{name}_mean`.
    """

    def __init__(
        self,
        name: str,
        dmax: int,
        hazard0_rv: RandomVariable,
        shape_process: TemporalProcess,
        time_process: TemporalProcess,
        offset_rv: RandomVariable,
        group_process: TemporalProcess,
        active: np.ndarray | None = None,
    ) -> None:
        r"""Initialize the delay.

        Parameters
        ----------
        name
            Prefix for the sample sites.
        dmax
            Maximum delay $D$.
        hazard0_rv
            $\beta(0)$, the logit hazard at delay 0.
        shape_process
            Process for $\beta(d)$ over delays $0..D - 1$.
        time_process
            Process for $\gamma(s)$ over reference days.
        offset_rv
            Offsets $\delta_p$, shape `(n_groups,)`
            (`PooledInitialValue` with mean 0).
        group_process
            Process for the sum-to-zero coordinates of $\eta_p(s)$.
        active
            Boolean `(n_groups,)`: groups with their own drift, the rest
            following $\gamma(s)$. `None` is all.

        Raises
        ------
        TypeError
            If `hazard0_rv` or `offset_rv` is not a `RandomVariable`.
        """
        for label, rv in (("hazard0_rv", hazard0_rv), ("offset_rv", offset_rv)):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        super().__init__(name=name)
        self.dmax = dmax
        self.hazard0_rv = hazard0_rv
        self.shape_process = shape_process
        self.time_process = time_process
        self.offset_rv = offset_rv
        self.group_process = group_process
        self.active = active

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HazardDelay(name={self.name!r}, dmax={self.dmax}, "
            f"hazard0_rv={self.hazard0_rv!r}, "
            f"shape_process={self.shape_process!r}, "
            f"time_process={self.time_process!r}, "
            f"offset_rv={self.offset_rv!r}, "
            f"group_process={self.group_process!r}, active={self.active!r})"
        )

    def sample(
        self,
        n_timepoints: int,
        n_groups: int = 1,
        first_day_dow: int | None = None,
        by_week: bool = False,
        **kwargs: object,
    ) -> Array:
        """Sample the delay of each group by day, or by week.

        Parameters
        ----------
        n_timepoints
            Days to sample.
        n_groups
            Number of groups, such as provinces.
        first_day_dow
            Weekday of the first day, passed by PyRenew.
        by_week
            Return one row per week when the delay is constant within
            weeks (a `WeeklyTemporalProcess` starting on `first_day_dow`),
            rather than repeating it for each day.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(n_groups, n_timepoints, dmax + 1)`, or
            `(n_groups, ceil(n_timepoints / 7), dmax + 1)` by week.
        """
        beta0 = self.hazard0_rv()
        beta = self.shape_process.sample(
            n_timepoints=self.dmax,
            initial_value=beta0,
            name_prefix=f"{self.name}_shape",
        )[:, 0]
        gamma, shifts = pooled_walk(
            self.time_process,
            self.group_process,
            self.offset_rv(),
            n_timepoints,
            n_groups,
            f"{self.name}_time",
            first_day_dow,
            self.active,
        )
        shift = jnp.asarray(shifts)
        # Recorded for the delay summaries written after a fit
        # (`delay_summary`); they change nothing in the model.
        numpyro.deterministic(f"{self.name}_beta", beta)
        numpyro.deterministic(f"{self.name}_shared_shift", gamma)
        numpyro.deterministic(f"{self.name}_shift", shift)
        weekly = (
            isinstance(self.time_process, WeeklyTemporalProcess)
            and first_day_dow == self.time_process.start_dow
        )
        if weekly:
            # The shifts are constant within weeks, so compute one row
            # per week and repeat it.
            shift = shift[::7]
        logit = beta[None, None, :] + shift.T[:, :, None]
        pmf = hazard_to_pmf(logit)
        mean = pmf @ jnp.arange(self.dmax + 1)
        if weekly:
            mean = jnp.repeat(mean, 7, axis=1)[:, :n_timepoints]
        numpyro.deterministic(f"{self.name}_mean", mean)
        if weekly and not by_week:
            pmf = jnp.repeat(pmf, 7, axis=1)[:, :n_timepoints]
        return pmf
