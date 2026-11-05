# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A delay pmf that varies with the reference day, through a logit hazard.

`HazardDelay` returns a matrix $f_s(d)$ of shape
`(n_timepoints, dmax + 1)`, the delay pmf for events on day $s$.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax import Array
from pyrenew.latent.temporal_processes import (
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.metaclass import RandomVariable


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
    r"""Nonparametric delay with a smooth baseline and a time shift.

    $$\operatorname{logit} h_s(d) = \beta(d) + \gamma(s),$$

    where $\beta$ is a temporal process over delay $d$ starting at
    $\beta(0)$ (a smoothness prior) and $\gamma$ is a temporal process
    over the reference day $s$, starting at 0, that shifts all hazards
    together (proportional odds). The pmf follows from `hazard_to_pmf`.
    When $\gamma$ has a small innovation SD, the delay is near constant.
    The shape $\beta$, the shift $\gamma$ and the mean delay by day
    are recorded as `{name}_beta`, `{name}_shift` and `{name}_mean`.
    """

    def __init__(
        self,
        name: str,
        dmax: int,
        hazard0_rv: RandomVariable,
        shape_process: TemporalProcess,
        time_process: TemporalProcess,
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

        Raises
        ------
        TypeError
            If `hazard0_rv` is not a `RandomVariable`.
        """
        if not isinstance(hazard0_rv, RandomVariable):
            raise TypeError("hazard0_rv must be a RandomVariable.")
        super().__init__(name=name)
        self.dmax = dmax
        self.hazard0_rv = hazard0_rv
        self.shape_process = shape_process
        self.time_process = time_process

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HazardDelay(name={self.name!r}, dmax={self.dmax}, "
            f"hazard0_rv={self.hazard0_rv!r}, "
            f"shape_process={self.shape_process!r}, "
            f"time_process={self.time_process!r})"
        )

    def sample(
        self,
        n_timepoints: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Array:
        """Sample the delay matrix.

        Parameters
        ----------
        n_timepoints
            Days to sample.
        first_day_dow
            Weekday of the first day, passed by PyRenew.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(n_timepoints, dmax + 1)`.
        """
        beta0 = self.hazard0_rv()
        beta = self.shape_process.sample(
            n_timepoints=self.dmax,
            initial_value=beta0,
            name_prefix=f"{self.name}_shape",
        )[:, 0]
        gamma = self.time_process.sample(
            n_timepoints=n_timepoints,
            initial_value=0.0,
            name_prefix=f"{self.name}_time",
            first_day_dow=first_day_dow,
        )
        # Recorded for the delay summaries written after a fit
        # (`delay_summary`); they change nothing in the model.
        numpyro.deterministic(f"{self.name}_beta", beta)
        numpyro.deterministic(f"{self.name}_shift", gamma)
        weekly = (
            isinstance(self.time_process, WeeklyTemporalProcess)
            and first_day_dow == self.time_process.start_dow
        )
        if weekly:
            # The shift is constant within weeks, so compute one row per
            # week and repeat it.
            gamma = gamma[::7]
        logit = beta[None, :] + gamma
        pmf = hazard_to_pmf(logit)
        mean = pmf @ jnp.arange(self.dmax + 1)
        if weekly:
            pmf = jnp.repeat(pmf, 7, axis=0)[:n_timepoints]
            mean = jnp.repeat(mean, 7, axis=0)[:n_timepoints]
        numpyro.deterministic(f"{self.name}_mean", mean)
        return pmf
