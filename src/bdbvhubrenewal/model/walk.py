# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Expected onsets as a weekly random walk on the log scale."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent import LatentSample
from pyrenew.latent.base import BaseLatentInfectionProcess
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class RandomWalkOnsets(BaseLatentInfectionProcess):
    r"""Expected onsets of each place as a random walk, with no renewal.

    $$\lambda_z(t) = e^{x_z(w)} / 7, \qquad
    x_z(w) = x_z(0) + \sigma_{g(z)} W_z(w),$$

    for day $t$ in week $w$, with $W_z$ a walk with unit steps from `walk`,
    $x_z(0)$ from `initial_rv` and the step SD $\sigma_g$ of group $g(z)$
    from `step_sd_rv`. The latent axis is PyRenew's: it starts
    `n_initialization_points` days before the data, a lead PyRenew's
    builder sets from `gen_int_rv`, which the walk does not otherwise use.
    PyRenew multiplies latent values by each place's population, so a
    walk model gives every place a population of 1.
    """

    def __init__(
        self,
        *,
        name: str,
        gen_int_rv: RandomVariable,
        n_initialization_points: int,
        initial_rv: RandomVariable,
        step_sd_rv: RandomVariable,
        walk: TemporalProcess,
        groups: ArrayLike | None = None,
    ) -> None:
        r"""Initialize the walk.

        Parameters
        ----------
        name
            Prefix for the sample sites.
        gen_int_rv
            Sets the lead of the latent axis (PyRenew's builder).
        n_initialization_points
            Days of the latent axis before the data.
        initial_rv
            $x_z(0)$, shape `(n_places,)`.
        step_sd_rv
            $\sigma_g$, shape `(n_groups,)`.
        walk
            Weekly process with unit step SD for $W_z$, such as
            `WeeklyTemporalProcess` around `NonCentredRandomWalk`.
        groups
            Group of each place, `(n_places,)`. By default each place is
            its own group.

        Raises
        ------
        TypeError
            If `initial_rv` or `step_sd_rv` is not a `RandomVariable`.
        """
        for label, rv in (
            ("initial_rv", initial_rv),
            ("step_sd_rv", step_sd_rv),
        ):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        super().__init__(
            name=name,
            gen_int_rv=gen_int_rv,
            n_initialization_points=n_initialization_points,
        )
        self.initial_rv = initial_rv
        self.step_sd_rv = step_sd_rv
        self.walk = walk
        self.groups = None if groups is None else np.asarray(groups)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RandomWalkOnsets(name={self.name!r}, "
            f"initial_rv={self.initial_rv!r}, step_sd_rv={self.step_sd_rv!r}, "
            f"walk={self.walk!r})"
        )

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: Array | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """Sample expected onsets for each place.

        Parameters
        ----------
        n_days_post_init
            Days after the lead.
        subpop_fractions
            Population share of each place.
        first_day_dow
            Weekday of the first day, passed by PyRenew.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        LatentSample
            `aggregate` of shape `(n_total,)` and `all_subpops` of shape
            `(n_total, n_places)`.
        """
        pop = self._parse_and_validate_fractions(subpop_fractions)
        group = np.arange(pop.n_subpops) if self.groups is None else self.groups
        steps = self.walk.sample(
            n_timepoints=self.n_initialization_points + n_days_post_init,
            initial_value=0.0,
            n_processes=pop.n_subpops,
            name_prefix=f"{self.name}_walk",
            first_day_dow=first_day_dow,
        )
        sigma = jnp.asarray(self.step_sd_rv())
        log_weekly = self.initial_rv() + sigma[group] * steps
        onsets = jnp.exp(log_weekly) / 7.0
        return LatentSample(
            aggregate=onsets @ pop.fractions, all_subpops=onsets
        )
