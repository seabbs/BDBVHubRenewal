# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Expected onsets as a weekly random walk on the log scale."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array
from pyrenew.latent import LatentSample
from pyrenew.latent.base import BaseLatentInfectionProcess
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class GroupPooledValue(RandomVariable):
    r"""One value per group, pooled, and taken by every place in the group.

    $$v_z = \mu + \tau\, \eta_{g(z)}, \qquad \eta_g \sim N(0, 1),$$

    with $g(z)$ the group of place $z$, such as the province of a zone.
    Non-centred: $\eta$ is the site `{name}_z`.
    """

    def __init__(
        self,
        name: str,
        groups: np.ndarray,
        mean_rv: RandomVariable,
        sd_rv: RandomVariable,
    ) -> None:
        r"""Initialize the pooled values.

        Parameters
        ----------
        name
            Prefix for the sample sites.
        groups
            Group index of every place, `(Z,)`.
        mean_rv
            The shared value $\mu$.
        sd_rv
            The spread $\tau$ of the group values.

        Raises
        ------
        TypeError
            If `mean_rv` or `sd_rv` is not a `RandomVariable`.
        """
        for label, rv in (("mean_rv", mean_rv), ("sd_rv", sd_rv)):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        super().__init__(name=name)
        self.groups = np.asarray(groups, dtype=int)
        self.mean_rv = mean_rv
        self.sd_rv = sd_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GroupPooledValue(name={self.name!r}, "
            f"n_groups={int(self.groups.max()) + 1}, "
            f"mean_rv={self.mean_rv!r}, sd_rv={self.sd_rv!r})"
        )

    def sample(self, **kwargs: object) -> Array:
        """Sample the value of each place.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(Z,)`.
        """
        eta = numpyro.sample(
            f"{self.name}_z",
            dist.Normal(0.0, 1.0).expand([int(self.groups.max()) + 1]),
        )
        value: Array = self.mean_rv() + self.sd_rv() * eta[self.groups]
        return value


class RandomWalkOnsets(BaseLatentInfectionProcess):
    r"""Expected onsets of each place as a random walk, with no renewal.

    $$\lambda_p(t) = e^{x_p(w)} / 7, \qquad
    x_p(w) = x_p(w - 1) + \sigma_p z_{p,w},$$

    for day $t$ in week $w$, with $x_p$ from `walk` (one process per
    place, its step SD $\sigma_p$ from the process's innovation SD)
    and $x_p$ at the first week from `initial_rv`. The latent axis is
    PyRenew's: it starts `n_initialization_points` days before the data,
    a lead PyRenew's builder sets from `gen_int_rv`, which the walk does
    not otherwise use. PyRenew multiplies latent values by each place's
    population, so a walk model gives every place a population of 1.
    """

    def __init__(
        self,
        *,
        name: str,
        gen_int_rv: RandomVariable,
        n_initialization_points: int,
        initial_rv: RandomVariable,
        walk: TemporalProcess,
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
            $x_p$ at the start, shape `(n_places,)`.
        walk
            Weekly process for $x_p$, such as `WeeklyTemporalProcess`
            around `NonCentredRandomWalk`.

        Raises
        ------
        TypeError
            If `initial_rv` is not a `RandomVariable`.
        """
        if not isinstance(initial_rv, RandomVariable):
            raise TypeError("initial_rv must be a RandomVariable.")
        super().__init__(
            name=name,
            gen_int_rv=gen_int_rv,
            n_initialization_points=n_initialization_points,
        )
        self.initial_rv = initial_rv
        self.walk = walk

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RandomWalkOnsets(name={self.name!r}, "
            f"initial_rv={self.initial_rv!r}, walk={self.walk!r})"
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
        log_weekly = self.walk.sample(
            n_timepoints=self.n_initialization_points + n_days_post_init,
            initial_value=self.initial_rv(),
            n_processes=pop.n_subpops,
            name_prefix=f"{self.name}_walk",
            first_day_dow=first_day_dow,
        )
        onsets = jnp.exp(log_weekly) / 7.0
        return LatentSample(
            aggregate=onsets @ pop.fractions, all_subpops=onsets
        )
