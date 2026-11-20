# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The simulator's latent infection process, a PyRenew latent process.

`StratifiedInfections` subclasses PyRenew's `BaseLatentInfectionProcess`:
the generation time is its `gen_int_rv`, the start a PyRenew
`InfectionInitializationProcess` (with `start.InitializeStrata`), and R a
component such as `rt.PlaceRt`. Infections come from `strata.strata_renewal`
with mixing, imports, shifts, variants and a choice of depletion.

It is not a `BaseSubpopulationInfections`: that class sets each place's R
as sum-to-zero deviations around a baseline and seeds by exponential
growth, where the simulator's places share, copy, perturb or lag a path
and may start at a level.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from pyrenew.latent.base import BaseLatentInfectionProcess, LatentSample
from pyrenew.metaclass import RandomVariable

from .strata import Depletion, Shifts, Variants, strata_renewal

__all__ = ["StratifiedInfections"]


class StratifiedInfections(BaseLatentInfectionProcess):
    """Infections by stratum in counts, from a renewal with depletion.

    The sample records `"{name}::r"` (R after shifts and variants),
    `"{name}::reff"` (effective R after each day's infections) and
    `"{name}::susceptible"`, each `(n, k)`.

    Parameters
    ----------
    name : str
        Name of the process.
    gen_int_rv : RandomVariable
        The generation time pmf, element 0 at lag 1 (PyRenew's
        `DeterministicPMF`).
    start_rv : RandomVariable
        Infections over the generation time's span before day 1,
        `(G, k)` (PyRenew's `InfectionInitializationProcess`).
    log_rt_rv : RandomVariable
        Log R by day and stratum, sampled with `n` and `place`
        (`rt.PlaceRt`).
    population : array_like
        Each stratum's population.
    place : array_like
        Each stratum's place index (0-based).
    age : array_like
        Each stratum's age group.
    mixing_rv : RandomVariable, optional
        The kernel and its daily share (`mixing.PlaceMixing`).
    imports_rv : RandomVariable, optional
        Imports by day and stratum, sampled with `n` and `unit`
        (`imports.Imports`).
    unit : array_like, optional
        Each stratum's import unit (its seed or level).
    shifts : Shifts, optional
        Shifts in R.
    variants : Variants, optional
        Variants.
    depletion : Depletion, optional
        `strata.LinearDepletion` (default) or `strata.ExponentialDepletion`.
    """

    def __init__(
        self,
        *,
        name: str,
        gen_int_rv: RandomVariable,
        start_rv: RandomVariable,
        log_rt_rv: RandomVariable,
        population: ArrayLike,
        place: ArrayLike,
        age: ArrayLike,
        mixing_rv: RandomVariable | None = None,
        imports_rv: RandomVariable | None = None,
        unit: ArrayLike | None = None,
        shifts: Shifts | None = None,
        variants: Variants | None = None,
        depletion: Depletion | None = None,
    ) -> None:
        super().__init__(
            name=name,
            gen_int_rv=gen_int_rv,
            n_initialization_points=len(gen_int_rv()),
        )
        self.start_rv = start_rv
        self.log_rt_rv = log_rt_rv
        self.population = jnp.asarray(population, dtype=float)
        self.place = place
        self.age = age
        self.mixing_rv = mixing_rv
        self.imports_rv = imports_rv
        self.unit = unit
        self.shifts = shifts
        self.variants = variants
        self.depletion = depletion

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: ArrayLike | None = None,
        **kwargs: Any,
    ) -> LatentSample:
        """Sample the infections.

        Parameters
        ----------
        n_days_post_init : int
            Days after the start.
        subpop_fractions : array_like, optional
            Ignored: the process holds each stratum's population.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        LatentSample
            `aggregate` `(G + n,)` and `all_subpops` `(G + n, k)`.
        """
        n = n_days_post_init
        log_rt = self.log_rt_rv(n=n, place=self.place)
        mixing = (
            None
            if self.mixing_rv is None
            else self.mixing_rv(n=n, place=self.place, age=self.age)
        )
        imports = (
            None
            if self.imports_rv is None
            else self.imports_rv(n=n, unit=self.unit)
        )
        run = strata_renewal(
            self.start_rv(),
            self.gen_int_rv(),
            jnp.exp(log_rt),
            self.population,
            mixing=mixing,
            imports=imports,
            shifts=self.shifts,
            variants=self.variants,
            depletion=self.depletion,
        )
        numpyro.deterministic(f"{self.name}::r", run.r)
        numpyro.deterministic(f"{self.name}::reff", run.reff)
        numpyro.deterministic(f"{self.name}::susceptible", run.susceptible)
        return LatentSample(
            aggregate=run.infections.sum(axis=1), all_subpops=run.infections
        )
