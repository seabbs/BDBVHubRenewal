# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Subpopulation renewal coupled by importation, with susceptible depletion.

`MetapopulationInfections` is a PyRenew `BaseSubpopulationInfections` with
a coupled renewal step (`coupled_renewal`) and the mixing matrix from an
`Importation` component.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent import BaseSubpopulationInfections
from pyrenew.latent.base import LatentSample, PopulationStructure
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import RandomVariable

from .pooled import pooled_walk


def mixing_matrix(
    kernel: ArrayLike, importation: ArrayLike, fractions: ArrayLike
) -> Array:
    r"""Per-capita mixing matrix from a row-normalised importation kernel.

    In counts the force on province $p$ is
    $(1 - \varepsilon) F_p + \varepsilon \sum_q K_{qp} F_q$. Per resident
    of $p$, with population fractions $f$,

    $$W = (1 - \varepsilon) I + \varepsilon
    \operatorname{diag}(1/f) K^\top \operatorname{diag}(f).$$

    Parameters
    ----------
    kernel
        Row-normalised kernel with zero diagonal, shape `(P, P)`.
    importation
        Share $\varepsilon$ of transmission sent to other provinces, a
        scalar or one per origin.
    fractions
        Population fractions, shape `(P,)`.

    Returns
    -------
    Array
        Shape `(P, P)`.
    """
    kernel, fractions = jnp.asarray(kernel), jnp.asarray(fractions)
    n = fractions.shape[0]
    # A vector of shares is one per origin: row p keeps 1 - eps_p of its
    # own force and takes eps_q of origin q's.
    eps = jnp.broadcast_to(jnp.asarray(importation), (n,))
    imported = kernel.T * (eps * fractions)[None, :] / fractions[:, None]
    return jnp.asarray(jnp.diag(1 - eps) + imported)


def coupled_renewal(
    I0: Array,
    rt: Array,
    gen_int: Array,
    mixing: Array,
    pool: Array | float | None = None,
    seeding: Array | None = None,
) -> tuple[Array, Array, Array]:
    r"""Run a per-capita renewal with importation and depletion.

    With $x_p(t)$ infections per resident, $\nu$ the effective share of
    the population at risk (`pool`), $g$ the generation interval and $W$
    from `mixing_matrix`,

    $$y_p(t) = R_p(t) \sum_q W_{pq} \sum_{\tau \ge 1}
    x_q(t - \tau) g(\tau),$$

    $$x_p(t) = S_p(t) \{1 - e^{-y_p(t) / \nu}\}, \qquad
    S_p(t + 1) = S_p(t) - x_p(t) = S_p(t) e^{-y_p(t) / \nu},$$

    with $S_p$ the susceptibles per resident, starting from $\nu$ less
    the seed, so cumulative infections never exceed the pool. This is
    PyRenew's `logistic_susceptibility_adjustment` (Bhatt et al. 2023,
    eq. 6) with population $\nu$, written with `expm1`: PyRenew's
    `1 - exp(-y)` loses most of its digits in float32 at per-capita
    rates near 1e-6, which makes the gradient noisy and stalls NUTS.
    To first order in $y$ it is the linear-depletion renewal $x = S y / \nu$
    (the NumPy reference in `tests/model/numpy_reference.py`). Without
    a pool there is no depletion and $x_p(t) = y_p(t)$.

    PyRenew's renewal scans (`compute_infections_from_rt` and
    `compute_infections_from_rt_with_mobility`) carry no
    transfer between provinces with depletion of a pool $\nu$. Hence
    this scan, written as theirs: a pure function of the seed, $R$ and
    the generation interval, called from the latent process.

    Parameters
    ----------
    I0
        Per-capita infections in the seeding period, shape `(n_init, P)`.
        All rows count towards depletion; the last `len(gen_int)` rows
        start the renewal.
    rt
        Reproduction numbers after seeding, shape `(n, P)`.
    gen_int
        Generation interval pmf, element 0 at lag 1 (PyRenew convention).
    mixing
        Mixing matrix $W$, shape `(P, P)`.
    pool
        Effective share of the population at risk, $\nu$. `None`
        turns depletion off.
    seeding
        Seed infections per resident added to $y_p(t)$, `(n, P)`, for
        provinces seeded after the start.

    Returns
    -------
    tuple of Array
        Infections `(n, P)`, the effective reproduction number
        $R_p(t) S_p(t) / \nu$, `(n, P)`, and the infections from
        importation, `(n, P)`: the off-diagonal part of $y$ as a share of
        $y$, times the infections.
    """
    off = mixing - jnp.diag(jnp.diag(mixing))
    rev_g = jnp.flip(jnp.asarray(gen_int))
    history = I0[-rev_g.shape[0] :]
    # The susceptible share of the pool is carried on the log scale:
    # S(t + 1) = S(t) exp(-y / nu) exactly, so log S needs no subtraction
    # of nearly equal numbers, and 1 - exp(-y / nu) is taken by expm1.
    if pool is None:
        log_share0 = jnp.zeros(rt.shape[1:])
    else:
        susceptible0 = jnp.clip(pool - I0.sum(axis=0), 0.0, pool)
        log_share0 = jnp.log(jnp.clip(susceptible0 / pool, 1e-30, 1.0))

    if seeding is None:
        seeding = jnp.zeros_like(rt)

    def step(
        carry: tuple[Array, Array], inputs: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        hist, log_share = carry
        r_t, seed_t = inputs
        force = rev_g @ hist
        imported = r_t * (off @ force)
        y = r_t * (mixing @ force) + seed_t
        share = jnp.exp(log_share)
        if pool is None:
            x = y
        else:
            x = -pool * share * jnp.expm1(-y / pool)
            log_share = log_share - y / pool
        hist = jnp.concatenate([hist[1:], x[None, :]])
        from_imports = x * jnp.where(
            y > 0, imported / jnp.where(y > 0, y, 1), 0
        )
        return (hist, log_share), (x, r_t * share, from_imports)

    _, (x, r_eff, from_imports) = jax.lax.scan(
        step, (history, log_share0), (rt, seeding)
    )
    return x, r_eff, from_imports


class Importation(RandomVariable):
    r"""Per-capita mixing matrix from an importation kernel and share.

    $$W = (1 - \varepsilon) I + \varepsilon
    \operatorname{diag}(1/f) K^\top \operatorname{diag}(f),$$

    from `mixing_matrix`, with $K$ from `kernel_rv` (row-normalised,
    rows are origins, zero diagonal), $\varepsilon$ from `share_rv` and
    the population fractions $f$ passed at sample time. It plays the
    part of PyRenew's `mobility_matrix_rv`, as one component the latent
    process calls, so the kernel may be fixed or random.
    """

    def __init__(
        self, name: str, kernel_rv: RandomVariable, share_rv: RandomVariable
    ) -> None:
        r"""Initialize the importation component.

        Parameters
        ----------
        name
            Name of the component.
        kernel_rv
            Returns $K$, shape `(P, P)`. A fixed kernel is a
            `DeterministicVariable`.
        share_rv
            Returns $\varepsilon$, the share of transmission sent to
            other provinces.

        Raises
        ------
        TypeError
            If `kernel_rv` or `share_rv` is not a `RandomVariable`.
        """
        for label, rv in (("kernel_rv", kernel_rv), ("share_rv", share_rv)):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        super().__init__(name=name)
        self.kernel_rv = kernel_rv
        self.share_rv = share_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Importation(name={self.name!r}, kernel_rv={self.kernel_rv!r}, "
            f"share_rv={self.share_rv!r})"
        )

    def sample(self, fractions: ArrayLike, **kwargs: object) -> Array:
        """Sample the mixing matrix.

        Parameters
        ----------
        fractions
            Population share of each province, `(P,)`.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            $W$, shape `(P, P)`.
        """
        return mixing_matrix(self.kernel_rv(), self.share_rv(), fractions)


class MetapopulationInfections(BaseSubpopulationInfections):
    r"""Province renewal with importation and susceptible depletion.

    Rt follows `BaseSubpopulationInfections`: a national temporal process
    plus sum-to-zero province deviations,
    $\log R_p(t) = \log R(t) + \delta_p(t)$, here sampled on the
    sum-to-zero basis (`pooled_walk`).
    Seeding is PyRenew's exponential seeding at the initial growth rate,
    but anchored so that `I0_rv` is the level on the first data day
    rather than the first seeding day.
    The renewal is `coupled_renewal`, which sends a share $\varepsilon$
    of transmission to other provinces through the mixing matrix of
    `mixing_rv` and depletes each province's effective susceptible pool.
    `sample` runs the steps of `BaseSubpopulationInfections.sample`, with
    the renewal in `_infections_from_rt`.

    Infections are per resident, so the population fractions passed at
    sample time set the transfer weights. Only a share $\nu$
    (`effective_population_rv`) of each province is at risk, and it is
    that pool that depletes. Because $\nu$ only enters through depletion,
    it does not trade off against the seed or the ascertainment.
    """

    def __init__(
        self,
        *,
        mixing_rv: RandomVariable | None = None,
        effective_population_rv: RandomVariable | None = None,
        rt_observation: RandomVariable | None = None,
        seed_days: ArrayLike | None = None,
        **kwargs: object,
    ) -> None:
        r"""Initialize the coupled infections process.

        Parameters
        ----------
        mixing_rv
            Returns the mixing matrix $W$ given the population fractions
            (`Importation`). `None` turns coupling off.
        effective_population_rv
            $\nu$. `None` turns susceptible depletion off.
        rt_observation
            Called with Rt, infections, the seeding length and the
            fractions, to add an observation of Rt
            (`RtPseudoObservation`). `None` adds none.
        seed_days
            Day (from the first data day) each province's seeding starts.
            Provinces with a day after 0 start (almost) empty and get
            their seed, the `I0_rv` level, as a constant daily input over
            the 7 days from it, plus whatever importation brings. The
            day leads the first case week by the incubation period
            (`model.seed_lead`). A province with day `inf` is never
            seeded and gains infections only by importation. `None`
            seeds every province at the start.
        **kwargs
            Passed to `BaseSubpopulationInfections`.
        """
        super().__init__(**kwargs)
        self.mixing_rv = mixing_rv
        self.effective_population_rv = effective_population_rv
        self.rt_observation = rt_observation
        self.seed_days = None if seed_days is None else np.asarray(seed_days)

    def _initial_infections(
        self, I0: Array, rt0: Array, gen_int: Array, n_days: int
    ) -> tuple[Array, Array | None]:
        """Seed infections, and the later seeding input of late provinces.

        The seed `I0` is set on the first data day and projected back
        over the seeding period at the initial growth rate. The rate is
        clipped so a steep early decline cannot project a seed larger
        than the pool. Late provinces start from 1e-6 of their seed
        level (PyRenew needs a positive start) and take the seed over
        the 7 days from their seed day.

        Returns
        -------
        tuple
            Infections per resident over the seeding period,
            `(n_init, P)`, and the seeding input after it, `(n_days, P)`,
            or None.
        """
        n_init = self.n_initialization_points
        r0 = r_approx_from_R(R=rt0, g=gen_int, n_newton_steps=4)
        r_seed = jnp.clip(r0, -0.05, 0.3)
        I0_all = I0[None, :] * jnp.exp(
            r_seed[None, :] * (jnp.arange(n_init) - n_init)[:, None]
        )
        if self.seed_days is None:
            return I0_all, None
        late = jnp.asarray(self.seed_days > 0)
        I0_all = jnp.where(late[None, :], 1e-6 * I0_all, I0_all)
        days = np.arange(n_days)[:, None]
        window = (days >= self.seed_days) & (days < self.seed_days + 7)
        seeding = jnp.asarray(window & late) * I0[None, :]
        return I0_all, seeding

    def _infections_from_rt(
        self,
        rt_raw: Array,
        I0: Array,
        gen_int: Array,
        pop: PopulationStructure,
        seeding: Array | None = None,
    ) -> tuple[Array, Array]:
        r"""Infections and adjusted Rt after seeding, by `coupled_renewal`.

        Records the infections from importation as `infections_imported`.

        Returns
        -------
        tuple of Array
            Infections per resident and $R_p(t) S_p(t) / \nu$, both
            `(n_days, P)`.
        """
        mixing = (
            jnp.eye(pop.n_subpops)
            if self.mixing_rv is None
            else self.mixing_rv(fractions=pop.fractions)
        )
        pool = (
            None
            if self.effective_population_rv is None
            else self.effective_population_rv()
        )
        post, r_eff, imported = coupled_renewal(
            I0, rt_raw, gen_int, mixing, pool, seeding
        )
        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic(
                "infections_imported",
                jnp.vstack([jnp.zeros_like(I0), imported]),
            )
        return post, r_eff

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: Array | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """Sample per-capita infections for each province.

        Parameters
        ----------
        n_days_post_init
            Days after the seeding period.
        subpop_fractions
            Population share of each province.
        first_day_dow
            Weekday of the first day, passed by PyRenew.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        LatentSample
            `aggregate` of shape `(n_total,)` and `all_subpops` of shape
            `(n_total, n_subpops)`.
        """
        pop = self._parse_and_validate_fractions(subpop_fractions)
        n_init = self.n_initialization_points
        log_rt_baseline, log_rt = pooled_walk(
            self.baseline_rt_process,
            self.subpop_rt_deviation_process,
            self.log_rt_time_0_rv(),
            n_init + n_days_post_init,
            pop.n_subpops,
            "log_rt",
            first_day_dow,
        )
        rt = jnp.exp(log_rt)
        gen_int = self.gen_int_rv()
        I0 = self._validate_and_prepare_I0(jnp.asarray(self.I0_rv()), pop)
        I0_all, seeding = self._initial_infections(
            I0, rt[0], gen_int, n_days_post_init
        )
        post, r_eff = self._infections_from_rt(
            rt[n_init:], I0_all, gen_int, pop, seeding
        )
        infections = jnp.vstack([I0_all, post])
        aggregate = infections @ pop.fractions

        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic("rt_baseline", jnp.exp(log_rt_baseline))
            numpyro.deterministic("rt_subpop", rt)
            numpyro.deterministic(
                "rt_subpop_adjusted", jnp.vstack([rt[:n_init], r_eff])
            )
        if self.rt_observation is not None:
            self.rt_observation(
                rt=rt,
                infections=infections,
                n_init=n_init,
                fractions=pop.fractions,
            )
        return LatentSample(aggregate=aggregate, all_subpops=infections)
