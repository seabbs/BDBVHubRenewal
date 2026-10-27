# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
# Synced from the prototype by scripts/sync_pyrenew.py. Edit there.
r"""Province renewal with importation and susceptible depletion.

`MetapopulationInfections` is PyRenew's `SubpopulationInfections` with a
different renewal step. Rt, seeding and the output contract are unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.typing import ArrayLike
from pyrenew.latent import SubpopulationInfections
from pyrenew.latent.base import LatentSample
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import RandomVariable

from .processes import pooled_walk


def adjacency_kernel(adjacency: ArrayLike) -> np.ndarray:
    """Build a row-normalised importation kernel from a 0/1 adjacency matrix.

    Rows are origins, as in `bdbv_hub_renewal.simulate.gravity_kernel`:
    `K[q, p]` is the share of province `q`'s exported transmission that
    lands in `p`.

    Parameters
    ----------
    adjacency : array_like
        Symmetric 0/1 matrix, shape `(P, P)`. Each province needs a
        neighbour.

    Returns
    -------
    numpy.ndarray
        Kernel with zero diagonal and rows summing to one.
    """
    adj = np.asarray(adjacency, dtype=float).copy()
    np.fill_diagonal(adj, 0.0)
    return adj / adj.sum(axis=1, keepdims=True)


def mixing_matrix(
    kernel: ArrayLike, importation: ArrayLike, fractions: ArrayLike
) -> ArrayLike:
    r"""Per-capita mixing matrix from a row-normalised importation kernel.

    In counts the force on province $p$ is
    $(1 - \varepsilon) F_p + \varepsilon \sum_q K_{qp} F_q$. Per resident
    of $p$, with population fractions $f$,

    $$W = (1 - \varepsilon) I + \varepsilon
    \operatorname{diag}(1/f) K^\top \operatorname{diag}(f).$$

    Parameters
    ----------
    kernel : ArrayLike
        Row-normalised kernel with zero diagonal, shape `(P, P)`.
    importation : ArrayLike
        Share $\varepsilon$ of transmission sent to other provinces.
    fractions : ArrayLike
        Population fractions, shape `(P,)`.

    Returns
    -------
    ArrayLike
        Shape `(P, P)`.
    """
    n = fractions.shape[0]
    imported = kernel.T * fractions[None, :] / fractions[:, None]
    return (1 - importation) * jnp.eye(n) + importation * imported


def coupled_renewal(
    I0: ArrayLike,
    rt: ArrayLike,
    gen_int: ArrayLike,
    mixing: ArrayLike,
    susceptible0: ArrayLike,
    deplete: bool = True,
    pool: ArrayLike = 1.0,
) -> tuple[ArrayLike, ArrayLike]:
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
    To first order in $y$ it is `bdbv_hub_renewal.latent.renewal` per
    capita, $x = S y / \nu$. Without
    depletion $x_p(t) = y_p(t)$.

    PyRenew's renewal scan (`compute_infections_from_rt`) runs each
    column on its own and its feedback term only looks back one
    generation interval, so it cannot carry transfer between provinces
    or cumulative depletion. Hence this scan.

    Parameters
    ----------
    I0 : ArrayLike
        Per-capita infections in the seeding period, shape `(n_init, P)`.
        Only the last `len(gen_int)` rows are used.
    rt : ArrayLike
        Reproduction numbers after seeding, shape `(n, P)`.
    gen_int : ArrayLike
        Generation interval pmf, element 0 at lag 1 (PyRenew convention).
    mixing : ArrayLike
        Mixing matrix $W$, shape `(P, P)`.
    susceptible0 : ArrayLike
        Susceptibles per resident after seeding, shape `(P,)`.
    deplete : bool, optional
        Apply susceptible depletion.
    pool : ArrayLike, optional
        Effective share of the population at risk, $\nu$.

    Returns
    -------
    tuple of ArrayLike
        Infections `(n, P)` and the effective reproduction number
        $R_p(t) S_p(t) / \nu$, `(n, P)`.
    """
    rev_g = jnp.flip(jnp.asarray(gen_int))
    history = I0[-rev_g.shape[0] :]
    # The susceptible share of the pool is carried on the log scale:
    # S(t + 1) = S(t) exp(-y / nu) exactly, so log S needs no subtraction
    # of nearly equal numbers, and 1 - exp(-y / nu) is taken by expm1.
    log_share0 = jnp.log(jnp.clip(susceptible0 / pool, 1e-30, 1.0))
    if not deplete:
        log_share0 = jnp.zeros_like(log_share0)

    def step(carry, r_t):
        hist, log_share = carry
        y = r_t * (mixing @ (rev_g @ hist))
        share = jnp.exp(log_share)
        if deplete:
            x = -pool * share * jnp.expm1(-y / pool)
            log_share = log_share - y / pool
        else:
            x = y
        hist = jnp.concatenate([hist[1:], x[None, :]])
        return (hist, log_share), (x, r_t * share)

    _, (x, r_eff) = jax.lax.scan(step, (history, log_share0), rt)
    return x, r_eff


class MetapopulationInfections(SubpopulationInfections):
    r"""Province renewal with importation and susceptible depletion.

    Rt follows `SubpopulationInfections`: a national temporal process
    plus sum-to-zero province deviations,
    $\log R_p(t) = \log R(t) + \delta_p(t)$.
    Seeding is PyRenew's exponential seeding at the initial growth rate,
    but anchored so that `I0_rv` is the level on the first data day
    rather than the first seeding day.
    The renewal step is replaced by `coupled_renewal`, which sends a share
    $\varepsilon$ of transmission to other provinces through a
    row-normalised kernel and depletes each province's effective
    susceptible pool.

    Infections are per resident, so the population fractions passed at
    sample time set the transfer weights. Only a share $\nu$
    (`effective_population_rv`) of each province is at risk, and it is
    that pool that depletes. Because $\nu$ only enters through depletion,
    it does not trade off against the seed or the ascertainment.

    Parameters
    ----------
    kernel : ArrayLike or None
        Row-normalised importation kernel (rows are origins). `None`
        turns coupling off.
    importation_rv : RandomVariable or None
        Importation share $\varepsilon$. Needed when `kernel` is given.
    deplete : bool
        Apply susceptible depletion.
    effective_population_rv : RandomVariable, optional
        $\nu$. Needed with depletion.
    rt_observation : RtPseudoObservation, optional
        External Rt estimates as a tempered pseudo-observation.
    **kwargs
        Passed to `SubpopulationInfections` (without infection feedback).
    """

    def __init__(
        self,
        *,
        kernel: ArrayLike | None = None,
        importation_rv: RandomVariable | None = None,
        deplete: bool = True,
        effective_population_rv: RandomVariable | None = None,
        rt_observation=None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.rt_observation = rt_observation
        if deplete and effective_population_rv is None:
            raise ValueError("Depletion needs effective_population_rv.")
        self.effective_population_rv = effective_population_rv
        if (kernel is None) != (importation_rv is None):
            raise ValueError("kernel and importation_rv go together.")
        self.kernel = None if kernel is None else jnp.asarray(kernel)
        self.importation_rv = importation_rv
        self.deplete = deplete

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: ArrayLike | None = None,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> LatentSample:
        """Sample per-capita infections for each province.

        Returns
        -------
        LatentSample
            `aggregate` of shape `(n_total,)` and `all_subpops` of shape
            `(n_total, n_subpops)`.
        """
        pop = self._parse_and_validate_fractions(subpop_fractions)
        n_init = self.n_initialization_points
        n_total = n_init + n_days_post_init
        log_rt_baseline, log_rt = pooled_walk(
            self.baseline_rt_process,
            self.subpop_rt_deviation_process,
            self.log_rt_time_0_rv(),
            n_total,
            pop.n_subpops,
            "log_rt",
            first_day_dow,
        )
        rt = jnp.exp(log_rt)
        gen_int = self.gen_int_rv()

        I0 = self._validate_and_prepare_I0(jnp.asarray(self.I0_rv()), pop)
        r0 = r_approx_from_R(R=rt[0], g=gen_int, n_newton_steps=4)
        # The seed is set on the first data day and projected back over
        # the seeding period at the initial growth rate. The rate is
        # clipped so a steep early decline cannot project a seed larger
        # than the pool.
        r_seed = jnp.clip(r0, -0.05, 0.3)
        I0_all = I0[None, :] * jnp.exp(
            r_seed[None, :] * (jnp.arange(n_init) - n_init)[:, None]
        )

        if self.kernel is None:
            mixing = jnp.eye(pop.n_subpops)
        else:
            mixing = mixing_matrix(
                self.kernel, self.importation_rv(), pop.fractions
            )
        pool = self.effective_population_rv() if self.deplete else 1.0
        susceptible0 = jnp.clip(pool - I0_all.sum(axis=0), 0.0, pool)
        post, r_eff = coupled_renewal(
            I0_all,
            rt[n_init:],
            gen_int,
            mixing,
            susceptible0,
            self.deplete,
            pool,
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
            self.rt_observation.sample(rt, infections, n_init)
        return LatentSample(aggregate=aggregate, all_subpops=infections)
