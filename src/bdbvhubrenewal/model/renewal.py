# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Province renewal with importation and susceptible depletion.

`MetapopulationInfections` is PyRenew's `SubpopulationInfections` with a
different renewal step. Rt, seeding and the output contract are unchanged.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent import SubpopulationInfections
from pyrenew.latent.base import LatentSample
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import RandomVariable

from .observation import RtPseudoObservation
from .rt import pooled_walk


def adjacency_kernel(adjacency: ArrayLike) -> np.ndarray:
    """Build a row-normalised importation kernel from a 0/1 adjacency matrix.

    Rows are origins, as in `bdbvhubrenewal.model.inputs.gravity_kernel`:
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
    return np.asarray(adj / adj.sum(axis=1, keepdims=True), dtype=float)


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
    kernel : Array
        Row-normalised kernel with zero diagonal, shape `(P, P)`.
    importation : Array
        Share $\varepsilon$ of transmission sent to other provinces, a
        scalar or one per origin.
    fractions : Array
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
    susceptible0: Array,
    deplete: bool = True,
    pool: Array | float = 1.0,
    seeding: Array | None = None,
    import_rt: str = "destination",
) -> tuple[Array, Array, Array]:
    r"""Run a per-capita renewal with importation and depletion.

    With $x_p(t)$ infections per resident, $\nu$ the effective share of
    the population at risk (`pool`), $g$ the generation interval and $W$
    from `mixing_matrix`,

    $$y_p(t) = R_p(t) \sum_q W_{pq} \sum_{\tau \ge 1}
    x_q(t - \tau) g(\tau),$$

    $$x_p(t) = S_p(t) \{1 - e^{-y_p(t) / \nu}\}, \qquad
    S_p(t + 1) = S_p(t) - x_p(t) = S_p(t) e^{-y_p(t) / \nu},$$

    With `import_rt="origin"` imports arrive at the origin's Rt instead,
    $y_p(t) = \sum_q W_{pq} R_q(t) \sum_{\tau \ge 1} x_q(t - \tau)
    g(\tau)$, as BVDOutbreakSize relocates infections (`renewal.jl`), so
    a low destination Rt does not switch its imports off.

    with $S_p$ the susceptibles per resident, starting from $\nu$ less
    the seed, so cumulative infections never exceed the pool. This is
    PyRenew's `logistic_susceptibility_adjustment` (Bhatt et al. 2023,
    eq. 6) with population $\nu$, written with `expm1`: PyRenew's
    `1 - exp(-y)` loses most of its digits in float32 at per-capita
    rates near 1e-6, which makes the gradient noisy and stalls NUTS.
    To first order in $y$ it is the linear-depletion renewal $x = S y / \nu$
    (the NumPy reference in `tests/model/numpy_reference.py`). Without
    depletion $x_p(t) = y_p(t)$.

    PyRenew's renewal scan (`compute_infections_from_rt`) runs each
    column on its own and its feedback term only looks back one
    generation interval, so it cannot carry transfer between provinces
    or cumulative depletion. Hence this scan.

    Parameters
    ----------
    I0 : Array
        Per-capita infections in the seeding period, shape `(n_init, P)`.
        Only the last `len(gen_int)` rows are used.
    rt : Array
        Reproduction numbers after seeding, shape `(n, P)`.
    gen_int : Array
        Generation interval pmf, element 0 at lag 1 (PyRenew convention).
    mixing : Array
        Mixing matrix $W$, shape `(P, P)`.
    susceptible0 : Array
        Susceptibles per resident after seeding, shape `(P,)`.
    deplete : bool, optional
        Apply susceptible depletion.
    pool : Array, optional
        Effective share of the population at risk, $\nu$.
    seeding : Array, optional
        Seed infections per resident added to $y_p(t)$, `(n, P)`, for
        provinces seeded after the start.
    import_rt : {"destination", "origin"}
        Whose Rt multiplies imported transmission.

    Returns
    -------
    tuple of Array
        Infections `(n, P)`, the effective reproduction number
        $R_p(t) S_p(t) / \nu$, `(n, P)`, and the infections from
        importation, `(n, P)`: the off-diagonal part of $y$ as a share of
        $y$, times the infections.
    """
    if import_rt not in ("destination", "origin"):
        raise ValueError(f"Unknown import_rt {import_rt!r}.")
    off = mixing - jnp.diag(jnp.diag(mixing))
    rev_g = jnp.flip(jnp.asarray(gen_int))
    history = I0[-rev_g.shape[0] :]
    # The susceptible share of the pool is carried on the log scale:
    # S(t + 1) = S(t) exp(-y / nu) exactly, so log S needs no subtraction
    # of nearly equal numbers, and 1 - exp(-y / nu) is taken by expm1.
    log_share0 = jnp.log(jnp.clip(susceptible0 / pool, 1e-30, 1.0))
    if not deplete:
        log_share0 = jnp.zeros_like(log_share0)

    if seeding is None:
        seeding = jnp.zeros_like(rt)

    def step(
        carry: tuple[Array, Array], inputs: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        hist, log_share = carry
        r_t, seed_t = inputs
        force = rev_g @ hist
        if import_rt == "origin":
            imported = off @ (r_t * force)
            y = mixing @ (r_t * force) + seed_t
        else:
            imported = r_t * (off @ force)
            y = r_t * (mixing @ force) + seed_t
        share = jnp.exp(log_share)
        if deplete:
            x = -pool * share * jnp.expm1(-y / pool)
            log_share = log_share - y / pool
        else:
            x = y
        hist = jnp.concatenate([hist[1:], x[None, :]])
        from_imports = x * jnp.where(
            y > 0, imported / jnp.where(y > 0, y, 1), 0
        )
        return (hist, log_share), (x, r_t * share, from_imports)

    _, (x, r_eff, from_imports) = jax.lax.scan(
        step, (history, log_share0), (rt, seeding)
    )
    return x, r_eff, from_imports


class LearntKernel(RandomVariable):
    r"""Gravity importation kernel with a learnt decay and destination weights.

    $$\log K_{qp} = \log N_p - D_{qp} / (300\,\text{km} \cdot e^{\kappa})
    + w_p + c_q, \qquad \kappa \sim N(0, 1),$$

    with $w = \sigma_w Q z$ on the sum-to-zero basis (destinations
    among the modelled provinces only), $z \sim N(0, 1)$,
    $\sigma_w \sim N^+(0, 0.5)$, and $c_q$ normalising each origin's
    row to one, which holds its export total, so the importation scale
    stays with $\varepsilon$ (BVDOutbreakSize #1030, decay and
    destination weights). With $\kappa = 0$ and $w = 0$ it is the gravity
    kernel. All on the log scale (`log_softmax`).

    With `flows`, the log flows between modelled provinces also get a
    multivariate normal deviation $W = L\zeta$, $\zeta \sim N(0, I)$
    (non-centred), with $L$ from `importation_flow_factor`: double-centred
    flows with $\sigma_\text{flow} \sim N^+(0, 0.5)$ and reciprocity
    $(\rho + 1)/2 \sim \text{Beta}(2, 2)$, as BVDOutbreakSize (#1030 and
    later). The flow basis, its scaling and these priors are copied from
    there; BVDOutbreakSize's destination effect inside the factor and its
    correlation with per-origin importation are not, as the destination
    weights above take that role and $\varepsilon$ is shared.

    `sample` returns the kernel between modelled provinces. The kernel
    from them to provinces with no cases (gravity with the learnt decay,
    no weights) is kept as `import_kernel` for `OnsetStreams` in the
    same model run.

    Parameters
    ----------
    name : str
        Prefix for the sample sites.
    distances : array_like
        Distances in km between all provinces, modelled first, `(P+Q, P+Q)`.
    populations : array_like
        Populations, `(P + Q,)`.
    n_active : int
        Number of modelled provinces $P$.
    decay_km : float, optional
        Prior median of the decay.
    weight_sd : float, optional
        Scale of the half-normal prior on $\sigma_w$.
    flows : bool, optional
        Add the multivariate normal on the log flows.
    """

    def __init__(
        self,
        name: str,
        distances: ArrayLike,
        populations: ArrayLike,
        n_active: int,
        decay_km: float = 300.0,
        weight_sd: float = 0.5,
        flows: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.flows = flows
        self.symmetric, self.antisymmetric = flow_pair_basis(n_active)
        self.distances = jnp.asarray(distances)
        self.log_pop = jnp.log(jnp.asarray(populations))
        self.n_active = n_active
        self.decay_km = decay_km
        self.weight_sd = weight_sd
        self.import_kernel: Array | None = None

    def sample(self, **kwargs: object) -> Array:
        """Sample the kernel between modelled provinces.

        Returns
        -------
        Array
            Row-normalised, zero diagonal, `(P, P)`.
        """
        import numpyro.distributions as dist

        from .rt import sum_to_zero_basis

        p = self.n_active
        log_decay = numpyro.sample(
            f"{self.name}_log_decay", dist.Normal(0.0, 1.0)
        )
        base = self.log_pop[None, :] - self.distances / (
            self.decay_km * jnp.exp(log_decay)
        )
        self_flow = jnp.eye(base.shape[0], dtype=bool)
        base = jnp.where(self_flow, -jnp.inf, base)
        if p > 1:
            sd = numpyro.sample(
                f"{self.name}_weight_sd", dist.HalfNormal(self.weight_sd)
            )
            z = numpyro.sample(
                f"{self.name}_weight_z", dist.Normal(0.0, 1.0).expand([p - 1])
            )
            w = sd * (jnp.asarray(sum_to_zero_basis(p)) @ z)
        else:
            w = jnp.zeros(1)
        flow = jnp.zeros((p, p))
        ds, da = self.symmetric.shape[1], self.antisymmetric.shape[1]
        if self.flows and ds + da > 0:
            sd_flow = numpyro.sample(
                f"{self.name}_flow_sd", dist.HalfNormal(0.5)
            )
            z = numpyro.sample(
                f"{self.name}_flow_z", dist.Normal(0.0, 1.0).expand([ds + da])
            )
            rho = jnp.asarray(-1.0)
            if ds > 0:
                unit = numpyro.sample(
                    f"{self.name}_reciprocity_unit", dist.Beta(2.0, 2.0)
                )
                rho = 2 * unit - 1
            factor = importation_flow_factor(
                self.symmetric, self.antisymmetric, sd_flow, rho
            )
            flow = (factor @ z).reshape(p, p)
            numpyro.deterministic(f"{self.name}_log_flow_deviation", flow)
        numpyro.deterministic(f"{self.name}_destination_weights", w)
        active = jax.nn.log_softmax(base[:p, :p] + w[None, :] + flow, axis=1)
        kernel = jnp.exp(active)
        numpyro.deterministic(f"{self.name}_kernel", kernel)
        if base.shape[0] > p:
            self.import_kernel = jnp.exp(jax.nn.log_softmax(base[:p], axis=1))[
                :, p:
            ]
        return kernel


def _orthonormal_complement(constraints: np.ndarray, m: int) -> np.ndarray:
    """Orthonormal basis of the complement of the rows' span in R^m."""
    _, sv, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int((sv > 1e-9 * sv.max()).sum())
    return np.asarray(vt[rank:].T, dtype=float).reshape(m, m - rank)


def flow_pair_basis(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Orthonormal bases of the double-centred flows between `n` provinces.

    As BVDOutbreakSize `flow_pair_basis`. A double-centred flow matrix has
    a zero diagonal and zero row and column sums. The symmetric part moves
    $q \to p$ and $p \to q$ together and has $n(n - 3)/2$ directions; the
    antisymmetric part moves them in opposite directions and has
    $(n - 1)(n - 2)/2$. Column $j$ reshapes (row-major) to an `(n, n)`
    matrix with rows as origins.

    Returns
    -------
    tuple of numpy.ndarray
        Symmetric and antisymmetric bases, `(n * n, d)` each.
    """
    idx = np.arange(n * n).reshape(n, n)
    rows = []
    for p in range(n):
        for cells in ([idx[p, p]], idx[p, :], idx[:, p]):
            v = np.zeros(n * n)
            v[cells] = 1.0
            rows.append(v)

    def swapped(sign: float) -> list[np.ndarray]:
        out = []
        for q in range(n):
            for p in range(q):
                v = np.zeros(n * n)
                v[idx[p, q]] += 1.0
                v[idx[q, p]] += sign
                out.append(v)
        return out

    bases = []
    for sign in (-1.0, 1.0):
        basis = _orthonormal_complement(np.array(rows + swapped(sign)), n * n)
        basis[np.diag(idx)] = 0.0
        bases.append(basis)
    return bases[0], bases[1]


def importation_flow_factor(
    symmetric: ArrayLike,
    antisymmetric: ArrayLike,
    sd_flow: ArrayLike,
    reciprocity: ArrayLike,
) -> Array:
    r"""Factor $L$ of the normal prior on the log flow deviations.

    The flow term of BVDOutbreakSize `importation_flow_factor`: the
    double-centred flows (`flow_pair_basis`) scaled so each flow has
    variance $\sigma_\text{flow}^2$ on average and correlation $\rho$ with
    its reverse, $L L^\top = \sigma_\text{flow}^2 R(\rho)$. The
    destination effect of BVDOutbreakSize's factor is left out here: the
    learnt destination weights of `LearntKernel` carry it. Rows are flows
    $q \to p$ at `q * n + p`; diagonal rows are zero.

    Returns
    -------
    Array
        `(n * n, d)`.
    """
    sym, anti = jnp.asarray(symmetric), jnp.asarray(antisymmetric)
    n2 = sym.shape[0]
    n = math.isqrt(n2)
    pairs = n * (n - 1) // 2
    ds, da = sym.shape[1], anti.shape[1]
    parts = []
    if ds > 0:
        parts.append(sd_flow * jnp.sqrt((1 + reciprocity) * pairs / ds) * sym)
    if da > 0:
        parts.append(sd_flow * jnp.sqrt((1 - reciprocity) * pairs / da) * anti)
    if not parts:
        return jnp.zeros((n2, 0))
    return jnp.concatenate(parts, axis=1)


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
    kernel : Array, LearntKernel or None
        Row-normalised importation kernel (rows are origins), fixed or
        learnt. `None` turns coupling off.
    importation_rv : RandomVariable or None
        Importation share $\varepsilon$. Needed when `kernel` is given.
    deplete : bool
        Apply susceptible depletion.
    effective_population_rv : RandomVariable, optional
        $\nu$. Needed with depletion.
    rt_observation : RtPseudoObservation, optional
        External Rt estimates as a tempered pseudo-observation.
    seed_days : array_like, optional
        First day (from the first data day) of each province's first
        case week. Provinces with a day after 0 start (almost) empty and
        get their seed, the `I0_rv` level, as a constant daily input over
        that week, plus whatever importation brings. Infections are dated
        by onset, so seeding the week before would put onsets where the
        data have none. `None` seeds every province at the start.
    deviation_weights : array_like, optional
        Case counts by province. Rt deviations then sum to zero weighted
        by them, so the national Rt walk is the case-weighted log Rt.
    outbreak : array_like, optional
        Outbreak provinces by week of the data, `(n_weeks, P)` boolean. A
        province is sporadic in the weeks it is not: no Rt deviation
        (national Rt), so its infections come from national Rt applied to
        its own and imported infections. The seeding weeks take the first
        data week's class and the forecast the last. `None` makes every
        province an outbreak province throughout.
    primary : int, optional
        Seed only this province (the first with cases); every other
        province starts empty and is seeded by importation, as
        BVDOutbreakSize. `None` seeds each province.
    import_rt : {"destination", "origin"}
        Whose Rt multiplies imported transmission (`coupled_renewal`).
    **kwargs
        Passed to `SubpopulationInfections` (without infection feedback).
    """

    def __init__(
        self,
        *,
        kernel: Array | None = None,
        importation_rv: RandomVariable | None = None,
        deplete: bool = True,
        effective_population_rv: RandomVariable | None = None,
        rt_observation: RtPseudoObservation | None = None,
        seed_days: Array | None = None,
        deviation_weights: Array | None = None,
        outbreak: Array | None = None,
        primary: int | None = None,
        import_rt: str = "destination",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.outbreak = None if outbreak is None else np.asarray(outbreak, bool)
        self.primary = primary
        self.import_rt = import_rt
        self.seed_days = None if seed_days is None else np.asarray(seed_days)
        self.deviation_weights = (
            None if deviation_weights is None else np.asarray(deviation_weights)
        )
        self.rt_observation = rt_observation
        if deplete and effective_population_rv is None:
            raise ValueError("Depletion needs effective_population_rv.")
        self.effective_population_rv = effective_population_rv
        if (kernel is None) != (importation_rv is None):
            raise ValueError("kernel and importation_rv go together.")
        if kernel is not None and not isinstance(kernel, RandomVariable):
            kernel = jnp.asarray(kernel)
        self.kernel = kernel
        self.importation_rv = importation_rv
        self.deplete = deplete

    def _daily_outbreak(self, n_init: int, n_total: int) -> np.ndarray | None:
        """Weekly outbreak classes on the daily model axis."""
        if self.outbreak is None:
            return None
        week = np.clip(
            (np.arange(n_total) - n_init) // 7, 0, self.outbreak.shape[0] - 1
        )
        return self.outbreak[week]

    def sample(
        self,
        n_days_post_init: int,
        subpop_fractions: Array | None = None,
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
            self.deviation_weights,
            self._daily_outbreak(n_init, n_total),
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
            kernel = (
                self.kernel()
                if isinstance(self.kernel, RandomVariable)
                else self.kernel
            )
            if self.importation_rv is None:
                raise ValueError("A kernel needs importation_rv.")
            mixing = mixing_matrix(kernel, self.importation_rv(), pop.fractions)
        seeding = None
        if self.seed_days is not None:
            late = jnp.asarray(self.seed_days > 0)
            # Late provinces start from 1e-6 of their seed level (PyRenew
            # needs a positive start) and take the seed over their first
            # case week.
            I0_all = jnp.where(late[None, :], 1e-6 * I0_all, I0_all)
            days = np.arange(rt.shape[0] - n_init)[:, None]
            window = (days >= self.seed_days) & (days < self.seed_days + 7)
            seeding = jnp.asarray(window & (self.seed_days > 0)) * I0[None, :]
        if self.primary is not None:
            # Only the primary province is seeded. The others start empty
            # (PyRenew needs a positive start) and importation seeds them.
            others = jnp.asarray(np.arange(pop.n_subpops) != self.primary)
            I0_all = jnp.where(others[None, :], 1e-12, I0_all)
            if seeding is not None:
                seeding = jnp.where(others[None, :], 0.0, seeding)
        pool: Array | float = 1.0
        if self.deplete:
            if self.effective_population_rv is None:
                raise ValueError("Depletion needs effective_population_rv.")
            pool = self.effective_population_rv()
        susceptible0 = jnp.clip(pool - I0_all.sum(axis=0), 0.0, pool)
        post, r_eff, imported = coupled_renewal(
            I0_all,
            rt[n_init:],
            gen_int,
            mixing,
            susceptible0,
            self.deplete,
            pool,
            seeding,
            self.import_rt,
        )
        infections = jnp.vstack([I0_all, post])
        aggregate = infections @ pop.fractions

        with numpyro.handlers.scope(prefix=self.name, divider="::"):
            numpyro.deterministic("rt_baseline", jnp.exp(log_rt_baseline))
            numpyro.deterministic("rt_subpop", rt)
            numpyro.deterministic(
                "rt_subpop_adjusted", jnp.vstack([rt[:n_init], r_eff])
            )
            numpyro.deterministic(
                "infections_imported",
                jnp.vstack([jnp.zeros_like(I0_all), imported]),
            )
        if self.rt_observation is not None:
            self.rt_observation.sample(rt, infections, n_init, pop.fractions)
        return LatentSample(aggregate=aggregate, all_subpops=infections)
