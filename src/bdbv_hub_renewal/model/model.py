# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Assemble, fit and forecast the province model with `PyrenewBuilder`."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import random
from numpyro.distributions.transforms import AffineTransform, PowerTransform
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from pyrenew.ascertainment import IndependentAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent.temporal_processes import WeeklyTemporalProcess
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from scipy import stats

from .delays import (
    FixedTimeVarying,
    HazardDelay,
    ParametricDelay,
    empirical_delay_matrix,
    empirical_missing_fraction,
)
from .observation import OnsetStreams, RtPseudoObservation
from .renewal import LearntKernel, MetapopulationInfections
from .rt import (
    MissingOnsetFraction,
    NonCentredRandomWalk,
    PooledInitialValue,
)

SUNDAY = 6
GEN_INT_LENGTH = 42
ALPHA0 = 0.6


def generation_interval(
    mean: float = 15.0, sd: float = 9.0, length: int = GEN_INT_LENGTH
) -> np.ndarray:
    r"""Double interval censored Gamma generation interval on lags 1..length.

    The daily pmf on $0..L$ is primarycensored's with a uniform primary
    over a day and daily secondary censoring,
    $f(d) = G(d + 1) - 2 G(d) + G(d - 1)$ with
    $G(x) = x F(x; k, \theta) - k \theta F(x; k + 1, \theta)$ for
    $x > 0$ (the analytic Gamma form), truncated at $L$. The renewal
    uses lags 1 to $L$, so lag 0 (about 0.05% of the mass) is dropped and
    the rest renormalised, as BVDOutbreakSize left-truncates its
    generation interval at one day. Computed once, in float64.
    Matches `primarycensored::dpcens(..., D = L + 1)`.

    Returns
    -------
    numpy.ndarray
        Pmf on lags 1..length (PyRenew order, element 0 at lag 1).
    """
    shape, scale = (mean / sd) ** 2, sd**2 / mean

    def integrated_cdf(x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, 0.0)
        return x * stats.gamma.cdf(x, shape, scale=scale) - (
            shape * scale * stats.gamma.cdf(x, shape + 1, scale=scale)
        )

    d = np.arange(length + 1, dtype=float)
    pmf = integrated_cdf(d + 1) - 2 * integrated_cdf(d) + integrated_cdf(d - 1)
    pmf = np.maximum(pmf, 0.0)[1:]
    return pmf / pmf.sum()


@dataclass(frozen=True)
class ModelConfig:
    r"""Structural choices of the model, for the ladder and the variants.

    Attributes
    ----------
    national : bool
        Fit one national renewal instead of provinces (ladder step 0).
    streams : {"split", "total"}
        Fit known and hub-imputed onsets as two streams with estimated
        delays and missingness, or one total stream.
    truncation : bool
        Right truncation by vintage. For the total stream it uses the
        hub's delay pmf as a fixed reporting delay.
    coupling : {"gravity", "learnt", "adjacency", "none"}
        Importation kernel between modelled provinces: fixed gravity,
        gravity with a learnt decay and destination weights
        (`LearntKernel`), adjacency, or none.
    delay : {"hazard", "gamma", "lognormal", "fixed"}
        Nonparametric hazard delay, a censored parametric family, or a
        smoothed empirical delay fixed from the hub's delay file.
    missing : {"estimated", "fixed"}
        Estimate $\pi_p(t)$ or fix it at the smoothed observed share.
    depletion : bool
        Susceptible depletion in the renewal.
    rt_cadence : {"daily", "weekly"}
        Step of the Rt random walks.
    rt_deviation : {"rw", "ar1"}
        Province deviations as random walks, or mean-reverting AR(1)
        with a sampled half-life (default, as BVDOutbreakSize).
    active : {"cases", "all"}
        Full renewal for provinces with cases only, or for every province.
    inactive : {"imports", "none"}
        Importation-only component for provinces with no cases, or none
        (their forecasts are zero).
    report_inactive : bool
        Return forecasts for provinces with no cases. If false their
        draws go to the national `unassigned` component instead.
    effective_population : float or None
        Share $\nu$ of each province at risk, which depletion uses.
        `None` (default) estimates it with a log-normal prior; a number
        fixes it.
    future_rt : {"walk", "hold"}
        Rt past the data: the random walks carry on, or Rt holds at its
        last value (EpiNow2 `future = "latest"`).
    seeding : {"first_case", "start"}
        Seed each province in the week before its first case week, or
        every province at the start of the grid.
    walk : {"knots", "noncentred", "centred"}
        Parameterisation of the national Rt walk: weekly knots of total
        log Rt (`KnotWalk`, default), standardised steps, or the path
        relative to its start. Province deviations, $\pi$, delays and
        ascertainment always use standardised steps.
    bvd : bool
        Add BVDOutbreakSize Rt as a tempered pseudo-observation
        (`RtPseudoObservation`), when the vintage carries it.
    bvd_alpha : float
        Its tempering. 0 gives the model without it, exactly.
    dmax : int
        Maximum onset-to-report delay in days.
    publication_lag : {"estimated", "fixed", "none"}
        Lag from report to publication in a vintage: a Dirichlet simplex
        on $0..L$, fixed at its prior mean (uniform), or none.
    max_publication_lag : int
        $L$, in days (`vintages.PUBLICATION_LAG_DAYS`).
    delay_counts : {"multinomial", "dirichlet-multinomial"}
        Likelihood of the delay file's counts given the weekly totals.
        The Dirichlet-multinomial has $1/\sqrt{\psi} \sim
        \text{HalfNormal}(0.1)$.
    """

    national: bool = False
    streams: Literal["split", "total"] = "split"
    truncation: bool = True
    coupling: Literal["gravity", "learnt", "adjacency", "none"] = "gravity"
    delay: Literal["hazard", "gamma", "lognormal", "fixed"] = "hazard"
    missing: Literal["estimated", "fixed"] = "estimated"
    depletion: bool = True
    rt_cadence: Literal["daily", "weekly"] = "weekly"
    rt_deviation: Literal["rw", "ar1"] = "ar1"
    active: Literal["cases", "all"] = "cases"
    inactive: Literal["imports", "none"] = "imports"
    report_inactive: bool = True
    effective_population: float | None = None
    walk: Literal["knots", "noncentred", "centred"] = "knots"
    seeding: Literal["first_case", "start"] = "first_case"
    future_rt: Literal["walk", "hold"] = "walk"
    bvd: bool = False
    bvd_alpha: float = 0.5
    dmax: int = 60
    publication_lag: Literal["estimated", "fixed", "none"] = "estimated"
    max_publication_lag: int = 3
    delay_counts: Literal["multinomial", "dirichlet-multinomial"] = (
        "multinomial"
    )


MAIN = ModelConfig()
BVD = ModelConfig(bvd=True)
"""Main model informed by BVDOutbreakSize Rt (`hubrenewalbvd`)."""
FALLBACK = ModelConfig(
    coupling="none",
    delay="fixed",
    missing="fixed",
    rt_cadence="weekly",
    publication_lag="fixed",
)
_SIMPLE = {
    "streams": "total",
    "truncation": False,
    "coupling": "none",
    "depletion": False,
    "inactive": "none",
}
LEARNT = ModelConfig(coupling="learnt")
"""Main model with a learnt gravity kernel."""
LADDER: dict[str, ModelConfig] = {
    "S0 national renewal": ModelConfig(national=True, **_SIMPLE),
    "S1 + provinces (pooled Rt)": ModelConfig(**_SIMPLE),
    "S2 + truncation (hub delay)": ModelConfig(
        **_SIMPLE | {"truncation": True}
    ),
    "S3 + delays, missingness, imputation": ModelConfig(
        coupling="none", depletion=False, inactive="none"
    ),
    "S4 + coupling and importation": ModelConfig(depletion=False),
    "S5 + depletion (main)": MAIN,
}
"""Configurations of the step-by-step build, in order."""


@dataclass
class Vintage:
    """One data vintage, on weeks starting `start_date` (a Sunday).

    Attributes
    ----------
    start_date : datetime.date
        First day of the first data week. Must be a Sunday.
    cutoff : int
        Data cut-off as a day index from `start_date`.
    known, imputed : numpy.ndarray
        Weekly onset counts, `(n_weeks, P)`, NaN where unobserved.
    delay_counts : numpy.ndarray
        Known-onset cases by onset week and delay, pooled over provinces,
        `(n_weeks, dmax + 1)`.
    hub_delay_pmf : numpy.ndarray
        The hub's empirical delay pmf for this vintage, `(dmax + 1,)`.
    populations : numpy.ndarray
        Province populations, `(P,)`.
    kernel : numpy.ndarray or None
        Row-normalised importation kernel between modelled provinces.
    horizon_weeks : int
        Forecast horizon in weeks.
    names : list of str
        Modelled province names.
    inactive : numpy.ndarray or None
        Weekly totals in provinces with no cases, `(n_weeks, Q)`.
    inactive_names : list of str
        Their names.
    import_kernel : numpy.ndarray or None
        Kernel rows from modelled to inactive provinces, `(P, Q)`.
    fit_mask : numpy.ndarray or None
        Boolean `(P,)`. False switches a province's likelihood off (it
        stays in the latent process and is still predicted).
    seed_days : numpy.ndarray or None
        Day (from `start_date`) of each province's first case week.
    distances, all_populations : numpy.ndarray or None
        Distances (km) and populations of modelled then idle provinces,
        for the learnt kernel.
    bvd : dict or None
        External Rt summaries: `day` (index of $s_b$ from the first data
        day), `groups` `(G, P)`, `values`, `scales` and `kinds` `(G,)`.
    """

    start_date: dt.date
    cutoff: int
    known: np.ndarray
    imputed: np.ndarray
    delay_counts: np.ndarray
    hub_delay_pmf: np.ndarray
    populations: np.ndarray
    kernel: np.ndarray | None = None
    horizon_weeks: int = 3
    names: list[str] = field(default_factory=list)
    inactive: np.ndarray | None = None
    inactive_names: list[str] = field(default_factory=list)
    import_kernel: np.ndarray | None = None
    fit_mask: np.ndarray | None = None
    bvd: dict | None = None
    distances: np.ndarray | None = None
    all_populations: np.ndarray | None = None
    seed_days: np.ndarray | None = None

    @property
    def n_weeks(self) -> int:
        """Number of data weeks."""
        return self.known.shape[0]

    @property
    def n_provinces(self) -> int:
        """Number of provinces."""
        return self.known.shape[1]


def _named_walk(
    name: str,
    sd: float,
    n_weeks: int,
    cadence: str = "weekly",
    ar: bool = False,
    centred: bool = False,
    knots: tuple[float, float] | None = None,
    future: str = "walk",
):
    """Walk frozen over the seeding period, fitted on the data weeks only.

    `n_weeks` data weeks set the fit axis; later steps are drawn in the
    forecast pass. With `ar`, the walk is AR(1) with a half-life of
    LogNormal(log 6 weeks, 0.6) (BVDOutbreakSize deviations: 42 days).
    """
    sd_rv = DistributionalVariable(name, dist.HalfNormal(sd))
    weekly = cadence == "weekly"
    frozen = GEN_INT_LENGTH // 7 if weekly else GEN_INT_LENGTH
    n_fit = frozen + n_weeks if weekly else frozen + 7 * n_weeks
    half_life = None
    if ar:
        half_life = DistributionalVariable(
            f"{name}_half_life",
            dist.LogNormal(np.log(6.0 if weekly else 42.0), 0.6),
        )
    inner = NonCentredRandomWalk(
        sd_rv,
        frozen=frozen,
        centred=centred,
        knots=knots,
        n_fit=n_fit,
        half_life_rv=half_life,
        future=future,
    )
    if cadence == "weekly":
        return WeeklyTemporalProcess(inner, start_dow=SUNDAY)
    return inner


def fit_weeks(data: Vintage, config: ModelConfig) -> int:
    """Weeks on the fit axis: the data weeks, extended to the BVD day.

    The BVD pseudo-observation sits on $s_b$, which can fall after the
    cut-off; the fit axis then runs to the end of that week (with no
    data) so the observation informs Rt there.

    Returns
    -------
    int
        Number of weeks.
    """
    if config.bvd and data.bvd is not None:
        return max(data.n_weeks, data.bvd["day"] // 7 + 1)
    return data.n_weeks


def _nb_noise(stream: str) -> NegativeBinomialNoise:
    r"""Negative binomial with $1 / \sqrt{\phi} \sim$ HalfNormal(0.25).

    As EpiNow2 (`obs_opts`): the prior shrinks towards Poisson.
    """
    inv_sqrt = DistributionalVariable(
        f"inv_sqrt_phi_{stream}", dist.HalfNormal(0.25)
    )
    return NegativeBinomialNoise(
        TransformedVariable(f"phi_{stream}", inv_sqrt, PowerTransform(-2.0))
    )


def build_model(data: Vintage, config: ModelConfig = MAIN) -> MultiSignalModel:
    """Build the province model for one vintage.

    Parameters
    ----------
    data : Vintage
        The vintage. Fixed inputs of the fallback are estimated from it.
    config : ModelConfig
        Structural choices.

    Returns
    -------
    MultiSignalModel
        Model with one observation, `onsets`.
    """
    n_prov = data.n_provinces
    daily = config.rt_cadence == "daily"
    centred = config.walk == "centred"
    knots = config.walk == "knots"
    rt_scale = 1 / np.sqrt(7) if daily else 1.0
    single = n_prov == 1
    kernel = None if config.coupling == "none" or single else data.kernel
    if kernel is not None and config.coupling == "learnt":
        kernel = LearntKernel(
            "kernel", data.distances, data.all_populations, n_prov
        )
    if kernel is None:
        importation = None
    else:
        importation = DistributionalVariable("importation", dist.Beta(1, 100))

    builder = PyrenewBuilder()
    builder.configure_latent(
        MetapopulationInfections,
        name="latent",
        gen_int_rv=DeterministicPMF("gen_int", generation_interval()),
        # The seed is the expected confirmed onsets per day at the start,
        # turned into infections per resident.
        I0_rv=TransformedVariable(
            "I0",
            DistributionalVariable(
                "initial_onsets", dist.LogNormal(0.0, 1.5).expand([n_prov])
            ),
            AffineTransform(0.0, 1 / (ALPHA0 * data.populations)),
        ),
        log_rt_time_0_rv=PooledInitialValue(
            "log_rt0", n_prov, 0.0, 0.5, 0.3, fixed_mean=knots
        ),
        baseline_rt_process=_named_walk(
            "rt_sd",
            0.2 * rt_scale,
            fit_weeks(data, config),
            config.rt_cadence,
            centred=centred,
            knots=(0.0, 0.5) if knots else None,
            future=config.future_rt,
        ),
        subpop_rt_deviation_process=_named_walk(
            "rt_dev_sd",
            0.1 * rt_scale,
            fit_weeks(data, config),
            config.rt_cadence,
            ar=config.rt_deviation == "ar1",
            future=config.future_rt,
        ),
        kernel=kernel,
        importation_rv=importation,
        deplete=config.depletion,
        seed_days=data.seed_days if config.seeding == "first_case" else None,
        rt_observation=RtPseudoObservation(alpha=config.bvd_alpha, **data.bvd)
        if config.bvd and data.bvd is not None
        else None,
        effective_population_rv=DeterministicVariable(
            "effective_population", config.effective_population
        )
        if config.effective_population is not None
        else DistributionalVariable(
            "effective_population", dist.LogNormal(np.log(0.05), 0.75)
        ),
    )

    # Only the product of ascertainment and the effective population is
    # identified (see NOTES.md), so the starting ascertainment is fixed and
    # the effective population carries the level.
    ascertainment = IndependentAscertainment(
        name="ascertainment",
        rate_rvs={"onsets": DeterministicVariable("alpha0", ALPHA0)},
        temporal_processes={
            "onsets": _named_walk("alpha_sd", 0.03, fit_weeks(data, config))
        },
    )
    builder.add_ascertainment(ascertainment)

    pad = GEN_INT_LENGTH  # The model axis starts GEN_INT_LENGTH days early.

    def fixed(name: str, daily: np.ndarray) -> FixedTimeVarying:
        return FixedTimeVarying(
            name, np.vstack([np.repeat(daily[:1], pad, 0), daily])
        )

    if config.streams == "total":
        if config.truncation:
            rows = data.hub_delay_pmf
        else:
            rows = np.eye(1, config.dmax + 1)[0]
        delay = fixed("delay", np.tile(rows, (7 * data.n_weeks, 1)))
        missing = None
    elif not config.truncation:
        raise ValueError("Split streams always use right truncation.")
    elif config.missing == "fixed":
        mean_delay = data.hub_delay_pmf @ np.arange(data.hub_delay_pmf.size)
        pi = empirical_missing_fraction(
            data.known, data.imputed, shift=round(float(mean_delay))
        )
        missing = fixed("missing_fraction", pi)
    else:
        missing = MissingOnsetFraction(
            "missing_fraction",
            PooledInitialValue("logit_pi0", n_prov, -2.0, 1.0, 0.5),
            _named_walk("pi_sd", 0.3, fit_weeks(data, config)),
            _named_walk("pi_dev_sd", 0.1, fit_weeks(data, config)),
        )

    if missing is not None:
        delay_time = _named_walk("delay_sd", 0.15, fit_weeks(data, config))
        if config.delay == "fixed":
            # Delays near the cut-off are part published, so treat the
            # last L days as truncated.
            f = empirical_delay_matrix(
                data.delay_counts, data.cutoff - config.max_publication_lag
            )
            delay = fixed("delay", f)
        elif config.delay in ("gamma", "lognormal"):
            delay = ParametricDelay(
                "delay", config.dmax, delay_time, config.delay
            )
        else:
            delay = HazardDelay("delay", config.dmax, delay_time)
    imports = None
    if config.inactive == "imports" and data.import_kernel is not None:
        imports = DistributionalVariable(
            "import_scale", dist.LogNormal(np.log(1e-3), 2.0)
        )

    n_lag = config.max_publication_lag + 1
    lag: DistributionalVariable | np.ndarray | None = None
    if config.publication_lag == "estimated":
        lag = DistributionalVariable(
            "onsets_publication_lag", dist.Dirichlet(np.ones(n_lag))
        )
    elif config.publication_lag == "fixed":
        lag = np.full(n_lag, 1 / n_lag)

    builder.add_observation(
        OnsetStreams(
            name="onsets",
            ascertainment_rate_rv=ascertainment.for_signal("onsets"),
            missing_fraction_rv=missing,
            delay_rv=delay,
            known_noise=_nb_noise("known"),
            imputed_noise=None if missing is None else _nb_noise("imputed"),
            import_scale_rv=imports,
            kernel_rv=kernel if isinstance(kernel, LearntKernel) else None,
            start_dow=SUNDAY,
            publication_lag=lag,
            delay_overdispersion_rv=DistributionalVariable(
                "delay_overdispersion", dist.HalfNormal(0.1)
            )
            if config.delay_counts == "dirichlet-multinomial"
            else None,
            # Delays walk weekly, fixed delays are weekly rows, and the
            # axis starts on a Sunday with whole weeks.
            weekly_delay=True,
        )
    )
    model = builder.build()
    model.config = config
    if model.latent.n_initialization_points != pad:
        raise ValueError(
            "The model axis must start a whole number of weeks early."
        )
    return model


def model_args(
    model: MultiSignalModel, data: Vintage, forecast: bool = False
) -> dict:
    """Arguments for `model.run` and `Predictive` for one vintage.

    The model axis runs from `n_init` days before `start_date` to the end
    of the forecast horizon. Weekly data are padded with NaN for the
    seeding weeks and the horizon.

    Returns
    -------
    dict
        Keyword arguments for `MultiSignalModel.sample`.
    """
    if data.start_date.weekday() != SUNDAY:
        raise ValueError("start_date must be a Sunday.")
    n_init = model.latent.n_initialization_points
    # The fit runs on the data weeks only; the forecast pass extends the
    # axis by the horizon and draws the new steps from their priors.
    extra = fit_weeks(data, model.config) - data.n_weeks
    horizon = max(data.horizon_weeks, extra) if forecast else extra
    n_weeks = data.n_weeks + horizon

    def pad(x):
        lead = np.full((n_init // 7, x.shape[1]), np.nan)
        tail = np.full((horizon, x.shape[1]), np.nan)
        return jnp.asarray(np.vstack([lead, x, tail]))

    obs = {
        "cutoff": n_init + data.cutoff,
        "hub_delay_pmf": jnp.asarray(data.hub_delay_pmf),
        "forecast": forecast,
    }
    use_imports = data.import_kernel is not None and (
        model.observations["onsets"].import_scale_rv is not None
    )
    if use_imports:
        obs["import_kernel"] = jnp.asarray(data.import_kernel)
    known, imputed = data.known.copy(), data.imputed.copy()
    if data.fit_mask is not None:
        known[:, ~data.fit_mask] = np.nan
        imputed[:, ~data.fit_mask] = np.nan
    if forecast:
        obs["vintage"] = jnp.stack([pad(known), pad(imputed)])
    else:
        obs |= {
            "known": pad(known),
            "imputed": pad(imputed),
            "delay_counts": jnp.nan_to_num(pad(data.delay_counts)),
        }
        if use_imports:
            obs["inactive"] = pad(data.inactive)
    pops = np.asarray(data.populations, dtype=float)
    return {
        "n_days_post_init": 7 * n_weeks,
        "population_size": float(pops.sum()),
        "subpop_fractions": jnp.asarray(pops / pops.sum()),
        "obs_start_date": data.start_date,
        "onsets": obs,
    }


EXTRA_FIELDS = (
    "diverging",
    "num_steps",
    "accept_prob",
    "energy",
    "adapt_state.step_size",
)
"""Sampler fields kept for the diagnostics."""


def fit(
    model: MultiSignalModel,
    data: Vintage,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    seed: int = 0,
    target_accept: float = 0.9,
    max_tree_depth: int = 10,
) -> MultiSignalModel:
    """Fit by NUTS and return the model with its MCMC attached.

    As `MultiSignalModel.run`, but keeping `EXTRA_FIELDS` for the
    diagnostics. Chains run in parallel when devices allow.

    Returns
    -------
    MultiSignalModel
        The same model, with `model.mcmc` holding the posterior.
    """
    args = model_args(model, data)
    model.validate_data(
        **{k: v for k, v in args.items() if k != "population_size"}
    )
    kernel = NUTS(
        model.sample,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_median(num_samples=15),
    )
    model.mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    model.mcmc.run(random.key(seed), extra_fields=EXTRA_FIELDS, **args)
    return model


def seed_from_imports(
    imports: np.ndarray,
    rt: np.ndarray,
    gen_int: np.ndarray,
    start: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Draw onsets in idle provinces: importations that may spread.

    For one posterior draw, importations are
    $m_q(t) \sim \text{Poisson}(\mu_q(t))$ from day `start` (the day
    after the cut-off; before it these provinces are observed with no
    cases). Each also seeds local transmission at the national
    reproduction number, weighted by infections,
    $\ell_q(t) \sim \text{Poisson}\big(R(t) \sum_\tau g(\tau)
    y_q(t - \tau)\big)$ with $y = m + \ell$. Susceptible depletion is
    ignored because these provinces have had no cases.

    Parameters
    ----------
    imports : numpy.ndarray
        Expected importations $\mu$, `(n, Q)`.
    rt : numpy.ndarray
        National $R(t)$, `(n,)`.
    gen_int : numpy.ndarray
        Generation interval, element 0 at lag 1.
    start : int
        First day of local spread (the day after the cut-off).
    rng : numpy.random.Generator
        Random numbers.

    Returns
    -------
    numpy.ndarray
        Onsets $y$, `(n, Q)`.
    """
    # Up to the cut-off these provinces are observed with no cases.
    y = rng.poisson(np.maximum(imports, 0.0)).astype(float)
    y[:start] = 0.0
    g = np.asarray(gen_int)
    for t in range(start, y.shape[0]):
        lags = y[max(0, t - g.size) : t][::-1]
        force = g[: lags.shape[0]] @ lags
        y[t] += rng.poisson(rt[t] * force)
    return y


def forecast_trajectories(
    model: MultiSignalModel,
    data: Vintage,
    n_trajectories: int = 100,
    seed: int = 1,
    posterior: dict | None = None,
) -> dict[str, np.ndarray]:
    """Draw joint trajectories of final weekly onsets (known plus imputed).

    Each trajectory uses one posterior draw for every province and week,
    so national counts are the sum over provinces within a trajectory.
    Provinces with no cases get `seed_from_imports` draws when the model
    has the importation component.

    Returns
    -------
    dict
        `province`: `(n_trajectories, n_weeks + horizon, P)`,
        `inactive`: `(n_trajectories, n_weeks + horizon, Q)` (Q may be 0),
        `national`: the sum over both, and `insample`, the predictive
        draws of the fitted streams at the vintage (see `ppc_draws`).
    """
    if posterior is None:
        posterior = model.mcmc.get_samples()
    n_draws = next(iter(posterior.values())).shape[0]
    rng = np.random.default_rng(seed)
    keep = rng.choice(n_draws, n_trajectories, replace=n_draws < n_trajectories)
    thinned = {k: v[keep] for k, v in posterior.items()}
    pred = Predictive(model.sample, posterior_samples=thinned)(
        random.key(seed), **model_args(model, data, forecast=True)
    )
    n_init = model.latent.n_initialization_points
    n_lead = n_init // 7
    province = np.asarray(pred["onsets_final_total"])[:, n_lead:]
    n_weeks = province.shape[1]
    inactive = np.zeros((n_trajectories, n_weeks, len(data.inactive_names)))
    if "onsets_expected_imports_daily" in pred:
        mu = np.asarray(pred["onsets_expected_imports_daily"])[:, n_init:]
        # National Rt weighted by infections (counts), not the geometric
        # mean of the province values.
        rt_p = np.asarray(pred["latent::rt_subpop"])[:, n_init:]
        w = np.asarray(pred["latent_infections_by_subpop"])[:, n_init:]
        rt = (rt_p * w).sum(-1) / np.maximum(w.sum(-1), 1e-12)
        g = np.asarray(model.latent.gen_int_rv())
        for k in range(n_trajectories):
            y = seed_from_imports(mu[k], rt[k], g, data.cutoff + 1, rng)
            inactive[k] = y[: 7 * n_weeks].reshape(n_weeks, 7, -1).sum(1)
    sites = [
        s for s in ("known", "imputed", "total") if f"onsets_{s}_obs" in pred
    ]
    insample = {
        s: np.asarray(pred[f"onsets_{s}_obs"])[:, 0, n_lead:] for s in sites
    }
    return {
        "province": province,
        "inactive": inactive,
        "national": province.sum(axis=2) + inactive.sum(axis=2),
        "insample": insample,
    }
