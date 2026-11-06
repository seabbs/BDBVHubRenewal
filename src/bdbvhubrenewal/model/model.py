# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Assemble, fit and forecast the province model with `PyrenewBuilder`."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import Array, random
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    PowerTransform,
)
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from pyrenew.ascertainment import IndependentAscertainment
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent.temporal_processes import (
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.model import MultiSignalModel, PyrenewBuilder
from pyrenew.observation import NegativeBinomialNoise
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from .censored import CensoredPMF, censored_pmf
from .composition_noise import DirichletMultinomialNoise, MultinomialNoise
from .delays import (
    FixedTimeVarying,
)
from .hazard_delay import HazardDelay
from .observation import MissingOnsetFraction, OnsetStreams
from .pooled import PooledInitialValue
from .renewal import Importation, MetapopulationInfections
from .rt_observation import RtPseudoObservation
from .temporal import NonCentredRandomWalk

SUNDAY = 6
"""Weekday number of Sunday (Monday is 0), the first day of a hub week."""
GEN_INT_LENGTH = 42
"""Lags of the generation interval, about its 98.5% quantile (a whole
number of weeks, as the model axis starts this many days early)."""
GI_MEAN = 15.3
"""Gamma generation interval mean in days. See `GI_SD`."""
GI_SD = 9.3
"""Gamma generation interval mean and SD in days, BVDOutbreakSize's prior
means (from the NEJM serial interval). After double censoring, dropping
lag 0 and truncating at 42 days the effective mean is 14.8 and SD 8.4."""
INCUBATION_MEAN = 6.3
"""Log-normal incubation period prior mean in days, Bundibugyo virus disease
in Uganda in 2007 (MacNeil et al. 2010, n = 24), as BVDOutbreakSize uses."""
INCUBATION_MEAN_SD = 0.54
"""Prior SD of the incubation period mean, BVDOutbreakSize's, which gives
MacNeil et al.'s 95% CI of 5.2 to 7.3 days."""
INCUBATION_SD = 3.5
"""Log-normal incubation period prior SD in days, BVDOutbreakSize's (MacNeil
et al. give no interval on the spread)."""
INCUBATION_SD_SD = 0.8
"""Prior SD of the incubation period SD, BVDOutbreakSize's."""
INCUBATION_LENGTH = 28
"""Longest incubation period in days, its 99.9% quantile at the prior means
rounded up."""
PHI_FLOOR = 1e-3
"""Added to $1/\\phi$ so the concentration stays at or below 1000."""
ALPHA0 = 0.6
"""Ascertainment that turns the seed's expected onsets into infections."""
FORECAST_RT_HALF_LIFE = 2.0
"""Half-life in weeks with which the pooled log Rt reverts towards 0
(Rt of 1) past the data weeks."""


def generation_interval(
    mean: float = GI_MEAN, sd: float = GI_SD, length: int = GEN_INT_LENGTH
) -> np.ndarray:
    r"""Double interval censored Gamma generation interval on lags 1..length.

    The daily pmf on $0..L$ from `censored_pmf`, truncated at $L$. The
    renewal uses lags 1 to $L$, so lag 0 (about 0.05% of the mass) is
    dropped and the rest renormalised, as BVDOutbreakSize left-truncates
    its generation interval at one day. Computed once, in float64.
    Matches `primarycensored::dpcens(1:L, ..., L = 1, D = L + 1)`.

    Parameters
    ----------
    mean, sd : float
        Gamma mean and SD in days. Default to `GI_MEAN` and `GI_SD`.
    length : int
        Longest lag in days. Defaults to `GEN_INT_LENGTH`.

    Returns
    -------
    numpy.ndarray
        Pmf on lags 1..length (PyRenew order, element 0 at lag 1).
    """
    with jax.enable_x64():
        pmf = np.asarray(censored_pmf("gamma", mean, sd / mean, length))[1:]
    return np.asarray(pmf / pmf.sum(), dtype=float)


def incubation_period(
    mean: float = INCUBATION_MEAN,
    sd: float = INCUBATION_SD,
    length: int = INCUBATION_LENGTH,
) -> np.ndarray:
    r"""Double interval censored log-normal incubation period on 0..length.

    The daily pmf from infection to onset, from `censored_pmf`, truncated
    at $L$ and renormalised. Computed once, in float64.
    Matches `primarycensored::dpcens(0:L, plnorm, D = L + 1)`.

    Parameters
    ----------
    mean, sd : float
        Log-normal mean and SD in days. Default to `INCUBATION_MEAN` and
        `INCUBATION_SD`.
    length : int
        Longest lag in days. Defaults to `INCUBATION_LENGTH`.

    Returns
    -------
    numpy.ndarray
        Pmf on lags 0..length.
    """
    with jax.enable_x64():
        pmf = np.asarray(censored_pmf("lognormal", mean, sd / mean, length))
    return np.asarray(pmf, dtype=float)


def seed_lead(incubation: np.ndarray) -> int:
    """Days to seed a late province before its first case week.

    The lead that puts the most onsets in the first case week from a week
    of constant seeding: the argmax over $L$ of the mass of the week-long
    seed convolved with the incubation pmf on days $L$ to $L + 6$ after
    the seed starts. At the prior mean incubation period it is 5 days,
    with 67% of onsets in the week and 9% before it.

    Parameters
    ----------
    incubation : numpy.ndarray
        Incubation pmf on lags 0..H.

    Returns
    -------
    int
        The lead in days, at most 6 so the seed starts after the
        previous case week began.
    """
    onsets = np.convolve(np.full(7, 1 / 7), incubation)
    in_week = [onsets[lead : lead + 7].sum() for lead in range(7)]
    return int(np.argmax(in_week))


PRIOR_PUBLICATION_LAG: Final[tuple[float, ...]] = tuple(
    [0.85 / 4] * 4 + [0.15 / 57] * 57
)
"""Report to publication lag pmf on 0..60 days where no pair of delay files
gives an estimate: 85% within 3 days and 15% spread over 4 to 60 days.
It is a prior shaped by the later files (29% of rows new between 10-09
and 10-17 were more than 6 days late), so for the 2026-10-09 backtest it
uses a little later information."""


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
    coupling : {"gravity", "none"}
        Importation between modelled provinces through the fixed gravity
        kernel, or none.
    depletion : bool
        Susceptible depletion in the renewal.
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
    bvd : bool
        Add BVDOutbreakSize Rt as a tempered pseudo-observation
        (`RtPseudoObservation`), when the vintage carries it.
    bvd_alpha : float
        Its tempering. 0 gives the model without it, exactly.
    dmax : int
        Maximum onset-to-report delay in days.
    publication_lag : {"empirical", "estimated", "fixed", "none"}
        Lag from report to publication in a vintage: the empirical pmf
        on $0..180$ days from rows new in earlier delay files
        (`Vintage.publication_lag_pmf`), a Dirichlet simplex on $0..L$,
        uniform on $0..L$ (the default), or none. Empirical falls back to
        `PRIOR_PUBLICATION_LAG` for a vintage with no earlier pair of
        delay files. It is not the default: late rows in the delay files
        are mostly onsets recorded later for cases already counted as
        imputed, so it overstates late reporting of the total.
    max_publication_lag : int
        $L$, in days (the model default).
    delay_counts : {"multinomial", "dirichlet-multinomial"}
        Likelihood of the delay file's counts given the weekly totals.
        The Dirichlet-multinomial (default) has $1/\sqrt{\psi} \sim
        \text{HalfNormal}(0.1)$, with $\psi$ capped at $10^4$, so that
        thousands of pooled delays do not make the delay overconfident.
    """

    national: bool = False
    streams: Literal["split", "total"] = "split"
    truncation: bool = True
    coupling: Literal["gravity", "none"] = "gravity"
    depletion: bool = True
    active: Literal["cases", "all"] = "cases"
    inactive: Literal["imports", "none"] = "imports"
    report_inactive: bool = True
    effective_population: float | None = None
    bvd: bool = False
    bvd_alpha: float = 0.5
    dmax: int = 60
    publication_lag: Literal["empirical", "estimated", "fixed", "none"] = (
        "fixed"
    )
    max_publication_lag: int = 3
    delay_counts: Literal["multinomial", "dirichlet-multinomial"] = (
        "dirichlet-multinomial"
    )


MAIN = ModelConfig()
"""The main model configuration, with every default of `ModelConfig`."""
BVD = ModelConfig(bvd=True)
"""Main model informed by BVDOutbreakSize Rt (`hubrenewalbvd`)."""
_SIMPLE: dict[str, Any] = {
    "streams": "total",
    "truncation": False,
    "coupling": "none",
    "depletion": False,
    "inactive": "none",
}
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
        The hub's imputation delay pmf for this release, `(dmax + 1,)`.
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
        Day (from `start_date`) each province's seeding starts: 0 for
        provinces with cases in the first data week, else `seed_lead` days
        before the first case week.
    bvd : dict or None
        External Rt summaries: `day` (index of $s_b$ from the first data
        day), `groups` `(G, P)`, `values`, `scales` and `kinds` `(G,)`.
    publication_lag_pmf : numpy.ndarray or None
        Empirical report to publication lag pmf on $0..180$ days
        (`data.releases.publication_lag_pmf`), or None.
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
    bvd: dict[str, Any] | None = None
    seed_days: np.ndarray | None = None
    publication_lag_pmf: np.ndarray | None = None

    @property
    def n_weeks(self) -> int:
        """Number of data weeks."""
        return int(self.known.shape[0])

    @property
    def n_provinces(self) -> int:
        """Number of provinces."""
        return int(self.known.shape[1])


def _pooled(
    name: str,
    n_groups: int,
    loc: float,
    scale: float,
    pooling_sd: float,
    centred: bool = True,
) -> PooledInitialValue:
    r"""Group values around $\mu \sim N(m, s)$, spread $N^+(0, \tau)$."""
    return PooledInitialValue(
        name,
        n_groups,
        DistributionalVariable(f"{name}_mean", dist.Normal(loc, scale)),
        DistributionalVariable(f"{name}_sd", dist.HalfNormal(pooling_sd)),
        centred=centred,
    )


def _named_walk(
    name: str,
    sd: float,
    n_weeks: int,
    ar: bool = False,
    forecast_half_life: float | None = None,
) -> TemporalProcess:
    """Weekly walk frozen over the seeding period, fitted on the data weeks.

    `n_weeks` data weeks set the fit axis; later steps are drawn in the
    forecast pass. With `ar`, the walk is AR(1) with a half-life of
    LogNormal(log 6 weeks, 0.6) (BVDOutbreakSize deviations: 42 days).
    With `forecast_half_life`, the forecast steps revert towards 0
    (`NonCentredRandomWalk`).
    """
    sd_rv = DistributionalVariable(name, dist.HalfNormal(sd))
    frozen = GEN_INT_LENGTH // 7
    half_life = None
    if ar:
        half_life = DistributionalVariable(
            f"{name}_half_life", dist.LogNormal(np.log(6.0), 0.6)
        )
    inner = NonCentredRandomWalk(
        sd_rv,
        frozen=frozen,
        n_fit=frozen + n_weeks,
        half_life_rv=half_life,
        forecast_half_life=forecast_half_life,
    )
    return WeeklyTemporalProcess(inner, start_dow=SUNDAY)


def fit_weeks(data: Vintage, config: ModelConfig) -> int:
    """Weeks on the fit axis: the data weeks, extended to the BVD day.

    The BVD pseudo-observation sits on $s_b$, which can fall after the
    cut-off; the fit axis then runs to the end of that week (with no
    data) so the observation informs Rt there.

    Parameters
    ----------
    data : Vintage
        The model inputs at one vintage.
    config : ModelConfig
        The model configuration.

    Returns
    -------
    int
        Number of weeks.
    """
    if config.bvd and data.bvd is not None:
        return max(data.n_weeks, int(data.bvd["day"]) // 7 + 1)
    return data.n_weeks


def capped_concentration(
    name: str, inv_sqrt_rv: DistributionalVariable, floor: float
) -> TransformedVariable:
    r"""Concentration $1 / (x^2 + c)$ from $x = 1 / \sqrt{\text{conc}}$.

    The prior on $x$ shrinks towards the Poisson or multinomial limit, as
    EpiNow2's `obs_opts`, and the floor $c$ caps the concentration at
    $1 / c$: as the concentration grows the float32 log density loses
    precision, which stalls NUTS.

    Parameters
    ----------
    name : str
        Name of the concentration.
    inv_sqrt_rv : DistributionalVariable
        $x$.
    floor : float
        $c$.

    Returns
    -------
    TransformedVariable
        The concentration.
    """
    capped = ComposeTransform(
        [PowerTransform(2.0), AffineTransform(floor, 1.0), PowerTransform(-1.0)]
    )
    return TransformedVariable(name, inv_sqrt_rv, capped)


def _nb_noise(stream: str) -> NegativeBinomialNoise:
    r"""Negative binomial with $1 / \sqrt{\phi} \sim$ HalfNormal(0.25).

    $\phi \le 1000$: the float32 negative binomial log density loses
    precision as $\phi \to \infty$ (second-difference noise 1.5 at
    $x = 3 \times 10^{-4}$ on real data), and at $\phi = 1000$ it is
    Poisson for these counts.
    """
    return NegativeBinomialNoise(
        capped_concentration(
            f"phi_{stream}",
            DistributionalVariable(
                f"inv_sqrt_phi_{stream}", dist.HalfNormal(0.25)
            ),
            PHI_FLOOR,
        )
    )


def _dm_noise() -> DirichletMultinomialNoise:
    r"""Dirichlet-multinomial with $1 / \sqrt{\psi} \sim$ HalfNormal(0.1).

    $\psi \le 10^4$, so that the pooled delays of thousands of cases do
    not make the delay overconfident.
    """
    return DirichletMultinomialNoise(
        capped_concentration(
            "delay_concentration",
            DistributionalVariable(
                "delay_overdispersion", dist.HalfNormal(0.1)
            ),
            1e-4,
        )
    )


def build_model(data: Vintage, config: ModelConfig = MAIN) -> MultiSignalModel:
    """Build the province model for one vintage.

    Parameters
    ----------
    data : Vintage
        The vintage.
    config : ModelConfig
        Structural choices.

    Returns
    -------
    MultiSignalModel
        Model with one observation, `onsets`.
    """
    n_prov = data.n_provinces
    mixing = None
    if config.coupling == "gravity" and data.kernel is not None:
        mixing = Importation(
            "mixing",
            DeterministicVariable("kernel", data.kernel),
            DistributionalVariable("importation", dist.Beta(1, 100)),
        )

    effective_population = None
    if config.depletion:
        effective_population = (
            DistributionalVariable(
                "effective_population", dist.LogNormal(np.log(0.05), 0.75)
            )
            if config.effective_population is None
            else DeterministicVariable(
                "effective_population", config.effective_population
            )
        )

    builder = PyrenewBuilder()
    builder.configure_latent(
        MetapopulationInfections,
        name="latent",
        gen_int_rv=DeterministicPMF("gen_int", generation_interval()),
        # The seed is the infections per day that will be confirmed, on
        # the first data day, turned into infections per resident.
        I0_rv=TransformedVariable(
            "I0",
            DistributionalVariable(
                "initial_infections",
                dist.LogNormal(0.0, 1.5).expand([n_prov]),
            ),
            AffineTransform(0.0, 1 / (ALPHA0 * data.populations)),
        ),
        log_rt_time_0_rv=_pooled("log_rt0", n_prov, 0.0, 0.5, 0.3),
        baseline_rt_process=_named_walk(
            "rt_sd",
            0.2,
            fit_weeks(data, config),
            forecast_half_life=FORECAST_RT_HALF_LIFE,
        ),
        subpop_rt_deviation_process=_named_walk(
            "rt_dev_sd", 0.1, fit_weeks(data, config), ar=True
        ),
        mixing_rv=mixing,
        effective_population_rv=effective_population,
        seed_days=data.seed_days,
        rt_observation=RtPseudoObservation(
            "bvd", alpha=config.bvd_alpha, **data.bvd
        )
        if config.bvd and data.bvd is not None
        else None,
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
    else:
        missing = MissingOnsetFraction(
            "missing_fraction",
            # Non-centred: the imputed share pins these offsets only
            # weakly, and centred they form a funnel (R-hat 1.68).
            _pooled("logit_pi0", n_prov, -2.0, 1.0, 0.5, centred=False),
            _named_walk("pi_sd", 0.3, fit_weeks(data, config)),
            _named_walk("pi_dev_sd", 0.1, fit_weeks(data, config)),
        )

    if missing is not None:
        delay = HazardDelay(
            "delay",
            config.dmax,
            DistributionalVariable("delay_hazard0", dist.Normal(-2.0, 1.0)),
            # Real fits put the step SD at about 0.2 (2026-10-24: 90% CrI
            # 0.16 to 0.25), in the tail of a scale of 0.1, hence 0.4.
            NonCentredRandomWalk(
                DistributionalVariable("delay_smooth_sd", dist.HalfNormal(0.4))
            ),
            _named_walk("delay_sd", 0.15, fit_weeks(data, config)),
        )
    imports = None
    if config.inactive == "imports" and data.import_kernel is not None:
        imports = DistributionalVariable(
            "import_scale", dist.LogNormal(np.log(1e-3), 2.0)
        )

    n_lag = config.max_publication_lag + 1
    lag: DistributionalVariable | np.ndarray = np.ones(1)
    if config.publication_lag == "estimated":
        lag = DistributionalVariable(
            "onsets_publication_lag", dist.Dirichlet(np.ones(n_lag))
        )
    elif config.publication_lag == "fixed":
        lag = np.full(n_lag, 1 / n_lag)
    elif config.publication_lag == "empirical":
        lag = np.asarray(
            PRIOR_PUBLICATION_LAG
            if data.publication_lag_pmf is None
            else data.publication_lag_pmf,
            dtype=float,
        )

    builder.add_observation(
        OnsetStreams(
            name="onsets",
            temporal_pmf_rv=CensoredPMF(
                "infection_to_onset",
                "lognormal",
                # BVDOutbreakSize's priors (onset_incidence_model).
                DistributionalVariable(
                    "incubation_mean",
                    dist.TruncatedNormal(
                        INCUBATION_MEAN, INCUBATION_MEAN_SD, low=1.0
                    ),
                ),
                DistributionalVariable(
                    "incubation_sd",
                    dist.TruncatedNormal(
                        INCUBATION_SD, INCUBATION_SD_SD, low=1.0
                    ),
                ),
                INCUBATION_LENGTH,
            ),
            ascertainment_rate_rv=ascertainment.for_signal("onsets"),
            missing_fraction_rv=missing,
            delay_rv=delay,
            known_noise=_nb_noise("known"),
            imputed_noise=None if missing is None else _nb_noise("imputed"),
            import_scale_rv=imports,
            start_dow=SUNDAY,
            publication_lag=lag,
            delay_noise=_dm_noise()
            if config.delay_counts == "dirichlet-multinomial"
            else MultinomialNoise(),
            # The axis starts on a Sunday with whole weeks.
            weekly_axis=True,
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
    model: MultiSignalModel,
    data: Vintage,
    forecast: bool = False,
    as_of: int | None = None,
) -> dict[str, Any]:
    """Arguments for `model.run` and `Predictive` for one vintage.

    The model axis runs from `n_init` days before `start_date` to the end
    of the forecast horizon. Weekly data are padded with NaN for the
    seeding weeks and the horizon.

    Parameters
    ----------
    model : MultiSignalModel
        The model, from `build_model`.
    data : Vintage
        The model inputs at one vintage.
    forecast : bool, default False
        Extend the axis by the horizon and pass the data as the vintage,
        not as observations, to draw forecasts.
    as_of : int, optional
        With `forecast`, a later cut-off in days from `start_date`, to
        draw counts as reported by it.

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

    def pad(x: np.ndarray) -> Array:
        lead = np.full((n_init // 7, x.shape[1]), np.nan)
        tail = np.full((horizon, x.shape[1]), np.nan)
        return jnp.asarray(np.vstack([lead, x, tail]))

    obs: dict[str, Any] = {
        "cutoff": n_init + data.cutoff,
        "hub_delay_pmf": jnp.asarray(data.hub_delay_pmf),
        "forecast": forecast,
        **({"as_of": n_init + as_of} if forecast and as_of is not None else {}),
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
        if use_imports and data.inactive is not None:
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

    Parameters
    ----------
    model : MultiSignalModel
        The model, from `build_model`.
    data : Vintage
        The model inputs at one vintage.
    num_warmup, num_samples : int, default 500
        Warm-up and kept iterations per chain.
    num_chains : int, default 1
        Number of chains.
    seed : int, default 0
        Random seed.
    target_accept : float, default 0.9
        NUTS target acceptance probability.
    max_tree_depth : int, default 10
        NUTS maximum tree depth.

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
    $R(t)$ is dated by infection and $y$ by onset, so local spread here
    runs about an incubation period early. Their counts are near zero.

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
    return np.asarray(y, dtype=float)


def forecast_trajectories(
    model: MultiSignalModel,
    data: Vintage,
    n_trajectories: int = 100,
    seed: int = 1,
    posterior: dict[str, Any] | None = None,
    as_of: Sequence[int] = (),
) -> dict[str, Any]:
    """Draw joint trajectories of final weekly onsets (known plus imputed).

    Each trajectory uses one posterior draw for every province and week,
    so national counts are the sum over provinces within a trajectory.
    Provinces with no cases get `seed_from_imports` draws when the model
    has the importation component.
    Those draws use the national Rt, dated by infection, on an onset-dated
    import series.

    Parameters
    ----------
    model : MultiSignalModel
        A fitted model, from `fit`.
    data : Vintage
        The model inputs it was fitted to.
    n_trajectories : int, default 100
        Number of trajectories.
    seed : int, default 1
        Random seed.
    posterior : dict, optional
        Posterior draws to use instead of `model.mcmc`.
    as_of : sequence of int
        Later cut-offs, in days from `start_date`, to draw counts as
        reported by.

    Returns
    -------
    dict
        `province`: `(n_trajectories, n_weeks + horizon, P)`,
        `inactive`: `(n_trajectories, n_weeks + horizon, Q)` (Q may be 0),
        `national`: the sum over both, `insample`, the predictive
        draws of the fitted streams at the vintage (see `ppc_draws`),
        `draw`, the index of the posterior draw behind each trajectory,
        and `as_reported`: for each later cut-off in `as_of` (days from
        `start_date`), the counts as reported by it, with the same
        posterior draws, as `{"province", "national"}`. These score
        forecasts against a later vintage as reported, like for like.
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
    as_reported = {}
    for cut in as_of:
        later = Predictive(model.sample, posterior_samples=thinned)(
            random.key(seed + cut),
            **model_args(model, data, forecast=True, as_of=cut),
        )
        prov = np.asarray(later["onsets_final_total"])[:, n_lead:]
        # Provinces with no cases: their draws up to the later cut-off.
        idle = inactive.copy()
        first = (cut // 7) + 1
        idle[:, first:] = 0.0
        as_reported[cut] = {
            "province": prov,
            "inactive": idle,
            "national": prov.sum(axis=2) + idle.sum(axis=2),
        }
    return {
        "province": province,
        "inactive": inactive,
        "national": province.sum(axis=2) + inactive.sum(axis=2),
        "insample": insample,
        "draw": keep,
        "as_reported": as_reported,
    }
