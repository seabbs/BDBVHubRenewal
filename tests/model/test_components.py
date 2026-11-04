from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent.temporal_processes import WeeklyTemporalProcess
from pyrenew.randomvariable import DistributionalVariable

from bdbvhubrenewal.model import (
    ConstantProcess,
    FixedTimeVarying,
    HazardDelay,
    MissingOnsetFraction,
    NonCentredRandomWalk,
    PooledInitialValue,
    censored_pmf,
    coupled_renewal,
    generation_interval,
    hazard_to_pmf,
    incubation_period,
    mixing_matrix,
    onset_expectations,
    pooled_walk,
)
from bdbvhubrenewal.model.observation import backsample

from .numpy_reference import (
    expected_imputed_onsets,
    expected_known_onsets,
    expected_missing_by_report,
    renewal,
)


def _hazard(name, time_process):
    """HazardDelay on delays 0..30 with the model's priors."""
    return HazardDelay(
        name,
        30,
        DistributionalVariable(f"{name}_hazard0", dist.Normal(-2.0, 1.0)),
        NonCentredRandomWalk(
            DistributionalVariable(f"{name}_smooth_sd", dist.HalfNormal(0.4))
        ),
        time_process,
    )


def uniform_kernel(n):
    """Row-normalised kernel with equal flows and a zero diagonal."""
    return (np.ones((n, n)) - np.eye(n)) / (n - 1)


def seeded(fn, seed=0):
    with numpyro.handlers.seed(rng_seed=seed):
        return fn()


def test_hazard_to_pmf_is_a_pmf_and_geometric_for_constant_hazard():
    pmf = np.asarray(hazard_to_pmf(jnp.full((3, 20), jnp.log(0.2 / 0.8))))
    assert pmf.shape == (3, 21)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(pmf[0, :5], 0.2 * 0.8 ** np.arange(5), rtol=1e-5)


def test_backsample_matches_reference():
    rng = np.random.default_rng(1)
    m, g = rng.gamma(2, 5, 50), rng.dirichlet(np.ones(15))
    np.testing.assert_allclose(
        backsample(jnp.asarray(m), jnp.asarray(g)),
        expected_imputed_onsets(m, g),
        rtol=1e-5,
    )


def test_onset_expectations_match_reference():
    rng = np.random.default_rng(2)
    n, dmax, cutoff = 70, 20, 60
    onsets = rng.gamma(5, 4, (n, 1))
    delay = rng.dirichlet(np.ones(dmax + 1), n)
    pi = rng.uniform(0.05, 0.3, (n, 1))
    hub = rng.dirichlet(np.ones(dmax + 1))
    out = onset_expectations(
        jnp.asarray(onsets), jnp.asarray(delay), jnp.asarray(pi), hub, cutoff
    )
    known = expected_known_onsets(onsets[:, 0], delay, pi[:, 0], cutoff)
    missing = expected_missing_by_report(onsets[:, 0], delay, pi[:, 0])
    imputed = expected_imputed_onsets(missing[: cutoff + 1], hub)
    np.testing.assert_allclose(out["known"][:, 0], known, rtol=1e-5)
    np.testing.assert_allclose(
        out["imputed"][: cutoff + 1, 0], imputed, rtol=1e-5
    )
    assert np.all(np.asarray(out["imputed"][cutoff + 1 :]) == 0)
    # Eventually every case is either known or imputed somewhere.
    total = out["known_final"].sum() + out["imputed_final"].sum()
    assert float(total) <= onsets.sum() + 1e-6
    known = np.asarray(out["known"])
    # Equal up to float32 round-off where everything is reported.
    assert np.all(
        np.asarray(out["known_final"]) >= known - 1e-5 * np.abs(known) - 1e-6
    )


def test_mixing_conserves_transmission():
    f = jnp.array([0.5, 0.3, 0.2])
    k = uniform_kernel(3)
    w = mixing_matrix(jnp.asarray(k), 0.1, f)
    # In counts, column q of diag(f) W diag(1/f) sums to one.
    counts = f[:, None] * w / f[None, :]
    np.testing.assert_allclose(counts.sum(axis=0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        mixing_matrix(jnp.asarray(k), 0.0, f), jnp.eye(3)
    )


@pytest.mark.parametrize("deplete", [False, True])
def test_coupled_renewal_matches_reference(deplete):
    rng = np.random.default_rng(3)
    g = generation_interval()
    n_init, n, p = len(g), 120, 3
    pops = np.array([2e5, 1e5, 5e4]) if deplete else np.array([2e4, 1e4, 5e3])
    pops_ref = np.full(p, 1e30) if not deplete else pops
    f = pops / pops.sum()
    k = uniform_kernel(p)
    rt = rng.uniform(1.2, 1.8, (n, p))
    seed = np.zeros((p, n_init + n))
    seed[:, :n_init] = rng.uniform(0, 2, (p, n_init))
    ref, _ = renewal(
        np.hstack([np.zeros((p, n_init)), rt.T]),
        np.concatenate([[0.0], g]),
        seed,
        pops_ref,
        kernel=k,
        importation=0.05,
    )
    x, _, _ = coupled_renewal(
        jnp.asarray(seed[:, :n_init].T / pops),
        jnp.asarray(rt),
        jnp.asarray(g),
        mixing_matrix(jnp.asarray(k), 0.05, jnp.asarray(f)),
        pool=1.0 if deplete else None,
    )
    # Exact without depletion. With it, PyRenew's exponential depletion
    # matches the reference's linear form to first order (small y).
    rtol = 1e-4 if not deplete else 2e-2
    np.testing.assert_allclose(
        np.asarray(x) * pops, ref[:, n_init:].T, rtol=rtol, atol=1e-6
    )


def test_depletion_bounds_attack_rate():
    g = generation_interval()
    I0 = jnp.full((len(g), 2), 1e-4)
    rt = jnp.full((400, 2), 3.0)
    w = jnp.eye(2)
    x, r_eff, _ = coupled_renewal(I0, rt, g, w, pool=1.0)
    assert float(x.sum(axis=0).max() + I0.sum(axis=0).max()) <= 1.0 + 1e-6
    assert float(r_eff[-1].max()) < 1.0
    x_free, _, _ = coupled_renewal(I0, rt, g, w)
    assert float(x_free.sum()) > 1e3


def test_random_walk_and_pooled_walk_shapes():
    rw = NonCentredRandomWalk(DeterministicVariable("sd", 0.1))
    out = seeded(
        lambda: rw.sample(10, initial_value=jnp.ones(3), n_processes=3)
    )
    assert out.shape == (10, 3)
    np.testing.assert_allclose(out[0], 1.0)
    weekly = WeeklyTemporalProcess(rw, start_dow=6)
    base, groups = seeded(
        lambda: pooled_walk(
            weekly, weekly, jnp.array([0.1, 0.2, 0.3]), 30, 3, "x", 6
        )
    )
    assert base.shape == (30, 1) and groups.shape == (30, 3)
    np.testing.assert_allclose(groups.mean(axis=1), base[:, 0], atol=1e-6)
    np.testing.assert_allclose(groups[0], [0.1, 0.2, 0.3], atol=1e-6)


def test_forecast_reverts_towards_zero():
    def walk(half_life):
        rw = NonCentredRandomWalk(
            DeterministicVariable("sd", 1e-6),
            n_fit=4,
            forecast_half_life=half_life,
        )
        z = jnp.array([[1.0e6], [1.0e6], [1.0e6]])
        with numpyro.handlers.condition(data={"rw_z": z}):
            return seeded(lambda: rw.sample(40, initial_value=0.0))

    free = walk(None)
    np.testing.assert_allclose(free[:4, 0], [0.0, 1.0, 2.0, 3.0], atol=1e-4)
    np.testing.assert_allclose(free[4:, 0], 3.0, atol=1e-3)
    damped = walk(1.0)
    np.testing.assert_allclose(damped[:4, 0], free[:4, 0], atol=1e-6)
    np.testing.assert_allclose(damped[4, 0], 1.5, atol=1e-3)
    np.testing.assert_allclose(damped[-1, 0], 0.0, atol=1e-3)


def test_missing_fraction_and_hazard_delay_shapes():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1)), start_dow=6
    )
    pi = MissingOnsetFraction(
        "pi",
        PooledInitialValue(
            "pi0",
            4,
            DistributionalVariable("pi0_mean", dist.Normal(-2.0, 0.5)),
            DistributionalVariable("pi0_sd", dist.HalfNormal(0.3)),
        ),
        weekly,
        weekly,
    )
    out = seeded(lambda: pi(n_timepoints=35, n_groups=4, first_day_dow=6))
    assert out.shape == (35, 4) and bool(jnp.all((out > 0) & (out < 1)))
    delay = _hazard("delay", weekly)
    pmf = seeded(lambda: delay(n_timepoints=35, first_day_dow=6))
    assert pmf.shape == (35, 31)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, rtol=1e-5)


def test_hazard_delay_records_its_shape_and_shift():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1)), start_dow=6
    )
    delay = _hazard("delay", weekly)
    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
        pmf = delay(n_timepoints=35, first_day_dow=6)
    beta = tr["delay_beta"]["value"]
    shift = tr["delay_shift"]["value"]
    assert beta.shape == (30,)
    assert shift.shape[0] == 35
    # The pmf is the hazard pmf of shape plus shift on each day.
    np.testing.assert_allclose(
        pmf[10], hazard_to_pmf(beta + shift[10].squeeze()), rtol=1e-5
    )


def test_near_constant_when_innovation_sd_is_small():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 1e-6)), start_dow=6
    )
    pmf = seeded(lambda: _hazard("d", weekly)(n_timepoints=70, first_day_dow=6))
    np.testing.assert_allclose(pmf[0], pmf[-1], atol=1e-5)


def test_constant_process_fixes_the_delay_over_onset_time():
    weekly = WeeklyTemporalProcess(ConstantProcess(), start_dow=6)
    pmf = seeded(
        lambda: _hazard("d", weekly)(n_timepoints=70, first_day_dow=6)
    )
    np.testing.assert_array_equal(pmf, jnp.broadcast_to(pmf[0], pmf.shape))


def test_fixed_inputs_pad_with_last_row():
    fixed = FixedTimeVarying("x", np.arange(6.0).reshape(3, 2))
    out = fixed(n_timepoints=5)
    np.testing.assert_allclose(out[-1], [4.0, 5.0])
    assert out.shape == (5, 2)


def test_fast_kernels_match_reference_values_and_gradients():
    import jax

    from .reference_kernels import (
        hazard_to_pmf_reference,
        onset_expectations_reference,
    )

    rng = np.random.default_rng(5)
    n, dmax, p, cutoff = 50, 12, 3, 40
    args = (
        jnp.asarray(rng.uniform(0.1, 5, (n, p))),
        jnp.asarray(rng.dirichlet(np.ones(dmax + 1), n)),
        jnp.asarray(rng.uniform(0.05, 0.3, (n, p))),
        jnp.asarray(rng.dirichlet(np.ones(dmax + 1))),
    )
    new = onset_expectations(*args, cutoff)
    old = onset_expectations_reference(*args, cutoff)
    for key in new:
        np.testing.assert_allclose(new[key], old[key], rtol=1e-4, atol=1e-5)

    def loss(fn):
        return lambda *a: sum(jnp.sum(v) for v in fn(*a, cutoff).values())

    g_new = jax.grad(loss(onset_expectations), argnums=(0, 1, 2))(*args)
    g_old = jax.grad(loss(onset_expectations_reference), argnums=(0, 1, 2))(
        *args
    )
    for a, b in zip(g_new, g_old, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    x = jnp.asarray(rng.normal(-2, 1, (4, 20)))
    np.testing.assert_allclose(
        hazard_to_pmf(x), hazard_to_pmf_reference(x), rtol=1e-5, atol=1e-7
    )


def test_province_centres_cover_all_provinces_and_match_reference():
    from bdbvhubrenewal.data import PROVINCE_CENTRES
    from bdbvhubrenewal.model.inputs import province_centres

    c = province_centres().set_index("location")
    assert len(c) == 26 and not c.isna().any().any()
    for loc, (lat, lon) in PROVINCE_CENTRES.items():
        assert c.loc[loc, "latitude"] == pytest.approx(lat, abs=1e-3)
        assert c.loc[loc, "longitude"] == pytest.approx(lon, abs=1e-3)


DPCENS = pd.read_csv(Path(__file__).parent / "fixtures" / "dpcens.csv")
"""primarycensored `dpcens` pmfs, from `fixtures/dpcens.R`."""


def test_generation_interval_matches_primarycensored():
    ref = DPCENS[DPCENS["case"] == "gi"]
    np.testing.assert_allclose(generation_interval(), ref["p"], atol=1e-8)


def test_incubation_period_matches_primarycensored():
    ref = DPCENS[DPCENS["case"] == "incubation"]
    np.testing.assert_allclose(incubation_period(), ref["p"], atol=1e-8)


@pytest.mark.parametrize(
    ("family", "mean", "cv"),
    list(
        DPCENS.loc[DPCENS["case"] == "delay", ["family", "mean", "cv"]]
        .drop_duplicates()
        .itertuples(index=False)
    ),
)
def test_censored_pmf_matches_primarycensored(family, mean, cv):
    ref = DPCENS[
        (DPCENS["case"] == "delay")
        & (DPCENS["family"] == family)
        & (DPCENS["mean"] == mean)
        & (DPCENS["cv"] == cv)
    ]
    with jax.enable_x64():
        ours = np.asarray(censored_pmf(family, mean, cv, 60))
    np.testing.assert_allclose(ours, ref["p"], atol=1e-8)
    # The model runs in float32, good to the docstring's 3e-5.
    with jax.enable_x64(False):
        ours32 = np.asarray(censored_pmf(family, mean, cv, 60))
    np.testing.assert_allclose(ours32, ref["p"], atol=3e-5)


def test_onsets_are_infections_convolved_with_the_incubation_period():
    from bdbvhubrenewal.model.observation import OnsetStreams

    eta = incubation_period()
    streams = OnsetStreams(
        "onsets",
        DeterministicPMF("incubation", eta),
        DeterministicVariable("alpha", 1.0),
        None,
        FixedTimeVarying("delay", np.ones((1, 1))),
        None,
    )
    x = np.random.default_rng(3).gamma(2.0, 1.0, (80, 2))
    expected = np.stack([np.convolve(c, eta)[:80] for c in x.T], axis=1)
    np.testing.assert_allclose(
        streams._predicted_obs(jnp.asarray(x)), expected, rtol=1e-5
    )
    assert streams.lookback_days() == len(eta) - 1


def test_idle_provinces_are_zero_up_to_the_cutoff():
    from bdbvhubrenewal.model.model import seed_from_imports

    rng = np.random.default_rng(0)
    y = seed_from_imports(
        np.full((60, 2), 0.5), np.full(60, 1.5), generation_interval(), 40, rng
    )
    assert np.all(y[:40] == 0)
    assert y[40:].sum() > 0


class _Daily:
    """A weekly process seen as a daily one, to force the daily path."""

    def __init__(self, inner):
        self.inner = inner

    def sample(self, **kwargs):
        return self.inner.sample(**kwargs)


@pytest.mark.parametrize("dow", [6, 2])
def test_hazard_delay_on_weekly_rows_matches_daily(dow):
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.3)), start_dow=6
    )

    def run(process):
        delay = _hazard("delay", process)
        return seeded(lambda: delay(n_timepoints=40, first_day_dow=dow), 4)

    np.testing.assert_allclose(run(weekly), run(_Daily(weekly)), rtol=1e-6)


def test_hazard_delay_smoothness_prior_allows_real_fits():
    # Real fits put the step SD of the delay shape at about 0.2
    # (2026-10-24: 90% CrI 0.16 to 0.25), so the prior is HalfNormal(0.4).
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1)), start_dow=6
    )
    delay = _hazard("delay", weekly)
    with numpyro.handlers.trace() as tr:
        seeded(lambda: delay(n_timepoints=14, first_day_dow=6))
    assert float(tr["delay_smooth_sd"]["fn"].scale) == pytest.approx(0.4)


def test_generation_interval_matches_bvdoutbreaksize():
    # BVDOutbreakSize's prior means (priors.jl): mean 15.3, SD 9.3 days.
    from bdbvhubrenewal.data.simulate import SimulationSettings
    from bdbvhubrenewal.model.model import GI_MEAN, GI_SD, generation_interval

    assert (GI_MEAN, GI_SD) == (15.3, 9.3)
    np.testing.assert_allclose(
        generation_interval(), generation_interval(15.3, 9.3)
    )
    s = SimulationSettings()
    assert (s.gi_mean, s.gi_sd) == (GI_MEAN, GI_SD)


def test_imported_infections_are_part_of_the_total():
    rng = np.random.default_rng(4)
    g = generation_interval()
    p = 3
    f = np.array([0.5, 0.3, 0.2])
    k = uniform_kernel(p)
    w = np.asarray(mixing_matrix(jnp.asarray(k), 0.05, jnp.asarray(f)))
    I0 = rng.uniform(0, 1e-5, (len(g), p))
    rt = rng.uniform(0.6, 1.8, (60, p))
    # Imported infections are part of the total, and none without coupling.
    x, _, imp = coupled_renewal(
        jnp.asarray(I0), jnp.asarray(rt), jnp.asarray(g), jnp.asarray(w),
        pool=0.05,
    )  # fmt: skip
    assert np.all(np.asarray(imp) <= np.asarray(x) + 1e-12)
    assert float(np.asarray(imp).sum()) > 0
    _, _, none = coupled_renewal(
        jnp.asarray(I0), jnp.asarray(rt), jnp.asarray(g), jnp.eye(p),
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(none), 0.0)


def _quad_censored_pmf(family, mean, cv, dmax):
    """Float64 double censored pmf by numerical integration, as a reference."""
    from scipy import integrate, stats

    if family == "gamma":
        shape = 1 / cv**2
        dist = stats.gamma(shape, scale=mean / shape)
    else:
        sdlog = np.sqrt(np.log1p(cv**2))
        dist = stats.lognorm(sdlog, scale=mean / np.sqrt(1 + cv**2))

    def cdf(x):
        return dist.cdf(x) if x > 0 else 0.0

    pmf = np.array(
        [
            integrate.quad(
                lambda u, d=d: cdf(d + 1 - u) - cdf(d - u),
                0,
                1,
                epsabs=1e-14,
                epsrel=1e-12,
                points=[0.0, 1.0],
            )[0]
            for d in range(dmax + 1)
        ]
    )
    return pmf / pmf.sum()


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize("cv", [0.3, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("mean", [2.0, 8.0, 15.0])
def test_censored_pmf_is_close_in_float32(family, cv, mean):
    # The baselinenowcast port enables float64 on import, so pin float32.
    with jax.enable_x64(False):
        ours = np.asarray(censored_pmf(family, mean, cv, 60))
    assert ours.dtype == np.float32
    ref = _quad_censored_pmf(family, mean, cv, 60)
    # float32 incomplete gamma is good to about 1e-6, and the closed form
    # scales that by the delay, so errors are a few 1e-5 at most.
    assert np.max(np.abs(ours - ref)) < 4e-5
    big = ref > 1e-4
    assert np.max(np.abs(ours[big] / ref[big] - 1)) < 2e-2
