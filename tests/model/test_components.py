import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.temporal_processes import WeeklyTemporalProcess

from bdbv_hub_renewal.data.delays import ParametricDelay
from bdbv_hub_renewal.model import (
    FixedTimeVarying,
    HazardDelay,
    MissingOnsetFraction,
    NonCentredRandomWalk,
    PooledInitialValue,
    adjacency_kernel,
    censored_pmf,
    coupled_renewal,
    empirical_delay_matrix,
    empirical_missing_fraction,
    generation_interval,
    hazard_to_pmf,
    mixing_matrix,
    onset_expectations,
    pooled_walk,
)
from bdbv_hub_renewal.model.observation import backsample

from .numpy_reference import (
    expected_imputed_onsets,
    expected_known_onsets,
    expected_missing_by_report,
    renewal,
)


def seeded(fn, seed=0):
    with numpyro.handlers.seed(rng_seed=seed):
        return fn()


def test_hazard_to_pmf_is_a_pmf_and_geometric_for_constant_hazard():
    pmf = np.asarray(hazard_to_pmf(jnp.full((3, 20), jnp.log(0.2 / 0.8))))
    assert pmf.shape == (3, 21)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(pmf[0, :5], 0.2 * 0.8 ** np.arange(5), rtol=1e-5)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_censored_pmf_matches_reference(family):
    ours = np.asarray(censored_pmf(family, 8.0, 0.8, 60))
    ref = ParametricDelay(family, 8.0, 6.4).pmf(60)
    np.testing.assert_allclose(ours, ref, atol=2e-4)


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
    k = adjacency_kernel(np.ones((3, 3)))
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
    k = adjacency_kernel(np.ones((p, p)))
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
    x, _ = coupled_renewal(
        jnp.asarray(seed[:, :n_init].T / pops),
        jnp.asarray(rt),
        jnp.asarray(g),
        mixing_matrix(jnp.asarray(k), 0.05, jnp.asarray(f)),
        jnp.asarray(1 - seed[:, :n_init].sum(axis=1) / pops),
        deplete=deplete,
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
    s0 = 1 - I0.sum(axis=0)
    x, r_eff = coupled_renewal(I0, rt, g, w, s0, deplete=True)
    assert float(x.sum(axis=0).max() + I0.sum(axis=0).max()) <= 1.0 + 1e-6
    assert float(r_eff[-1].max()) < 1.0
    x_free, _ = coupled_renewal(I0, rt, g, w, s0, deplete=False)
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


def test_missing_fraction_and_hazard_delay_shapes():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1)), start_dow=6
    )
    pi = MissingOnsetFraction(
        "pi", PooledInitialValue("pi0", 4, -2.0, 0.5, 0.3), weekly, weekly
    )
    out = seeded(lambda: pi(n_timepoints=35, n_groups=4, first_day_dow=6))
    assert out.shape == (35, 4) and bool(jnp.all((out > 0) & (out < 1)))
    delay = HazardDelay("delay", 30, weekly)
    pmf = seeded(lambda: delay(n_timepoints=35, first_day_dow=6))
    assert pmf.shape == (35, 31)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, rtol=1e-5)


def test_near_constant_when_innovation_sd_is_small():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 1e-6)), start_dow=6
    )
    pmf = seeded(
        lambda: HazardDelay("d", 30, weekly)(n_timepoints=70, first_day_dow=6)
    )
    np.testing.assert_allclose(pmf[0], pmf[-1], atol=1e-5)


def test_fixed_inputs_pad_with_last_row():
    fixed = FixedTimeVarying("x", np.arange(6.0).reshape(3, 2))
    out = fixed(n_timepoints=5)
    np.testing.assert_allclose(out[-1], [4.0, 5.0])
    assert out.shape == (5, 2)


def test_empirical_inputs_are_valid(sim_vintage):
    f = empirical_delay_matrix(sim_vintage.delay_counts, sim_vintage.cutoff)
    assert f.shape == (
        7 * sim_vintage.n_weeks,
        sim_vintage.delay_counts.shape[1],
    )
    np.testing.assert_allclose(f.sum(axis=1), 1.0, rtol=1e-8)
    pi = empirical_missing_fraction(sim_vintage.known, sim_vintage.imputed)
    assert pi.shape == (7 * sim_vintage.n_weeks, sim_vintage.n_provinces)
    assert np.all((pi > 0) & (pi < 1))


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
    from bdbv_hub_renewal.data import PROVINCE_CENTRES
    from bdbv_hub_renewal.model.inputs import province_centres

    c = province_centres().set_index("location")
    assert len(c) == 26 and not c.isna().any().any()
    for loc, (lat, lon) in PROVINCE_CENTRES.items():
        assert c.loc[loc, "latitude"] == pytest.approx(lat, abs=1e-3)
        assert c.loc[loc, "longitude"] == pytest.approx(lon, abs=1e-3)


# R: primarycensored::dpcens(0:42, pgamma, shape = (15/9)^2, rate = 15/81,
#    pwindow = 1, swindow = 1, D = 43), primarycensored 1.x, R 4.6.
DPCENS_GI = np.array(
    [0.0004862021, 0.0050286802, 0.0135041309, 0.0229102532, 0.0317137783,
     0.0391890828, 0.0450491334, 0.0492578353, 0.0519197765, 0.0532123394,
     0.0533434874, 0.0525258716, 0.0509615373, 0.0488335504, 0.0463021113,
     0.0435035255, 0.0405509272, 0.0375360192, 0.0345313317, 0.0315926802,
     0.0287616150, 0.0260677387, 0.0235308224, 0.0211626864, 0.0189688388,
     0.0169498778, 0.0151026721, 0.0134213432, 0.0118980692, 0.0105237347,
     0.0092884482, 0.0081819476, 0.0071939120, 0.0063141955, 0.0055329985,
     0.0048409869, 0.0042293714, 0.0036899538, 0.0032151488, 0.0027979866,
     0.0024321017, 0.0021117111, 0.0018315857]
)  # fmt: skip
# R: dpcens(0:60, pgamma, shape = 1/0.64, rate = 1/(8*0.64), 1, 1, D = 61)
# and dpcens(0:60, plnorm, meanlog = log(8) - log1p(0.64)/2,
#    sdlog = sqrt(log1p(0.64)), 1, 1, D = 61), first 12 values.
DPCENS_DELAY = {
    "gamma": [0.0201013592, 0.0691281551, 0.0865284119, 0.0899649926,
              0.0872268498, 0.0814602581, 0.0743087386, 0.0667008352,
              0.0591717391, 0.0520241311, 0.0454182071, 0.0394255981],
    "lognormal": [0.0008402915, 0.0229248670, 0.0741250861, 0.1075417400,
                  0.1148966476, 0.1075311290, 0.0943743233, 0.0801201601,
                  0.0668394085, 0.0552668086, 0.0455189862, 0.0374546304],
}  # fmt: skip


def test_generation_interval_is_double_censored():
    # Lags 1..42: the double-censored pmf on 0..42 with lag 0 dropped and
    # the rest renormalised (as BVDOutbreakSize left-truncates at a day).
    expected = DPCENS_GI[1:] / DPCENS_GI[1:].sum()
    np.testing.assert_allclose(generation_interval(), expected, rtol=1e-6)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_censored_pmf_matches_primarycensored(family):
    ours = np.asarray(censored_pmf(family, 8.0, 0.8, 60))
    np.testing.assert_allclose(ours[:12], DPCENS_DELAY[family], rtol=2e-4)


def test_idle_provinces_are_zero_up_to_the_cutoff():
    from bdbv_hub_renewal.model.model import seed_from_imports

    rng = np.random.default_rng(0)
    y = seed_from_imports(
        np.full((60, 2), 0.5), np.full(60, 1.5), generation_interval(), 40, rng
    )
    assert np.all(y[:40] == 0)
    assert y[40:].sum() > 0


def test_per_origin_mixing_conserves_transmission():
    f = jnp.array([0.5, 0.3, 0.2])
    k = adjacency_kernel(np.ones((3, 3)))
    eps = jnp.array([0.1, 0.0, 0.3])
    w = mixing_matrix(jnp.asarray(k), eps, f)
    counts = f[:, None] * w / f[None, :]
    np.testing.assert_allclose(counts.sum(axis=0), 1.0, rtol=1e-6)
    # An origin with no share sends nothing.
    np.testing.assert_allclose(np.asarray(counts)[[0, 2], 1], 0.0, atol=1e-7)
    same = mixing_matrix(jnp.asarray(k), jnp.full(3, 0.1), f)
    np.testing.assert_allclose(same, mixing_matrix(jnp.asarray(k), 0.1, f))


def test_linear_weekly_process_interpolates_knots():
    from bdbv_hub_renewal.model.rt import LinearWeeklyProcess

    rw = NonCentredRandomWalk(DeterministicVariable("sd", 0.3))
    out = seeded(lambda: LinearWeeklyProcess(rw).sample(29, initial_value=0.0))
    weekly = out[::7, 0]
    np.testing.assert_allclose(
        out[3, 0], (weekly[0] * 4 + weekly[1] * 3) / 7, rtol=1e-5
    )
    assert out.shape == (29, 1)


def test_empirical_missing_fraction_moves_to_report_days():
    # Onset-week shares are 0.1 then 0.4 from week 10. Missing onsets are
    # dated by report, so the report-day fraction steps `shift` days later.
    imputed = np.where(np.arange(20)[:, None] < 10, 10.0, 40.0)
    known = 100.0 - imputed
    base = empirical_missing_fraction(known, imputed, window=0, shrink=0)
    pi = empirical_missing_fraction(known, imputed, window=0, shrink=0, shift=9)
    assert pi.shape == base.shape
    np.testing.assert_allclose(pi[9:], base[:-9])
    np.testing.assert_allclose(pi[:9], np.repeat(base[:1], 9, axis=0))
    assert pi[78, 0] == pytest.approx(0.1)
    assert pi[79, 0] == pytest.approx(0.4)


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
        delay = HazardDelay("delay", 30, process)
        return seeded(lambda: delay(n_timepoints=40, first_day_dow=dow), 4)

    np.testing.assert_allclose(run(weekly), run(_Daily(weekly)), rtol=1e-6)
