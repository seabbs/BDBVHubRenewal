import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.temporal_processes import WeeklyTemporalProcess

from bdbv_hub_renewal.delays import ParametricDelay
from bdbv_hub_renewal.latent import renewal
from bdbv_hub_renewal.observation import (
    expected_imputed_onsets,
    expected_known_onsets,
    expected_missing_by_report,
)
from bdbv_hub_renewal.pyrenew import (
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
from bdbv_hub_renewal.pyrenew.observation import backsample


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
    ours = np.asarray(censored_pmf(family, 8.0, 0.8, 60, n_grid=200))
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
    assert np.all(np.asarray(out["known_final"]) >= np.asarray(out["known"]))


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
    from bdbv_hub_renewal.pyrenew.adapter import province_centres

    c = province_centres().set_index("location")
    assert len(c) == 26 and not c.isna().any().any()
    for loc, (lat, lon) in PROVINCE_CENTRES.items():
        assert c.loc[loc, "latitude"] == pytest.approx(lat, abs=1e-3)
        assert c.loc[loc, "longitude"] == pytest.approx(lon, abs=1e-3)
