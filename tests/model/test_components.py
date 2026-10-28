import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.temporal_processes import WeeklyTemporalProcess

from bdbvhubrenewal.data.delays import ParametricDelay
from bdbvhubrenewal.model import (
    FixedTimeVarying,
    HazardDelay,
    MissingOnsetFraction,
    NonCentredRandomWalk,
    PooledInitialValue,
    censored_pmf,
    coupled_renewal,
    generation_interval,
    hazard_to_pmf,
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
    x, r_eff, _ = coupled_renewal(I0, rt, g, w, s0, deplete=True)
    assert float(x.sum(axis=0).max() + I0.sum(axis=0).max()) <= 1.0 + 1e-6
    assert float(r_eff[-1].max()) < 1.0
    x_free, _, _ = coupled_renewal(I0, rt, g, w, s0, deplete=False)
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
    gi = generation_interval(15.0, 9.0)
    np.testing.assert_allclose(gi, expected, rtol=1e-6)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_censored_pmf_matches_primarycensored(family):
    ours = np.asarray(censored_pmf(family, 8.0, 0.8, 60))
    np.testing.assert_allclose(ours[:12], DPCENS_DELAY[family], rtol=2e-4)


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
        delay = HazardDelay("delay", 30, process)
        return seeded(lambda: delay(n_timepoints=40, first_day_dow=dow), 4)

    np.testing.assert_allclose(run(weekly), run(_Daily(weekly)), rtol=1e-6)


def test_hazard_delay_smoothness_prior_allows_real_fits():
    # Real fits put the step SD of the delay shape at about 0.2
    # (2026-10-24: 90% CrI 0.16 to 0.25), so the prior is HalfNormal(0.4).
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1)), start_dow=6
    )
    delay = HazardDelay("delay", 30, weekly)
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


def test_case_weighted_basis_is_orthonormal_and_orthogonal_to_weights():
    from bdbvhubrenewal.model.rt import sum_to_zero_basis

    w = np.array([6000.0, 2000.0, 400.0, 90.0])
    q = sum_to_zero_basis(4, w)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(w @ q, 0.0, atol=1e-8)
    plain = sum_to_zero_basis(4)
    np.testing.assert_allclose(plain.sum(axis=0), 0.0, atol=1e-10)


def test_basis_over_outbreak_provinces_leaves_sporadic_rows_zero():
    from bdbvhubrenewal.model.rt import sum_to_zero_basis

    w = np.array([6000.0, 3.0, 400.0, 12.0])
    active = np.array([True, False, True, False])
    q = sum_to_zero_basis(4, w, active)
    assert q.shape == (4, 1)
    np.testing.assert_allclose(q[~active], 0.0)
    np.testing.assert_allclose(q.T @ q, np.eye(1), atol=1e-10)
    np.testing.assert_allclose(w @ q, 0.0, atol=1e-8)


def test_pooled_walk_holds_sporadic_provinces_at_the_national_value():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1)), start_dow=6
    )
    active = np.array([True, False, True, True])
    base, groups = seeded(
        lambda: pooled_walk(
            weekly,
            weekly,
            jnp.array([0.1, 0.9, 0.2, 0.3]),
            30,
            4,
            "x",
            6,
            np.array([5.0, 1.0, 2.0, 3.0]),
            active,
        )
    )
    np.testing.assert_allclose(groups[:, 1], base[:, 0], atol=1e-6)
    w = np.array([5.0, 2.0, 3.0])
    np.testing.assert_allclose((groups[:, active] - base) @ w, 0.0, atol=1e-5)
    # A single outbreak province leaves everyone on the national walk.
    _, alone = seeded(
        lambda: pooled_walk(
            weekly, weekly, 0.1, 30, 3, "y", 6,
            active=np.array([True, False, False]),
        )
    )  # fmt: skip
    np.testing.assert_allclose(alone, np.repeat(alone[:, :1], 3, 1))


def _origin_renewal(init, rt, g, w):
    """Reference loop with imports at the origin's Rt, no depletion."""
    rev = np.flip(g)
    hist = np.asarray(init[-len(g) :], dtype=float)
    out, imported = [], []
    off = w - np.diag(np.diag(w))
    for r in rt:
        force = rev @ hist
        out.append(w @ (r * force))
        imported.append(off @ (r * force))
        hist = np.vstack([hist[1:], out[-1]])
    return np.array(out), np.array(imported)


def test_imports_at_the_origin_rt_and_imported_infections():
    rng = np.random.default_rng(4)
    g = generation_interval()
    p = 3
    f = np.array([0.5, 0.3, 0.2])
    k = uniform_kernel(p)
    w = np.asarray(mixing_matrix(jnp.asarray(k), 0.05, jnp.asarray(f)))
    I0 = rng.uniform(0, 1e-5, (len(g), p))
    rt = rng.uniform(0.6, 1.8, (60, p))
    s0 = np.ones(p)
    x, _, imp = coupled_renewal(
        jnp.asarray(I0), jnp.asarray(rt), jnp.asarray(g), jnp.asarray(w),
        jnp.asarray(s0), deplete=False, import_rt="origin",
    )  # fmt: skip
    ref, ref_imp = _origin_renewal(I0, rt, g, w)
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(imp), ref_imp, rtol=1e-4)
    # With one Rt for all, origin and destination agree.
    flat = np.repeat(rt[:, :1], p, 1)
    args = (jnp.asarray(I0), jnp.asarray(flat), jnp.asarray(g))
    xd, _, _ = coupled_renewal(
        *args, jnp.asarray(w), jnp.asarray(s0), deplete=False
    )
    xo, _, _ = coupled_renewal(
        *args, jnp.asarray(w), jnp.asarray(s0), deplete=False,
        import_rt="origin",
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(xd), np.asarray(xo), rtol=1e-5)
    # Imported infections are part of the total, and none without coupling.
    xdep, _, impdep = coupled_renewal(
        jnp.asarray(I0), jnp.asarray(rt), jnp.asarray(g), jnp.asarray(w),
        jnp.asarray(s0) * 0.05, deplete=True, pool=0.05,
    )  # fmt: skip
    assert np.all(np.asarray(impdep) <= np.asarray(xdep) + 1e-12)
    assert float(np.asarray(impdep).sum()) > 0
    _, _, none = coupled_renewal(
        jnp.asarray(I0), jnp.asarray(rt), jnp.asarray(g), jnp.eye(p),
        jnp.asarray(s0), deplete=False,
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(none), 0.0)


def test_pooled_walk_masks_deviations_by_time():
    weekly = WeeklyTemporalProcess(
        NonCentredRandomWalk(DeterministicVariable("sd", 0.3)), start_dow=6
    )
    mask = np.ones((28, 4), bool)
    mask[:14, 3] = False  # Starts sporadic, becomes an outbreak.
    mask[14:, 1] = False  # Goes quiet.
    w = np.array([5.0, 1.0, 2.0, 3.0])
    base, groups = seeded(
        lambda: pooled_walk(
            weekly, weekly, jnp.array([0.1, 0.9, 0.2, 0.3]), 28, 4, "x", 6,
            w, mask,
        )
    )  # fmt: skip
    dev = np.asarray(groups - base)
    np.testing.assert_allclose(dev[~mask], 0.0, atol=1e-6)
    np.testing.assert_allclose((dev * mask) @ w, 0.0, atol=1e-5)
    assert np.abs(dev[mask]).max() > 1e-3


def test_flow_pair_basis_is_double_centred():
    from bdbvhubrenewal.model import flow_pair_basis

    n = 5
    sym, anti = flow_pair_basis(n)
    assert sym.shape == (n * n, n * (n - 3) // 2)
    assert anti.shape == (n * n, (n - 1) * (n - 2) // 2)
    for basis, sign in ((sym, 1.0), (anti, -1.0)):
        np.testing.assert_allclose(
            basis.T @ basis, np.eye(basis.shape[1]), atol=1e-9
        )
        for col in basis.T:
            m = col.reshape(n, n)
            np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-12)
            np.testing.assert_allclose(m.sum(axis=0), 0.0, atol=1e-9)
            np.testing.assert_allclose(m.sum(axis=1), 0.0, atol=1e-9)
            np.testing.assert_allclose(m, sign * m.T, atol=1e-9)


def test_flow_factor_gives_flow_variance_and_reciprocity():
    from bdbvhubrenewal.model import flow_pair_basis, importation_flow_factor

    n = 5
    sym, anti = flow_pair_basis(n)
    cov = np.asarray(importation_flow_factor(sym, anti, 0.7, 0.4))
    cov = cov @ cov.T
    off = ~np.eye(n, dtype=bool).reshape(-1)
    # On average over flows, as BVDOutbreakSize scales the two parts.
    np.testing.assert_allclose(np.diag(cov)[off].mean(), 0.49, rtol=1e-6)
    np.testing.assert_allclose(np.diag(cov)[~off], 0.0, atol=1e-12)
    rev = np.arange(n * n).reshape(n, n).T.reshape(-1)
    pairs = cov[np.arange(n * n), rev][off]
    np.testing.assert_allclose(pairs.mean(), 0.4 * 0.49, rtol=1e-6)


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
    ours = np.asarray(censored_pmf(family, mean, cv, 60))
    assert ours.dtype == np.float32
    ref = _quad_censored_pmf(family, mean, cv, 60)
    # float32 incomplete gamma is good to about 1e-6, and the closed form
    # scales that by the delay, so errors are a few 1e-5 at most.
    assert np.max(np.abs(ours - ref)) < 4e-5
    big = ref > 1e-4
    assert np.max(np.abs(ours[big] / ref[big] - 1)) < 2e-2
