import numpy as np
import numpyro
import pytest
from jax import random
from numpyro.infer import Predictive, init_to_median

from bdbvhubrenewal.model import (
    MAIN,
    ModelConfig,
    build_model,
    model_args,
)
from bdbvhubrenewal.model.inputs import vintage_from_tables
from bdbvhubrenewal.model.model import LADDER

from .conftest import SIM_LOCATIONS, VINTAGE_DAYS, vintage_date


def trace(model, data, forecast=False):
    """Trace the model at prior medians (as NUTS initialises)."""
    median = numpyro.handlers.substitute(
        numpyro.handlers.seed(model.sample, rng_seed=0),
        substitute_fn=init_to_median(num_samples=15),
    )
    with numpyro.handlers.trace() as tr:
        median(**model_args(model, data, forecast=forecast))
    return tr


def test_vintage_shapes(sim_vintage):
    v = sim_vintage
    assert v.names == ["CD-IT", "CD-NK", "CD-HU", "CD-TO"]
    assert v.inactive_names == ["CD-BU", "CD-SK"]
    assert v.known.shape == v.imputed.shape == (v.n_weeks, 4)
    assert v.inactive.shape == (v.n_weeks, 2)
    assert v.import_kernel.shape == (4, 2)
    assert v.delay_counts.shape == (v.n_weeks, 61)
    assert v.start_date.weekday() == 6
    assert v.cutoff // 7 + 1 == v.n_weeks
    np.testing.assert_allclose(v.hub_delay_pmf.sum(), 1.0)


def test_active_set_follows_the_data(outbreak, sim_tables, table):
    # At day 30 only the provinces with cases so far are active.
    target, delays = outbreak.vintage(vintage_date(outbreak, 41))
    v = vintage_from_tables(
        target, delays, vintage_date(outbreak, 41), SIM_LOCATIONS, table
    )
    assert set(v.names) < set(SIM_LOCATIONS)
    assert set(v.names) | set(v.inactive_names) == set(SIM_LOCATIONS)


@pytest.mark.parametrize(
    "cfg",
    [
        *LADDER.values(),
        ModelConfig(delay="gamma"),
        ModelConfig(delay="lognormal"),
        ModelConfig(coupling="learnt"),
        ModelConfig(rt_weighting="cases"),
        ModelConfig(outbreak_threshold=20, import_rt="origin"),
        ModelConfig(coupling="learnt", kernel_flows=True),
        ModelConfig(seeding="primary"),
    ],
)
def test_model_sites_and_shapes(outbreak, sim_tables, table, cfg):
    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    data = vintage_from_tables(
        target, delays, vintage_date(outbreak, day), SIM_LOCATIONS, table, cfg
    )
    model = build_model(data, cfg)
    tr = trace(model, data, forecast=True)
    n_weeks = model.latent.n_initialization_points // 7 + data.n_weeks + 3
    p = 1 if cfg.national else 4
    assert tr["onsets_final_total"]["value"].shape == (n_weeks, p)
    stream = "total" if cfg.streams == "total" else "known"
    assert tr[f"onsets_{stream}_obs"]["value"].shape == (2, n_weeks, p)
    assert tr["latent::rt_subpop"]["value"].shape == (7 * n_weeks, p)
    assert ("onsets_inactive_obs" in tr) == (cfg.inactive == "imports")
    # The fit sees the data weeks only; the horizon is drawn separately.
    if not cfg.national:
        assert "log_rt_deviations_z_future" in tr
    tr = trace(model, data)
    sampled = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    log_density = numpyro.infer.util.log_density(
        model.sample, (), model_args(model, data), sampled
    )[0]
    assert np.isfinite(float(log_density))


def test_masked_province_has_no_likelihood(outbreak, sim_tables, table):
    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    vintage = vintage_date(outbreak, day)
    data = vintage_from_tables(
        target, delays, vintage, SIM_LOCATIONS, table, masked=("CD-NK",)
    )
    assert data.fit_mask.tolist() == [True, False, True, True]
    model = build_model(data)
    tr = trace(model, data)
    params = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }

    def density(scale):
        changed = data.known.copy()
        changed[:, 1] *= scale
        data.known = changed
        return float(
            numpyro.infer.util.log_density(
                model.sample, (), model_args(model, data), params
            )[0]
        )

    original = data.known.copy()
    assert density(1.0) == pytest.approx(density(3.0))
    data.known = original
    # Still predicted.
    assert np.all(np.asarray(tr["onsets_expected_known"]["value"])[:, 1] >= 0)


def test_wrong_data_shape_is_rejected(sim_vintage):
    model = build_model(sim_vintage)
    args = model_args(model, sim_vintage)
    args["onsets"]["known"] = args["onsets"]["known"][1:]
    with pytest.raises(ValueError, match="known has shape"):
        model.validate_data(
            **{k: v for k, v in args.items() if k != "population_size"}
        )


@pytest.mark.parametrize("depletion", [True, False])
def test_prior_predictive_is_bounded_by_depletion(sim_vintage, depletion):
    cfg = ModelConfig(depletion=depletion)
    model = build_model(sim_vintage, cfg)
    args = model_args(model, sim_vintage, forecast=True)
    prior = Predictive(model.sample, num_samples=200)(random.key(4), **args)
    expected = np.asarray(
        prior["onsets_expected_known_final"]
        + prior["onsets_expected_imputed_final"]
    )
    cumulative = expected.sum(axis=1)
    nu = prior.get("effective_population", np.ones(200))
    pool = np.asarray(nu)[:, None] * sim_vintage.populations[None, :]
    share = cumulative / pool
    draws = np.asarray(prior["onsets_final_total"])
    if depletion:
        # Confirmed onsets can never exceed the effective population, so
        # the 3-week forecasts stay bounded and the count draws are valid.
        assert share.max() <= 1.0 + 1e-6
        assert np.all(draws >= 0)
    else:
        # Without depletion some prior draws run away (and overflow).
        assert np.nanmax(share) > 1.0 or np.isnan(share).any()


def test_main_is_the_last_ladder_step():
    assert list(LADDER.values())[-1] == MAIN


def test_real_data_kernels_and_log_density_are_finite(table):
    from bdbvhubrenewal.data import load_delays, load_target

    from .conftest import MAIN_REPO_DATA

    if not MAIN_REPO_DATA.exists():
        pytest.skip("main repo data absent")
    target, delays = load_target(MAIN_REPO_DATA), load_delays(MAIN_REPO_DATA)
    locations = tuple(sorted(set(target["location"]) - {"CD"}))
    for vintage in ("2026-10-09", "2026-10-24"):
        data = vintage_from_tables(
            target,
            delays.loc[delays["vintage"] == vintage],
            vintage,
            locations,
            table,
        )
        assert len(data.names) + len(data.inactive_names) == 26
        assert np.all(np.isfinite(data.kernel))
        assert np.all(np.isfinite(data.import_kernel))
        model = build_model(data)
        tr = trace(model, data)
        sampled = {
            k: v["value"]
            for k, v in tr.items()
            if v["type"] == "sample" and not v["is_observed"]
        }
        ld = numpyro.infer.util.log_density(
            model.sample, (), model_args(model, data), sampled
        )[0]
        assert np.isfinite(float(ld))


def test_learnt_kernel_is_gravity_at_prior_centre(sim_vintage):
    from bdbvhubrenewal.model import LearntKernel

    data = sim_vintage
    k = LearntKernel("k", data.distances, data.all_populations, 4)
    sub = numpyro.handlers.substitute(
        k.sample,
        data={
            "k_log_decay": 0.0,
            "k_weight_z": np.zeros(3),
            "k_weight_sd": 0.3,
        },
    )
    kernel = np.asarray(sub())
    np.testing.assert_allclose(kernel, data.kernel, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0, rtol=1e-5)
    assert k.import_kernel.shape == (4, 2)


def test_single_patch_locations_are_independent(outbreak, sim_tables, table):
    from bdbvhubrenewal.model.inputs import RENEWALSP

    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    vintage = vintage_date(outbreak, day)

    def one(target):
        return vintage_from_tables(
            target, delays, vintage, ("CD-IT",), table, RENEWALSP
        )

    changed = target.copy()
    changed.loc[changed["location"] == "CD-NK", "observed"] *= 5
    a, b = one(target), one(changed)
    assert a.names == ["CD-IT"]
    np.testing.assert_array_equal(a.known, b.known)
    np.testing.assert_array_equal(a.imputed, b.imputed)
    np.testing.assert_array_equal(a.delay_counts, b.delay_counts)
    model_a, model_b = build_model(a, RENEWALSP), build_model(b, RENEWALSP)
    tr = trace(model_a, a)
    params = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    lds = [
        float(
            numpyro.infer.util.log_density(
                m.sample, (), model_args(m, d), params
            )[0]
        )
        for m, d in ((model_a, a), (model_b, b))
    ]
    assert lds[0] == lds[1]


def test_late_provinces_are_seeded_at_their_first_cases(outbreak, table):
    # At day 166 Nord-Kivu (seeded day 30 in the simulation) has its first
    # cases weeks after Ituri, so it starts near empty.
    target, delays = outbreak.vintage(vintage_date(outbreak, 166))
    data = vintage_from_tables(
        target, delays, vintage_date(outbreak, 166), SIM_LOCATIONS, table
    )
    assert data.seed_days[0] == 0 and data.seed_days[1] > 0
    # Without coupling, nothing reaches it before its seed.
    model = build_model(data, ModelConfig(coupling="none"))
    tr = trace(model, data)
    x = np.asarray(tr["latent_infections_by_subpop"]["value"])
    n_init = model.latent.n_initialization_points
    day = n_init + data.seed_days[1]
    # Infections are dated by onset, so nothing comes before the first
    # case week; the seed lands in it.
    assert x[:day, 1].sum() < 1e-3 * x[day : day + 7, 1].sum()


def test_missingness_offsets_are_non_centred(sim_vintage):
    model = build_model(sim_vintage)
    tr = trace(model, sim_vintage)
    assert "logit_pi0_z" in tr and "logit_pi0_offsets" not in tr
    assert "log_rt0_offsets" in tr


def test_as_reported_draws_match_the_vintage_at_its_own_cutoff(sim_vintage):
    from bdbvhubrenewal.model import forecast_trajectories

    model = build_model(sim_vintage)
    prior = Predictive(model.sample, num_samples=20)(
        random.key(1), **model_args(model, sim_vintage)
    )
    sampled = {
        k: v
        for k, v in prior.items()
        if not k.endswith("_obs") and k != "onsets_delays"
    }
    later = sim_vintage.cutoff + 14
    draws = forecast_trajectories(
        model,
        sim_vintage,
        20,
        posterior=sampled,
        as_of=(sim_vintage.cutoff, later),
    )
    now = draws["as_reported"][sim_vintage.cutoff]["province"]
    seen = sim_vintage.known + sim_vintage.imputed
    n = sim_vintage.n_weeks
    # At the vintage's own cut-off the known counts are the vintage's, and
    # the imputed ones are centred on it.
    np.testing.assert_allclose(
        np.median(now[:, : n - 1].sum(-1), axis=0),
        seen[: n - 1].sum(-1),
        rtol=0.25,
        atol=3,
    )
    two_weeks = draws["as_reported"][later]["national"]
    assert two_weeks.shape == draws["national"].shape
    # More is reported two weeks on, and no more than eventually.
    assert np.median(two_weeks[:, n - 1]) >= np.median(now.sum(-1)[:, n - 1])
    assert np.all(
        np.median(two_weeks, 0)[: n - 1]
        <= np.median(draws["national"], 0)[: n - 1] * 1.25 + 3
    )


def _log_density(model, data):
    tr = trace(model, data)
    sampled = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    return float(
        numpyro.infer.util.log_density(
            model.sample, (), model_args(model, data), sampled
        )[0]
    )


def test_outbreak_classes_use_recent_cases():
    from bdbvhubrenewal.model.model import outbreak_classes

    weekly = np.array(
        [[30, 0], [5, 1], [0, np.nan], [0, 25], [0, 0], [40, 0]], dtype=float
    )
    got = outbreak_classes(weekly, 2, 20)
    expected = np.array(
        [[1, 0], [1, 0], [0, 0], [0, 1], [0, 1], [1, 0]], dtype=bool
    )
    np.testing.assert_array_equal(got, expected)
    assert outbreak_classes(weekly, 2, None) is None


def test_sporadic_weeks_take_national_rt(sim_vintage):
    from bdbvhubrenewal.model.model import outbreak_classes

    data = sim_vintage
    cfg = ModelConfig(
        outbreak_threshold=20,
        outbreak_window=6,
        rt_weighting="cases",
        import_rt="origin",
    )
    classes = outbreak_classes(data.known + data.imputed, 6, 20)
    assert classes.any() and not classes.all()
    model = build_model(data, cfg)
    tr = trace(model, data)
    n_init = model.latent.n_initialization_points
    log_rt = np.log(np.asarray(tr["latent::rt_subpop"]["value"]))[n_init:]
    base = np.log(np.asarray(tr["latent::rt_baseline"]["value"]))[n_init:]
    daily = np.repeat(classes, 7, axis=0)[: log_rt.shape[0]]
    dev = log_rt - base
    np.testing.assert_allclose(dev[~daily], 0.0, atol=1e-5)
    assert np.abs(dev[daily]).max() > 1e-4
    # Case weights over the outbreak provinces at each time.
    w = np.nansum(data.known + data.imputed, axis=0) + 1.0
    np.testing.assert_allclose((dev * daily) @ w, 0.0, atol=1e-2)
    imported = np.asarray(tr["latent::infections_imported"]["value"])
    x = np.asarray(tr["latent_infections_by_subpop"]["value"])
    f = np.asarray(data.populations) / np.sum(data.populations)
    assert imported.shape == x.shape
    assert np.all(imported * f * np.sum(data.populations) <= x + 1e-6)
    assert np.isfinite(_log_density(model, data))


def test_primary_seeding_leaves_other_provinces_to_importation(sim_vintage):
    data = sim_vintage
    model = build_model(data, ModelConfig(seeding="primary", coupling="none"))
    tr = trace(model, data)
    assert tr["initial_onsets"]["value"].shape == (1,)
    x = np.asarray(tr["latent_infections_by_subpop"]["value"])
    primary = int(np.argmin(data.seed_days))
    others = np.arange(x.shape[1]) != primary
    # Without coupling nothing reaches the other provinces.
    assert x[:, others].max() < 1e-3
    assert x[:, primary].max() > 1.0
    coupled = build_model(data, ModelConfig(seeding="primary"))
    xc = np.asarray(
        trace(coupled, data)["latent_infections_by_subpop"]["value"]
    )
    assert xc[:, others].max(axis=0).min() > 1e-3
    assert np.isfinite(_log_density(coupled, data))


def test_learnt_kernel_flows_are_zero_at_prior_centre(sim_vintage):
    from bdbvhubrenewal.model import LearntKernel

    data = sim_vintage
    p = data.n_provinces
    k = LearntKernel(
        "kernel", data.distances, data.all_populations, p, flows=True
    )
    n_flow = k.symmetric.shape[1] + k.antisymmetric.shape[1]
    zero = {
        "kernel_log_decay": 0.0,
        "kernel_weight_z": np.zeros(p - 1),
        "kernel_flow_z": np.zeros(n_flow),
    }
    with (
        numpyro.handlers.seed(rng_seed=0),
        numpyro.handlers.substitute(data=zero),
    ):
        got = np.asarray(k())
    plain = LearntKernel("kernel", data.distances, data.all_populations, p)
    with (
        numpyro.handlers.seed(rng_seed=0),
        numpyro.handlers.substitute(data=zero),
    ):
        base = np.asarray(plain())
    np.testing.assert_allclose(got, base, rtol=1e-5, atol=1e-7)
    with numpyro.handlers.seed(rng_seed=1), numpyro.handlers.trace() as tr:
        draw = np.asarray(k())
    np.testing.assert_allclose(draw.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.diag(draw), 0.0, atol=1e-12)
    assert "kernel_flow_sd" in tr and "kernel_destination_weights" in tr


def test_default_has_no_sporadic_class():
    cfg = ModelConfig()
    assert cfg.outbreak_threshold is None and cfg.import_rt == "destination"

