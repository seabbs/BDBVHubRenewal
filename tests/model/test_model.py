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
from bdbvhubrenewal.model.model import LADDER, incubation_period, seed_lead

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
    assert v.names == list(SIM_LOCATIONS)
    assert v.known.shape == v.imputed.shape == (v.n_weeks, 6)
    assert v.delay_counts.shape == (v.n_weeks, 6, 61)
    assert v.start_date.weekday() == 6
    assert v.cutoff // 7 + 1 == v.n_weeks
    np.testing.assert_allclose(v.hub_delay_pmf.sum(), 1.0)


def test_provinces_with_no_case_are_never_seeded(outbreak, sim_tables, table):
    # At day 41 some provinces have no case yet: they are modelled, but
    # never seeded, so their infections come from importation only.
    target, delays = outbreak.vintage(vintage_date(outbreak, 41))
    v = vintage_from_tables(
        target, delays, vintage_date(outbreak, 41), SIM_LOCATIONS, table
    )
    assert v.names == list(SIM_LOCATIONS)
    none = (v.known + v.imputed).sum(axis=0) == 0
    assert none.any()
    assert np.all(np.isinf(v.seed_days[none]))
    assert np.all(np.isfinite(v.seed_days[~none]))


@pytest.mark.parametrize(
    "cfg",
    [
        *LADDER.values(),
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
    p = 1 if cfg.national else 6
    assert tr["onsets_final_total"]["value"].shape == (n_weeks, p)
    stream = "total" if cfg.streams == "total" else "known"
    assert tr[f"onsets_{stream}_obs"]["value"].shape == (2, n_weeks, p)
    assert tr["latent::rt_subpop"]["value"].shape == (7 * n_weeks, p)
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
    assert data.fit_mask.tolist() == [True, False, True, True, True, True]
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


def test_real_data_kernels_are_finite(table):
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
        assert np.all(np.isfinite(data.kernel))


def test_late_provinces_are_seeded_at_their_first_cases(outbreak, table):
    # At day 166 Nord-Kivu (seeded day 30 in the simulation) has its first
    # cases weeks after Ituri, so it starts near empty.
    target, delays = outbreak.vintage(vintage_date(outbreak, 166))
    data = vintage_from_tables(
        target, delays, vintage_date(outbreak, 166), SIM_LOCATIONS, table
    )
    assert data.seed_days[0] == 0 and data.seed_days[1] > 0
    # The seed leads the first case week by the incubation lead.
    lead = seed_lead(incubation_period())
    assert lead == 5
    assert (data.seed_days[1] + lead) % 7 == 0
    # Without coupling, nothing reaches it before its seed.
    model = build_model(data, ModelConfig(coupling="none"))
    tr = trace(model, data)
    x = np.asarray(tr["latent_infections_by_subpop"]["value"])
    n_init = model.latent.n_initialization_points
    day = n_init + int(data.seed_days[1])
    assert x[:day, 1].sum() < 1e-3 * x[day : day + 7, 1].sum()


def test_unseeded_provinces_get_infections_only_by_importation(sim_vintage):
    # CD-BU and CD-SK have no case: with coupling they gain infections
    # from the other provinces, without it they stay near zero.
    j = [sim_vintage.names.index(p) for p in ("CD-BU", "CD-SK")]
    other = [i for i in range(len(sim_vintage.names)) if i not in j]

    def share(config):
        model = build_model(sim_vintage, config)
        tr = trace(model, sim_vintage)
        x = np.asarray(tr["latent_infections_by_subpop"]["value"])
        pops = np.asarray(sim_vintage.populations)
        cases = x * pops
        return cases[:, j].sum() / cases[:, other].sum()

    alone = share(ModelConfig(coupling="none"))
    assert alone < 1e-4
    assert share(ModelConfig()) > 100 * alone


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


def test_unseeded_provinces_sample_no_seed_or_deviation(sim_vintage):
    # Barely informed, they follow the pooled Rt trend and missing-onset
    # share and take a fixed seed.
    unseeded = ~sim_vintage.seeded
    k = int((~unseeded).sum())
    assert unseeded.any()
    model = build_model(sim_vintage)
    tr = trace(model, sim_vintage)
    assert tr["initial_infections"]["value"].shape == (k,)
    assert tr["log_rt0_offsets"]["value"].shape == (k - 1,)
    rt = np.asarray(tr["latent::rt_subpop"]["value"])
    base = np.asarray(tr["latent::rt_baseline"]["value"])
    np.testing.assert_allclose(
        rt[:, unseeded], base[:, :1].repeat(unseeded.sum(), 1), rtol=1e-6
    )
