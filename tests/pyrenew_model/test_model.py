import numpy as np
import numpyro
import pytest
from jax import random
from numpyro.infer import Predictive, init_to_median

from bdbv_hub_renewal.pyrenew import (
    FALLBACK,
    MAIN,
    ModelConfig,
    build_model,
    model_args,
)
from bdbv_hub_renewal.pyrenew.adapter import vintage_from_tables
from bdbv_hub_renewal.pyrenew.model import LADDER

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
        FALLBACK,
        ModelConfig(delay="gamma"),
        ModelConfig(delay="lognormal", coupling="adjacency"),
        ModelConfig(rt_deviation="ar1"),
        ModelConfig(rt_cadence="daily"),
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
    from bdbv_hub_renewal.data import load_delays, load_target

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
