"""Nowcasts of partly reported weeks keep the counts already reported."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from jax import random
from numpyro.infer import Predictive

from bdbvhubrenewal.model import FALLBACK, MAIN, build_model, model_args
from bdbvhubrenewal.model.model import forecast_trajectories
from bdbvhubrenewal.model.observation import onset_expectations


def prior_latent(model, data, n=20):
    """Prior draws of every site but the observation draws."""
    prior = Predictive(model.sample, num_samples=n)(
        random.key(3), **model_args(model, data)
    )
    return {k: v for k, v in prior.items() if not k.endswith("_obs")}


def test_nowcast_keeps_known_and_recentres_imputed(sim_vintage, config):
    model = build_model(sim_vintage, config)
    draws = forecast_trajectories(
        model,
        sim_vintage,
        n_trajectories=50,
        posterior=prior_latent(model, sim_vintage, n=50),
    )
    province = draws["province"][:, : sim_vintage.n_weeks]
    # Known onsets only accumulate, so no draw is below them.
    assert np.all(province >= sim_vintage.known[None])
    # The hub re-imputes missing onsets at each vintage, so the imputed
    # count is drawn around its current value and can fall below it.
    reported = sim_vintage.known + sim_vintage.imputed
    assert np.any(province < reported[None])
    imputed = province - sim_vintage.known[None]
    big = sim_vintage.imputed >= 5
    centre = np.median(imputed, axis=0)[big] / sim_vintage.imputed[big]
    assert 0.5 < np.median(centre) < 2.0


def test_masked_provinces_do_not_see_their_counts(sim_vintage, config):
    mask = np.ones(len(sim_vintage.names), dtype=bool)
    mask[0] = False
    data = dataclasses.replace(sim_vintage, fit_mask=mask)
    shifted = dataclasses.replace(
        data, known=data.known + 1000 * ~mask[None, :]
    )
    model = build_model(data, config)
    latent = prior_latent(model, data)
    a = forecast_trajectories(model, data, 20, posterior=latent)
    b = forecast_trajectories(model, shifted, 20, posterior=latent)
    np.testing.assert_array_equal(a["province"], b["province"])


def test_publication_lag_mixes_cut_offs():
    rng = np.random.default_rng(4)
    n, dmax, cutoff = 70, 20, 60
    args = (
        jnp.asarray(rng.gamma(5, 4, (n, 2))),
        jnp.asarray(rng.dirichlet(np.ones(dmax + 1), n)),
        jnp.asarray(rng.uniform(0.05, 0.3, (n, 2))),
        jnp.asarray(rng.dirichlet(np.ones(dmax + 1))),
    )
    lag = np.array([0.1, 0.2, 0.3, 0.4])
    out = onset_expectations(*args, cutoff, publication_lag=jnp.asarray(lag))
    # Reports are published by T with probability F(T - t), so the
    # expectations mix those at cut-offs T - l with weights p(l).
    parts = [onset_expectations(*args, cutoff - k) for k in range(4)]
    for key in ("known", "imputed", "triangle", "known_final"):
        mixed = sum(q * p[key] for q, p in zip(lag, parts, strict=True))
        np.testing.assert_allclose(out[key], mixed, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        onset_expectations(*args, cutoff, publication_lag=jnp.ones(1))["known"],
        parts[0]["known"],
    )


def test_model_takes_the_empirical_publication_lag(sim_vintage):
    lag = np.r_[np.full(10, 0.08), np.zeros(50), 0.2]
    data = dataclasses.replace(sim_vintage, publication_lag_pmf=lag)
    for cfg in (MAIN, FALLBACK):
        assert cfg.publication_lag == "empirical"
        model = build_model(data, cfg)
        obs = model.observations["onsets"]
        np.testing.assert_allclose(obs.publication_lag, lag)
        assert "onsets_publication_lag" not in prior_latent(model, data, 2)
    # With no estimate, a fixed long-tail prior pmf stands in: 85% on
    # 0..3 days and 15% spread over 4..60.
    model = build_model(
        dataclasses.replace(sim_vintage, publication_lag_pmf=None), MAIN
    )
    prior = np.asarray(model.observations["onsets"].publication_lag)
    assert prior.shape == (61,)
    assert prior[:4].sum() == pytest.approx(0.85)
    assert prior[4:].sum() == pytest.approx(0.15)
    est = build_model(
        sim_vintage, dataclasses.replace(MAIN, publication_lag="estimated")
    )
    prior = prior_latent(est, sim_vintage, n=5)
    assert prior["onsets_publication_lag"].shape == (5, 4)


def test_dirichlet_multinomial_delays(sim_vintage):
    cfg = dataclasses.replace(MAIN, delay_counts="dirichlet-multinomial")
    args = model_args(build_model(sim_vintage, MAIN), sim_vintage)

    def delay_site(config, values=None):
        model = build_model(sim_vintage, config)
        run = numpyro.handlers.seed(model.sample, rng_seed=0)
        if values is not None:
            run = numpyro.handlers.substitute(run, values)
        with numpyro.handlers.trace() as tr:
            run(**args)
        return tr, tr["onsets_delays"]["fn"].base_dist

    tr, dm = delay_site(cfg)
    assert isinstance(dm, dist.DirichletMultinomial)
    latent = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    _, mn = delay_site(
        dataclasses.replace(MAIN, delay_counts="multinomial"), latent
    )
    # Same expected delay probabilities, with concentration psi.
    # Capped at 1e4, as the negative binomial concentrations are.
    psi = 1 / (latent["delay_overdispersion"] ** 2 + 1e-4)
    conc = np.asarray(dm.concentration)
    np.testing.assert_allclose(conc.sum(-1), psi, rtol=1e-4)
    np.testing.assert_allclose(conc / psi, mn.probs, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("final", [True, False])
@pytest.mark.parametrize("daily", [False, True])
def test_weekly_blocks_match_the_dense_expectations(final, daily):
    rng = np.random.default_rng(5)
    n_w, dmax = 12, 20
    n = 7 * n_w
    delay = np.repeat(rng.dirichlet(np.ones(dmax + 1), n_w), 7, axis=0)
    if daily:
        # A delay that also changes within weeks, as a day-of-week effect.
        delay = rng.dirichlet(np.ones(dmax + 1), n)
    args = (
        jnp.asarray(rng.gamma(5, 4, (n, 3))),
        jnp.asarray(delay),
        jnp.asarray(rng.uniform(0.05, 0.3, (n, 3))),
        jnp.asarray(rng.dirichlet(np.ones(dmax + 1))),
        n - 10,
        final,
    )
    lag = jnp.asarray([0.4, 0.3, 0.2, 0.1])
    dense = onset_expectations(*args, publication_lag=lag)
    blocks = onset_expectations(*args, publication_lag=lag, weekly=True)
    for key, value in dense.items():
        np.testing.assert_allclose(blocks[key], value, rtol=1e-5, atol=1e-6)


def test_model_log_density_is_the_same_with_weekly_blocks(sim_vintage, config):
    from numpyro.infer.util import log_density

    model = build_model(sim_vintage, config)
    args = model_args(model, sim_vintage)
    with numpyro.handlers.trace() as tr:
        numpyro.handlers.seed(model.sample, rng_seed=0)(**args)
    values = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    obs = model.observations["onsets"]
    assert obs.weekly_axis
    fast = log_density(model.sample, (), args, values)[0]
    obs.weekly_axis = False
    dense = log_density(model.sample, (), args, values)[0]
    assert float(fast) == pytest.approx(float(dense), rel=1e-5)


def test_dispersion_prior_is_half_normal_on_inverse_root_phi(sim_vintage):
    model = build_model(sim_vintage, MAIN)
    with numpyro.handlers.trace() as tr:
        numpyro.handlers.seed(model.sample, rng_seed=0)(
            **model_args(model, sim_vintage)
        )
    for stream in ("known", "imputed"):
        site = tr[f"inv_sqrt_phi_{stream}"]
        assert isinstance(site["fn"], dist.HalfNormal)
        assert float(site["fn"].scale) == pytest.approx(0.25)


def test_negative_and_long_delays_are_clipped_not_dropped(
    outbreak, sim_tables, table
):
    from bdbvhubrenewal.model.inputs import vintage_from_tables

    from .conftest import SIM_LOCATIONS, VINTAGE_DAYS, vintage_date

    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    odd = delays.sort_values("onset_week").iloc[-2:].copy()
    odd["delay"] = [-3, 90]
    both = vintage_from_tables(
        target,
        pd.concat([delays, odd]),
        vintage_date(outbreak, day),
        SIM_LOCATIONS,
        table,
    )
    base = vintage_from_tables(
        target, delays, vintage_date(outbreak, day), SIM_LOCATIONS, table
    )
    extra = both.delay_counts - base.delay_counts
    assert extra.sum() == 2
    assert extra[:, 0].sum() == 1
    assert extra[:, -1].sum() == 1


def test_trajectories_name_their_posterior_draws(sim_vintage):
    model = build_model(sim_vintage, MAIN)
    latent = prior_latent(model, sim_vintage, n=8)
    draws = forecast_trajectories(model, sim_vintage, 5, posterior=latent)
    assert draws["draw"].shape == (5,)
    assert np.all((draws["draw"] >= 0) & (draws["draw"] < 8))


def test_delay_counts_are_dirichlet_multinomial_by_default():
    assert MAIN.delay_counts == "dirichlet-multinomial"
    assert FALLBACK.delay_counts == "dirichlet-multinomial"
