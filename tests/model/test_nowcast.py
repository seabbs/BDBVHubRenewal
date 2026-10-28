"""Nowcasts of partly reported weeks keep the counts already reported."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import Predictive

from bdbv_hub_renewal.model import FALLBACK, MAIN, build_model, model_args
from bdbv_hub_renewal.model.model import forecast_trajectories
from bdbv_hub_renewal.model.observation import onset_expectations


def prior_latent(model, data, n=20):
    """Prior draws of every site but the observation draws."""
    prior = Predictive(model.sample, num_samples=n)(
        random.key(3), **model_args(model, data)
    )
    return {k: v for k, v in prior.items() if not k.endswith("_obs")}


def test_nowcast_is_never_below_the_reported_counts(sim_vintage, config):
    model = build_model(sim_vintage, config)
    draws = forecast_trajectories(
        model,
        sim_vintage,
        n_trajectories=20,
        posterior=prior_latent(model, sim_vintage),
    )
    reported = sim_vintage.known + sim_vintage.imputed
    province = draws["province"][:, : sim_vintage.n_weeks]
    assert np.all(province >= reported[None])
    # Weeks reported in full keep their counts in every draw.
    assert np.any(np.all(province == reported[None], axis=0))


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


def test_model_estimates_the_publication_lag(sim_vintage):
    model = build_model(sim_vintage, MAIN)
    prior = prior_latent(model, sim_vintage, n=5)
    assert prior["onsets_publication_lag"].shape == (5, 4)
    fixed = build_model(sim_vintage, FALLBACK)
    assert "onsets_publication_lag" not in prior_latent(fixed, sim_vintage, 2)


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
    _, mn = delay_site(MAIN, latent)
    # Same expected delay probabilities, with concentration psi.
    psi = 1 / latent["delay_overdispersion"] ** 2
    conc = np.asarray(dm.concentration)
    np.testing.assert_allclose(conc.sum(-1), psi, rtol=1e-4)
    np.testing.assert_allclose(conc / psi, mn.probs, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("final", [True, False])
def test_weekly_blocks_match_the_dense_expectations(final):
    rng = np.random.default_rng(5)
    n_w, dmax = 12, 20
    n = 7 * n_w
    delay = np.repeat(rng.dirichlet(np.ones(dmax + 1), n_w), 7, axis=0)
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
    assert obs.weekly_delay
    fast = log_density(model.sample, (), args, values)[0]
    obs.weekly_delay = False
    dense = log_density(model.sample, (), args, values)[0]
    assert float(fast) == pytest.approx(float(dense), rel=1e-5)
