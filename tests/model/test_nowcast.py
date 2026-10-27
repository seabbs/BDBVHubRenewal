"""Nowcasts of partly reported weeks keep the counts already reported."""

import dataclasses

import numpy as np
from jax import random
from numpyro.infer import Predictive

from bdbv_hub_renewal.model import build_model, model_args
from bdbv_hub_renewal.model.model import forecast_trajectories


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
