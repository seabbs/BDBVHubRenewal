"""Tests for the NonCentredRandomWalk temporal process."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.temporal_processes import TemporalProcess

from bdbvhubrenewal.model import NonCentredRandomWalk


def test_is_a_temporal_process():
    rw = NonCentredRandomWalk(DeterministicVariable("sd", 0.1))
    assert isinstance(rw, TemporalProcess)
    assert rw.step_size == 1


def test_fit_and_forecast_steps_are_separate_sites():
    rw = NonCentredRandomWalk(
        DeterministicVariable("sd", 0.1), frozen=2, n_fit=6
    )
    with trace() as tr:
        x = seed(rw.sample, rng_seed=0)(
            10, initial_value=jnp.ones(2), n_processes=2, name_prefix="w"
        )
    assert x.shape == (10, 2)
    np.testing.assert_allclose(x[:3], 1.0)
    assert tr["w_z"]["value"].shape == (3, 2)
    assert tr["w_z_future"]["value"].shape == (4, 2)


def test_needs_random_variables():
    with pytest.raises(TypeError, match="innovation_sd_rv"):
        NonCentredRandomWalk(0.1)
    with pytest.raises(TypeError, match="half_life_rv"):
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1), half_life_rv=6)


def test_repr():
    rw = NonCentredRandomWalk(DeterministicVariable("sd", 0.1), frozen=6)
    assert repr(rw).startswith("NonCentredRandomWalk(innovation_sd_rv=")
    assert "frozen=6" in repr(rw)


def test_centred_samples_the_levels():
    # Centred, the fitted levels themselves are the site, so data that
    # pin the level of the walk pin each parameter, not a sum of steps
    # or a sum with the starting value.
    from numpyro.handlers import condition

    rw = NonCentredRandomWalk(
        DeterministicVariable("sd", 0.5), frozen=2, n_fit=6, centred=True
    )
    levels = jnp.arange(6.0).reshape(3, 2)
    with trace() as tr:
        x = seed(condition(rw.sample, {"w_x": levels.T}), rng_seed=0)(
            10, initial_value=jnp.ones(2), n_processes=2, name_prefix="w"
        )
    assert "w_z" not in tr
    np.testing.assert_allclose(x[:3], 1.0)
    np.testing.assert_allclose(x[3:6], levels)
    assert tr["w_z_future"]["value"].shape == (4, 2)
    # Steps of the levels from the starting value are N(0, sd).
    steps = jnp.diff(jnp.vstack([jnp.ones((1, 2)), levels]), axis=0)
    expected = float(
        __import__("numpyro")
        .distributions.Normal(0.0, 0.5)
        .log_prob(steps)
        .sum()
    )
    got = float(tr["w_x"]["fn"].log_prob(tr["w_x"]["value"]).sum())
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_centred_matches_non_centred_in_distribution():
    rw_nc = NonCentredRandomWalk(DeterministicVariable("sd", 0.3), n_fit=8)
    rw_c = NonCentredRandomWalk(
        DeterministicVariable("sd", 0.3), n_fit=8, centred=True
    )
    from jax import random, vmap

    def draw(rw, key):
        return seed(rw.sample, rng_seed=key)(8, n_processes=1)

    keys = random.split(random.key(1), 4000)
    a = vmap(lambda k: draw(rw_nc, k))(keys)
    b = vmap(lambda k: draw(rw_c, k))(keys)
    np.testing.assert_allclose(a.std(0), b.std(0), rtol=0.06)


def test_centred_needs_a_random_walk():
    with pytest.raises(ValueError, match="centred"):
        NonCentredRandomWalk(
            DeterministicVariable("sd", 0.1),
            half_life_rv=DeterministicVariable("h", 6.0),
            centred=True,
        )
