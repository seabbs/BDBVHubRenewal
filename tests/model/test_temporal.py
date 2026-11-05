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
