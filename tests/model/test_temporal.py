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


def test_anchored_start_keeps_the_start_prior():
    # A change of variables with Jacobian 1: the joint density of the
    # anchor and the steps is the start prior times the step prior.
    import numpyro.distributions as dist
    from numpyro.infer.util import log_density

    rw = NonCentredRandomWalk(
        DeterministicVariable("sd", 0.3),
        frozen=1,
        n_fit=8,
        half_life_rv=DeterministicVariable("h", 3.0),
    )
    prior = dist.Normal(0.2, 0.5)

    def model():
        return rw.sample(8, initial_value=prior, n_processes=2, name_prefix="w")

    with trace() as tr:
        x = seed(model, rng_seed=3)()
    params = {k: tr[k]["value"] for k in ("w_anchor", "w_z")}
    lp, _ = log_density(model, (), {}, params)
    expected = (
        prior.log_prob(x[0]).sum()
        + dist.Normal(0, 1).log_prob(params["w_z"]).sum()
    )
    np.testing.assert_allclose(lp, expected, rtol=1e-6)
    # The anchor is the start plus the mean steps' part over its weights.
    phi = 0.5 ** (1 / 3.0)
    decay = phi ** np.arange(1, 7)[:, None]
    free = np.asarray(x[2:8]) - decay * np.asarray(x[0])
    np.testing.assert_allclose(
        tr["w_anchor"]["value"],
        x[0] + free.sum(0) / (1 + decay.sum()),
        rtol=1e-5,
    )


def test_needs_random_variables():
    with pytest.raises(TypeError, match="innovation_sd_rv"):
        NonCentredRandomWalk(0.1)
    with pytest.raises(TypeError, match="half_life_rv"):
        NonCentredRandomWalk(DeterministicVariable("sd", 0.1), half_life_rv=6)


def test_repr():
    rw = NonCentredRandomWalk(DeterministicVariable("sd", 0.1), frozen=6)
    assert repr(rw).startswith("NonCentredRandomWalk(innovation_sd_rv=")
    assert "frozen=6" in repr(rw)
