import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer.util
import pytest
from pyrenew.deterministic import DeterministicVariable

from bdbvhubrenewal.model.composition_noise import (
    DirichletMultinomialNoise,
    MultinomialNoise,
)

PROBS = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
COUNTS = jnp.array([[5.0, 3.0, 2.0], [0.0, 1.0, 9.0]])


def log_density(noise, mask=None):
    def model():
        noise.sample("x", PROBS, COUNTS.sum(-1), obs=COUNTS, mask=mask)

    return float(numpyro.infer.util.log_density(model, (), {}, {})[0])


def test_multinomial_is_the_multinomial_log_density():
    expected = dist.Multinomial(COUNTS.sum(-1), PROBS).log_prob(COUNTS)
    np.testing.assert_allclose(
        log_density(MultinomialNoise()), expected.sum(), rtol=1e-6
    )


def test_dirichlet_multinomial_approaches_multinomial():
    noise = DirichletMultinomialNoise(DeterministicVariable("c", 1e4))
    np.testing.assert_allclose(
        log_density(noise), log_density(MultinomialNoise()), rtol=1e-2
    )


def test_mask_drops_rows():
    row = dist.Multinomial(COUNTS[0].sum(), PROBS[0]).log_prob(COUNTS[0])
    lp = log_density(MultinomialNoise(), jnp.array([True, False]))
    np.testing.assert_allclose(lp, row, rtol=1e-6)


def test_prior_draws_keep_the_totals():
    with numpyro.handlers.seed(rng_seed=1):
        draws = DirichletMultinomialNoise(
            DeterministicVariable("c", 5.0)
        ).sample("x", PROBS, jnp.array([10.0, 4.0]))
    np.testing.assert_allclose(draws.sum(-1), [10.0, 4.0])


def test_validate_and_repr():
    MultinomialNoise.validate()
    good = DirichletMultinomialNoise(DeterministicVariable("c", 2.0))
    good.validate()
    assert "DirichletMultinomialNoise(concentration_rv=" in repr(good)
    assert repr(MultinomialNoise()) == "MultinomialNoise()"
    with pytest.raises(ValueError, match="positive"):
        DirichletMultinomialNoise(DeterministicVariable("c", 0.0)).validate()
    with pytest.raises(TypeError, match="RandomVariable"):
        DirichletMultinomialNoise(2.0)
