import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from bdbvhubrenewal.model.distributions import ObservedCellsDirichletMultinomial


@pytest.fixture
def counts():
    rng = np.random.default_rng(0)
    x = rng.poisson(rng.gamma(0.3, 5.0, (6, 9))).astype(float)
    x[2] = 0.0  # a row with no counts
    return jnp.asarray(x)


def test_matches_dirichlet_multinomial(counts):
    rng = np.random.default_rng(1)
    log_alpha = jnp.asarray(rng.normal(0.0, 1.5, counts.shape))

    def log_prob(cls, log_alpha):
        d = cls(jnp.exp(log_alpha), total_count=counts.sum(-1))
        return d.log_prob(counts)

    ref = log_prob(dist.DirichletMultinomial, log_alpha)
    new = log_prob(ObservedCellsDirichletMultinomial, log_alpha)
    np.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-6)
    assert new[2] == 0.0

    def grad(cls):
        return jax.grad(lambda a: log_prob(cls, a).sum())(log_alpha)

    np.testing.assert_allclose(
        grad(ObservedCellsDirichletMultinomial),
        grad(dist.DirichletMultinomial),
        rtol=1e-5,
        atol=1e-6,
    )
