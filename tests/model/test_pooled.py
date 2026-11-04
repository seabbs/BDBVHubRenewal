"""Tests for the pooled group values."""

import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

from bdbvhubrenewal.model import PooledInitialValue
from bdbvhubrenewal.model.pooled import sum_to_zero_basis


def _pooled(centred: bool = True) -> PooledInitialValue:
    return PooledInitialValue(
        "v",
        4,
        DistributionalVariable("v_mean", dist.Normal(1.0, 0.5)),
        DistributionalVariable("v_sd", dist.HalfNormal(0.3)),
        centred=centred,
    )


@pytest.mark.parametrize("centred", [True, False])
def test_offsets_sum_to_zero_around_the_mean(centred):
    with trace() as tr:
        v = seed(_pooled(centred), rng_seed=1)()
    assert v.shape == (4,)
    np.testing.assert_allclose(
        float(v.mean()), float(tr["v_mean"]["value"]), atol=1e-6
    )
    assert ("v_offsets" if centred else "v_z") in tr


def test_one_group_has_no_spread():
    one = PooledInitialValue(
        "v",
        1,
        DeterministicVariable("v_mean", 0.2),
        DistributionalVariable("v_sd", dist.HalfNormal(0.3)),
    )
    with trace() as tr:
        v = seed(one, rng_seed=0)()
    np.testing.assert_allclose(v, [0.2])
    assert "v_sd" not in tr


def test_basis_is_orthonormal_and_sums_to_zero():
    q = sum_to_zero_basis(5)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(q.sum(axis=0), 0.0, atol=1e-12)


def test_needs_random_variables_and_repr():
    with pytest.raises(TypeError, match="mean_rv"):
        PooledInitialValue("v", 4, 1.0, DeterministicVariable("s", 0.3))
    assert repr(_pooled()).startswith("PooledInitialValue(name='v'")
