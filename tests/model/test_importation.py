"""Tests for the Importation component."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

from bdbvhubrenewal.model import Importation, mixing_matrix


def _kernel(n: int) -> np.ndarray:
    k = np.ones((n, n)) - np.eye(n)
    return k / k.sum(axis=1, keepdims=True)


def test_importation_is_the_mixing_matrix():
    f = jnp.array([0.5, 0.3, 0.2])
    imp = Importation(
        "mixing",
        DeterministicVariable("kernel", _kernel(3)),
        DeterministicVariable("importation", 0.1),
    )
    np.testing.assert_allclose(
        imp(fractions=f), mixing_matrix(_kernel(3), 0.1, f)
    )


def test_importation_samples_its_share():
    imp = Importation(
        "mixing",
        DeterministicVariable("kernel", _kernel(2)),
        DistributionalVariable("importation", dist.Beta(1, 100)),
    )
    w = seed(imp, rng_seed=0)(fractions=jnp.array([0.6, 0.4]))
    assert w.shape == (2, 2)
    assert float(w[0, 1]) > 0


def test_importation_needs_random_variables():
    with pytest.raises(TypeError, match="kernel_rv"):
        Importation(
            "mixing", _kernel(2), DeterministicVariable("importation", 0.1)
        )
    with pytest.raises(TypeError, match="share_rv"):
        Importation("mixing", DeterministicVariable("kernel", _kernel(2)), 0.1)


def test_importation_repr():
    imp = Importation(
        "mixing",
        DeterministicVariable("kernel", _kernel(2)),
        DeterministicVariable("importation", 0.1),
    )
    assert repr(imp).startswith("Importation(name='mixing'")
