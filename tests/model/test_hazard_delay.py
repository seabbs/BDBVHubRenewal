"""Interface tests for HazardDelay (its maths is in test_components)."""

import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent.temporal_processes import WeeklyTemporalProcess
from pyrenew.randomvariable import DistributionalVariable

from bdbvhubrenewal.model import HazardDelay, NonCentredRandomWalk


def _delay(hazard0_rv=None) -> HazardDelay:
    return HazardDelay(
        "delay",
        20,
        hazard0_rv or DistributionalVariable("h0", dist.Normal(-2.0, 1.0)),
        NonCentredRandomWalk(DeterministicVariable("smooth", 0.2)),
        WeeklyTemporalProcess(
            NonCentredRandomWalk(DeterministicVariable("sd", 0.1)),
            start_dow=6,
        ),
    )


def test_rows_are_pmfs():
    pmf = seed(_delay(), rng_seed=0)(n_timepoints=28, first_day_dow=6)
    assert pmf.shape == (28, 21)
    np.testing.assert_allclose(np.asarray(pmf).sum(axis=1), 1.0, rtol=1e-5)


def test_needs_a_random_hazard0_and_has_a_repr():
    with pytest.raises(TypeError, match="hazard0_rv"):
        _delay(hazard0_rv=-2.0)
    assert repr(_delay()).startswith("HazardDelay(name='delay', dmax=20")
