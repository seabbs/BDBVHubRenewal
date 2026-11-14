"""Fixtures for the baselinenowcast port.

The oracle tests run in float64, as the R package does.
"""

from __future__ import annotations

import jax
import pytest
from bnc_helpers import CASES, SMALL, Case, load_case


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


@pytest.fixture(params=CASES)
def case(request: pytest.FixtureRequest) -> Case:
    return load_case(request.param)


@pytest.fixture(params=SMALL)
def small(request: pytest.FixtureRequest) -> Case:
    return load_case(request.param)
