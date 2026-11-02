"""Fixtures for the baselinenowcast port.

Tests run in float64.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import pytest
from bnc_helpers import CASES, SMALL, Case, load_case


@pytest.fixture(autouse=True)
def _x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


@pytest.fixture(params=CASES)
def case(request: pytest.FixtureRequest) -> Case:
    return load_case(request.param)


@pytest.fixture(params=SMALL)
def small(request: pytest.FixtureRequest) -> Case:
    return load_case(request.param)
