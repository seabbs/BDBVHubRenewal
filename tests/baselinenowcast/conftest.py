"""Fixtures for the baselinenowcast port.

Importing the package enables float64.
"""

from __future__ import annotations

import pytest
from bnc_helpers import CASES, SMALL, Case, load_case


@pytest.fixture(params=CASES)
def case(request: pytest.FixtureRequest) -> Case:
    return load_case(request.param)


@pytest.fixture(params=SMALL)
def small(request: pytest.FixtureRequest) -> Case:
    return load_case(request.param)
