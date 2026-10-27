from __future__ import annotations

import os
from pathlib import Path

import pytest
from hv_fixtures import SNAPSHOT

from bdbv_hub_renewal import hub_validate as hv


@pytest.fixture(scope="session")
def hub_path() -> Path:
    return Path(os.environ.get("HUB_PATH", SNAPSHOT))


@pytest.fixture(scope="session")
def hub(hub_path: Path) -> hv.Hub:
    return hv.Hub.load(hub_path)
