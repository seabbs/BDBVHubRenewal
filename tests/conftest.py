from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbv_forecasts.backtest import BacktestConfig
from bdbv_forecasts.data import load_delays, load_target

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def target() -> pd.DataFrame:
    return load_target(DATA)


@pytest.fixture(scope="session")
def delays() -> pd.DataFrame:
    return load_delays(DATA)


@pytest.fixture(scope="session")
def tasks_json() -> Path:
    return DATA / "hub-config" / "tasks.json"


@pytest.fixture(scope="session")
def schema_json() -> Path:
    return DATA / "hub-config" / "model-metadata-schema.json"


@pytest.fixture
def config(tmp_path: Path) -> BacktestConfig:
    return BacktestConfig(
        n_samples=100,
        seed=1,
        development_until="2026-10-10",
        model_choice=tmp_path / "model-choice.yml",
        reconstructed_dates=("2026-08-29", "2026-10-10", "2026-10-17"),
        simulated_dates=("2026-08-01", "2026-09-05"),
    )
