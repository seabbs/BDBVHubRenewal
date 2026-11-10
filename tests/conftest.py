from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily, load_delays, load_target, tidy
from bdbvhubrenewal.evaluation.backtest import BacktestConfig
from bdbvhubrenewal.forecasters import ForecastTask

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"
"""A frozen copy of the tidy hub data. It is never refreshed: the
workflows fetch the live data, and the tests read only this copy."""

FIXTURE_VINTAGES = (
    "2026-08-18",
    "2026-10-09",
    "2026-10-17",
    "2026-10-24",
    "2026-11-01",
    "2026-11-08",
)
"""The real vintages in the fixture, oldest first."""

# The loaders and the CLI read `data/` by default. Point both at the
# fixture before any test builds a parser or loads a table.
os.environ["BDBV_DATA_DIR"] = str(DATA)
tidy._DEFAULT_DATA_DIR = DATA


@pytest.fixture(autouse=True)
def _one_province_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit per-province variants in this process, where stand-ins reach.

    Tests that check the worker processes set the variable themselves.
    """
    monkeypatch.setenv("BDBV_PROVINCE_WORKERS", "1")


@pytest.fixture(scope="session")
def target() -> pd.DataFrame:
    return load_target(DATA)


@pytest.fixture(scope="session")
def vintages() -> list[str]:
    """`FIXTURE_VINTAGES`, for tests that pin expectations to them."""
    return list(FIXTURE_VINTAGES)


@pytest.fixture(scope="session")
def delays() -> pd.DataFrame:
    return load_delays(DATA)


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    return load_daily(DATA)


@pytest.fixture(scope="session")
def tasks_json() -> Path:
    return DATA / "hub-config" / "tasks.json"


@pytest.fixture(scope="session")
def schema_json() -> Path:
    return DATA / "hub-config" / "model-metadata-schema.json"


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(
        n_samples=100,
        seed=1,
        simulated_dates=("2026-08-01", "2026-09-05"),
    )


@pytest.fixture(scope="session")
def daily_zones() -> pd.DataFrame:
    from bdbvhubrenewal.data import load_daily_zones

    return load_daily_zones(DATA)


class FlatModel:
    """Latest reported week carried forward, a stand-in for harness tests.

    The random walk models need release snapshots and cannot forecast
    most origins, so tests of the backtest, scoring and report machinery
    run this model instead.
    """

    name = "flat"
    supports_mask = False

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        rng = task.rng()
        weeks = list(task.target_end_dates)
        frames = []
        for loc in task.locations:
            series = (
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .sort_index()
            )
            level = float(series.iloc[-1]) if len(series) else 0.0
            values = np.column_stack(
                [
                    np.full(task.n_samples, float(series[w]))
                    if w in series.index
                    else rng.poisson(level, task.n_samples).astype(float)
                    for w in weeks
                ]
            )
            frames.append(
                pd.DataFrame(
                    {
                        "location": loc,
                        "target_end_date": np.tile(weeks, task.n_samples),
                        "sample": np.repeat(
                            np.arange(1, task.n_samples + 1), len(weeks)
                        ),
                        "value": values.reshape(-1),
                    }
                )
            )
        final = pd.concat(frames, ignore_index=True)
        if not task.report_by:
            return final
        # Counts reported by a later date are the final counts to then.
        later = [
            final.loc[final["target_end_date"] <= date].assign(as_of=date)
            for date in task.report_by
        ]
        return pd.concat(
            [final.assign(as_of=np.nan), *later], ignore_index=True
        )


@pytest.fixture
def flat_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register `FlatModel` as `flat` for one test and return its name."""
    from bdbvhubrenewal.forecasters import base

    monkeypatch.setitem(
        base._REGISTRY,
        "flat",
        base.ModelSpec("flat", FlatModel, "Test stand-in."),
    )
    return "flat"
