"""The random walk models' samples are pinned, and a rerun compiles nothing.

The hashes are of each model's samples at the 2026-10-24 vintage with
seed 3, from the eager code before the walks were compiled as whole
programs. Compiling must not change a single draw.

pprandomwalknowcast is not pinned: its NUTS fit is compiled for the CPU it
runs on, so its draws differ between machines (archie and the CI runners).
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest
from jax import monitoring

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask, get_model

PINNED = {
    "randomwalk": "b5aecfce6d5f7a9e",
    "randomwalknowcast": "8ac351f1d9ac8546",
}

_COMPILES: list[float] = []


def _count(event: str, duration: float, **kwargs: object) -> None:
    del kwargs
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES.append(duration)


monitoring.register_event_duration_secs_listener(_count)


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture
def compiles() -> list[float]:
    _COMPILES.clear()
    return _COMPILES


def _digest(samples: pd.DataFrame) -> str:
    rows = samples.sort_values(["location", "target_end_date", "sample"])
    values = rows["value"].to_numpy().astype("float64").tobytes()
    return hashlib.sha256(values).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_samples_are_unchanged(name: str, task: ForecastTask) -> None:
    assert _digest(get_model(name).forecast(task)) == PINNED[name]


# pprandomwalknowcast's pooled fit compiles its own programs each time.
@pytest.mark.parametrize("name", ["randomwalk", "randomwalknowcast"])
def test_a_second_forecast_of_the_same_shape_compiles_nothing(
    name: str, task: ForecastTask, compiles: list[float]
) -> None:
    model = get_model(name)
    model.forecast(task)
    compiles.clear()
    model.forecast(task)
    assert compiles == []
