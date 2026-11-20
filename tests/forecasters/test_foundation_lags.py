"""The foundation model's lag count, from the data's reporting delay."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import ZoneData, hub_zones
from bdbvhubrenewal.forecasters import ForecastTask, get_model
from nowcasteval.backtest import BacktestConfig, make_tasks
from nowcastfoundation.forecaster import (
    LAG_QUANTILE,
    LAG_WINDOW,
    MAX_LAG,
    HubTriangle,
    lag_count,
)

D = LAG_WINDOW["daily"][1]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    health_zones: pd.DataFrame,
    tasks_json,
) -> ForecastTask:
    """The task test_foundation.py forecasts, zones included."""
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
    )[0]


def _triangle(last: int, share, final: float = 100.0) -> HubTriangle:
    """One place's daily triangle: `share(t, d)` of `final` at lag d."""
    n = last + 8
    partial = np.array(
        [[final * share(t, d) for d in range(D)] for t in range(n)]
    )[None]
    times = [str(t) for t in range(n)]
    return HubTriangle(times, np.full((1, n), final), partial, last)


def test_the_lag_count_is_the_delay_s_99_point_in_steps():
    assert LAG_QUANTILE == 0.99
    # A tenth more of each count at each lag: complete at lag 9.
    tri = _triangle(200, lambda _t, d: min(1.0, (d + 1) / 10))
    assert lag_count(tri, "daily") == 9


def test_only_steps_complete_at_the_cut_off_inform_it():
    lo, hi = LAG_WINDOW["daily"]
    last = 300

    def share(t, d):
        # Recent steps (and very old ones) report slowly; the window's
        # steps are complete at lag 4.
        if last - hi <= t <= last - lo:
            return min(1.0, (d + 1) / 5)
        return min(1.0, (d + 1) / 60)

    assert lag_count(_triangle(last, share), "daily") == 4


def test_releases_missing_at_some_lags_are_pooled_from_the_rest():
    # Weekly releases of daily data: each step is seen at every 7th lag.
    def share(t, d):
        return min(1.0, (d + 1) / 10) if (t + d) % 7 == 0 else np.nan

    assert lag_count(_triangle(200, share), "daily") == 9


def test_no_evidence_takes_as_many_lags_as_the_history_allows():
    tri = _triangle(200, lambda _t, _d: np.nan)
    assert lag_count(tri, "daily") == D
    # With two steps of context before the longest lag.
    short = _triangle(30, lambda _t, _d: np.nan)
    assert lag_count(short, "daily") == 29


def test_a_delay_not_reached_in_the_window_takes_the_history():
    tri = _triangle(200, lambda _t, d: min(0.9, (d + 1) / 10))
    assert lag_count(tri, "daily") == D


def test_the_estimate_is_capped_by_the_history():
    tri = _triangle(8, lambda _t, d: min(1.0, (d + 1) / 10))
    assert lag_count(tri, "daily") == 7


def _lags_seen(model, task: ForecastTask) -> set[str]:
    store: list = []

    def sample(groups, horizon, n, seed):  # noqa: ARG001
        store.append(groups)
        return [
            np.zeros((n, np.atleast_2d(g["target"]).shape[0], horizon))
            for g in groups
        ]

    dataclasses.replace(model, sample=sample).forecast(task)
    names = store[0][0]["future_covariates"]
    return {name.rsplit("_lag_", 1)[1] for name in names}


@pytest.mark.parametrize("source", ["daily", "weekly"])
def test_a_fixed_lag_count_is_used_as_given(source, task: ForecastTask):
    # The SI compares the estimate with the fixed cap of 28 days or 4 weeks.
    model = dataclasses.replace(
        get_model(f"fm_chronos_{source}_ragged"), lags=MAX_LAG[source]
    )
    assert len(_lags_seen(model, task)) == MAX_LAG[source]


def test_the_ragged_block_takes_the_estimated_lag_count(task: ForecastTask):
    model = get_model("fm_chronos_daily_ragged")
    assert model.lags is None
    got = len(_lags_seen(model, task))
    assert got == model.lag_count(task)
    assert 1 <= got <= D
