"""Chronos-2's trajectories joined by the fitted coupling (#356)."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import ZoneData, hub_zones
from bdbvhubrenewal.forecasters import ForecastTask, get_model
from nowcasteval.backtest import BacktestConfig, make_tasks
from nowcastfoundation import forecaster as foundation
from nowcastutils import trajectories

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


class _Pipeline:
    """Quantiles from each group's own level, spread by step; counts calls."""

    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __init__(self) -> None:
        self.groups = 0

    def predict_quantiles(
        self,
        inputs,
        prediction_length,
        quantile_levels,
        batch_size=256,  # noqa: ARG002
    ):
        import torch

        self.groups += len(inputs)
        out = []
        for g in inputs:
            target = np.atleast_2d(g["target"])
            own = np.nan_to_num(target[:, -7:]).mean(axis=1) + 1.0
            step = 1.0 + 0.1 * np.arange(prediction_length)
            z = np.asarray(quantile_levels) - 0.5
            q = own[:, None, None] * step[None, :, None] * (1 + z)[None, None]
            out.append(torch.tensor(q, dtype=torch.float32))
        return out, None


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    health_zones: pd.DataFrame,
    tasks_json,
) -> ForecastTask:
    """The latest fixture release's task, so earlier releases have matured."""
    latest = target["vintage"].max()
    rows = target.loc[target["vintage"] == latest]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
    )[0]


@pytest.fixture
def couplings(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record each `couple` call's method and fit."""
    seen: list = []
    real = trajectories.couple

    def couple(quantiles, levels, method, fit, n_samples, seed):
        seen.append((method, fit))
        return real(quantiles, levels, method, fit, n_samples, seed)

    monkeypatch.setattr(foundation, "couple", couple)
    return seen


def test_the_models_couple_by_the_fitted_gaussian_copula():
    assert get_model("fm_chronos_daily_ragged_ft").coupling == "gaussian"
    assert set(trajectories.METHODS) >= {"independent", "comonotone"}


def test_the_sampler_draws_through_couple(
    monkeypatch: pytest.MonkeyPatch, couplings: list
):
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Pipeline())
    group = {"target": np.full((2, 20), 5.0)}
    draws = foundation.chronos_sampler(
        [group], 6, 200, 1, coupling={"method": "comonotone"}
    )[0]
    assert couplings[-1][0] == "comonotone"
    rank = draws.argsort(axis=0).argsort(axis=0)
    # Comonotone: each trajectory keeps its rank across a series' steps.
    assert (rank[:, 0, 0] == rank[:, 0, -1]).all()
    foundation.chronos_sampler([group], 6, 200, 1)
    assert couplings[-1][0] == "independent"


def test_the_gaussian_fit_reads_the_model_s_earlier_forecasts(
    monkeypatch: pytest.MonkeyPatch, couplings: list, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    get_model("fm_chronos_daily_ragged").forecast(task)
    # The task's block and the same block at earlier releases.
    assert pipe.groups > 1
    method, fit = couplings[-1]
    assert method == "gaussian"
    assert {"rho", "s", "n_origins"} <= set(fit)
    assert fit["n_origins"] >= 1


@pytest.mark.parametrize("method", ["independent", "comonotone"])
def test_the_si_couplings_need_no_earlier_forecasts(
    monkeypatch: pytest.MonkeyPatch,
    couplings: list,
    task: ForecastTask,
    method: str,
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    model = dataclasses.replace(
        get_model("fm_chronos_daily_ragged"), coupling=method
    )
    model.forecast(task)
    assert pipe.groups == 1
    assert couplings[-1] == (method, {"method": method})


def test_prefetch_holds_the_earlier_forecasts_too(
    monkeypatch: pytest.MonkeyPatch, task: ForecastTask
):
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    model = get_model("fm_chronos_daily_ragged")
    alone = model.forecast(task)
    ready = foundation.prefetch(model, [task])
    pipe.groups = 0
    pd.testing.assert_frame_equal(ready.forecast(task), alone)
    assert pipe.groups == 0


def test_an_unknown_coupling_is_an_error(task: ForecastTask):
    model = dataclasses.replace(
        get_model("fm_chronos_daily_ragged"), coupling="schaake"
    )
    with pytest.raises(ValueError, match="unknown"):
        model.forecast(task)
