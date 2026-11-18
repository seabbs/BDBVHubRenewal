"""Forecasts as quantiles: a model's own where it has them, else draws."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.evaluation.marginals import forecast_quantiles
from bdbvhubrenewal.forecasters import ForecastTask, foundation, get_model

LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


class _FakePipeline:
    """Chronos-2's native levels, with quantiles 1,000 times the level."""

    quantiles = (
        0.01,
        0.05,
        *np.round(np.arange(0.1, 0.91, 0.1), 2),
        0.95,
        0.99,
    )

    def predict_quantiles(
        self, groups, prediction_length, quantile_levels, **_kw
    ):
        import torch

        out = []
        for g in groups:
            n_var = np.atleast_2d(g["target"]).shape[0]
            level = 1000 * torch.tensor(quantile_levels, dtype=torch.float32)
            q = level[None, None, :].expand(n_var, prediction_length, -1)
            out.append(q)
        return out, None


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )[0]


@pytest.fixture
def fake_chronos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _FakePipeline())


@needs_torch
@pytest.mark.usefixtures("fake_chronos")
def test_a_quantile_model_s_quantiles_do_not_depend_on_the_seed(
    task: ForecastTask,
) -> None:
    model = get_model("fm_chronos_weekly_ragged")
    one = forecast_quantiles(model, task, LEVELS, seed=1)
    two = forecast_quantiles(model, task, LEVELS, seed=2)
    pd.testing.assert_frame_equal(one, two)
    assert sorted(one["quantile"].unique()) == list(LEVELS)


@needs_torch
@pytest.mark.usefixtures("fake_chronos")
def test_native_quantiles_are_interpolated_to_the_levels(
    task: ForecastTask,
) -> None:
    model = get_model("fm_chronos_weekly_ragged")
    out = forecast_quantiles(model, task, LEVELS, seed=1)
    # Steps after the reported counts: no floor lifts them, so each is
    # the native quantile read at the level (0.025 between 0.01 and 0.05).
    last = out.loc[out["target_end_date"] == out["target_end_date"].max()]
    at = last.groupby("quantile")["value"].min()
    assert at[0.025] == 25.0
    assert at[0.5] == 500.0
    assert at[0.975] == 975.0


@needs_torch
@pytest.mark.usefixtures("fake_chronos")
def test_sums_use_the_same_draws_for_every_model(
    task: ForecastTask,
) -> None:
    model = get_model("fm_chronos_daily_ragged")
    other = dataclasses.replace(model, name="fm_chronos_other")
    one = forecast_quantiles(model, task, LEVELS, n_draws=200, seed=4)
    two = forecast_quantiles(other, task, LEVELS, n_draws=200, seed=4)
    np.testing.assert_array_equal(one["value"], two["value"])


@needs_torch
@pytest.mark.usefixtures("fake_chronos")
def test_daily_sums_come_from_draws_seeded_by_the_seed(
    task: ForecastTask,
) -> None:
    model = get_model("fm_chronos_daily_ragged")
    one = forecast_quantiles(model, task, LEVELS, n_draws=200, seed=4)
    again = forecast_quantiles(model, task, LEVELS, n_draws=200, seed=4)
    other = forecast_quantiles(model, task, LEVELS, n_draws=200, seed=5)
    pd.testing.assert_frame_equal(one, again)
    assert not np.array_equal(one["value"], other["value"])


def test_a_sample_model_gives_the_quantiles_of_its_draws(
    task: ForecastTask,
) -> None:
    from bdbvhubrenewal.evaluation.backtest import run_forecasts

    model = get_model("randomwalk")
    out = forecast_quantiles(model, task, LEVELS, n_draws=300, seed=7)
    drawn = run_forecasts(
        [model], [dataclasses.replace(task, n_samples=300, seed=7)]
    )
    keys = [c for c in drawn.columns if c not in ("sample", "value")]
    want = (
        drawn.groupby(keys, dropna=False)["value"]
        .quantile(np.array(LEVELS))
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), want[list(out.columns)]
    )


@needs_torch
@pytest.mark.usefixtures("fake_chronos")
def test_an_unavailable_quantile_model_has_no_rows(task: ForecastTask) -> None:
    no_history = dataclasses.replace(task, target_history=None)
    model = get_model("fm_chronos_weekly_ragged")
    assert forecast_quantiles(model, no_history, LEVELS).empty
