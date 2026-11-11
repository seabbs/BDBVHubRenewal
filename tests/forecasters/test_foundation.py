"""The Chronos-2 forecasters, with a stand-in for Chronos-2."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    FoundationModel,
    ForecastTask,
    ModelUnavailableError,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.foundation import (
    LAYOUTS,
    N_DELAYS,
    chronos_sampler,
    delay_triangle,
    quantile_draws,
    report_series,
)

MODELS = ("fm_complete", "fm_delay_series", "fm_delay_cov")


def _constant(value: float):
    """A sampler whose every draw is `value` on the log1p scale."""

    def sample(inputs, horizon, n, seed):
        assert seed >= 0
        return [
            np.full((n, np.atleast_2d(x["target"]).shape[0], horizon), value)
            for x in inputs
        ]

    return sample


def _recording(store: list):
    """A sampler that keeps its inputs and draws zeros."""

    def sample(inputs, horizon, n, seed):
        store.append((inputs, horizon))
        return _constant(0.0)(inputs, horizon, n, seed)

    return sample


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


def _reported(task: ForecastTask, out: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(out[["location", "target_end_date"]])
    return (
        task.target.set_index(["location", "target_end_date"])["total"]
        .reindex(keys)
        .fillna(0.0)
        .to_numpy()
    )


def test_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in MODELS:
        assert specs[name].fitted
        assert not specs[name].baseline
        model = get_model(name)
        assert isinstance(model, FoundationModel)
        assert model.layout in LAYOUTS


def test_triangle_classes_and_complete_week() -> None:
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "A", "B", "B", "C"],
            "onset_week": ["2026-10-03"] * 3 + ["2026-10-10"] * 3,
            "delay": [-2, 6, 7, 20, 7 * N_DELAYS, 3],
        }
    )
    weeks = ("2026-10-03", "2026-10-10", "2026-10-17")
    tri = delay_triangle(delays, "2026-10-24", ("A", "B"), weeks)
    assert tri.counts.shape == (2, 3, N_DELAYS)
    # -2 and 6 days are class 0 and 7 days class 1; 7K days is dropped
    # and place C is not asked for.
    assert tri.counts[0, 0, :2].tolist() == [2.0, 1.0]
    assert tri.counts[1, 1, 2] == 1.0
    assert tri.counts.sum() == 4.0
    # Week ending 10-17 has its last delay-0 report day on 10-23.
    assert weeks[tri.last] == "2026-10-17"
    tri = delay_triangle(delays, "2026-10-22", ("A", "B"), weeks)
    assert weeks[tri.last] == "2026-10-10"


def test_report_series_puts_each_class_on_the_report_clock() -> None:
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "A"],
            "onset_week": ["2026-10-03", "2026-10-03", "2026-10-10"],
            "delay": [0, 8, 1],
        }
    )
    weeks = ("2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24")
    tri = delay_triangle(delays, "2026-10-31", ("A",), weeks, 2)
    series = report_series(tri)
    assert series.shape == (1, 2, tri.last + 1)
    # Class 0 at its onset week, class 1 one week later.
    assert series[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert series[0, 1].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_quantile_draws_invert_the_quantile_function() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([[[0.0, 1.0, 3.0]], [[2.0, 2.0, 2.0]]])
    u = np.array([[0.5], [0.3]])
    out = quantile_draws(levels, values, u)
    assert out.shape == (2, 1)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(2.0)
    # Outside the levels the draw is the outermost quantile.
    low = quantile_draws(levels, values[:1], np.array([[0.01]]))
    assert low[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("name", MODELS)
def test_draws_are_counts_at_least_the_reported(
    name: str, task: ForecastTask
) -> None:
    reported = None
    for value in (-50.0, 0.0, 3.0):
        model = dataclasses.replace(get_model(name), sample=_constant(value))
        out = model.forecast(task)
        check_samples(out, task)
        values = out["value"].to_numpy()
        assert (values >= 0).all()
        assert np.array_equal(values, np.round(values))
        reported = _reported(task, out)
        assert (values >= reported).all()
    # Draws above the reported counts raise the forecast past them.
    assert (values > reported).any()


@pytest.mark.parametrize("name", MODELS)
def test_low_draws_give_the_reported_counts(
    name: str, task: ForecastTask
) -> None:
    model = dataclasses.replace(get_model(name), sample=_constant(-50.0))
    out = model.forecast(task)
    assert np.array_equal(out["value"].to_numpy(), _reported(task, out))


@pytest.mark.parametrize("name", MODELS)
def test_maps_final_counts_through_a_delay(
    name: str, task: ForecastTask
) -> None:
    model = dataclasses.replace(get_model(name), sample=_constant(0.0))
    model.forecast(task)
    summary = model.delay_summary
    assert summary is not None
    assert set(summary["model"]) == {name}
    assert set(summary["quantity"]) == {"reported_share"}


def test_delay_cov_masks_cells_not_yet_reported(task: ForecastTask) -> None:
    seen: list = []
    model = dataclasses.replace(
        get_model("fm_delay_cov"), sample=_recording(seen)
    )
    model.forecast(task)
    inputs, horizon = seen[0]
    first = inputs[0]
    past = np.stack(list(first["past_covariates"].values()))
    future = np.stack(list(first["future_covariates"].values()))
    assert past.shape == (N_DELAYS, first["target"].shape[-1])
    assert future.shape == (N_DELAYS, horizon)
    # The context is complete; the future holds the reported cells of
    # each delay class, fewer for longer delays, then nothing.
    assert not np.isnan(past).any()
    known = (~np.isnan(future)).sum(axis=1)
    assert (np.diff(known) <= 0).all()
    assert known[0] == N_DELAYS - 1
    assert known[-1] == 0
    assert np.isnan(future[:, N_DELAYS - 1 :]).all()


def test_complete_forecasts_from_the_last_complete_week(
    task: ForecastTask,
) -> None:
    complete: list = []
    cov: list = []
    dataclasses.replace(
        get_model("fm_complete"), sample=_recording(complete)
    ).forecast(task)
    dataclasses.replace(
        get_model("fm_delay_cov"), sample=_recording(cov)
    ).forecast(task)
    (a, h_a), (b, h_b) = complete[0], cov[0]
    assert h_a == h_b
    assert len(a) == len(b) == len(task.locations)
    assert "past_covariates" not in a[0]
    assert np.array_equal(a[0]["target"], b[0]["target"])


@pytest.mark.parametrize("name", ("fm_delay_series", "fm_delay_cov"))
def test_no_delay_data_is_unavailable(name: str, task: ForecastTask) -> None:
    bare = dataclasses.replace(task, delays=None)
    model = dataclasses.replace(get_model(name), sample=_constant(0.0))
    with pytest.raises(ModelUnavailableError, match="delay line data"):
        model.forecast(bare)


@pytest.mark.skipif(
    importlib.util.find_spec("chronos") is not None,
    reason="Chronos-2 is installed",
)
def test_chronos_missing_is_unavailable() -> None:
    with pytest.raises(ModelUnavailableError, match="extra chronos"):
        chronos_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0)
