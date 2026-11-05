"""The Toto 1.0 delay forecaster, with a stand-in for Toto."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    FMDelay,
    ForecastTask,
    ModelUnavailableError,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.foundation import (
    N_DELAYS,
    delay_triangle,
    report_series,
    toto_sampler,
)


def _constant(value: float):
    def sample(series: np.ndarray, horizon: int, n: int, seed: int):
        assert seed >= 0
        return np.full((n, series.shape[0], horizon), value)

    return sample


@pytest.fixture(scope="module")
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    return make_tasks(rows, delays, BacktestConfig(seed=3))[0]


def test_registered_as_a_fitted_model() -> None:
    spec = {s.name: s for s in list_models()}["fm_delay"]
    assert spec.fitted
    assert not spec.baseline
    assert isinstance(get_model("fm_delay"), FMDelay)


def test_triangle_classes_and_complete_week() -> None:
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "A", "B", "B", "C"],
            "onset_week": ["2026-10-03"] * 3 + ["2026-10-10"] * 3,
            "delay": [-2, 6, 7, 20, 7 * N_DELAYS, 3],
        }
    )
    tri = delay_triangle(delays, "2026-10-24", ("A", "B"), "2026-10-17")
    assert tri.weeks == ("2026-10-03", "2026-10-10", "2026-10-17")
    assert tri.counts.shape == (2, 3, N_DELAYS)
    # -2 and 6 days are class 0 and 7 days class 1; 7K days is dropped
    # and place C is not asked for.
    assert tri.counts[0, 0, :2].tolist() == [2.0, 1.0]
    assert tri.counts[1, 1, 2] == 1.0
    assert tri.counts.sum() == 4.0
    # Week ending 10-17 has its last delay-0 report day on 10-23.
    assert tri.weeks[tri.last] == "2026-10-17"
    tri = delay_triangle(delays, "2026-10-22", ("A", "B"), "2026-10-17")
    assert tri.weeks[tri.last] == "2026-10-10"


def test_report_series_puts_each_class_on_the_report_clock() -> None:
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "A"],
            "onset_week": ["2026-10-03", "2026-10-03", "2026-10-10"],
            "delay": [0, 8, 1],
        }
    )
    tri = delay_triangle(delays, "2026-10-31", ("A",), "2026-10-24", 2)
    series = report_series(tri)
    assert series.shape == (2, tri.last + 1)
    # Class 0 at its onset week, class 1 one week later.
    assert series[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert series[1].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_zero_draws_give_the_reported_totals(task: ForecastTask) -> None:
    out = FMDelay(sample=_constant(0.0)).forecast(task)
    check_samples(out, task)
    reported = (
        task.target.set_index(["location", "target_end_date"])["total"]
        .reindex(pd.MultiIndex.from_frame(out[["location", "target_end_date"]]))
        .fillna(0.0)
        .to_numpy()
    )
    assert np.array_equal(out["value"].to_numpy(), reported)


def test_draws_add_new_reports_of_incomplete_cells(task: ForecastTask) -> None:
    out = FMDelay(sample=_constant(5.0)).forecast(task)
    check_samples(out, task)
    last = max(task.target_end_dates)
    # A week after the data has every cell to come: 5 for each class.
    after = out.loc[out["target_end_date"] == last, "value"]
    assert (after == 5.0 * N_DELAYS).all()
    # Every sample is the same draw here, so trajectories agree.
    assert (
        out.groupby(["location", "target_end_date"])["value"]
        .nunique()
        .eq(1)
        .all()
    )
    # No week falls below what is reported.
    zero = FMDelay(sample=_constant(0.0)).forecast(task)
    assert (out["value"].to_numpy() >= zero["value"].to_numpy()).all()


def test_no_delay_data_is_unavailable(task: ForecastTask) -> None:
    bare = dataclasses.replace(task, delays=None)
    with pytest.raises(ModelUnavailableError, match="delay line data"):
        FMDelay(sample=_constant(0.0)).forecast(bare)


@pytest.mark.skipif(
    importlib.util.find_spec("toto") is not None, reason="Toto is installed"
)
def test_toto_missing_is_unavailable() -> None:
    with pytest.raises(ModelUnavailableError, match="extra toto"):
        toto_sampler(np.zeros((2, 5)), 3, 10, 0)
