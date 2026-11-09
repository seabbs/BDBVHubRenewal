"""The random walk nowcast on known onsets, scaled up for missing onsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bdbvhubrenewal.forecasters.knownonset import (
    RandomWalkNowcastKnown,
    add_missing,
    missing_ratio,
)

from bdbvhubrenewal.data.snapshots import snapshot_arrays, weekly_sums
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.chain import Settings, fit_delay


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def run(task: ForecastTask) -> tuple[RandomWalkNowcastKnown, pd.DataFrame]:
    model = get_model("rwnc_known")
    assert isinstance(model, RandomWalkNowcastKnown)
    return model, model.forecast(task)


def test_ratio_is_own_where_known_counts_are_big_and_pooled_otherwise() -> None:
    # Country, then two provinces. Province 1 is big with 1.5 times its
    # known count in total; province 2 has too few known onsets and takes
    # the country's pool, its qualifying children (province 1).
    parent = np.array([-1, 0, 0])
    known = np.array([[40.0, 40.0], [40.0, 40.0], [1.0, 1.0]])
    total = np.array([[62.0, 62.0], [60.0, 60.0], [2.0, 2.0]])
    ratio = missing_ratio(known, total, parent, min_level=10.0)
    assert ratio[0] == pytest.approx(62 / 40)
    assert ratio[1] == pytest.approx(1.5)
    assert ratio[2] == pytest.approx(1.5)


def test_ratio_is_one_with_no_known_onsets() -> None:
    parent = np.array([-1, 0])
    zero = np.zeros((2, 3))
    ratio = missing_ratio(zero, zero + 5.0, parent, min_level=10.0)
    assert (ratio == 1.0).all()


def test_scale_up_recovers_the_total_on_average() -> None:
    rng = np.random.default_rng(1)
    known = np.full((20000, 2, 3), 30.0)
    known[:, 1] = 0.0
    ratio = np.array([1.5, 2.0])
    total = add_missing(rng, known, ratio)
    assert (total >= known).all()
    assert total[:, 0].mean() == pytest.approx(45.0, rel=0.01)
    # A place with no known onset gets no added cases.
    assert (total[:, 1] == 0).all()
    # The added cases are negative binomial, so they vary more than a
    # Poisson count with the same mean.
    added = total[:, 0] - known[:, 0]
    assert added.var() > added.mean()


def test_the_delay_is_fitted_to_known_onsets_only(
    run: tuple[RandomWalkNowcastKnown, pd.DataFrame],
    task: ForecastTask,
    daily: pd.DataFrame,
) -> None:
    model, _ = run
    origin = model.origin_at(task, errors=False)
    known = snapshot_arrays(daily, task.vintage, task.locations, ("observed",))
    total = snapshot_arrays(daily, task.vintage, task.locations)
    np.testing.assert_array_equal(origin.snap.counts, known.counts)
    o = len(known.releases) - 1
    total_fit = fit_delay(total, o, Settings())
    assert total_fit is not None
    assert not np.allclose(origin.fit.g, total_fit.g)


def test_samples_are_valid_and_scaled_to_the_total(
    run: tuple[RandomWalkNowcastKnown, pd.DataFrame],
    task: ForecastTask,
    daily: pd.DataFrame,
) -> None:
    model, samples = run
    check_samples(samples, task)
    assert (samples["value"] >= 0).all()
    assert (samples["value"] == samples["value"].round()).all()
    pd.testing.assert_frame_equal(
        samples, get_model("rwnc_known").forecast(task)
    )
    # Every data week is at least the total the hub already reports.
    total = snapshot_arrays(daily, task.vintage, task.locations)
    shown = weekly_sums(total, len(total.releases) - 1)
    lowest = samples.groupby(["location", "target_end_date"])["value"].min()
    for (loc, week), v in lowest.items():
        if week in total.week_ends:
            i = total.locations.index(loc)
            assert v >= shown[i, total.week_ends.index(week)]
    # The scale-up is reported per place, and is at least 1.
    shares = {r["location"]: r["missing_ratio"] for r in model.fit_diagnostics}
    assert set(shares) == set(task.locations)
    assert min(shares.values()) >= 1.0


def test_registered_and_delay_summary_is_of_the_total(
    run: tuple[RandomWalkNowcastKnown, pd.DataFrame],
) -> None:
    specs = {s.name: s for s in list_models()}
    assert not specs["rwnc_known"].baseline
    model, _ = run
    assert model.delay_summary is not None
    assert set(model.delay_summary["model"]) == {"rwnc_known"}
