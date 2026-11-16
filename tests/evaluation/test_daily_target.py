"""The daily target: each day's count by Meldedatum, not summed to weeks.

The checkpoint scores (`scripts/checkpoint_scores.py`) score a second
target beside the hub's 7-day one. These tests pin the source side: the
daily 40-day truth, the models' daily forecasts, and that every 7-day
output is unchanged (`tests/fixtures/pins`, written by `make_pins.py` on
the code before the daily target).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.germany import hub_target, read_triangle
from bdbvhubrenewal.evaluation.germany import (
    DAILY_HORIZONS,
    QUANTILE_LEVELS,
    germany_daily_forecasts,
    germany_forecasts,
    with_truth,
)
from bdbvhubrenewal.evaluation.marginals import daily_quantiles
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.forecasters import NATIONAL, get_model

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "tests" / "fixtures" / "pins"
GERMANY = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


@pytest.fixture(scope="module")
def pins():
    spec = importlib.util.spec_from_file_location(
        "make_pins", PINS / "make_pins.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stand_in(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    from bdbvhubrenewal.forecasters import foundation

    monkeypatch.setattr(foundation, "_chronos", lambda *_: pins.Pipeline())


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(GERMANY)


# Daily truth -----------------------------------------------------------


def _toy() -> pd.DataFrame:
    """Two Meldedaten of Germany, each revised over its report dates."""
    rows = [
        ("2021-10-01", 5, "2021-10-01"),
        ("2021-10-01", 8, "2021-10-03"),
        ("2021-10-01", 9, "2021-11-10"),
        ("2021-10-01", 12, "2021-11-20"),
        ("2021-10-02", 4, "2021-10-02"),
        ("2021-10-02", 6, "2021-11-11"),
    ]
    return pd.DataFrame(
        {
            "reference_date": [r[0] for r in rows],
            "location": "DE",
            "age_group": "00+",
            "confirm": [r[1] for r in rows],
            "report_date": [r[2] for r in rows],
        }
    )


def test_a_one_day_window_gives_each_day_s_count_by_report_date() -> None:
    out = hub_target(_toy(), ["2021-10-01", "2021-10-02"], window=1)
    by = out.set_index(["target_end_date", "vintage"])["total"]
    # Each day alone, not summed with the days before it.
    assert by[("2021-10-01", "2021-10-01")] == 5
    assert by[("2021-10-01", "2021-10-02")] == 5
    assert by[("2021-10-01", "2021-10-03")] == 8
    assert by[("2021-10-02", "2021-10-02")] == 4
    assert by[("2021-10-02", "2021-11-10")] == 4


def test_daily_truth_is_the_first_report_40_days_after_each_day() -> None:
    days = ["2021-10-01", "2021-10-02"]
    truth = rolling_truth(hub_target(_toy(), days, window=1), 40)
    truth = truth.set_index("target_end_date")
    # 2021-10-01 + 40 days is 2021-11-10, which reports 9 (not 12).
    assert truth.loc["2021-10-01", "truth"] == 9
    assert truth.loc["2021-10-01", "truth_vintage"] == "2021-11-10"
    # 2021-10-02 + 40 days is 2021-11-11, which reports 6.
    assert truth.loc["2021-10-02", "truth"] == 6
    assert truth.loc["2021-10-02", "truth_vintage"] == "2021-11-11"


def test_the_seven_day_sum_is_unchanged_by_the_window(
    triangle: pd.DataFrame,
) -> None:
    ends = ["2021-10-01", "2021-10-08"]
    pd.testing.assert_frame_equal(
        hub_target(triangle, ends), hub_target(triangle, ends, window=7)
    )


def test_with_truth_takes_the_window(triangle: pd.DataFrame) -> None:
    fc = pd.DataFrame(
        {
            "location": ["DE"],
            "target_end_date": ["2021-09-01"],
            "value": [1.0],
        }
    )
    daily = with_truth(fc, triangle, 10, window=1)
    weekly = with_truth(fc, triangle, 10)
    want = rolling_truth(hub_target(triangle, ["2021-09-01"], window=1), 10)
    want = want.loc[want["location"] == "DE"]
    assert daily["truth"].item() == want["truth"].item()
    assert daily["truth"].item() < weekly["truth"].item()


# The 7-day output is unchanged ------------------------------------------


@needs_torch
@pytest.mark.usefixtures("stand_in")
def test_seven_day_draws_are_unchanged_by_the_daily_target(pins) -> None:
    want = pd.read_csv(PINS / "bdbv-7day.csv.gz")
    frames = []
    for v, task in pins.bdbv_tasks().items():
        for name in want["model"].unique():
            got = get_model(name).forecast(task)
            frames.append(got.assign(model=name, vintage=v))
    got = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(
        got, want, check_dtype=False, check_exact=True
    )


@needs_torch
@pytest.mark.usefixtures("stand_in")
def test_german_seven_day_quantiles_are_unchanged(
    triangle: pd.DataFrame,
) -> None:
    want = pd.read_csv(PINS / "germany-7day.csv.gz")
    got = germany_forecasts(
        list(want["model"].unique()),
        triangle,
        ["2021-10-04"],
        n_samples=40,
        seed=1,
    )
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want, check_dtype=False, check_exact=True
    )


# Daily forecasts ---------------------------------------------------------


def _weekly_of(daily: pd.DataFrame) -> pd.DataFrame:
    from bdbvhubrenewal.data.dates import epiweek_end

    week = daily["target_end_date"].map(lambda d: epiweek_end(d).isoformat())
    return (
        daily.assign(target_end_date=week)
        .groupby(["location", "target_end_date", "sample"], as_index=False)[
            "value"
        ]
        .sum()
    )


def test_random_walk_nowcast_days_sum_to_its_weeks(pins) -> None:
    model = get_model("randomwalknowcast")
    for task in pins.bdbv_tasks().values():
        weekly = model.forecast(task)
        daily = model.daily(task)
        assert set(daily.columns) == {
            "location",
            "target_end_date",
            "sample",
            "value",
        }
        assert set(daily["location"]) == set(task.locations)
        values = daily["value"].to_numpy()
        assert (values >= 0).all()
        assert np.array_equal(values, np.round(values))
        # Each day is a day of a target week up to the cut-off week.
        summed = _weekly_of(daily)
        both = summed.merge(
            weekly, on=["location", "target_end_date", "sample"]
        )
        assert len(both) == len(summed)
        np.testing.assert_array_equal(both["value_x"], both["value_y"])
        assert set(summed["target_end_date"]) <= set(task.target_end_dates)
        assert len(set(summed["target_end_date"])) >= 4


def test_random_walk_nowcast_days_are_at_least_their_reported_count(
    pins, daily: pd.DataFrame
) -> None:
    task = pins.bdbv_tasks()["2026-10-24"]
    out = get_model("randomwalknowcast").daily(task)
    seen = daily.loc[daily["vintage"] == "2026-10-24"]
    seen = seen.assign(reported=seen["observed"] + seen["imputed"]).rename(
        columns={"onset_date": "target_end_date"}
    )
    both = out.merge(
        seen[["location", "target_end_date", "reported"]],
        on=["location", "target_end_date"],
    )
    assert not both.empty
    assert (both["value"] >= both["reported"]).all()


def test_random_walk_nowcast_daily_draws_are_reproducible(pins) -> None:
    task = pins.bdbv_tasks()["2026-10-17"]
    a = get_model("randomwalknowcast").daily(task)
    b = get_model("randomwalknowcast").daily(task)
    pd.testing.assert_frame_equal(a, b)


@needs_torch
@pytest.mark.usefixtures("stand_in")
def test_chronos_daily_quantiles_are_its_native_per_day_quantiles(pins) -> None:
    task = pins.bdbv_tasks()["2026-10-24"]
    levels = np.asarray(QUANTILE_LEVELS)
    model = get_model("fm_chronos_daily_ragged")
    q = model.quantiles(task, levels, target="daily")
    assert set(q.columns) == {
        "location",
        "target_end_date",
        "quantile",
        "value",
    }
    assert set(q["location"]) == {NATIONAL, *task.locations}
    assert set(q["quantile"]) == set(levels)
    wide = q.pivot_table(
        index=["location", "target_end_date"],
        columns="quantile",
        values="value",
    )
    assert (np.diff(wide.to_numpy(), axis=1) >= 0).all()
    # The 7-day call is the default and still gives no own quantiles for
    # the daily source, whose weeks are sums of steps.
    assert model.quantiles(task, levels) is None
    assert model.quantiles(task, levels, target="7day") is None
    # The days are those of the target weeks.
    days = pd.to_datetime(q["target_end_date"])
    first = pd.Timestamp(min(task.target_end_dates)) - pd.Timedelta(days=6)
    assert days.min() >= first
    assert days.max() <= pd.Timestamp(max(task.target_end_dates))


@needs_torch
@pytest.mark.usefixtures("stand_in")
def test_weekly_chronos_has_no_daily_target(pins) -> None:
    task = pins.bdbv_tasks()["2026-10-24"]
    model = get_model("fm_chronos_weekly_ragged")
    assert model.quantiles(task, QUANTILE_LEVELS, target="daily") is None
    assert model.daily(task) is None
    got = daily_quantiles(model, task, QUANTILE_LEVELS, 20, 1)
    assert got is None


@needs_torch
@pytest.mark.usefixtures("stand_in")
def test_chronos_daily_draws_sum_to_its_weekly_draws(pins) -> None:
    task = pins.bdbv_tasks()["2026-10-24"]
    model = get_model("fm_chronos_daily_ragged")
    daily = model.daily(task)
    weekly = model.forecast(task)
    both = _weekly_of(daily).merge(
        weekly, on=["location", "target_end_date", "sample"]
    )
    assert not both.empty
    # Weeks are floored at their reported count after summing, so a week's
    # draw is at least the sum of its days.
    assert (both["value_y"] >= both["value_x"]).all()


def test_daily_quantiles_of_a_sample_model_come_from_its_draws(pins) -> None:
    task = pins.bdbv_tasks()["2026-10-24"]
    model = get_model("randomwalknowcast")
    q = daily_quantiles(model, task, QUANTILE_LEVELS, 30, 5)
    assert q is not None
    draws = model.daily(dataclasses.replace(task, n_samples=30, seed=5))
    want = draws.groupby(["location", "target_end_date"])["value"].quantile(0.5)
    got = q.loc[q["quantile"] == 0.5].set_index(
        ["location", "target_end_date"]
    )["value"]
    pd.testing.assert_series_equal(
        got.sort_index(), want.reindex(got.index).sort_index(),
        check_names=False,
    )  # fmt: skip
    # The country is the sum of the provinces by draw.
    assert NATIONAL in set(q["location"])


def test_german_daily_forecasts_cover_each_day_back_28_days(
    triangle: pd.DataFrame,
) -> None:
    assert tuple(range(-28, 1)) == DAILY_HORIZONS
    out = germany_daily_forecasts(
        ["randomwalknowcast"], triangle, ["2021-10-04"], 30, 1
    )
    assert set(out.columns) == {
        "model",
        "origin",
        "location",
        "target_end_date",
        "horizon",
        "quantile",
        "value",
    }
    assert set(out["horizon"]) == set(DAILY_HORIZONS)
    assert set(out["quantile"]) == set(QUANTILE_LEVELS)
    per = out.groupby(["model", "origin", "location"]).size()
    assert (per == len(DAILY_HORIZONS) * len(QUANTILE_LEVELS)).all()
    assert "DE" in set(out["location"])
    assert "80+" in set(out["location"])
    days = pd.to_datetime(out["target_end_date"])
    back = (days - pd.Timestamp("2021-10-04")).dt.days
    assert (back == out["horizon"]).all()


@needs_torch
@pytest.mark.usefixtures("stand_in")
def test_german_daily_forecasts_leave_out_weekly_models(
    triangle: pd.DataFrame,
) -> None:
    out = germany_daily_forecasts(
        ["fm_chronos_daily_ragged", "fm_chronos_weekly_ragged"],
        triangle,
        ["2021-10-04"],
        30,
        1,
    )
    assert set(out["model"]) == {"fm_chronos_daily_ragged"}
    assert set(out["horizon"]) == set(DAILY_HORIZONS)
