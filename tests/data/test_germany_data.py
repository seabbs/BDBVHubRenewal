"""The German hospitalisation nowcasting hub data as forecast tasks."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data.germany import (
    HORIZONS,
    hub_forecasts,
    hub_target,
    hub_triangle,
    level,
    nowcast_tasks,
    place_code,
    read_triangle,
)
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.forecasters import NATIONAL, get_model

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)

NOWCAST = "2021-10-20"
"""A Wednesday, the last report date of the fixture."""


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.fixture(scope="module")
def tasks(triangle: pd.DataFrame):
    return nowcast_tasks(triangle, NOWCAST, n_samples=20, seed=1)


def test_hub_triangle_accumulates_changes_by_report_date(
    tmp_path: Path,
) -> None:
    nan = float("nan")
    # The second Meldedatum has two days reported when the file was made.
    values = {f"value_{d}d": [0.0, nan] for d in range(81)}
    values["value_0d"] = [5.0, 3.0]
    values["value_1d"] = [0.0, 0.0]
    values["value_2d"] = [-1.0, nan]
    values["value_>80d"] = [4.0, nan]
    wide = pd.DataFrame(
        {
            "date": ["2021-11-01", "2021-11-02"],
            "location": ["DE", "DE-BY"],
            "age_group": ["00+", "00+"],
            **values,
        }
    )
    path = tmp_path / "triangle.csv"
    wide.to_csv(path, index=False)
    long = hub_triangle(path).set_index(
        ["location", "reference_date", "report_date"]
    )["confirm"]
    first = long.loc[("DE", "2021-11-01")]
    assert first.loc["2021-11-01"] == 5
    assert first.loc["2021-11-03"] == 4
    # Changes after 80 days are put at 81 days, as epinowcast does.
    assert first.loc["2022-01-21"] == 8
    assert len(first) == 82
    # Missing changes are report dates after the file was made.
    assert list(long.loc[("DE-BY", "2021-11-02")]) == [3, 3]


def test_states_and_age_groups_are_separate_tasks_summing_to_germany(
    tasks,
) -> None:
    states, ages = tasks
    assert states.locations == (place_code("DE-BE"), place_code("DE-BY"))
    assert ages.locations == tuple(
        place_code("DE", a)
        for a in ("00-04", "05-14", "15-34", "35-59", "60-79", "80+")
    )
    for task in tasks:
        places = set(task.daily_history["location"])
        assert places == {NATIONAL, *task.locations}


def test_dates_move_to_the_next_saturday(tasks) -> None:
    for task in tasks:
        # 2021-10-20 is a Wednesday, three days before a Saturday.
        assert task.reference_date == dt.date(2021, 10, 23)
        assert task.reference_date.weekday() == 5
        assert task.target_end_dates == tuple(
            (task.reference_date + dt.timedelta(days=h)).isoformat()
            for h in sorted(HORIZONS)
        )
        assert task.daily_history["vintage"].max() == "2021-10-23"


def test_releases_are_the_counts_reported_by_each_report_date(
    triangle: pd.DataFrame, tasks
) -> None:
    states = tasks[0]
    daily = states.daily_history.set_index(
        ["vintage", "location", "onset_date"]
    )
    # Meldedatum 2021-09-01 as reported by 2021-09-10, moved three days.
    want = triangle.loc[
        (triangle["location"] == "DE-BY")
        & (triangle["age_group"] == "00+")
        & (triangle["reference_date"] == "2021-09-01")
        & (triangle["report_date"] == "2021-09-10"),
        "confirm",
    ].item()
    got = daily.loc[("2021-09-13", place_code("DE-BY"), "2021-09-04")]
    assert got["observed"] == want
    assert got["imputed"] == 0
    assert got["cutoff"] == "2021-09-13"


def test_weekly_target_sums_each_release_to_hub_weeks(tasks) -> None:
    states = tasks[0]
    daily = states.daily_history
    week = daily.loc[
        (daily["vintage"] == states.vintage)
        & (daily["location"] == NATIONAL)
        & (daily["onset_date"] > "2021-10-16")
    ]
    target = states.target.set_index(["location", "target_end_date"])
    assert target.loc[(NATIONAL, "2021-10-23"), "total"] == (
        week["observed"].sum()
    )
    assert set(states.target_history["vintage"]) == set(daily["vintage"])


def test_models_forecast_the_tasks_and_map_back(tasks) -> None:
    model = get_model("randomwalk")
    out = pd.concat(
        hub_forecasts(run_forecasts([model], [task]), task, NOWCAST)
        for task in tasks
    )
    assert set(out["horizon"]) == set(HORIZONS)
    assert set(out["forecast_date"]) == {NOWCAST}
    assert (
        pd.to_datetime(out["target_end_date"])
        - pd.to_datetime(out["forecast_date"])
    ).dt.days.equals(out["horizon"])
    assert set(out["location"]) == {
        "DE",
        "DE-BE",
        "DE-BY",
        "00-04",
        "05-14",
        "15-34",
        "35-59",
        "60-79",
        "80+",
    }
    # Germany comes from the states task, the sum of its states.
    assert out.groupby(["location", "age_group", "horizon"]).size().eq(20).all()


def test_hub_target_is_each_7_day_sum_as_each_release_reports_it(
    triangle: pd.DataFrame,
) -> None:
    out = hub_target(triangle, ["2021-10-01"]).set_index(
        ["location", "vintage"]
    )
    rows = triangle.loc[
        (triangle["location"] == "DE-BY")
        & (triangle["reference_date"].between("2021-09-25", "2021-10-01"))
        & (triangle["report_date"] <= "2021-10-05")
    ]
    want = rows.groupby("reference_date")["confirm"].last().sum()
    got = out.loc[("DE-BY", "2021-10-05")]
    assert got["total"] == want
    assert got["cutoff"] == "2021-10-05"
    assert got["target_end_date"] == "2021-10-01"
    assert out.index.get_level_values("vintage").min() == "2021-10-01"
    assert set(out.index.get_level_values("location")) == {
        "DE", "DE-BE", "DE-BY", "00-04", "05-14", "15-34", "35-59",
        "60-79", "80+",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("place", "want"),
    [("DE", "national"), ("DE-BY", "states"), ("80+", "age")],
)
def test_levels_are_the_paper_s(place: str, want: str) -> None:
    assert level(place) == want
