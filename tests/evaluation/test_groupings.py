"""The grouping arms: the paired arm and the merged German task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.groupings import (
    GROUPING_ARMS,
    arm_name,
    merged_task,
    pair_and_sum,
)
from bdbvhubrenewal.forecasters.base import NATIONAL, ForecastTask

ZONES = {"P1-a": "P1", "P1-b": "P1", "P2-a": "P2"}
WEEKS = ["2026-10-10", "2026-10-17"]
N = 200


@pytest.fixture
def draws() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = [
        {
            "model": "m:bottom",
            "origin": "2026-10-17",
            "location": z,
            "target_end_date": w,
            "sample": s + 1,
            "value": float(rng.poisson(20)),
        }
        for z in ZONES
        for w in WEEKS
        for s in range(N)
    ]
    return pd.DataFrame(rows)


def _no_truth() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["location", "target_end_date", "truth", "truth_vintage"]
    )


def test_arms_are_the_fitted_groupings_and_paired() -> None:
    assert GROUPING_ARMS == ("joint", "level", "bottom", "paired")
    assert arm_name("m", "level") == "m:level"


def test_paired_keeps_each_cells_draws_and_sums_up(draws) -> None:
    out, fits = pair_and_sum(draws, _no_truth(), 3, parent=ZONES.get)
    assert len(fits) == 1
    keys = ["location", "target_end_date"]
    zones = out.loc[out["location"].isin(set(ZONES))]
    for (place, week), own in zones.groupby(keys):
        before = draws.loc[
            (draws["location"] == place) & (draws["target_end_date"] == week)
        ]
        assert sorted(own["value"]) == sorted(before["value"])
    wide = out.pivot_table(
        index=["target_end_date", "sample"], columns="location", values="value"
    )
    np.testing.assert_allclose(wide["P1"], wide["P1-a"] + wide["P1-b"])
    np.testing.assert_allclose(wide["P2"], wide["P2-a"])
    np.testing.assert_allclose(wide[NATIONAL], wide["P1"] + wide["P2"])
    # The copula ties the zones' draws together, unlike independent draws.
    assert np.corrcoef(wide["P1-a"], wide["P1-b"])[0, 1] > 0.3


def test_sums_are_raised_to_their_floors(draws) -> None:
    floors = pd.DataFrame(
        {
            "origin": "2026-10-17",
            "location": ["P1", NATIONAL],
            "target_end_date": WEEKS[0],
            "floor": [1000.0, 5000.0],
        }
    )
    out, _ = pair_and_sum(
        draws, _no_truth(), 3, parent=ZONES.get, floors=floors
    )
    first = out.loc[out["target_end_date"] == WEEKS[0]]
    assert (first.loc[first["location"] == "P1", "value"] == 1000).all()
    assert (first.loc[first["location"] == NATIONAL, "value"] == 5000).all()


def test_no_parent_sums_the_places_to_the_country(draws) -> None:
    out, _ = pair_and_sum(draws, _no_truth(), 3)
    assert set(out["location"]) == {*ZONES, NATIONAL}


def _task(places: tuple[str, ...]) -> ForecastTask:
    rows = pd.DataFrame(
        {"location": [NATIONAL, *places], "total": range(len(places) + 1)}
    )
    return ForecastTask(
        vintage="2022-01-01",
        reference_date=pd.Timestamp("2022-01-01").date(),
        kind="real",
        cutoff="2022-01-01",
        target=rows,
        delays=None,
        locations=places,
        target_end_dates=("2022-01-01",),
        n_samples=10,
        seed=1,
        target_history=rows,
        daily_history=rows,
    )


def test_merged_task_holds_both_tasks_places_once() -> None:
    task = merged_task(_task(("S1", "S2")), _task(("A1",)))
    assert task.locations == ("S1", "S2", "A1")
    for table in (task.target, task.target_history, task.daily_history):
        assert table["location"].tolist() == [NATIONAL, "S1", "S2", "A1"]


def test_options_are_named_for_the_paper() -> None:
    from bdbvhubrenewal.evaluation.groupings import split_arm

    assert split_arm("m:joint") == ("m", "joint")
    assert split_arm("m@abc:bottom") == ("m@abc", "zones_summed")
    assert split_arm("m:paired") == ("m", "zones_copula")
    assert split_arm("randomwalknowcast") == ("randomwalknowcast", "reference")


def test_aggregation_rows_keep_cells_the_reference_scored() -> None:
    from bdbvhubrenewal.evaluation.groupings import (
        AGGREGATION_COLUMNS,
        aggregation_rows,
    )

    scores = pd.DataFrame(
        {
            "model": ["randomwalknowcast", "m:joint", "m:joint"],
            "origin": "2026-10-17",
            "location": ["CD", "CD", "CD-KS"],
            "target_end_date": "2026-10-10",
            "level": ["national", "national", "provinces"],
            "wis": [2.0, 1.0, 3.0],
            "log_wis": [0.2, 0.1, 0.3],
            "cover50": [1.0, 0.0, 1.0],
            "cover95": [1.0, 1.0, 1.0],
        }
    )
    out = aggregation_rows(scores, "ebola")
    assert list(out.columns) == list(AGGREGATION_COLUMNS)
    assert out["location"].tolist() == ["CD", "CD"]
    assert set(out["option"]) == {"reference", "joint"}
    assert (out["study"] == "ebola").all()
    assert out["target_date"].tolist() == ["2026-10-10"] * 2


def test_sample_coverage_is_the_central_interval() -> None:
    from bdbvhubrenewal.evaluation.groupings import sample_coverage

    samples = pd.DataFrame(
        {
            "model": "m",
            "origin": "o",
            "location": "x",
            "target_end_date": "w",
            "sample": range(1, 101),
            "value": np.arange(100.0),
        }
    )
    truth = pd.DataFrame(
        {"location": ["x"], "target_end_date": ["w"], "truth": [98.5]}
    )
    out = sample_coverage(samples, truth, 0.95)
    assert out["cover95"].tolist() == [0.0]
    out = sample_coverage(samples, truth.assign(truth=50.0), 0.95)
    assert out["cover95"].tolist() == [1.0]
