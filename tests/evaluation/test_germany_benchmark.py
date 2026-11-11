"""The German hospitalisation nowcast benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data.germany import HORIZONS, hub_target, read_triangle
from bdbvhubrenewal.evaluation.germany import (
    germany_backtest,
    load_germany_config,
    nowcast_dates,
)
from bdbvhubrenewal.evaluation.nowcast import relative_wis, rolling_truth
from bdbvhubrenewal.evaluation.scoring import SCORE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.fixture(scope="module")
def scores(triangle: pd.DataFrame) -> pd.DataFrame:
    return germany_backtest(
        ["randomwalk", "randomwalknowcast"],
        triangle,
        ["2021-10-05", "2021-10-06"],
        n_samples=30,
        seed=1,
        truth_delay=10,
    )


def test_config_gives_the_paper_s_dates_and_an_explicit_model_set() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    dates = nowcast_dates(config)
    assert (dates[0], dates[-1], len(dates)) == (
        "2021-11-22",
        "2022-04-29",
        159,
    )
    assert config.models == ("randomwalk", "randomwalknowcast")
    assert config.truth_delay == 40


def test_scores_are_the_backtest_s_on_the_d_day_truth(
    triangle: pd.DataFrame, scores: pd.DataFrame
) -> None:
    assert set(SCORE_COLUMNS) <= set(scores.columns)
    assert set(scores["origin"]) == {"2021-10-05", "2021-10-06"}
    assert set(scores["horizon"]) <= set(HORIZONS)
    assert set(scores["pool"]) == {"national", "states", "age"}
    ends = sorted(set(scores["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), 10)
    both = scores.merge(truth, on=["location", "target_end_date"])
    assert len(both) == len(scores)
    assert (both["truth_x"] == both["truth_y"]).all()
    assert (both["truth_vintage_x"] == both["truth_vintage_y"]).all()


def test_sums_with_no_release_d_days_on_are_not_scored(
    scores: pd.DataFrame,
) -> None:
    # The fixture ends on 2021-10-20, so sums ending after 2021-10-10
    # have no truth.
    assert scores["target_end_date"].max() <= "2021-10-10"


def test_relative_wis_summarises_the_scores(scores: pd.DataFrame) -> None:
    rel = relative_wis(scores, ("pool",)).set_index(["pool", "model"])
    assert rel.loc[("national", "randomwalk"), "rel_wis"] == pytest.approx(1)
    assert (rel["n"] > 0).all()
