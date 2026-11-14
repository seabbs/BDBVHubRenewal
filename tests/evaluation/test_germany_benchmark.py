"""The German hospitalisation nowcast benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from bdbvhubrenewal.data.germany import HORIZONS, hub_target, read_triangle
from bdbvhubrenewal.evaluation.germany import (
    PAPER_COLUMNS,
    QUANTILE_LEVELS,
    SCORE_COLUMNS,
    asof_sums,
    germany_forecasts,
    load_germany_config,
    nowcast_dates,
    paper_forecasts,
    relative_scores,
    score_germany,
    team_forecasts,
    with_truth,
)
from bdbvhubrenewal.evaluation.nowcast import rolling_truth

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.fixture(scope="module")
def forecasts(triangle: pd.DataFrame) -> pd.DataFrame:
    drawn = germany_forecasts(
        ["randomwalk", "randomwalknowcast"],
        triangle,
        ["2021-10-04", "2021-10-05"],
        n_samples=30,
        seed=1,
    )
    return with_truth(drawn, triangle, truth_delay=10)


@pytest.fixture(scope="module")
def scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    return score_germany(forecasts, truth_delay=10)


def test_config_gives_the_paper_s_dates_and_an_explicit_model_set() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    every = nowcast_dates(config, held_out=False) + nowcast_dates(
        config, held_out=True
    )
    assert (every[0], every[-1], len(every)) == (
        "2021-11-22",
        "2022-04-29",
        159,
    )
    assert config.models == ("randomwalk", "randomwalknowcast")
    assert config.truth_delay == 40
    assert config.submissions
    assert config.baseline == "randomwalknowcast"
    assert config.paper_baseline == "KIT-frozen_baseline"


def test_the_later_half_of_the_dates_is_held_out() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    assert config.split_date == "2022-02-10"
    development = nowcast_dates(config)
    held = nowcast_dates(config, held_out=True)
    assert development[-1] < config.split_date <= held[0]
    assert len(development) - len(held) in (0, 1)


def test_our_quantiles_are_the_samples_quantiles_at_the_hub_s_levels(
    forecasts: pd.DataFrame,
) -> None:
    per = forecasts.groupby(["model", "origin", "location", "horizon"])
    assert (per["quantile"].apply(tuple) == QUANTILE_LEVELS).all()
    assert per["value"].is_monotonic_increasing.all()


def test_scores_are_quantile_scores_on_the_d_day_truth(
    triangle: pd.DataFrame, scores: pd.DataFrame
) -> None:
    assert set(SCORE_COLUMNS) <= set(scores.columns)
    assert "log_wis" not in scores.columns
    assert set(scores["origin"]) == {"2021-10-04", "2021-10-05"}
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


def test_relative_scores_are_against_the_baseline_given(
    scores: pd.DataFrame,
) -> None:
    for base in ("randomwalknowcast", "randomwalk"):
        rel = relative_scores(scores, ("pool",), base)
        rel = rel.set_index(["pool", "model"])
        assert rel.loc[("national", base), "rel_wis"] == pytest.approx(1)
    assert (rel["n"] > 0).all()
    assert {"coverage_50", "coverage_95", "rel_ae_median"} <= set(rel.columns)


def _team(model: str, target: int, values: dict[float, float], **kw) -> list:
    end = (pd.Timestamp("2021-10-04") + pd.Timedelta(days=target)).date()
    return [
        {
            "location": kw.get("location", "DE"),
            "age_group": kw.get("age_group", "00+"),
            "forecast_date": "2021-10-04",
            "target_end_date": str(end),
            "target": f"{target} day ahead inc hosp",
            "type": "quantile",
            "quantile": q,
            "value": v,
            "pathogen": "COVID-19",
            "model": model,
            "retrospective": False,
        }
        for q, v in values.items()
    ]


def _levels(scale: float) -> dict[float, float]:
    return {q: scale * (1 + i) for i, q in enumerate(QUANTILE_LEVELS)}


@pytest.fixture(scope="module")
def submissions() -> pd.DataFrame:
    no_tenth = {q: v for q, v in _levels(100).items() if q != 0.1}
    rki = _levels(100) | {0.5: 2e9}
    rows = [
        *_team("KIT-simple_nowcast", 0, _levels(100)),
        *_team("KIT-simple_nowcast", -7, _levels(100)),
        *_team("KIT-simple_nowcast", -3, _levels(100)),
        *_team("KIT-simple_nowcast", 0, _levels(10), age_group="80+"),
        *_team("KIT-simple_nowcast", 0, _levels(10), location="DE-BY"),
        *_team("Epiforecasts-independent", 0, no_tenth),
        *_team("RKI-weekly_report", 0, rki),
        *_team("RKI-weekly_report", -1, _levels(90)),
    ]
    mean = _team("KIT-simple_nowcast", 0, {0.5: 1.0})[0] | {
        "type": "mean",
        "quantile": np.nan,
    }
    return pd.DataFrame([*rows, mean])


def test_team_forecasts_are_the_hub_s_quantiles_at_our_horizons(
    submissions: pd.DataFrame,
) -> None:
    got = team_forecasts(submissions)
    assert set(got["model"]) == {"KIT", "Epiforecasts", "RKI"}
    assert set(got["horizon"]) == {0, -7}
    kit = got.loc[got["model"] == "KIT"]
    assert set(kit["location"]) == {"DE", "80+", "DE-BY"}
    assert (got.groupby(["model", "location", "horizon"]).size() == 7).all()
    assert set(got["origin"]) == {"2021-10-04"}


def test_team_forecasts_fill_epiforecasts_tenth_percentile(
    submissions: pd.DataFrame,
) -> None:
    # As the paper: the normal through the 2.5% and 25% quantiles.
    got = team_forecasts(submissions).set_index(["model", "quantile"])
    lo, hi = 100.0, 300.0
    sd = (hi - lo) / (norm.ppf(0.25) - norm.ppf(0.025))
    want = lo + sd * (norm.ppf(0.1) - norm.ppf(0.025))
    assert got.loc[("Epiforecasts", 0.1), "value"] == pytest.approx(want)


def test_only_epiforecasts_tenth_percentile_is_filled(
    submissions: pd.DataFrame,
) -> None:
    no_tenth = {q: v for q, v in _levels(100).items() if q != 0.1}
    other = pd.DataFrame(_team("SU-hier_bayes", 0, no_tenth))
    got = team_forecasts(pd.concat([submissions, other]))
    assert got.loc[got["model"] == "SU", "quantile"].nunique() == 6


def test_team_quantiles_are_sorted_within_a_forecast(
    submissions: pd.DataFrame,
) -> None:
    crossed = _levels(100) | {0.1: 50.0}
    other = pd.DataFrame(_team("SU-hier_bayes", 0, crossed))
    got = team_forecasts(pd.concat([submissions, other]))
    su = got.loc[got["model"] == "SU"].sort_values("quantile")
    assert su["value"].is_monotonic_increasing
    assert sorted(su["value"]) == sorted(crossed.values())


def test_unknown_teams_are_an_error(submissions: pd.DataFrame) -> None:
    other = pd.DataFrame(_team("New-model", 0, _levels(100)))
    with pytest.raises(ValueError, match="New-model"):
        team_forecasts(pd.concat([submissions, other]))


def test_team_forecasts_replace_rki_errors_by_the_minus_one_day_nowcast(
    submissions: pd.DataFrame,
) -> None:
    got = team_forecasts(submissions).set_index(["model", "quantile"])
    assert got.loc[("RKI", 0.5), "value"] == pytest.approx(90 * 4)
    assert got.loc[("RKI", 0.25), "value"] == pytest.approx(300)


def test_teams_are_scored_as_our_models(
    triangle: pd.DataFrame,
    submissions: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    quantiles = team_forecasts(submissions, ["2021-10-04"])
    teams = score_germany(with_truth(quantiles, triangle, 10), 10)
    assert set(SCORE_COLUMNS) <= set(teams.columns)
    assert list(teams.columns) == list(scores.columns)
    cell = ["origin", "location", "target_end_date"]
    both = teams.merge(scores, on=cell)
    assert len(both)
    assert (both["truth_x"] == both["truth_y"]).all()
    assert (both["truth_vintage_x"] == both["truth_vintage_y"]).all()
    assert set(teams["pool"]) == {"national", "states", "age"}


def test_team_forecasts_keep_the_dates_given(
    submissions: pd.DataFrame,
) -> None:
    assert team_forecasts(submissions, ["2021-10-05"]).empty


def test_paper_table_follows_the_paper_schema(
    forecasts: pd.DataFrame,
) -> None:
    paper = paper_forecasts(forecasts)
    assert list(paper.columns) == list(PAPER_COLUMNS)
    assert set(paper["study"]) == {"germany"}
    assert paper["sample_id"].isna().all()
    ages = paper.loc[paper["stratum"] != "all"]
    assert set(ages["location"]) == {"DE"}
    assert set(ages["stratum"]) == {
        "00-04", "05-14", "15-34", "35-59", "60-79", "80+"
    }  # fmt: skip
    days = pd.to_datetime(paper["target_date"]) - pd.to_datetime(
        paper["origin_date"]
    )
    assert (days.dt.days == paper["horizon"]).all()
    assert len(paper) == len(forecasts)


def test_asof_sums_are_the_data_each_nowcast_date_saw(
    triangle: pd.DataFrame,
) -> None:
    got = asof_sums(triangle, ["2021-10-04"], weeks=1)
    assert set(got.columns) == {
        "origin", "location", "target_end_date", "total"
    }  # fmt: skip
    assert got["target_end_date"].min() == "2021-09-27"
    assert got["target_end_date"].max() == "2021-10-04"
    ends = sorted(set(got["target_end_date"]))
    want = hub_target(triangle, ends)
    want = want.loc[want["vintage"] == "2021-10-04"]
    both = got.merge(want, on=["location", "target_end_date"])
    assert len(both) == len(got) == len(want)
    assert (both["total_x"] == both["total_y"]).all()
