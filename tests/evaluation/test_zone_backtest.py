from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily, load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation.zones import (
    SUMMED,
    ZONE_BASELINE,
    zone_backtest,
    zone_relative_skill,
)

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def scores(target: pd.DataFrame, delays: pd.DataFrame) -> pd.DataFrame:
    return zone_backtest(
        target,
        delays,
        load_daily(DATA),
        load_daily_zones(DATA),
        hub_zones(DATA / "hub-config" / "tasks.json"),
    )


def test_both_baselines_are_scored_at_zone_level(
    scores: pd.DataFrame,
) -> None:
    names = {"randomwalk", "randomwalknowcast"}
    assert set(scores["model"]) == names | {n + SUMMED for n in names}
    assert (scores["location"].str.count("-") == 2).all()
    assert set(scores["truth_vintage"]) == {"2026-11-01"}
    # Earlier origins have no pair of releases in the delay window.
    assert set(scores["vintage"]) == {"2026-10-17", "2026-10-24"}
    # Only zones with cases at the origin are forecast.
    per = scores.groupby(["model", "vintage"])["location"].nunique()
    assert (per.groupby("vintage").nunique() == 1).all()
    assert per.between(20, 300).all()


def test_every_model_scores_the_same_zones_and_weeks(
    scores: pd.DataFrame,
) -> None:
    keys = scores.groupby("model")[["location", "target_end_date"]].apply(
        lambda d: set(map(tuple, d.to_numpy()))
    )
    assert keys.map(len).nunique() == 1
    assert (keys == keys.iloc[0]).all()


def test_relative_skill_is_against_the_zone_randomwalk(
    scores: pd.DataFrame,
) -> None:
    rel = zone_relative_skill(scores).set_index("model")
    assert rel.loc[ZONE_BASELINE, "relative_wis"] == pytest.approx(1.0)
    assert (rel["relative_wis"] > 0).all()
    by = zone_relative_skill(scores, by=("horizon",))
    mine = by.loc[by["model"] == "randomwalknowcast"].set_index("horizon")
    assert (mine["relative_wis"] > 0).all()
    assert mine.index.min() == -4
