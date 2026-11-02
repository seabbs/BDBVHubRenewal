from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily, load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation.zones import (
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
        models=("randomwalk",),
    )


def test_zone_backtest_scores_every_zone_against_the_latest_vintage(
    scores: pd.DataFrame,
) -> None:
    assert set(scores["model"]) == {"randomwalk", ZONE_BASELINE}
    assert (scores["location"].str.count("-") == 2).all()
    assert set(scores["truth_vintage"]) == {"2026-10-24"}
    # Earlier origins have no pair of releases in the delay window.
    assert set(scores["vintage"]) == {"2026-10-17"}
    # Zones with no cases are scored too, against zero.
    assert (scores["truth"] == 0).any()
    per = scores.groupby(["model", "vintage"])["location"].nunique()
    assert per.min() > 150


def test_relative_skill_is_against_the_zone_randomwalk(
    scores: pd.DataFrame,
) -> None:
    rel = zone_relative_skill(scores).set_index("model")
    assert rel.loc[ZONE_BASELINE, "relative_wis"] == pytest.approx(1.0)
    assert rel.loc["randomwalk", "relative_wis"] > 0
    by = zone_relative_skill(scores, by=("horizon",))
    mine = by.loc[by["model"] == "randomwalk"].set_index("horizon")
    assert (mine["relative_wis"] > 0).all()
    assert mine.index.min() == -4
