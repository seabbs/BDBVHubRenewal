from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily_zones
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
        load_daily_zones(DATA),
        hub_zones(DATA / "hub-config" / "tasks.json"),
        models=("persistence",),
    )


def test_zone_backtest_scores_every_zone_against_the_latest_vintage(
    scores: pd.DataFrame,
) -> None:
    assert set(scores["model"]) == {"persistence", ZONE_BASELINE}
    assert (scores["location"].str.count("-") == 2).all()
    assert set(scores["truth_vintage"]) == {"2026-10-24"}
    assert set(scores["vintage"]) == {"2026-08-18", "2026-10-09", "2026-10-17"}
    # Zones with no cases are scored too, against zero.
    assert (scores["truth"] == 0).any()
    per = scores.groupby(["model", "vintage"])["location"].nunique()
    assert per.min() > 150


def test_allocated_zones_beat_zone_persistence(scores: pd.DataFrame) -> None:
    rel = zone_relative_skill(scores)
    row = rel.set_index("model").loc["persistence"]
    assert row["relative_wis"] < 1
    assert row["relative_log_wis"] < 1
    by = zone_relative_skill(scores, by=("horizon",))
    mine = by.loc[by["model"] == "persistence"].set_index("horizon")
    # Weeks both copy from the data tie; later weeks are better.
    assert (mine["relative_wis"] <= 1 + 1e-9).all()
    assert (mine.loc[mine.index >= -2, "relative_wis"] < 1).all()
