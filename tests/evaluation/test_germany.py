"""The German hospitalisation nowcast benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.germany import HORIZONS, read_triangle
from bdbvhubrenewal.evaluation.germany import (
    QUANTILE_LEVELS,
    forecast_germany,
    level_of,
    load_germany_config,
    nowcast_dates,
    portable_models,
    score_quantiles,
    summarise_germany,
    team_scores,
)
from bdbvhubrenewal.forecasters import VARIANTS

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)

KEYS = ["location", "age_group", "forecast_date", "target_end_date"]


def test_config_gives_the_paper_s_nowcast_dates() -> None:
    config = load_germany_config(ROOT / "config" / "germany.toml")
    dates = nowcast_dates(config)
    assert (dates[0], dates[-1], len(dates)) == (
        "2021-11-22",
        "2022-04-29",
        159,
    )


def test_portable_models_leave_out_the_renewal_family() -> None:
    names = portable_models()
    assert {"randomwalk", "randomwalknowcast", "pprandomwalknowcast"} <= set(
        names
    )
    assert not set(names) & set(VARIANTS)


def test_forecasts_are_the_paper_s_quantiles_of_each_nowcast() -> None:
    triangle = read_triangle(FIXTURE)
    out = forecast_germany(
        ["randomwalk"], triangle, ["2021-10-19", "2021-10-20"], 50, 1
    )
    per = out.groupby(["model", *KEYS, "horizon"])["quantile"]
    assert per.apply(tuple).map(QUANTILE_LEVELS.__eq__).all()
    assert set(out["horizon"]) == set(HORIZONS)
    assert set(out["forecast_date"]) == {"2021-10-19", "2021-10-20"}
    rising = out.groupby(["model", *KEYS])["value"].apply(
        lambda v: bool(np.all(np.diff(v.to_numpy()) >= 0))
    )
    assert rising.all()


def _quantiles(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": "m",
            "location": "DE",
            "age_group": "00+",
            "forecast_date": "2021-12-01",
            "target_end_date": "2021-12-01",
            "horizon": 0,
            "quantile": QUANTILE_LEVELS,
            "value": values,
        }
    )


def _truth(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2021-12-01"],
            "location": ["DE"],
            "age_group": ["00+"],
            "truth": [value],
        }
    )


def test_quantile_scores_match_the_paper_s_formula() -> None:
    q = [2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 20.0]
    out = score_quantiles(_quantiles(q), _truth(13.0)).iloc[0]
    tau = np.array(QUANTILE_LEVELS)
    want = np.mean(2 * ((np.array(q) > 13.0) - tau) * (np.array(q) - 13.0))
    assert out["wis"] == pytest.approx(want)
    assert out["coverage_50"] == 0.0
    assert out["coverage_95"] == 1.0
    assert out["level"] == "national"


def test_team_scores_take_the_paper_s_wis_and_published_quantiles() -> None:
    sub = _quantiles([2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 20.0]).assign(
        model="KIT-simple_nowcast",
        type="quantile",
        target="0 day ahead inc hosp",
        retrospective=False,
    )
    # A retrospective file replaces the prospective one.
    retro = sub.assign(
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], retrospective=True
    )
    paper = sub.drop(columns=["value", "retrospective"]).assign(
        model="KIT", score=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )
    # The paper scored a forecast with no published quantiles: left out.
    other = paper.assign(location="DE-BY")
    out = team_scores(
        pd.concat([paper, other]), pd.concat([sub, retro]), _truth(6.5)
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["model"] == "KIT"
    assert row["wis"] == pytest.approx(4.0)
    assert row["coverage_50"] == 0.0
    assert row["coverage_95"] == 1.0


def test_summary_is_by_model_and_level() -> None:
    scores = pd.DataFrame(
        {
            "model": ["a", "a", "b"],
            "level": ["national", "states", "national"],
            "wis": [1.0, 3.0, 2.0],
            "coverage_50": [1.0, 0.0, 1.0],
            "coverage_95": [1.0, 1.0, 1.0],
        }
    )
    out = summarise_germany(scores).set_index(["level", "model"])
    assert out.loc[("national", "a"), "wis"] == 1.0
    assert out.loc[("states", "a"), "n"] == 1


@pytest.mark.parametrize(
    ("location", "age_group", "level"),
    [
        ("DE", "00+", "national"),
        ("DE-BY", "00+", "states"),
        ("DE", "80+", "age"),
    ],
)
def test_levels_are_the_paper_s(
    location: str, age_group: str, level: str
) -> None:
    assert (
        level_of(pd.Series([location]), pd.Series([age_group])).item() == level
    )
