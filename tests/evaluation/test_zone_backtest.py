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

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "data"


@pytest.fixture(scope="module")
def result(
    target: pd.DataFrame, delays: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return zone_backtest(
        target,
        delays,
        load_daily(DATA),
        load_daily_zones(DATA),
        hub_zones(DATA / "hub-config" / "tasks.json"),
    )


@pytest.fixture(scope="module")
def scores(
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    return result[0]


def test_energy_scores_every_zone_and_week_as_one_vector(
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    scores, energy, _ = result
    assert set(energy["scale"]) == {"zones"}
    assert set(energy["model"]) == set(scores["model"])
    unit = ["origin", "target", "truth_vintage"]
    cells = scores.groupby([*unit, "model"]).size().groupby(unit).max()
    got = energy.groupby(unit)["n_cells"].agg(["min", "max"])
    assert (got["min"] == got["max"]).all()
    assert got["max"].equals(cells.rename("max"))


def test_both_baselines_are_scored_at_zone_level(
    scores: pd.DataFrame, vintages: list[str]
) -> None:
    names = {"randomwalk", "randomwalknowcast"}
    assert set(scores["model"]) == names | {n + SUMMED for n in names}
    assert (scores["location"].str.count("-") == 2).all()
    # Each origin is scored on what every later release reported, and on
    # the latest release's complete weeks.
    later = vintages[vintages.index("2026-10-17") :]
    latest = pd.Timestamp(later[-1])
    assert set(scores["target"]) == {"reported", "complete"}
    done = scores.loc[scores["target"] == "complete"]
    assert set(done["truth_vintage"]) == {later[-1]}
    assert set(done["origin"]) == set(later[:-1])
    # Complete weeks end at least three weeks before the latest cut-off.
    cutoff = latest - pd.Timedelta(days=(latest.day_of_week - 5) % 7)
    oldest = (cutoff - pd.Timedelta(weeks=3)).strftime("%Y-%m-%d")
    assert (done["target_end_date"] <= oldest).all()
    reported = scores.loc[scores["target"] == "reported"]
    pairs = set(zip(reported["origin"], reported["truth_vintage"], strict=True))
    assert pairs == {(o, t) for o in later for t in later if o < t}
    # Only zones with cases at the origin are forecast.
    per = scores.groupby(["model", "origin"])["location"].nunique()
    assert (per.groupby("origin").nunique() == 1).all()
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


def test_zone_forecasts_are_the_summed_forms_by_origin(
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    scores, _, forecasts = result
    assert set(forecasts["model"]) == {"randomwalk", "randomwalknowcast"}
    assert set(forecasts["origin"]) == set(scores["origin"])
    assert (forecasts["location"].str.count("-") == 2).all()
    # The summed forms are the ones scored under the SUMMED names.
    summed = scores.loc[scores["model"].str.endswith(SUMMED)]
    assert set(forecasts["location"]) == set(summed["location"])


def test_zone_backtest_by_origin_chunk_matches_one_run(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    vintages: list[str],
) -> None:
    from bdbvhubrenewal.cli import chunk

    origins = vintages[:-1]
    parts = [
        zone_backtest(
            target,
            delays,
            load_daily(DATA),
            load_daily_zones(DATA),
            hub_zones(DATA / "hub-config" / "tasks.json"),
            origins=chunk(origins, f"{k}/2"),
        )
        for k in range(2)
    ]
    for i in range(3):
        pd.testing.assert_frame_equal(
            pd.concat([p[i] for p in parts], ignore_index=True), result[i]
        )


def test_zone_backtest_of_no_origin_is_empty(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # A block whose origins every model skips forecasts nothing.
    out = zone_backtest(
        target,
        delays,
        load_daily(DATA),
        load_daily_zones(DATA),
        hub_zones(DATA / "hub-config" / "tasks.json"),
        origins=[],
    )
    assert all(t.empty for t in out)


def test_zone_forecasts_made_apart_by_model_score_as_one_run(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.evaluation.zones import (
        ZoneForecasts,
        score_zones,
        zone_forecasts,
    )

    args = (
        target,
        delays,
        load_daily(DATA),
        load_daily_zones(DATA),
        hub_zones(DATA / "hub-config" / "tasks.json"),
    )
    parts = []
    for name in ("randomwalk", "randomwalknowcast"):
        zone_forecasts(*args, models=[name]).write(tmp_path / name)
        parts.append(tmp_path / name)
    run = ZoneForecasts.read(parts)
    out = score_zones(target, delays, args[2], args[3], run)
    for got, want in zip(out, result, strict=True):
        pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_zone_forecasts_read_skips_an_empty_part(tmp_path: Path) -> None:
    from bdbvhubrenewal.evaluation.zones import ZoneForecasts

    empty = pd.DataFrame(columns=["origin", "model"])
    ZoneForecasts(empty, empty, empty).write(tmp_path / "a")
    samples = pd.DataFrame(
        {"key": ["randomwalk"], "origin": ["2026-10-24"], "value": [1.0]}
    )
    runs = pd.DataFrame({"origin": ["2026-10-24"], "model": ["randomwalk"]})
    ZoneForecasts(runs, samples, empty).write(tmp_path / "b")
    run = ZoneForecasts.read([tmp_path / "a", tmp_path / "b"])
    assert list(run.samples.columns) == ["key", "origin", "value"]
    assert len(run.runs) == 1
