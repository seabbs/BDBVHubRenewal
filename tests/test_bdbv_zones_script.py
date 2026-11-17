"""Tests of scripts/bdbv_zones.py: BDBV health-zone checkpoint scores."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import (
    hub_zones,
    is_zone,
    province_of,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import RandomWalk, RandomWalkNowcast
from bdbvhubrenewal.forecasters.zones import check_zone_coherence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "tests" / "fixtures" / "data"
VINTAGE = "2026-10-17"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bz():
    return _load("bdbv_zones")


@pytest.fixture(scope="module")
def cs():
    return _load("checkpoint_scores")


@pytest.fixture(scope="module")
def zones() -> tuple[str, ...]:
    return hub_zones(DATA / "hub-config" / "tasks.json")


@pytest.fixture(scope="module")
def task(target, delays, daily):
    rows = target.loc[target["vintage"] == VINTAGE]
    full = make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]
    return dataclasses.replace(full, locations=full.locations[:2], n_samples=40)


@pytest.fixture(scope="module")
def counts(daily_zones):
    return weekly_zone_counts(daily_zones, VINTAGE)


def _stand_in_zones(task, prov, counts, zones, daily_zones) -> pd.DataFrame:
    """Zone rows for the Chronos-2 stand-in, as its own forecast holds."""
    from bdbvhubrenewal.forecasters.zones import zones_for_model

    own = zones_for_model("randomwalk", task, prov, counts, zones, daily_zones)
    return own.assign(value=own["value"] + 1.0)


@pytest.fixture(scope="module")
def draws(task, counts, zones, daily_zones):
    """A cached origin file: the baseline and a Chronos-2 stand-in.

    Each with its national rows and a daily target, which the zones
    must ignore. The stand-in's file holds its own zone rows, as a
    main forecast does (#299); the baseline's holds none.
    """
    frames = []
    for name, model in (
        ("randomwalknowcast", RandomWalkNowcast()),
        ("fm_chronos_daily_ragged_ft", RandomWalk()),
    ):
        prov = model.forecast(task)
        if name != "randomwalknowcast":
            own = _stand_in_zones(task, prov, counts, zones, daily_zones)
            frames.append(own.assign(target="7day", model=name))
        national = (
            prov.groupby(["target_end_date", "sample"], as_index=False)["value"]
            .sum()
            .assign(location="CD")
        )
        daily = prov.assign(target_end_date="2026-10-01", target="daily")
        frames += [
            prov.assign(target="7day", model=name),
            national.assign(target="7day", model=name),
            daily.assign(model=name),
        ]
    out = pd.concat(frames, ignore_index=True).assign(origin=VINTAGE)
    out["value"] = out["value"].astype("float32")
    return out


@pytest.fixture(scope="module")
def made(bz, task, draws, counts, zones, daily_zones):
    return bz.origin_zones(task, draws, counts, zones, daily_zones)


def _provinces(draws: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = draws.loc[
        (draws["model"] == model)
        & (draws["target"] == "7day")
        & (draws["location"] != "CD")
    ]
    return rows[["location", "target_end_date", "sample", "value"]]


def test_a_model_s_own_zone_rows_are_kept_as_cached(made, draws) -> None:
    assert set(made["model"]) == {
        "randomwalknowcast",
        "fm_chronos_daily_ragged_ft",
    }
    model = "fm_chronos_daily_ragged_ft"
    keys = ["location", "target_end_date", "sample"]
    cached = draws.loc[
        (draws["model"] == model) & draws["location"].map(is_zone)
    ]
    got = made.loc[made["model"] == model]
    got = got.set_index(keys)["value"].sort_index()
    want = cached.set_index(keys)["value"].astype("float32").sort_index()
    assert np.array_equal(got.to_numpy(), want.to_numpy())


def test_a_model_with_no_zone_rows_is_split_from_its_provinces(
    bz, task, draws, counts, zones, daily_zones
) -> None:
    # As the old poller scored: the backtest tasks carry no zone data,
    # so a Chronos-2 checkpoint's zones are its province draws split.
    other = draws.loc[
        (draws["model"] == "randomwalknowcast")
        & ~draws["location"].map(is_zone)
    ].assign(model="fm_chronos_weekly_ragged_ft")
    got = bz.origin_zones(task, other, counts, zones, daily_zones)
    assert set(got["zone_method"]) == {"split"}
    zone = got[["location", "target_end_date", "sample", "value"]]
    assert (
        check_zone_coherence(
            _provinces(other, "fm_chronos_weekly_ragged_ft"), zone
        )
        == []
    )


def test_an_origin_with_no_zones_at_all_stops_the_unit(
    bz, task, draws, zones, daily_zones
) -> None:
    empty = pd.DataFrame(columns=["location", "target_end_date", "count"])
    other = draws.loc[
        (draws["model"] == "randomwalknowcast")
        & ~draws["location"].map(is_zone)
    ].assign(model="fm_chronos_weekly_ragged_ft")
    with pytest.raises(ValueError, match="no zone draws"):
        bz.origin_zones(task, other, empty, zones, daily_zones)


def test_zones_cover_only_the_forecast_provinces(made, task, zones) -> None:
    want = {z for z in zones if province_of(z) in task.locations}
    for _, own in made.groupby("model"):
        assert set(own["location"]) == want
    assert (made["target_end_date"].isin(task.target_end_dates)).all()


def test_randomwalknowcast_runs_natively_and_the_rest_are_their_own(
    made,
) -> None:
    method = made.groupby("model")["zone_method"].unique()
    assert method["randomwalknowcast"].tolist() == ["native"]
    assert method["fm_chronos_daily_ragged_ft"].tolist() == ["own"]


def test_the_zone_method_follows_the_model(bz) -> None:
    assert bz.zone_method("randomwalknowcast") == "native"
    for model in (
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_ragged_ft",
        "fm_chronos_daily_complete",
    ):
        assert bz.zone_method(model) == "split"


def test_the_native_zone_run_is_the_submission_s(
    made, task, draws, counts, zones, daily_zones
) -> None:
    # The same seed and stream as zones_for_model in a submission.
    from bdbvhubrenewal.forecasters.zones import zones_for_model

    want = zones_for_model(
        "randomwalknowcast",
        task,
        _provinces(draws, "randomwalknowcast"),
        counts,
        zones,
        daily_zones,
    )
    got = made.loc[made["model"] == "randomwalknowcast"]
    keys = ["location", "target_end_date", "sample"]
    got = got.set_index(keys)["value"].sort_index()
    want = want.set_index(keys)["value"].sort_index()
    assert np.array_equal(got.to_numpy(), want.to_numpy())


def test_only_zones_with_cases_at_the_origin_are_kept(
    bz, task, draws, counts, zones, daily_zones
) -> None:
    active = bz.active_zones(counts)
    assert active == set(counts["location"])
    kept = bz.origin_zones(
        task, draws, counts, zones, daily_zones, active=active
    )
    assert set(kept["location"]) <= active
    assert not kept.empty


def _zone_daily(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """Daily zone rows: (vintage, cutoff, location, onset_date, count)."""
    out = pd.DataFrame(
        rows, columns=["vintage", "location", "onset_date", "observed"]
    )
    return out.assign(imputed=0.0)


def test_the_truth_is_each_zone_week_in_the_first_release_d_days_on(
    bz,
) -> None:
    # Weeks end on Saturdays: 2026-09-05 and 2026-09-12.
    daily = _zone_daily(
        [
            ("v1", "CD-AA-01", "2026-09-01", 2),
            ("v2", "CD-AA-01", "2026-09-01", 3),
            ("v2", "CD-AA-01", "2026-09-08", 4),
            ("v3", "CD-AA-01", "2026-09-01", 9),
            ("v3", "CD-AA-02", "2026-09-08", 1),
        ]
    )
    cutoffs = {"v1": "2026-09-10", "v2": "2026-09-13", "v3": "2026-09-30"}
    truth = bz.zone_truth(daily, cutoffs, delay=7)
    got = truth.set_index(["location", "target_end_date"])
    # 2026-09-05 is due by 09-12: v2 (cut 09-13) is the first.
    assert got.loc[("CD-AA-01", "2026-09-05"), "truth"] == 3
    assert got.loc[("CD-AA-01", "2026-09-05"), "truth_vintage"] == "v2"
    # A zone with no onset that week counts 0 in the release.
    assert got.loc[("CD-AA-02", "2026-09-05"), "truth"] == 0
    # 2026-09-12 is due by 09-19: v3 is the first.
    assert got.loc[("CD-AA-01", "2026-09-12"), "truth"] == 0
    assert got.loc[("CD-AA-02", "2026-09-12"), "truth"] == 1
    assert got.loc[("CD-AA-02", "2026-09-12"), "truth_vintage"] == "v3"
    # No release is 7 days after 2026-09-26, so it has no truth.
    assert "2026-09-26" not in set(truth["target_end_date"])
    # Zero-filled weeks run to the release's cut-off week.
    assert ("CD-AA-01", "2026-09-19") in got.index


def _cell_draws(
    model: str, cells: list[tuple[str, str]], centre: float, origin: str
) -> pd.DataFrame:
    rng = np.random.default_rng(len(model))
    frames = [
        pd.DataFrame(
            {
                "location": loc,
                "target_end_date": week,
                "sample": np.arange(1, 51),
                "value": rng.poisson(centre, 50).astype(float),
            }
        )
        for loc, week in cells
    ]
    return pd.concat(frames, ignore_index=True).assign(
        model=model, origin=origin
    )


def test_zone_rows_are_matched_and_relative_to_randomwalknowcast(
    bz, cs
) -> None:
    cells = [
        ("CD-AA-01", "2026-09-05"),
        ("CD-AA-02", "2026-09-05"),
        ("CD-AA-01", "2026-09-12"),
    ]
    truth = pd.DataFrame(
        {
            "location": [c[0] for c in cells],
            "target_end_date": [c[1] for c in cells],
            "truth": [3.0, 0.0, 5.0],
            "truth_vintage": "v3",
        }
    )
    ref = pd.concat(
        [
            _cell_draws("randomwalknowcast", cells, 3, "v1"),
            _cell_draws("fm_chronos_daily_ragged", cells, 6, "v1"),
        ]
    )
    # The checkpoint misses one cell, so no model is scored on it.
    ck = _cell_draws("fm_chronos_daily_ragged_ft", cells[:2], 4, "v1")
    rows = bz.score_rows({"reference": [ref], "slots-abc": [ck]}, truth)
    assert list(rows.columns) == cs.COLUMNS
    assert set(rows["level"]) == {"zones"}
    assert set(rows["study"]) == {"ebola"}
    assert set(rows["target"]) == {"7day"}
    assert set(rows["rule"]) == {cs.RULE}
    assert (rows["n"] == 2).all()
    by = rows.set_index(["model", "checkpoint"])
    base = by.loc[("randomwalknowcast", "zero-shot")]
    assert base["rel_crps"] == pytest.approx(1.0)
    assert base["zone_method"] == "native"
    assert by.loc[("fm_chronos_daily_ragged", "slots-abc"), "zone_method"] == (
        "split"
    )
    assert by.loc[("fm_chronos_daily_ragged", "zero-shot"), "zone_method"] == (
        "split"
    )


def test_no_zone_rows_without_the_reference(bz, cs, tmp_path) -> None:
    rows = bz.rows(tmp_path, ["slots-abc"], origins=["v1"])
    assert rows.empty
    assert list(rows.columns) == cs.COLUMNS


def _touch(path: Path, when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    os.utime(path, (when, when))


def test_a_zone_unit_is_done_when_each_province_file_has_newer_zones(
    bz, tmp_path
) -> None:
    origins = ["2026-10-09", "2026-10-17"]
    src = tmp_path / "bdbv" / "slots-abc"
    out = tmp_path / "bdbv_zones" / "slots-abc"
    for o in origins:
        _touch(src / f"{o}.csv.gz", 1000)
    assert not bz.unit_done(tmp_path, "slots-abc", origins)
    _touch(out / "2026-10-09.csv.gz", 2000)
    assert not bz.unit_done(tmp_path, "slots-abc", origins)
    _touch(out / "2026-10-17.csv.gz", 2000)
    assert bz.unit_done(tmp_path, "slots-abc", origins)
    # A province file rewritten (new models) makes its zones stale.
    _touch(src / "2026-10-17.csv.gz", 3000)
    assert not bz.unit_done(tmp_path, "slots-abc", origins)


def test_a_zone_unit_waits_for_its_province_draws(bz, tmp_path) -> None:
    origins = ["2026-10-09", "2026-10-17"]
    _touch(tmp_path / "bdbv" / "slots-abc" / "2026-10-09.csv.gz", 1000)
    _touch(tmp_path / "bdbv_zones" / "slots-abc" / "2026-10-09.csv.gz", 2000)
    assert not bz.unit_done(tmp_path, "slots-abc", origins)


def test_checkpoint_scores_knows_the_zone_unit(cs, tmp_path) -> None:
    assert "zone_method" in cs.COLUMNS
    # New columns go last, so older tables keep their order.
    assert cs.COLUMNS[-3:] == ["zone_method", "factor", "factor_level"]
    assert not cs.unit_done(tmp_path, "bdbv_zones", "reference", "daily")


def test_zone_rows_stay_while_a_rewritten_province_file_waits(
    bz, tmp_path
) -> None:
    # The daily target rewrites a province file: the unit runs again,
    # but the label keeps its zone rows meanwhile.
    origins = ["2026-10-17"]
    _touch(tmp_path / "bdbv" / "slots-abc" / "2026-10-17.csv.gz", 3000)
    _touch(tmp_path / "bdbv_zones" / "slots-abc" / "2026-10-17.csv.gz", 2000)
    assert not bz.unit_done(tmp_path, "slots-abc", origins)
    assert bz.has_zones(tmp_path, "slots-abc", origins)
    assert not bz.has_zones(tmp_path, "slots-abc", [*origins, "2026-10-24"])


def test_zone_rows_score_frames_one_at_a_time(bz) -> None:
    cells = [("CD-AA-01", "2026-09-05")]
    truth = pd.DataFrame(
        {
            "location": ["CD-AA-01"],
            "target_end_date": ["2026-09-05"],
            "truth": [3.0],
            "truth_vintage": "v3",
        }
    )
    seen = []

    def frames():
        for origin in ("v1", "v2"):
            seen.append(origin)
            yield _cell_draws("randomwalknowcast", cells, 3, origin)

    rows = bz.score_rows({"reference": frames()}, truth)
    assert seen == ["v1", "v2"]
    assert rows["n_origins"].tolist() == [2]
