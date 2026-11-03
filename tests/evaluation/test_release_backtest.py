from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    check_distinct,
    make_tasks,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.nowcast import (
    BASELINE,
    clock_sensitivity,
    complete_weeks,
    expected_reported,
    pooled_summary,
    relative_wis,
    release_backtest,
    reported_share,
    stored_finals,
    summarise,
    weeks_back_from_cutoff,
)
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.chain import MissingSnapshots
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk


@pytest.fixture(scope="module")
def result(target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame):  # type: ignore[no-untyped-def]
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real,
        delays.assign(kind="real"),
        BacktestConfig(n_samples=50, seed=2),
        daily=daily,
    )
    return release_backtest(tasks, daily, target)


def test_origins_without_a_delay_pair_are_listed_with_no_forecasts(
    result,  # type: ignore[no-untyped-def]
) -> None:
    o = result.origins
    early = o.loc[o["origin"].isin(["2026-08-18", "2026-10-09"])]
    assert set(early["status"]) == {"no forecast"}
    assert (early["n_reported"] == 0).all()
    assert (early["n_complete"] == 0).all()
    ran = o.loc[(o["origin"] == "2026-10-17") & (o["status"] == "ran")]
    assert set(ran["model"]) == {"randomwalk", "randomwalknowcast"}
    assert (ran["n_complete"] > 0).all()
    assert (ran["n_reported"] > 0).all()


def test_scores_cover_the_two_targets_and_the_floor(
    result,  # type: ignore[no-untyped-def]
) -> None:
    s = result.scores
    assert set(s["target"]) == {"reported", "complete"}
    assert set(s["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        BASELINE,
    }
    assert set(s["origin"]) == {"2026-10-17", "2026-10-24"}
    rep = s.loc[s["target"] == "reported"]
    assert set(rep["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Every model is scored on the same forecasts of a target.
    n = summarise(s)
    for _, g in n.groupby("target"):
        assert g["n"].nunique() == 1


def test_reported_target_scores_forecast_weeks_the_release_covers(
    result,  # type: ignore[no-untyped-def]
) -> None:
    rep = result.scores.loc[result.scores["target"] == "reported"]
    late = rep.loc[(rep["origin"] == "2026-10-17")]
    # 10-24 is a forecast week from 10-17 and the 11-01 release covers it.
    assert (late["weeks_back"] < 0).any()
    assert set(late.loc[late["weeks_back"] < 0, "truth_vintage"]) == {
        "2026-10-24",
        "2026-11-01",
    }


@pytest.fixture(scope="module")
def origin_1017(
    delays: pd.DataFrame, target: pd.DataFrame, daily: pd.DataFrame
):  # type: ignore[no-untyped-def]
    real = target.loc[target["kind"] == "real"]
    task = make_tasks(
        real.loc[real["vintage"] == "2026-10-17"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )[0]
    return task, RandomWalk().origin_at(task, errors=False)


def test_reported_share_matches_the_origin_and_grows_with_time(
    origin_1017,  # type: ignore[no-untyped-def]
) -> None:
    _, origin = origin_1017
    snap = origin.snap
    cut = snap.days[snap.cut[origin.o]].date()
    weeks = list(snap.week_ends)
    now = reported_share(origin, cut, weeks)
    # At the origin's own cut-off it is the origin's completeness.
    got = now.pivot(
        index="location", columns="target_end_date", values="share"
    ).loc[list(snap.locations), weeks]
    np.testing.assert_allclose(got.to_numpy(), origin.complete)
    later = reported_share(origin, cut + pd.Timedelta(days=8), weeks)
    both = now.merge(later, on=["location", "target_end_date"])
    assert (both["share_y"] >= both["share_x"] - 1e-12).all()
    # Every full week is reported in full long after.
    far = reported_share(origin, cut + pd.Timedelta(days=400), weeks[1:])
    assert np.allclose(far["share"], 1.0)
    # A week after the cut-off has nothing reported.
    ahead = reported_share(origin, cut, ["2026-11-07"])
    assert (ahead["share"] == 0).all()


def test_expected_reported_runs_from_the_count_reported_to_the_final(
    origin_1017,  # type: ignore[no-untyped-def]
    target: pd.DataFrame,
) -> None:
    task, origin = origin_1017
    snap = origin.snap
    cut = snap.days[snap.cut[origin.o]].date()
    rows = target.loc[target["vintage"] == "2026-10-17"]
    reported = rows[["location", "target_end_date", "total"]].rename(
        columns={"total": "reported"}
    )
    weeks = list(task.target_end_dates)
    samples = pd.DataFrame(
        {
            "location": "CD-IT",
            "target_end_date": np.repeat(weeks, 2),
            "sample": np.tile([1, 2], len(weeks)),
            "value": np.tile([50.0, 500.0], len(weeks)),
        }
    )
    have = reported.set_index(["location", "target_end_date"])["reported"]
    now = expected_reported(samples, origin, reported, cut)
    # At the origin's release, the count expected is the count reported.
    want = [float(have.get(("CD-IT", w), 0.0)) for w in now["target_end_date"]]
    np.testing.assert_allclose(now["value"], want)
    # Long after, it is the forecast of the final count.
    far = expected_reported(
        samples, origin, reported, cut + pd.Timedelta(days=400)
    )
    np.testing.assert_allclose(far["value"], samples["value"])
    # In between it lies between the two and keeps the samples' order.
    mid = expected_reported(
        samples, origin, reported, cut + pd.Timedelta(days=8)
    )
    lo = np.minimum(now["value"], far["value"])
    hi = np.maximum(now["value"], far["value"])
    assert ((mid["value"] >= lo - 1e-9) & (mid["value"] <= hi + 1e-9)).all()
    pair = mid["value"].to_numpy().reshape(-1, 2)
    assert (pair[:, 0] <= pair[:, 1]).all()


def test_complete_weeks_come_from_later_releases_only(
    target: pd.DataFrame, daily: pd.DataFrame
) -> None:
    done = complete_weeks(target, daily, "2026-11-01")
    cd = done.loc[done["location"] == "CD"].set_index("target_end_date")
    # National 09-26 grew 420 to 423 from 10-24 to 11-01 and is 5 weeks
    # old, 10-03 grew 478 to 492 (2.9%), and 10-17 is too young.
    assert cd.loc["2026-09-26", "complete"]
    assert not cd.loc["2026-10-03", "complete"]
    assert not cd.loc["2026-10-17", "complete"]
    assert cd.loc["2026-10-03", "growth"] == pytest.approx(14 / 478)
    by_age = complete_weeks(target, daily, "2026-11-01", rule="age", weeks=6)
    cd = by_age.loc[by_age["location"] == "CD"].set_index("target_end_date")
    assert cd.loc["2026-09-19", "complete"]
    assert not cd.loc["2026-09-26", "complete"]
    with pytest.raises(ValueError, match="rule"):
        complete_weeks(target, daily, "2026-11-01", rule="pi")


def test_complete_target_only_has_complete_weeks(
    result,  # type: ignore[no-untyped-def]
    target: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    s = result.scores.loc[result.scores["target"] == "complete"]
    done = complete_weeks(target, daily, "2026-11-01")
    ok = set(
        zip(
            done.loc[done["complete"], "location"],
            done.loc[done["complete"], "target_end_date"],
            strict=True,
        )
    )
    assert set(zip(s["location"], s["target_end_date"], strict=True)) <= ok
    assert s["complete_at_latest"].all()


def test_weeks_back_count_from_the_cutoff_week(
    daily: pd.DataFrame,
) -> None:
    weeks = pd.Series(["2026-10-31", "2026-10-24", "2026-11-07"])
    # 11-01 is a Sunday whose cut-off is Saturday 10-31.
    got = weeks_back_from_cutoff(daily, "2026-11-01", weeks)
    assert got.tolist() == [0, 1, -1]


def test_pooled_summaries_say_which_places_they_hold(
    result,  # type: ignore[no-untyped-def]
) -> None:
    s = result.scores
    assert {"near_zero", "level"} <= set(s)
    small = set(s.loc[s["near_zero"], "location"])
    assert {"CD-BU", "CD-SU"} <= small
    assert not small & {"CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO"}
    pooled = pooled_summary(s, by=("target", "model"))
    assert set(pooled["pool"]) == {
        "national",
        "provinces",
        "provinces without near-zero",
    }
    for _, g in pooled.groupby(["target", "pool"]):
        assert g["n"].nunique() == 1
    every = pooled.loc[pooled["pool"] == "provinces", "n"].max()
    some = pooled.loc[pooled["pool"] == "provinces without near-zero", "n"]
    assert some.max() < every


def test_relative_wis_is_against_randomwalk_or_the_floor(
    result,  # type: ignore[no-untyped-def]
) -> None:
    rel = relative_wis(result.scores, by=("target",))
    done = rel.loc[rel["target"] == "complete"].set_index("model")
    assert done.loc["randomwalk", "rel_wis"] == pytest.approx(1.0)
    assert (done["rel_to"] == "randomwalk").all()
    rep = rel.loc[rel["target"] == "reported"].set_index("model")
    assert rep.loc["randomwalk", "rel_wis"] == pytest.approx(1.0)
    assert (rep["rel_to"] == "randomwalk").all()
    # Coverage is of the forecasts matched with the baseline.
    assert {"coverage_50", "coverage_90"} <= set(rel)
    assert rel["coverage_90"].between(0, 1).all()


def test_stored_forecasts_are_scored_on_both_targets(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real.loc[real["vintage"] == "2026-10-17"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )
    task = tasks[0]
    weeks = list(task.target_end_dates)
    stored = pd.DataFrame(
        {
            "model": "flat",
            "vintage": "2026-10-17",
            "location": np.repeat(["CD-IT", "CD-NK"], len(weeks) * 20),
            "target_end_date": np.tile(np.repeat(weeks, 20), 2),
            "sample": np.tile(np.arange(1, 21), 2 * len(weeks)),
            "value": 100.0,
        }
    )
    out = release_backtest(
        tasks, daily, target, models=("randomwalk",), stored=stored
    )
    flat = out.scores.loc[out.scores["model"] == "flat"]
    assert set(flat["target"]) == {"complete", "reported"}
    assert set(flat["location"]) == {"CD-IT", "CD-NK"}
    rw = out.scores.loc[out.scores["model"] == "randomwalk"]
    assert set(rw["target"]) == {"complete", "reported"}


def test_clock_sensitivity_moves_the_newest_week(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real.loc[real["vintage"] == "2026-10-24"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=50, seed=3),
        daily=daily,
    )
    table = clock_sensitivity(tasks, daily, target, shifts=(-1, 0, 1))
    cd = table.loc[
        (table["location"] == "CD")
        & (table["weeks_back"] == 0)
        & (table["model"] == "randomwalknowcast")
    ].set_index("clock_shift")
    assert cd.loc[-1, "q50"] > cd.loc[0, "q50"] > cd.loc[1, "q50"]
    assert set(table["model"]) == {"randomwalknowcast", "randomwalk"}


def test_quantiles_carry_the_reported_count_and_truth(
    result,  # type: ignore[no-untyped-def]
) -> None:
    q = result.quantiles
    assert {"q5", "q25", "q50", "q75", "q95", "reported", "truth"} <= set(q)
    assert (q["q5"] <= q["q95"]).all()


def test_missing_snapshots_stop_the_backtest(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(real, delays.assign(kind="real"), BacktestConfig())
    with pytest.raises(MissingSnapshots):
        release_backtest(tasks, daily, target)
    with pytest.raises(MissingSnapshots):
        run_forecasts([get_model("randomwalk")], tasks)


def test_identical_samples_from_two_models_stop_the_run() -> None:
    base = pd.DataFrame(
        {
            "kind": "real",
            "vintage": "2026-10-17",
            "location": "CD",
            "target_end_date": "2026-10-10",
            "sample": np.arange(1, 4),
            "value": [1.0, 2.0, 3.0],
        }
    )
    a, b = base.assign(model="a"), base.assign(model="b")
    with pytest.raises(ValueError, match="identical"):
        check_distinct(pd.concat([a, b]))
    check_distinct(
        pd.concat([a, dataclasses.replace(b) if False else b.assign(value=4.0)])
    )


def test_stored_finals_take_final_counts_and_match_hub_rounds() -> None:
    row = {"location": "CD", "target_end_date": "2026-10-24", "sample": 1}
    forecasts = pd.DataFrame(
        [
            {**row, "model": "renewal", "kind": "real",
             "vintage": "2026-10-17", "value": 5.0, "as_of": np.nan},
            {**row, "model": "renewal", "kind": "real",
             "vintage": "2026-10-17", "value": 4.0, "as_of": "2026-10-24"},
            {**row, "model": "randomwalk", "kind": "real",
             "vintage": "2026-10-17", "value": 3.0, "as_of": np.nan},
        ]
    )  # fmt: skip
    hub = pd.DataFrame(
        [
            {**row, "model": "team", "source": "hub",
             "reference_date": "2026-10-31", "value": 7.0},
            {**row, "model": "team", "source": "hub",
             "reference_date": "2026-12-05", "value": 8.0},
        ]
    )  # fmt: skip
    got = stored_finals(
        forecasts,
        hub,
        ["2026-10-17", "2026-10-24", "2026-11-01"],
        exclude=("randomwalk",),
    )
    assert list(got.columns) == [
        "model", "vintage", "location", "target_end_date", "sample", "value",
    ]  # fmt: skip
    got = got.set_index("model")
    # The vintage-matched row is not a final count, and the round with no
    # serving vintage is dropped.
    assert got.loc["renewal", "value"] == 5.0
    assert got.loc["team", "vintage"] == "2026-10-24"
    assert len(got) == 2
