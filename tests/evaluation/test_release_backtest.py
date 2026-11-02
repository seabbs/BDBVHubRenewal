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
    pooled_summary,
    relative_wis,
    release_backtest,
    summarise,
    weeks_back_from_cutoff,
)
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.chain import MissingSnapshots


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
    assert (early["n_next"] == 0).all()
    assert (early["n_complete"] == 0).all()
    ran = o.loc[(o["origin"] == "2026-10-17") & (o["status"] == "ran")]
    assert set(ran["model"]) == {"randomwalk", "randomwalknowcast"}
    assert (ran["n_next"] > 0).all()


def test_scores_cover_the_three_targets_and_the_floor(
    result,  # type: ignore[no-untyped-def]
) -> None:
    s = result.scores
    assert set(s["target"]) == {"next", "complete", "latest"}
    assert set(s["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        BASELINE,
    }
    assert set(s["origin"]) == {"2026-10-17", "2026-10-24"}
    nxt = s.loc[s["target"] == "next"]
    assert set(nxt["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Every model is scored on the same forecasts of a target.
    n = summarise(s)
    for _, g in n.groupby("target"):
        assert g["n"].nunique() == 1


def test_randomwalk_is_not_scored_on_the_next_release(
    result,  # type: ignore[no-untyped-def]
) -> None:
    # Its samples are of the final count, so they do not predict the
    # count at the next release.
    nxt = result.scores.loc[result.scores["target"] == "next"]
    assert set(nxt["model"]) == {"randomwalknowcast", BASELINE}
    o = result.origins
    rw = o.loc[(o["model"] == "randomwalk") & (o["status"] == "ran")]
    assert (rw["n_next"] == 0).all()
    assert rw["next"].str.contains("not scored").all()


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
    for (_, pool), g in pooled.groupby(["target", "pool"]):
        assert g["n"].nunique() == 1
    every = pooled.loc[pooled["pool"] == "provinces", "n"].max()
    some = pooled.loc[pooled["pool"] == "provinces without near-zero", "n"]
    assert some.max() < every


def test_relative_wis_is_against_randomwalk_or_the_floor(
    result,  # type: ignore[no-untyped-def]
) -> None:
    table = summarise(result.scores, by=("target", "model"))
    rel = relative_wis(table, by=("target",))
    done = rel.loc[rel["target"] == "complete"].set_index("model")
    assert done.loc["randomwalk", "rel_wis"] == pytest.approx(1.0)
    assert (done["rel_to"] == "randomwalk").all()
    nxt = rel.loc[rel["target"] == "next"].set_index("model")
    assert nxt.loc[BASELINE, "rel_wis"] == pytest.approx(1.0)
    assert (nxt["rel_to"] == BASELINE).all()


def test_stored_forecasts_are_scored_on_the_final_targets_only(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real.loc[real["vintage"] == "2026-10-24"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )
    task = tasks[0]
    weeks = list(task.target_end_dates)
    stored = pd.DataFrame(
        {
            "model": "flat",
            "vintage": "2026-10-24",
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
    assert set(flat["target"]) == {"complete", "latest"}
    assert set(flat["location"]) == {"CD-IT", "CD-NK"}


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
        (table["location"] == "CD") & (table["weeks_back"] == 0)
    ].set_index("clock_shift")
    assert cd.loc[-1, "q50"] > cd.loc[0, "q50"] > cd.loc[1, "q50"]
    assert set(table["model"]) == {"randomwalknowcast", "randomwalk"}


def test_next_target_uses_the_chain_over_the_gap(
    result,  # type: ignore[no-untyped-def]
) -> None:
    o = result.origins
    row = o.loc[
        (o["origin"] == "2026-10-17") & (o["model"] == "randomwalknowcast")
    ]
    assert row["mismatch_2026-10-24"].iloc[0] == 1


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
