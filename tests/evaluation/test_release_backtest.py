from __future__ import annotations

import dataclasses
import datetime as dt

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
    complete_weeks,
    relative_wis,
    release_backtest,
    reported_share,
    summarise,
    thin,
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
    ran = o.loc[o["origin"] == "2026-10-17"]
    assert set(ran["model"]) == {"randomwalk", "randomwalknowcast", BASELINE}
    assert (ran["n_complete"] > 0).all()
    assert (ran["n_reported"] > 0).all()


def test_every_model_is_scored_on_each_later_release(
    result,  # type: ignore[no-untyped-def]
) -> None:
    s = result.scores
    assert set(s["target"]) == {"reported", "complete"}
    assert set(s["model"]) == {"randomwalk", "randomwalknowcast", BASELINE}
    assert set(s["origin"]) == {"2026-10-17", "2026-10-24"}
    rep = s.loc[(s["target"] == "reported") & (s["origin"] == "2026-10-17")]
    assert set(rep["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Every model is scored on the same forecasts of a target.
    for _, g in summarise(s).groupby("target"):
        assert g["n"].nunique() == 1
    # Forecast weeks are scored once a later release reports them.
    assert (rep["weeks_back"] < 0).any()


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


def test_relative_wis_is_against_randomwalk(
    result,  # type: ignore[no-untyped-def]
) -> None:
    rel = relative_wis(result.scores, by=("target",)).set_index(
        ["target", "model"]
    )
    for tgt in ("reported", "complete"):
        assert rel.loc[(tgt, "randomwalk"), "rel_wis"] == pytest.approx(1.0)


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
    assert set(flat["target"]) == {"reported", "complete"}
    assert set(flat["location"]) == {"CD-IT", "CD-NK"}
    # Thinned to the count each release shows, so never above the final.
    rep = out.quantiles.loc[
        (out.quantiles["model"] == "flat")
        & (out.quantiles["target"] == "reported")
    ]
    assert (rep["q95"] <= 100).all()
    assert (rep.loc[rep["weeks_back"] >= 4, "q50"] >= 95).all()


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


@pytest.fixture(scope="module")
def origin(target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame):  # type: ignore[no-untyped-def]
    real = target.loc[target["kind"] == "real"]
    task = make_tasks(
        real.loc[real["vintage"] == "2026-10-24"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )[0]
    return RandomWalk().origin_at(task, errors=False)


def test_share_reported_at_the_origin_is_its_completeness(
    origin,  # type: ignore[no-untyped-def]
) -> None:
    snap = origin.snap
    cut = snap.days[snap.cut[origin.o]].date()
    share = reported_share(origin, cut, snap.week_ends)
    got = share["share"].to_numpy().reshape(len(snap.locations), -1)
    # The first week holds fewer than seven onset days.
    np.testing.assert_allclose(got[:, 1:], origin.complete[:, 1:], rtol=1e-6)


def test_share_is_zero_after_the_cutoff_and_one_when_old(
    origin,  # type: ignore[no-untyped-def]
) -> None:
    cut = dt.date(2026, 10, 31)
    weeks = ["2026-08-29", "2026-10-31", "2026-11-07"]
    share = reported_share(origin, cut, weeks).set_index(
        ["location", "target_end_date"]
    )["share"]
    assert share.between(0, 1).all()
    assert (share.xs("2026-08-29", level=1) == 1).all()
    assert (share.xs("2026-11-07", level=1) == 0).all()
    assert (share.xs("2026-10-31", level=1) < 1).all()


def test_thinning_draws_the_reported_count() -> None:
    samples = pd.DataFrame(
        {
            "location": "CD",
            "target_end_date": ["2026-10-24"] * 4000 + ["2026-10-31"] * 4000,
            "sample": np.tile(np.arange(1, 4001), 2),
            "value": 100.0,
        }
    )
    share = pd.DataFrame(
        {
            "location": "CD",
            "target_end_date": ["2026-10-24", "2026-10-31"],
            "share": [1.0, 0.3],
        }
    )
    out = thin(samples, share, np.random.default_rng(1))
    full = out.loc[out["target_end_date"] == "2026-10-24", "value"]
    part = out.loc[out["target_end_date"] == "2026-10-31", "value"]
    assert (full == 100).all()
    assert part.mean() == pytest.approx(30, abs=0.5)
    assert part.var() == pytest.approx(21, rel=0.1)
