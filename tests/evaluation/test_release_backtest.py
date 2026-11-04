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
    draw_reported,
    relative_wis,
    release_backtest,
    reported_share,
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


def test_energy_scores_each_scale_on_the_same_samples(
    result,  # type: ignore[no-untyped-def]
) -> None:
    e = result.energy
    unit = ["origin", "target", "truth_vintage", "scale"]
    assert set(e["scale"]) == {"national", "provinces"}
    assert set(e["model"]) == {"randomwalk", "randomwalknowcast", BASELINE}
    # One vector per unit, the same cells for every model.
    assert (e.groupby(unit)["n_cells"].nunique() == 1).all()
    assert (e.groupby(unit)["model"].nunique() == 3).all()
    # The vector holds every scored cell of the scale.
    s = result.scores.loc[result.scores["model"] == "randomwalk"]
    nat = s.loc[s["location"] == "CD"].groupby(unit[:3]).size()
    got = e.loc[(e["model"] == "randomwalk") & (e["scale"] == "national")]
    assert got.set_index(unit[:3])["n_cells"].equals(nat.rename("n_cells"))


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


def test_relative_wis_uses_only_matched_baseline_forecasts() -> None:
    def row(model: str, week: str, wis: float) -> dict[str, object]:
        return {
            "model": model,
            "target": "reported",
            "origin": "2026-10-17",
            "truth_vintage": "2026-10-24",
            "location": "CD",
            "target_end_date": week,
            "wis": wis,
            "log_wis": wis / 10,
            "coverage_50": 1.0,
            "coverage_90": 1.0,
            "bias": 0.5,
        }

    scores = pd.DataFrame(
        [
            row("randomwalk", "2026-10-17", 2.0),
            row("a", "2026-10-17", 1.0),
            # No randomwalk forecast for this week, so it is left out.
            row("a", "2026-10-24", 100.0),
        ]
    )
    rel = relative_wis(scores).set_index("model")
    assert rel.loc["a", "n"] == 1
    assert rel.loc["a", "rel_wis"] == pytest.approx(0.5)
    assert rel.loc["a", "rel_log_wis"] == pytest.approx(0.5)
    assert rel.loc["a", "bias"] == pytest.approx(0.5)


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
    # Old weeks keep the forecast final count at every later release.
    rep = out.quantiles.loc[
        (out.quantiles["model"] == "flat")
        & (out.quantiles["target"] == "reported")
    ]
    assert (rep.loc[rep["weeks_back"] >= 6, "q50"] == 100).all()
    # flat covers two provinces, so the energy score's shared cells do too,
    # unless it is left out of the compared models.
    prov = out.energy.loc[out.energy["scale"] == "provinces"]
    assert set(prov["model"]) == {"flat", "randomwalk", BASELINE}
    cells = prov.groupby("truth_vintage")["n_cells"].max()
    ours = release_backtest(
        tasks,
        daily,
        target,
        models=("randomwalk",),
        stored=stored,
        compare=("randomwalk", BASELINE),
    ).energy
    ours = ours.loc[ours["scale"] == "provinces"]
    assert "flat" not in set(ours["model"])
    assert (ours.groupby("truth_vintage")["n_cells"].max() > cells).all()


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


def test_draw_keeps_the_reported_count_and_thins_the_outstanding_count() -> (
    None
):
    weeks = ["2026-10-17", "2026-10-24", "2026-10-31"]
    n = 4000
    keys = {"location": "CD", "target_end_date": np.repeat(weeks, n)}
    samples = pd.DataFrame({**keys, "sample": np.tile(np.arange(n), 3)})
    samples["value"] = 100.0
    reported = pd.DataFrame(
        {
            "location": "CD",
            "target_end_date": weeks[:2],
            "reported": [100.0, 40.0],
        }
    )
    now = pd.DataFrame(
        {"location": "CD", "target_end_date": weeks, "share": [1.0, 0.4, 0.0]}
    )
    then = now.assign(share=[1.0, 0.7, 0.5])
    out = draw_reported(samples, reported, now, then, np.random.default_rng(1))
    got = out.groupby("target_end_date")["value"]
    # Complete at the origin: the final count stands. Part reported: half
    # of the outstanding 60 arrives. After the origin: half of 100.
    assert got.mean().tolist() == pytest.approx([100.0, 70.0, 50.0], abs=0.5)
    assert got.var().tolist() == pytest.approx([0.0, 15.0, 25.0], rel=0.1)
    assert (out.loc[out["target_end_date"] == weeks[1], "value"] >= 40).all()
    floor = samples.assign(value=np.repeat([100.0, 40.0, 0.0], n))
    out = draw_reported(floor, reported, now, then, np.random.default_rng(1))
    assert out.groupby("target_end_date")["value"].unique().map(
        list
    ).tolist() == [
        [100.0],
        [40.0],
        [0.0],
    ]
