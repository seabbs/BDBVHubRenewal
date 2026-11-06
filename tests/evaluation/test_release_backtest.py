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
    complete_weeks,
    delay_curves,
    draw_reported,
    relative_wis,
    release_backtest,
    summarise,
    week_shares,
    weeks_back_from_cutoff,
)
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.chain import MissingSnapshots
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalkNowcast,
    chain_delay_summary,
)


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


def test_a_model_that_cannot_forecast_is_listed_with_no_forecasts(
    result,  # type: ignore[no-untyped-def]
) -> None:
    o = result.origins.set_index(["origin", "model"])
    for early in ("2026-08-18", "2026-10-09"):
        # No pair of releases is in the delay window, so no nowcast.
        assert o.loc[(early, "randomwalknowcast"), "status"] == "no forecast"
        assert o.loc[(early, "randomwalknowcast"), "n_reported"] == 0
        # The walk on the counts as reported needs no delay.
        assert o.loc[(early, "randomwalk"), "status"] == "scored"
        assert o.loc[(early, "randomwalk"), "n_reported"] > 0
    ran = o.loc["2026-10-17"]
    assert set(ran.index) == {"randomwalk", "randomwalknowcast"}
    assert (ran["n_complete"] > 0).all()
    assert (ran["n_reported"] > 0).all()


def test_every_model_is_scored_on_each_later_release(
    result,  # type: ignore[no-untyped-def]
) -> None:
    s = result.scores
    assert set(s["target"]) == {"reported", "complete"}
    assert set(s["model"]) == {"randomwalk", "randomwalknowcast"}
    both = s.loc[s["origin"].isin(["2026-10-17", "2026-10-24"])]
    rep = both.loc[(both["target"] == "reported")]
    rep = rep.loc[rep["origin"] == "2026-10-17"]
    assert set(rep["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Where both forecast, both are scored on the same forecasts.
    for _, g in summarise(both).groupby("target"):
        assert g["n"].nunique() == 1
    # Forecast weeks are scored once a later release reports them.
    assert (rep["weeks_back"] < 0).any()


def test_randomwalk_is_scored_on_its_samples_as_they_are(
    result,  # type: ignore[no-untyped-def]
) -> None:
    q = result.quantiles
    rw = q.loc[(q["model"] == "randomwalk") & (q["origin"] == "2026-10-17")]
    cols = ["location", "target_end_date", "q5", "q50", "q95"]
    rep = rw.loc[rw["target"] == "reported"]
    by_release = [
        g[cols].reset_index(drop=True) for _, g in rep.groupby("truth_vintage")
    ]
    pd.testing.assert_frame_equal(by_release[0], by_release[1])


def test_energy_scores_each_scale_on_the_same_samples(
    result,  # type: ignore[no-untyped-def]
) -> None:
    e = result.energy.loc[result.energy["origin"] >= "2026-10-17"]
    unit = ["origin", "target", "truth_vintage", "scale"]
    assert set(e["scale"]) == {"national", "provinces"}
    assert set(e["model"]) == {"randomwalk", "randomwalknowcast"}
    # One vector per unit, the same cells for every model.
    assert (e.groupby(unit)["n_cells"].nunique() == 1).all()
    assert (e.groupby(unit)["model"].nunique() == 2).all()
    # The vector holds every scored cell of the scale.
    s = result.scores.loc[
        (result.scores["model"] == "randomwalk")
        & (result.scores["origin"] >= "2026-10-17")
    ]
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


def test_a_stored_model_is_mapped_by_its_own_delay_or_scored_complete(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real.loc[real["vintage"] == "2026-10-17"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )
    weeks = list(tasks[0].target_end_dates)
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
    # A delay that reports every case on its onset day.
    learnt = pd.DataFrame(
        {
            "model": "flat",
            "vintage": "2026-10-17",
            "location": "pooled",
            "x": [0, 1],
            "quantity": "reported_share",
            "q50": [1.0, 1.0],
        }
    )
    out = release_backtest(
        tasks,
        daily,
        target,
        models=("randomwalk",),
        stored=stored,
        delays=learnt,
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
    # Without a modelled delay, only complete weeks are scored.
    bare = release_backtest(
        tasks, daily, target, models=("randomwalk",), stored=stored
    )
    flat = bare.scores.loc[bare.scores["model"] == "flat"]
    assert set(flat["target"]) == {"complete"}


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
        run_forecasts([get_model("randomwalknowcast")], tasks)


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


def test_randomwalknowcast_s_delay_gives_its_completeness(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    task = make_tasks(
        real.loc[real["vintage"] == "2026-10-24"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )[0]
    origin = RandomWalkNowcast().origin_at(task, errors=False)
    snap = origin.snap
    curves = delay_curves(chain_delay_summary(origin, "m", task.vintage))
    cut = snap.days[snap.cut[origin.o]].date()
    grid = pd.MultiIndex.from_product(
        [snap.locations, snap.week_ends],
        names=["location", "target_end_date"],
    ).to_frame(index=False)
    share = week_shares(curves, cut, grid.assign(value=1.0)).set_index(
        ["location", "target_end_date"]
    )["share"]
    got = share.reindex(pd.MultiIndex.from_frame(grid)).to_numpy()
    got = got.reshape(len(snap.locations), -1)
    # The first week holds fewer than seven onset days.
    np.testing.assert_allclose(got[:, 1:], origin.complete[:, 1:], rtol=1e-6)


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


def test_a_models_reported_scores_do_not_depend_on_other_models(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real.loc[real["vintage"] == "2026-10-17"],
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=3),
        daily=daily,
    )
    weeks = list(tasks[0].target_end_dates)
    stored = pd.DataFrame(
        {
            "model": "flat",
            "vintage": "2026-10-17",
            "location": "CD-IT",
            "target_end_date": np.repeat(weeks, 20),
            "sample": np.tile(np.arange(1, 21), len(weeks)),
            "value": 100.0,
        }
    )

    def reported(out: pd.DataFrame) -> pd.DataFrame:
        keep = out.loc[
            (out["model"] == "randomwalk") & (out["target"] == "reported")
        ]
        return keep.sort_values(list(keep.columns)).reset_index(drop=True)

    alone = release_backtest(tasks, daily, target, models=("randomwalk",))
    both = release_backtest(
        tasks, daily, target, models=("randomwalk",), stored=stored
    )
    pd.testing.assert_frame_equal(reported(alone.scores), reported(both.scores))
