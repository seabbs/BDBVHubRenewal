"""The random walk baselines on release snapshots."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    run_forecasts,
)
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    ModelUnavailableError,
    RandomWalk,
    RandomWalkNowcast,
    add_national,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.chain import MissingSnapshots, NoDelayPair
from bdbvhubrenewal.forecasters.zones import zone_task


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, "2026-10-24")


def test_registry_has_both_models_and_no_daily_variant() -> None:
    specs = {s.name: s for s in list_models()}
    assert "randomwalknowcastdaily" not in specs
    assert specs["randomwalk"].baseline
    assert not specs["randomwalknowcast"].baseline
    assert isinstance(get_model("randomwalk"), RandomWalk)
    assert isinstance(get_model("randomwalknowcast"), RandomWalkNowcast)


@pytest.mark.parametrize("name", ["randomwalk", "randomwalknowcast"])
def test_samples_are_valid_and_reproducible(
    task: ForecastTask, name: str
) -> None:
    a = get_model(name).forecast(task)
    b = get_model(name).forecast(task)
    check_samples(a, task)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["location"]) == set(task.locations)
    assert (a["value"] >= 0).all()
    assert (a["value"] == a["value"].round()).all()


def test_the_two_models_give_different_samples(task: ForecastTask) -> None:
    a = RandomWalk().forecast(task)
    b = RandomWalkNowcast().forecast(task)
    assert not np.array_equal(a["value"].to_numpy(), b["value"].to_numpy())


def test_national_is_the_sum_of_provinces_by_sample(task: ForecastTask) -> None:
    out = add_national(RandomWalkNowcast().forecast(task))
    prov = out.loc[out["location"] != "CD"]
    nat = out.loc[out["location"] == "CD"].set_index(
        ["target_end_date", "sample"]
    )["value"]
    summed = prov.groupby(["target_end_date", "sample"])["value"].sum()
    assert (summed == nat.reindex(summed.index)).all()


def test_randomwalk_returns_complete_weeks_as_reported(
    task: ForecastTask,
) -> None:
    weeks = tuple(
        w.date().isoformat()
        for w in pd.date_range("2026-08-29", "2026-11-21", freq="7D")
    )
    task = dataclasses.replace(task, target_end_dates=weeks)
    model = RandomWalk()
    out = model.forecast(task)
    info = pd.DataFrame(model.fit_diagnostics).set_index("location")
    seen = task.target.set_index(["location", "target_end_date"])["total"]
    for loc in ("CD-IT", "CD-NK"):
        start = info.loc[loc, "start_week"]
        rows = out.loc[
            (out["location"] == loc) & (out["target_end_date"] <= start)
        ]
        assert len(rows) > 0
        for week, grp in rows.groupby("target_end_date"):
            assert (grp["value"] == seen[(loc, week)]).all()
    # Later weeks are not flat: the walk adds noise that grows with horizon.
    late = out.loc[
        (out["location"] == "CD-IT") & (out["target_end_date"] == "2026-11-14")
    ]
    assert late["value"].nunique() > 5


def test_the_start_week_is_the_latest_week_at_95_per_cent(
    task: ForecastTask,
) -> None:
    model = RandomWalk()
    model.forecast(task)
    info = pd.DataFrame(model.fit_diagnostics).set_index("location")
    assert info.loc["CD-IT", "start_week"] <= "2026-10-10"
    assert info.loc["CD-IT", "start_week"] >= "2026-09-12"


def test_nowcast_starts_from_the_latest_full_week(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    # Cut-off 10-24 is a Saturday, so its week is full.
    model = RandomWalkNowcast()
    model.forecast(_task(target, delays, daily, "2026-10-24"))
    assert {r["start_week"] for r in model.fit_diagnostics} == {"2026-10-24"}
    # Cut-off 10-16 is a Friday, so the week of 10-17 is partial.
    model.forecast(_task(target, delays, daily, "2026-10-17"))
    assert {r["start_week"] for r in model.fit_diagnostics} == {"2026-10-10"}


def test_a_partial_week_adds_the_days_after_the_cutoff(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    task = _task(target, delays, daily, "2026-10-17")
    nowcast = RandomWalkNowcast().forecast(task)
    plain = RandomWalk().forecast(task)
    wk = "2026-10-17"
    mine = nowcast.loc[
        (nowcast["location"] == "CD-IT") & (nowcast["target_end_date"] == wk)
    ]["value"]
    # The week has 6 of 7 days, so no draw is below the nowcast of 6 days.
    assert mine.mean() > 100
    assert mine.nunique() > 10
    assert plain["target_end_date"].min() <= wk


def test_diagnostics_name_the_sources(task: ForecastTask) -> None:
    model = RandomWalkNowcast()
    model.forecast(task)
    info = pd.DataFrame(model.fit_diagnostics).set_index("location")
    assert {
        "start_week", "n_err", "delay_source", "rho_source", "sigma_source"
    } <= set(info.columns)  # fmt: skip
    assert info.loc["CD-IT", "delay_source"] == "own:CD-IT"
    assert info.loc["CD-TO", "delay_source"] == "own:CD" or info.loc[
        "CD-TO", "delay_source"
    ].endswith("CD")
    assert info.loc["CD-IT", "rho_source"] == "pooled:CD"
    assert info.loc["CD-IT", "n_err"] == 4


def test_both_models_use_the_one_random_walk(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bdbvhubrenewal.forecasters import randomwalk as mod

    walk = mod.random_walk
    calls: list[int] = []

    def spy(*args, **kwargs):
        calls.append(1)
        return walk(*args, **kwargs)

    monkeypatch.setattr(mod, "random_walk", spy)
    mod.RandomWalk().forecast(task)
    assert calls
    n = len(calls)
    mod.RandomWalkNowcast().forecast(task)
    assert len(calls) > n


def test_models_raise_without_snapshots_or_a_delay_pair(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    bare = make_tasks(
        target.loc[target["vintage"] == "2026-10-24"],
        delays,
        BacktestConfig(seed=3),
    )[0]
    early = _task(target, delays, daily, "2026-10-09")
    for name in ("randomwalk", "randomwalknowcast"):
        with pytest.raises(MissingSnapshots):
            get_model(name).forecast(bare)
        with pytest.raises(NoDelayPair):
            get_model(name).forecast(early)
    assert issubclass(NoDelayPair, ModelUnavailableError)


def test_the_backtest_skips_origins_a_model_cannot_forecast(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real, delays, BacktestConfig(seed=3), history=real, daily=daily
    )
    out = run_forecasts([get_model("randomwalk")], tasks)
    assert set(out["vintage"]) == {"2026-10-17", "2026-10-24", "2026-11-01"}


def test_zones_run_from_the_same_code(
    task: ForecastTask, daily: pd.DataFrame, daily_zones: pd.DataFrame
) -> None:
    zones = tuple(sorted(set(daily_zones["location"])))
    counts = weekly_zone_counts(daily_zones, task.vintage)
    zt = zone_task(task, counts, zones, pd.concat([daily, daily_zones]))
    zt = dataclasses.replace(zt, locations=zt.locations[:60])
    out = RandomWalkNowcast().forecast(zt)
    check_samples(out, zt)
