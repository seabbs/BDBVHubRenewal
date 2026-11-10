"""The random walk baselines on release snapshots."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

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
from bdbvhubrenewal.forecasters.chain import (
    MissingSnapshots,
    NoCompleteWeek,
    NoDelayPair,
    Settings,
)


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, "2026-10-24")


@pytest.fixture(scope="module")
def runs(task: ForecastTask) -> dict[str, tuple[RandomWalk, pd.DataFrame]]:
    """Each model and its samples at 2026-10-24, run once for the module."""
    out = {}
    for name in ("randomwalk", "randomwalknowcast"):
        model = get_model(name)
        out[name] = (model, model.forecast(task))
    return out


def test_registry_has_both_models() -> None:
    specs = {s.name: s for s in list_models()}
    assert specs["randomwalk"].baseline
    assert not specs["randomwalknowcast"].baseline
    assert isinstance(get_model("randomwalk"), RandomWalk)
    assert isinstance(get_model("randomwalknowcast"), RandomWalkNowcast)


@pytest.mark.parametrize("name", ["randomwalk", "randomwalknowcast"])
def test_samples_are_valid_and_reproducible(
    task: ForecastTask,
    runs: dict[str, tuple[RandomWalk, pd.DataFrame]],
    name: str,
) -> None:
    a = runs[name][1]
    b = get_model(name).forecast(task)
    check_samples(a, task)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["location"]) == set(task.locations)
    assert (a["value"] >= 0).all()
    assert (a["value"] == a["value"].round()).all()
    other = runs[({"randomwalk", "randomwalknowcast"} - {name}).pop()][1]
    assert not np.array_equal(a["value"].to_numpy(), other["value"].to_numpy())


def test_national_is_the_sum_of_provinces_by_sample(
    runs: dict[str, tuple[RandomWalk, pd.DataFrame]],
) -> None:
    out = add_national(runs["randomwalknowcast"][1])
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


def test_nowcast_runs_to_the_latest_week_at_least_5_per_cent_reported(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    assert not hasattr(Settings(), "start_count")
    model = RandomWalkNowcast()
    model.forecast(_task(target, delays, daily, "2026-10-24"))
    start = {r["location"]: r["start_week"] for r in model.fit_diagnostics}
    # Cut-off 10-24 is a Saturday, and its week is at least 5% reported.
    assert start["CD-IT"] == "2026-10-24"
    assert max(start.values()) == "2026-10-24"
    # Cut-off 10-16 is a Friday, so the week of 10-17 is partial. Its
    # missing Saturday counts as unreported, and the week is still at least
    # 5% reported, so it is nowcast and the walk starts from it.
    task = _task(target, delays, daily, "2026-10-17")
    model.forecast(task)
    start = {r["location"]: r["start_week"] for r in model.fit_diagnostics}
    assert start["CD-IT"] == "2026-10-17"
    origin = model.origin_at(task, errors=False)
    i = origin.snap.locations.index("CD-IT")
    assert origin.complete[i, -1] >= 0.05


def test_the_partial_week_is_nowcast_whole(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    task = _task(target, delays, daily, "2026-10-17")
    model = RandomWalkNowcast()
    samples = model.forecast(task)
    origin = model.origin_at(task, errors=True)
    i = origin.snap.locations.index("CD-IT")
    mine = samples.loc[
        (samples["location"] == "CD-IT")
        & (samples["target_end_date"] == "2026-10-17")
    ]["value"]
    # The draws of the week holding the cut-off are its nowcast, with no
    # walk draws added for the days after the cut-off.
    assert mine.mean() == pytest.approx(origin.point[i, -1], rel=0.1)
    assert mine.nunique() > 10


def test_diagnostics_name_the_sources(
    runs: dict[str, tuple[RandomWalk, pd.DataFrame]],
) -> None:
    model = runs["randomwalknowcast"][0]
    info = pd.DataFrame(model.fit_diagnostics).set_index("location")
    assert {
        "start_week", "n_err", "delay_source", "obs_source", "sigma_source"
    } <= set(info.columns)  # fmt: skip
    assert info.loc["CD-IT", "delay_source"].startswith("own at ")
    n_it = int(info.loc["CD-IT", "delay_source"].split()[2])
    n_to = int(info.loc["CD-TO", "delay_source"].split()[2])
    assert n_to < n_it <= 42
    assert info.loc["CD-IT", "obs_source"] == "pooled:CD"
    assert info.loc["CD-IT", "n_err"] == 3


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


def test_randomwalk_raises_when_no_week_is_complete_and_the_backtest_skips(
    task: ForecastTask,
) -> None:
    strict = RandomWalk(settings=Settings(completeness=1.5))
    with pytest.raises(NoCompleteWeek, match="no start week"):
        strict.forecast(task)
    assert issubclass(NoCompleteWeek, ModelUnavailableError)
    out = run_forecasts([strict], [task])
    assert out.empty


def test_places_with_no_falls_of_their_own_have_no_removals(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    task = _task(target, delays, daily, "2026-11-01")
    model = RandomWalkNowcast()
    origin = model.origin_at(task, errors=True)
    values, weeks = model.trajectories(origin, task, origin.point)
    u = weeks.index("2026-10-10")
    locs = list(origin.snap.locations)
    for code in ("CD-IT", "CD-NK"):
        i = locs.index(code)
        assert (origin.removal[i] == 0).all()
        assert (values[:, i, u] >= origin.reported[i, u]).all()


def test_a_silent_place_waits_for_its_first_introduction() -> None:
    import jax

    from bdbvhubrenewal.forecasters.introductions import silent_trajectories

    silent = np.array([False, True, True, True])
    q = np.array([0.5, 0.0, 0.3, 0.3])
    unreported = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.5]])
    start = np.array([1, 1, 1, 0])
    sigma = np.array([0.0, 0.2, 0.0, 0.0])
    y = silent_trajectories(
        jax.random.key(1), silent, q, unreported, start, sigma, 4, 4000
    )
    # A place that is not silent has no draws.
    assert (y[0] == 0).all()
    # No introduction can come with a probability of 0, and the path is
    # exactly 0 without one.
    assert (y[1] == 0).all()
    # The share of samples with a first case by week h is 1 - 0.7^(h + 1),
    # and the introduction week has at least 1 case.
    started = (np.cumsum(y[2] > 0, axis=1) > 0).mean(axis=0)
    np.testing.assert_allclose(started, 1 - 0.7 ** np.arange(1, 5), atol=0.03)
    assert (y[2, :, 0] > 0).mean() == pytest.approx(0.3, abs=0.03)
    # A week half reported halves the chance of an introduction in it.
    assert (y[3, :, 0] > 0).mean() == pytest.approx(0.15, abs=0.03)


def test_a_silent_place_is_not_nowcast(runs) -> None:
    # Sud-Kivu has no case in the 9 weeks to 2026-10-24, so after its start
    # week it follows its introductions, nowcast weeks included.
    _, samples = runs["randomwalknowcast"]
    sk = samples.loc[samples["location"] == "CD-SK"]
    by_week = sk.groupby("target_end_date")["value"]
    # The weekly introduction probability is about 0.06, so most samples
    # stay at 0 in every week and none starts above a few cases.
    assert (by_week.apply(lambda v: (v == 0).mean()) > 0.7).all()
    assert by_week.median().max() == 0
    # Up to its start week it keeps its reported 0s exactly.
    model, _ = runs["randomwalknowcast"]
    start = {r["location"]: r["start_week"] for r in model.fit_diagnostics}
    assert (sk.loc[sk["target_end_date"] <= start["CD-SK"], "value"] == 0).all()


def test_a_small_place_walks_only_over_the_forecast_weeks(
    task: ForecastTask,
) -> None:
    # Sud Ubangi has 2 cases ever, in the week of 10-03. Its walk starts
    # from its nowcast of the latest week at least 5% reported, so even a
    # wide walk spread, as
    # pprandomwalknowcast fits it, compounds over a few weeks only.
    model = RandomWalkNowcast()
    origin = model.origin_at(task, errors=True)
    i = list(origin.snap.locations).index("CD-SU")
    origin.sigma[i] = 0.9
    values, _ = model.trajectories(origin, task, origin.point)
    assert np.quantile(values[:, i, :], 0.9) < 30


def test_the_walk_forecasts_on_from_each_samples_nowcast(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    # Cut-off 10-16 is a Friday, so the week of 10-17 is partial. Each
    # sample walks on from its own nowcast of the start week, so the
    # forecast is at least as spread as that nowcast and follows it.
    task = _task(target, delays, daily, "2026-10-17")
    model = RandomWalkNowcast()
    out = model.forecast(task)
    info = pd.DataFrame(model.fit_diagnostics).set_index("location")
    for loc in ("CD-IT", "CD-NK"):
        mine = out.loc[out["location"] == loc].pivot(
            index="sample", columns="target_end_date", values="value"
        )
        start = info.loc[loc, "start_week"]
        later = [w for w in mine.columns if w > "2026-10-17"]
        x0 = np.log1p(mine[start].to_numpy())
        x1 = np.log1p(mine[later[0]].to_numpy())
        assert x1.std() >= x0.std()
        assert np.corrcoef(x0, x1)[0, 1] > 0.2
