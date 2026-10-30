from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastModel,
    Persistence,
    add_national,
    check_samples,
    get_model,
    last_complete_week,
    list_models,
    register_model,
)
from bdbvhubrenewal.forecasters.persistence import completeness_lag


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame):  # type: ignore[no-untyped-def]
    rows = target.loc[target["vintage"] == "2026-10-17"]
    return make_tasks(rows, delays, BacktestConfig(seed=3))[0]


def test_registry_has_persistence_as_baseline() -> None:
    specs = list_models()
    assert specs[0].name == "persistence"
    assert specs[0].baseline
    assert isinstance(get_model("persistence"), ForecastModel)
    with pytest.raises(KeyError):
        get_model("nope")
    with pytest.raises(ValueError, match="already registered"):
        register_model("persistence", Persistence, "dup")


def test_last_complete_week() -> None:
    # Midpoint (Wednesday) at least 10 days before a Saturday vintage.
    assert last_complete_week("2026-10-24", 10) == dt.date(2026, 10, 17)
    assert last_complete_week("2026-10-24", 14) == dt.date(2026, 10, 10)
    assert last_complete_week("2026-10-24", 3) == dt.date(2026, 10, 24)
    # A Tuesday vintage: the week ending the Saturday before is not complete.
    assert last_complete_week("2026-08-18", 10) == dt.date(2026, 8, 8)


def test_completeness_lag_uses_delays(delays: pd.DataFrame) -> None:
    latest = delays.loc[delays["vintage"] == "2026-10-24"]
    lag = completeness_lag(latest, "2026-10-24")
    assert 8 <= lag <= 20
    assert completeness_lag(None, "2026-08-18", default=12) == 12


def test_persistence_output(task) -> None:  # type: ignore[no-untyped-def]
    samples = Persistence().forecast(task)
    check_samples(samples, task)
    lag = completeness_lag(task.delays, task.vintage)
    last = last_complete_week(task.vintage, lag).isoformat()
    for loc in task.locations:
        s = samples.loc[samples["location"] == loc]
        obs = task.target.set_index(["location", "target_end_date"])["total"]
        # Complete weeks are the published counts.
        for week in s["target_end_date"].unique():
            vals = s.loc[s["target_end_date"] == week, "value"]
            if week <= last:
                assert vals.nunique() == 1
                assert vals.iloc[0] == obs.get((loc, week), 0)
        # Later weeks are centred on the last complete week.
        level = obs.get((loc, last), 0)
        later = s.loc[s["target_end_date"] > last]
        med = later.groupby("target_end_date")["value"].median()
        assert np.allclose(med, level, rtol=0.35, atol=3)


def test_persistence_spread_grows(task) -> None:  # type: ignore[no-untyped-def]
    samples = Persistence().forecast(task)
    it = samples.loc[samples["location"] == "CD-IT"]
    q = it.groupby("target_end_date")["value"].quantile([0.05, 0.95]).unstack()
    width = (q[0.95] - q[0.05]).to_numpy()
    assert width[-1] > width[-4]


def test_persistence_is_reproducible(task) -> None:  # type: ignore[no-untyped-def]
    a = Persistence().forecast(task)
    b = Persistence().forecast(task)
    pd.testing.assert_frame_equal(a, b)


def test_add_national_sums_provinces(task) -> None:  # type: ignore[no-untyped-def]
    samples = add_national(Persistence().forecast(task))
    nat = samples.loc[samples["location"] == "CD"].set_index(
        ["target_end_date", "sample"]
    )["value"]
    prov = (
        samples.loc[samples["location"] != "CD"]
        .groupby(["target_end_date", "sample"])["value"]
        .sum()
    )
    pd.testing.assert_series_equal(
        nat.sort_index(), prov.sort_index(), check_names=False
    )


def test_check_samples_rejects_bad_output(task) -> None:  # type: ignore[no-untyped-def]
    samples = Persistence().forecast(task)
    with pytest.raises(ValueError, match="numbered"):
        check_samples(samples.assign(sample=samples["sample"] + 1), task)
    with pytest.raises(ValueError, match="non-negative"):
        check_samples(samples.assign(value=-1.0), task)


def test_a_model_may_give_its_own_national(task) -> None:  # type: ignore[no-untyped-def]
    samples = Persistence().forecast(task)
    own = (
        samples.groupby(["target_end_date", "sample"], as_index=False)["value"]
        .sum()
        .assign(location="CD", value=lambda d: d["value"] + 7.0)
    )[list(samples.columns)]
    both = pd.concat([samples, own], ignore_index=True)
    check_samples(both, task)
    out = add_national(both)
    nat = out.loc[out["location"] == "CD"]
    assert len(nat) == len(own)
    assert (nat["value"].to_numpy() == own["value"].to_numpy()).all()


def test_persistence_predicts_counts_as_reported_later(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    rows = target.loc[target["vintage"] == "2026-10-09"]
    task = make_tasks(
        rows, delays, BacktestConfig(seed=3), report_by=("2026-10-24",)
    )[0]
    assert task.report_by == ("2026-10-24",)
    samples = Persistence().forecast(task)
    check_samples(samples, task)
    final = samples.loc[samples["as_of"].isna()]
    later = samples.loc[samples["as_of"] == "2026-10-24"]
    # Weeks up to the later vintage only, with the same samples.
    assert later["target_end_date"].max() <= "2026-10-24"
    assert set(later["sample"]) == set(final["sample"])
    # At least what was reported at the vintage, and no more than the
    # final count where that is higher.
    keys = ["location", "target_end_date", "sample"]
    both = later.merge(final, on=keys, suffixes=("", "_final"))
    seen = rows.set_index(["location", "target_end_date"])["total"]
    floor = [
        seen.get((loc, week), 0)
        for loc, week in zip(
            both["location"], both["target_end_date"], strict=True
        )
    ]
    assert (both["value"] >= floor).all()
    assert (both["value"] <= np.maximum(both["value_final"], floor)).all()


def test_add_national_sums_each_as_of(task) -> None:  # type: ignore[no-untyped-def]
    samples = Persistence().forecast(task)
    later = samples.assign(as_of="2026-10-24", value=1.0)
    out = add_national(pd.concat([samples.assign(as_of=np.nan), later]))
    nat = out.loc[(out["location"] == "CD") & (out["as_of"] == "2026-10-24")]
    assert len(nat) == len(task.target_end_dates) * task.n_samples
    assert (nat["value"] == len(task.locations)).all()


def test_check_samples_rejects_unrequested_as_of(task) -> None:  # type: ignore[no-untyped-def]
    samples = Persistence().forecast(task)
    extra = samples.assign(as_of="2026-10-24")
    with pytest.raises(ValueError, match="as_of"):
        check_samples(pd.concat([samples.assign(as_of=np.nan), extra]), task)


def test_low_count_places_take_the_pooled_spread() -> None:
    # Places with at least `low_count` cases a week keep their own spread.
    # The rest take the spread of the pooled weekly changes of those
    # places, as persistencenow pools its steps.
    big = np.array([100.0, 120, 90, 130, 80, 140, 70, 150])
    calm = np.array([60.0, 61, 62, 63, 64, 65, 66, 67])
    small = np.array([0.0, 9, 0, 9, 0, 9, 0, 9])
    model = Persistence(low_count=10.0)
    sds = model._spreads([big, calm, small])
    assert sds[0] == pytest.approx(model._spread(big))
    assert sds[1] == pytest.approx(model._spread(calm))
    pooled = np.r_[np.diff(np.log1p(big)), np.diff(np.log1p(calm))]
    assert sds[2] == pytest.approx(np.std(pooled, ddof=1))
    # With no place above the limit, each keeps its own.
    alone = model._spreads([small])
    assert alone[0] == pytest.approx(model._spread(small))


def test_random_walk_is_a_log_scale_walk_with_poisson_draws() -> None:
    from bdbvhubrenewal.forecasters.persistence import random_walk

    rng = np.random.default_rng(1)
    start = np.full(4000, np.log1p(100.0))
    draws = random_walk(start, 0.3, 3, rng)
    assert draws.shape == (4000, 3)
    # The median stays at the start and the spread grows with the horizon.
    med = np.median(draws, axis=0)
    assert med == pytest.approx([100, 100, 100], rel=0.1)
    width = np.quantile(draws, 0.95, axis=0) - np.quantile(draws, 0.05, axis=0)
    assert width[0] < width[1] < width[2]
    # With no spread it is Poisson around the start.
    flat = random_walk(start, 0.0, 2, rng)
    assert flat.var() == pytest.approx(100, rel=0.15)


def test_both_baselines_use_the_one_random_walk(
    task, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    import bdbvhubrenewal.forecasters.persistence as mod

    calls: list[str] = []
    walk = mod.random_walk

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("walk")
        return walk(*args, **kwargs)

    monkeypatch.setattr(mod, "random_walk", spy)
    mod.Persistence().forecast(task)
    n = len(calls)
    assert n > 0
    mod.PersistenceNow().forecast(task)
    assert len(calls) > n


def test_completeness_from_releases(target: pd.DataFrame) -> None:
    from bdbvhubrenewal.forecasters.persistence import release_complete_weeks

    # Hub totals of 1 to 3 week old weeks still grow between releases, so
    # at 10-24 a week is complete only from about three weeks old.
    weeks = release_complete_weeks(target, "2026-10-24")
    assert weeks is not None
    assert 2 <= weeks <= 5
    # One release has no growth to learn from.
    assert release_complete_weeks(target, "2026-08-18") is None
