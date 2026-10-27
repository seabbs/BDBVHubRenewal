from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.backtest import BacktestConfig, make_tasks
from bdbv_hub_renewal.forecasters import (
    ForecastModel,
    Persistence,
    add_national,
    check_samples,
    get_model,
    last_complete_week,
    list_models,
    register_model,
)
from bdbv_hub_renewal.forecasters.persistence import completeness_lag


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
