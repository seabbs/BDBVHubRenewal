from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.triangle import delay_pmf, reporting_triangle
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    Persistence,
    PersistenceNow,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.persistence import nowcast_draws


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    vintage: str,
    report_by: tuple[str, ...] = (),
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), report_by)[0]


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    return _task(target, delays, "2026-10-17")


def _national(samples: pd.DataFrame) -> pd.DataFrame:
    final = samples
    if "as_of" in samples:
        final = samples.loc[samples["as_of"].isna()]
    return final.groupby(["target_end_date", "sample"])["value"].sum().unstack()


def test_registry_has_persistencenow_and_not_the_lag_variant() -> None:
    names = [s.name for s in list_models()]
    assert "persistencenow" in names
    assert "persistence_fu" not in names
    spec = next(s for s in list_models() if s.name == "persistencenow")
    assert not spec.baseline
    assert not spec.fitted
    assert isinstance(get_model("persistencenow"), PersistenceNow)


def test_nowcast_draws_add_negative_binomial_counts() -> None:
    rng = np.random.default_rng(1)
    observed = np.array([0.0, 40.0, 40.0, 40.0])
    ratio = np.array([1.0, 0.0, 0.25, 1.5])
    size = np.array([5.0, 5.0, np.inf, 5.0])
    draws = nowcast_draws(observed, ratio, size, 20000, rng)
    assert draws.shape == (20000, 4)
    # Never below the published count; nothing added without a ratio.
    assert (draws >= observed).all()
    assert (draws[:, :2] == observed[:2]).all()
    added = draws[:, 2:] - observed[2:]
    # The point nowcast is the mean of what is added.
    assert np.allclose(added.mean(axis=0), [10, 60], rtol=0.05)
    # Poisson with an infinite size, overdispersed otherwise.
    assert added[:, 0].var() == pytest.approx(10, rel=0.1)
    assert added[:, 1].var() == pytest.approx(60 + 60**2 / 5, rel=0.1)


def test_persistencenow_output(task: ForecastTask) -> None:
    samples = PersistenceNow().forecast(task)
    check_samples(samples, task)
    seen = task.target.set_index(["location", "target_end_date"])["total"]
    floor = [
        seen.get((loc, week), 0)
        for loc, week in zip(
            samples["location"], samples["target_end_date"], strict=True
        )
    ]
    # Never below the published total.
    assert (samples["value"].to_numpy() >= floor).all()


def test_persistencenow_nowcasts_recent_weeks(task: ForecastTask) -> None:
    nat = _national(PersistenceNow().forecast(task))
    seen = task.target.loc[task.target["location"] == "CD"].set_index(
        "target_end_date"
    )["total"]
    # The oldest forecast week is all but complete: little added, little
    # spread. The last full week before the vintage is mostly unreported,
    # so its nowcast is well above the published count, with spread.
    old, recent = "2026-09-26", "2026-10-10"
    assert nat.loc[old].median() - seen[old] < 0.08 * seen[old]
    assert nat.loc[recent].median() > 1.15 * seen[recent]
    assert nat.loc[recent].std() > 3 * nat.loc[old].std()


def test_persistencenow_carries_the_nowcast_forward(task: ForecastTask) -> None:
    nat = _national(PersistenceNow().forecast(task))
    plain = _national(Persistence().forecast(task))
    # Persistence carries forward a published week; the nowcast lifts the
    # level carried into the forecast weeks.
    future = "2026-11-07"
    assert nat.loc[future].median() > plain.loc[future].median()
    # The spread still grows with the horizon.
    q = nat.quantile([0.05, 0.95], axis=1)
    width = (q.loc[0.95] - q.loc[0.05]).to_numpy()
    assert width[-1] > width[-3]


def test_persistencenow_is_reproducible(task: ForecastTask) -> None:
    a = PersistenceNow().forecast(task)
    b = PersistenceNow().forecast(task)
    pd.testing.assert_frame_equal(a, b)


def test_persistencenow_without_delays_is_persistence(
    target: pd.DataFrame,
) -> None:
    rows = target.loc[target["vintage"] == "2026-08-18"]
    task = make_tasks(rows, None, BacktestConfig(seed=3))[0]
    assert task.delays is None
    pd.testing.assert_frame_equal(
        PersistenceNow().forecast(task), Persistence().forecast(task)
    )


def test_persistencenow_needs_enough_recent_delays(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Too few delay rows in the window give persistence, not a noisy nowcast.
    task = _task(target, delays, "2026-10-17")
    assert task.delays is not None
    few = replace(task, delays=task.delays.head(20))
    pd.testing.assert_frame_equal(
        PersistenceNow().forecast(few), Persistence().forecast(few)
    )


def test_persistencenow_predicts_counts_as_reported_later(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    task = _task(target, delays, "2026-10-09", ("2026-10-24",))
    samples = PersistenceNow().forecast(task)
    check_samples(samples, task)
    final = samples.loc[samples["as_of"].isna()]
    later = samples.loc[samples["as_of"] == "2026-10-24"]
    keys = ["location", "target_end_date", "sample"]
    both = later.merge(final, on=keys, suffixes=("", "_final"))
    seen = task.target.set_index(["location", "target_end_date"])["total"]
    floor = np.array(
        [
            seen.get((loc, week), 0)
            for loc, week in zip(
                both["location"], both["target_end_date"], strict=True
            )
        ]
    )
    # Between the published count and the final count.
    assert (both["value"] >= floor).all()
    assert (both["value"] <= both["value_final"]).all()
    # A week reported by then in part: the national median sits between
    # the published count and the final nowcast, near its expected share.
    nat = both.loc[both["target_end_date"] == "2026-10-03"]
    by_sample = nat.groupby("sample")[["value", "value_final"]].sum()
    published = sum(seen.get((loc, "2026-10-03"), 0) for loc in task.locations)
    assert published < by_sample["value"].median()
    assert by_sample["value"].median() < by_sample["value_final"].median()
    assert task.delays is not None
    _, tri = reporting_triangle(task.delays, "2026-10-09")
    cdf = np.cumsum(delay_pmf(tri[-14:]))
    s_v, s_t = cdf[6], cdf[21]
    expected = published + (by_sample["value_final"].median() - published) * (
        (s_t - s_v) / (1 - s_v)
    )
    assert by_sample["value"].median() == pytest.approx(expected, rel=0.1)
