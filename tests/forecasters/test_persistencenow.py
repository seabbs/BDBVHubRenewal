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
from bdbvhubrenewal.forecasters.persistence import (
    horizon_scales,
    nowcast_draws,
    revision_variance,
)


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


def test_nowcast_draws_add_negative_binomial_counts_per_place() -> None:
    rng = np.random.default_rng(1)
    # Two places, four weeks, each with its own ratio and size.
    observed = np.array([[0.0, 40.0, 40.0, 40.0], [0.0, 0.0, 10.0, 10.0]])
    ratio = np.array([[1.0, 0.0, 0.25, 1.5], [1.0, 1.0, 1.0, 1.0]])
    size = np.array([[5.0, 5.0, np.inf, 5.0], [5.0, 5.0, 5.0, 5.0]])
    draws = nowcast_draws(observed, ratio, size, 20000, rng)
    assert draws.shape == (20000, 2, 4)
    # Without revision noise, never below the published count.
    assert (draws >= observed).all()
    assert (draws[:, 0, :2] == observed[0, :2]).all()
    added = (draws - observed)[:, 0, 2:]
    # The point nowcast is the mean of what is added: Poisson with an
    # infinite size, overdispersed otherwise.
    assert np.allclose(added.mean(axis=0), [10, 60], rtol=0.05)
    assert added[:, 0].var() == pytest.approx(10, rel=0.1)
    assert added[:, 1].var() == pytest.approx(60 + 60**2 / 5, rel=0.1)
    # Places are drawn independently.
    other = (draws - observed)[:, 1, 3]
    assert abs(np.corrcoef(added[:, 1], other)[0, 1]) < 0.05


def test_high_count_places_use_their_own_delay(task: ForecastTask) -> None:
    sources = PersistenceNow().delay_sources(task)
    assert sources["CD-IT"] == "own"
    assert sources["CD-NK"] == "own"
    assert sources["CD-BU"] == "pooled"


def test_the_delay_window_sets_the_longest_delay() -> None:
    assert PersistenceNow(n_history=4).max_delay == 21
    assert PersistenceNow(n_history=14).max_delay == 60


def test_nowcast_draws_can_fall_below_the_published_total() -> None:
    rng = np.random.default_rng(2)
    observed = np.array([[100.0]])
    known = np.array([[80.0]])
    draws = nowcast_draws(
        observed, np.zeros((1, 1)), np.full((1, 1), np.inf), 20000, rng,
        known=known, revision=0.5,
    )  # fmt: skip
    # Re-imputation moves the total both ways, with variance
    # revision * imputed, and never below the known onsets.
    assert (draws < observed).mean() > 0.3
    assert draws.mean() == pytest.approx(100, abs=0.5)
    assert draws.var() == pytest.approx(0.5 * 20, rel=0.15)
    assert (draws >= known).all()


def test_revision_variance_from_past_vintages(target: pd.DataFrame) -> None:
    real = target.loc[target["kind"] == "real"]
    kappa = revision_variance(real, "2026-10-24")
    assert 0 < kappa < 5
    # One vintage has no revisions to learn from.
    one = real.loc[real["vintage"] == "2026-10-24"]
    assert revision_variance(one, "2026-10-24") == 0.0


def test_horizon_scales_shrink_to_one() -> None:
    rng = np.random.default_rng(3)
    wide = [rng.normal(0, 2.0, 400), rng.normal(0, 2.0, 400), np.array([])]
    scales = horizon_scales(wide, prior=10)
    assert scales[0] == pytest.approx(2.0, rel=0.1)
    assert scales[2] == pytest.approx(scales[1], rel=0.01)
    assert horizon_scales([np.array([])], prior=10)[0] == 1.0
    few = horizon_scales([np.array([5.0])], prior=10)[0]
    assert 1.0 < few < 2.0


def test_persistencenow_calibrates_the_walk_on_reconstructed_vintages(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    from bdbvhubrenewal.evaluation.backtest import (
        development_tasks,
        load_config,
    )

    config = load_config()
    tasks = development_tasks(target, delays, config)
    task = next(
        t for t in tasks
        if t.kind == "reconstructed" and t.vintage == "2026-10-03"
    )  # fmt: skip
    model = PersistenceNow()
    errors = model.walk_errors(task)
    assert len(errors) >= 3
    assert sum(e.size for e in errors) > 50
    check_samples(model.forecast(task), task)


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
    model = PersistenceNow()
    _, tri = reporting_triangle(task.delays, "2026-10-09", model.max_delay)
    cdf = np.cumsum(delay_pmf(tri[-model.n_history :]))
    s_v, s_t = cdf[6], cdf[21]
    expected = published + (by_sample["value_final"].median() - published) * (
        (s_t - s_v) / (1 - s_v)
    )
    assert by_sample["value"].median() == pytest.approx(expected, rel=0.1)
