from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.triangle import (
    delay_pmf,
    reporting_triangle,
)
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
    baseline_walk,
    nowcast_draws,
    nowcast_means,
    revision_variance,
    week_share,
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
    draws = nowcast_draws(observed, observed * ratio, size, 20000, rng)
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
    samples = PersistenceNow().forecast(task)
    it = samples.loc[samples["location"] == "CD-IT"]
    it = it.pivot(index="target_end_date", columns="sample", values="value")
    # The walk keeps its median at the nowcast of the week it starts from,
    # the week ending 10-10, the latest at least half reported.
    start = it.loc["2026-10-10"].median()
    assert it.loc["2026-11-07"].median() == pytest.approx(start, rel=0.25)
    # The spread still grows with the horizon.
    q = it.quantile([0.05, 0.95], axis=1)
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


def test_nowcast_means_scale_the_total_by_the_share_reported() -> None:
    totals = np.array([12.0, 0.0, 5.0])
    share = np.array([1.0, 0.1, 0.25])
    added = nowcast_means(totals, share)
    # Y / s - Y: nothing for a complete week or a week with no cases.
    assert added.tolist() == pytest.approx([0.0, 0.0, 15.0])


def test_week_share_averages_over_onset_days_and_lags() -> None:
    # A case with onset on day W - u of the week ending W, delay d and
    # publication lag l is in a vintage t days after W if d + l - u <= t.
    cdf = np.cumsum([0.5, 0.25, 0.25])
    share = week_share(cdf, np.array([-1, 0, 10]), lag=0)
    assert share[0] == pytest.approx((0.5 + 0.75 + 1.0 * 4) / 7)
    assert share[1] == pytest.approx((0.5 + 0.75 + 1.0 * 5) / 7)
    assert share[2] == 1.0
    # A lag uniform on 0 to 1 averages the shares with lags 0 and 1.
    both = week_share(cdf, np.array([0]), lag=1)[0]
    assert both == pytest.approx((share[1] + share[0]) / 2)


def test_baseline_walk_is_a_symmetrised_empirical_random_walk() -> None:
    rng = np.random.default_rng(5)
    centre = np.full(4000, np.log1p(100.0))
    steps = np.array([0.1, 0.3, -0.2, 0.05])
    draws = baseline_walk(centre, steps, 3, rng)
    assert draws.shape == (4000, 3)
    # The median stays at the centre and the spread grows with the horizon.
    assert np.allclose(np.median(draws, axis=0), 100, rtol=0.05)
    iqr = np.subtract(*np.percentile(draws, [75, 25], axis=0))
    assert iqr[2] > iqr[0]
    # Every one-week change is one of the past changes or its negative, up
    # to rounding to whole cases.
    first = np.log1p(draws[:, 0]) - centre
    pool = np.r_[steps, -steps]
    assert (np.abs(first[:, None] - pool[None, :]).min(axis=1) < 0.01).all()
    # Fewer than three past changes: Poisson around the centre.
    flat = baseline_walk(centre, np.array([0.2]), 2, rng)
    assert flat.var(axis=0) == pytest.approx([100, 100], rel=0.15)


def test_persistencenow_nowcasts_the_latest_week(task: ForecastTask) -> None:
    # The week ending on the vintage is nowcast and the walk starts from it.
    nat = _national(PersistenceNow().forecast(task))
    seen = task.target.loc[task.target["location"] == "CD"].set_index(
        "target_end_date"
    )["total"]
    assert nat.loc["2026-10-17"].median() > 2 * seen["2026-10-17"]


def test_sparse_places_stay_plausible(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Bas-Uele has had about 13 cases; its 97.5% quantile must stay small.
    task = _task(target, delays, "2026-10-24")
    samples = PersistenceNow().forecast(task)
    bu = samples.loc[samples["location"] == "CD-BU"]
    top = bu.groupby("target_end_date")["value"].quantile(0.975)
    assert top.max() <= 20
    # With 7 cases in the last 6 weeks, a few cases a week stay plausible.
    ahead = top.loc[top.index > task.vintage]
    assert (ahead >= 3).all()
    # Ituri's walk is centred between its last two nowcasts, nearer the
    # one with more of its cases reported.
    it = samples.loc[samples["location"] == "CD-IT"]
    med = it.groupby("target_end_date")["value"].median()
    low, high = sorted(med.loc[["2026-10-17", "2026-10-24"]])
    assert 0.8 * low <= med.iloc[-1] <= high


def test_ituri_turning_point_at_08_22(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Ituri peaked in the week ending 08-15 (413 cases) and fell after it
    # (337, 358, 298, 336, 277). At the reconstructed 08-22 vintage the
    # week ending 08-22 is about 7% reported. Its nowcast enters the walk's
    # centre weighted by that share, so the walk is not held at the peak.
    from bdbvhubrenewal.evaluation.backtest import (
        development_tasks,
        load_config,
    )

    task = next(
        t for t in development_tasks(target, delays, load_config())
        if t.kind == "reconstructed" and t.vintage == "2026-08-22"
    )  # fmt: skip
    samples = PersistenceNow().forecast(task)
    samples = samples.loc[samples["as_of"].isna()]
    it = samples.loc[samples["location"] == "CD-IT"]
    q = it.groupby("target_end_date")["value"].quantile([0.05, 0.5, 0.95])
    q = q.unstack()
    walk = ["2026-08-29", "2026-09-05", "2026-09-12", "2026-09-19"]
    assert (q.loc[walk, 0.5] < 400).all()
    for week, truth in zip(walk, [358, 298, 336, 277], strict=True):
        assert q.loc[week, 0.05] <= truth <= q.loc[week, 0.95]


def test_walk_spread_stays_plausible(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # At 10-24 Ituri's latest week is about 17% reported, so its nowcast is
    # noisy. Its 95% quantile four weeks ahead stays within twice the
    # recent nowcasts.
    task = _task(target, delays, "2026-10-24")
    samples = PersistenceNow().forecast(task)
    it = samples.loc[samples["location"] == "CD-IT"]
    q = it.groupby("target_end_date")["value"].quantile([0.5, 0.95]).unstack()
    recent = q.loc[q.index <= task.vintage, 0.5].iloc[-4:].max()
    assert q[0.95].iloc[-1] <= 2 * recent
