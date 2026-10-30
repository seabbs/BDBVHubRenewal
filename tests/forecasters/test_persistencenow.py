from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    nowcast_draws,
    release_factors,
)


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    vintage: str,
    report_by: tuple[str, ...] = (),
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows, delays, BacktestConfig(seed=3), report_by, history=real
    )[0]


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


def test_nowcast_draws_add_skellam_increments_per_place() -> None:
    rng = np.random.default_rng(1)
    # Two places, three weeks: growth, none, and a revision down.
    observed = np.array([[40.0, 40.0, 5.0], [10.0, 0.0, 10.0]])
    mean = np.array([[20.0, 0.0, -3.0], [5.0, 0.0, 0.0]])
    spread = np.array([[4.0, 0.0, 2.0], [1.0, 0.0, 3.0]])
    draws = nowcast_draws(observed, mean, spread, 40000, rng)
    assert draws.shape == (40000, 2, 3)
    added = draws - observed
    # The increment is Skellam: mean as given, variance |mean| + 2 spread.
    assert added[:, 0, 0].mean() == pytest.approx(20, rel=0.05)
    assert added[:, 0, 0].var() == pytest.approx(20 + 8, rel=0.1)
    assert (added[:, 0, 1] == 0).all()
    # Revisions down are allowed, and totals are floored at zero only.
    assert (added[:, 0, 2] < 0).mean() > 0.5
    assert (draws >= 0).all()
    assert (draws[:, 0, 2] == 0).any()
    # Places are drawn independently.
    assert abs(np.corrcoef(added[:, 0, 0], added[:, 1, 0])[0, 1]) < 0.05


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
    assert nat.loc[recent].std() > 2 * nat.loc[old].std()


def test_persistencenow_carries_the_nowcast_forward(task: ForecastTask) -> None:
    samples = PersistenceNow().forecast(task)
    it = samples.loc[samples["location"] == "CD-IT"]
    it = it.pivot(index="target_end_date", columns="sample", values="value")
    # The walk keeps its median at the nowcast of the week it starts from,
    # the week ending on the vintage.
    start = it.loc["2026-10-17"].median()
    assert it.loc["2026-11-07"].median() == pytest.approx(start, rel=0.25)
    # The spread still grows with the horizon.
    q = it.quantile([0.05, 0.95], axis=1)
    width = (q.loc[0.95] - q.loc[0.05]).to_numpy()
    assert width[-1] > width[-3]


def test_persistencenow_is_reproducible(task: ForecastTask) -> None:
    a = PersistenceNow().forecast(task)
    b = PersistenceNow().forecast(task)
    pd.testing.assert_frame_equal(a, b)


def test_persistencenow_predicts_counts_as_reported_later(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    task = _task(target, delays, "2026-10-17", ("2026-10-24",))
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
    # Between the published count and the final count, either way round,
    # as totals can be revised down.
    low = np.minimum(floor, both["value_final"])
    high = np.maximum(floor, both["value_final"])
    assert ((both["value"] >= low) & (both["value"] <= high)).all()
    # The week ending 10-10, one week old: by 10-24 it gains the share of
    # its remaining growth that the factors give for one more week.
    nat = both.loc[both["target_end_date"] == "2026-10-10"]
    by_sample = nat.groupby("sample")[["value", "value_final"]].sum()
    published = sum(seen.get((loc, "2026-10-10"), 0) for loc in task.locations)
    f = release_factors(task.target_history, task.vintage)
    assert f is not None
    q = (f[1] - 1) / (np.prod(f[1:]) - 1)
    expected = published + (by_sample["value_final"].median() - published) * q
    assert by_sample["value"].median() == pytest.approx(expected, rel=0.1)


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
    # Its latest weeks have no cases and the walk starts there, but its
    # spread still leaves a case a week plausible.
    ahead = top.loc[top.index > task.vintage]
    assert (ahead >= 1).all()
    # Ituri's walk starts from its latest nowcast, so its median stays
    # near its last two nowcasts.
    it = samples.loc[samples["location"] == "CD-IT"]
    med = it.groupby("target_end_date")["value"].median()
    low, high = sorted(med.loc[["2026-10-17", "2026-10-24"]])
    assert 0.8 * low <= med.iloc[-1] <= 1.25 * high


def test_ituri_turning_point_at_08_22(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Ituri peaked in the week ending 08-15 (413 cases) and fell after it
    # (337, 358, 298, 336, 277). At the reconstructed 08-22 vintage the
    # week ending 08-22 is about 7% reported. The walk starts from its
    # nowcast, so it is not held at the peak.
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


@pytest.mark.xfail(
    reason="The walk starts from the latest week, about 13% reported, and "
    "its steps include the noisy change into it.",
    strict=True,
)
def test_walk_spread_stays_plausible(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # At 10-24 Ituri's latest week is about 18% reported, so its nowcast,
    # where the walk starts, is noisy. Its 95% quantile four weeks ahead
    # stays within three times the recent nowcasts.
    task = _task(target, delays, "2026-10-24")
    samples = PersistenceNow().forecast(task)
    it = samples.loc[samples["location"] == "CD-IT"]
    q = it.groupby("target_end_date")["value"].quantile([0.5, 0.95]).unstack()
    recent = q.loc[q.index <= task.vintage, 0.5].iloc[-4:].max()
    assert q[0.95].iloc[-1] <= 3 * recent


def test_the_walk_starts_from_each_trajectory_latest_nowcast(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Each forecast trajectory continues from its own nowcast of the latest
    # week: the next week's median matches that week's, and draws that are
    # high in the latest week stay high in the next.
    task = _task(target, delays, "2026-10-24")
    samples = PersistenceNow().forecast(task)
    it = samples.loc[samples["location"] == "CD-IT"]
    it = it.pivot(index="target_end_date", columns="sample", values="value")
    latest, ahead = it.loc["2026-10-24"], it.loc["2026-10-31"]
    assert ahead.median() == pytest.approx(latest.median(), rel=0.15)
    assert np.corrcoef(np.log1p(latest), np.log1p(ahead))[0, 1] > 0.5


def test_release_factors_index_ages_by_release_date() -> None:
    # Ages are whole weeks from a week's end to the release date. A week
    # ending after the release date (10-10 in the Friday 10-09 release) is
    # partial and left out. Week 10-03 grows from 5 at 10-09 (0 weeks old)
    # to 15 at 10-17 (2 weeks old): f_0 f_1 = 3. Week 10-17 grows from 8 to
    # 16 (f_0 = 2) and week 10-10 from 20 to 30 (f_1 = 1.5).
    rows = pd.DataFrame(
        {
            "vintage": ["2026-10-09", "2026-10-09", "2026-10-17",
                        "2026-10-17", "2026-10-17", "2026-10-24",
                        "2026-10-24"],
            "location": "CD-IT",
            "target_end_date": ["2026-10-03", "2026-10-10", "2026-10-03",
                                "2026-10-10", "2026-10-17", "2026-10-10",
                                "2026-10-17"],
            "total": [5, 99, 15, 20, 8, 30, 16],
        }
    )  # fmt: skip
    f = release_factors(rows, "2026-10-24", max_weeks=3)
    assert f is not None
    assert f == pytest.approx([2.0, 1.5, 1.0, 1.0])
    # One release alone gives no factors.
    assert release_factors(rows, "2026-10-09") is None


def test_the_release_nowcast_lifts_weeks_one_and_two_weeks_old(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Hub totals of weeks one and two weeks old still grow by about 40% and
    # 15% between releases, so their nowcasts sit well above them.
    task = _task(target, delays, "2026-10-24")
    nat = _national(PersistenceNow().forecast(task))
    seen = task.target.loc[task.target["location"] == "CD"].set_index(
        "target_end_date"
    )["total"]
    assert nat.loc["2026-10-17"].median() > 1.4 * seen["2026-10-17"]
    assert nat.loc["2026-10-10"].median() > 1.1 * seen["2026-10-10"]


def test_the_uncertainty_is_learnt_from_past_releases(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # At 10-24 the nowcast made at 10-17 can be checked against 10-24, so
    # the negative binomial sizes are fitted, not Poisson.
    prep = PersistenceNow()._prepare(_task(target, delays, "2026-10-24"))
    assert prep is not None
    latest = prep.size[:, prep.last]
    assert np.isfinite(latest).all()


def test_with_one_release_it_is_persistence(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    task = _task(target, delays, "2026-08-18")
    pd.testing.assert_frame_equal(
        PersistenceNow().forecast(task), Persistence().forecast(task)
    )


def test_the_skellam_spread_is_fitted_to_retrospective_errors() -> None:
    from bdbvhubrenewal.forecasters.persistence import fit_skellam_spread

    # Increments with mean m and extra spread 0.2 Y on each side.
    rng = np.random.default_rng(2)
    y = rng.integers(20, 200, size=(400, 2)).astype(float)
    mean = np.c_[0.3 * y[:, 0], -0.02 * y[:, 1]]
    lam = 0.2 * y
    obs = rng.poisson(np.maximum(mean, 0) + lam) - rng.poisson(
        np.maximum(-mean, 0) + lam
    )
    rates = fit_skellam_spread(obs.astype(float), mean, y)
    assert rates == pytest.approx([0.2, 0.2], rel=0.25)
    # A horizon with too few points takes the rate fitted to all of them.
    few = fit_skellam_spread(obs[:2].astype(float), mean[:2], y[:2])
    assert few[0] == few[1]
