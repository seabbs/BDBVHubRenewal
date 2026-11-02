"""The daily chain ladder on release snapshots."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.baselinenowcast import pair_factors, pair_sums
from bdbvhubrenewal.data.snapshots import snapshot_arrays, weekly_sums
from bdbvhubrenewal.forecasters.chain import (
    NoDelayPair,
    Settings,
    build_origin,
    fit_delay,
    retro_errors,
    retro_plan,
    weeks_back,
)

PLACES = ("CD-HU", "CD-IT", "CD-NK", "CD-TO")


@pytest.fixture(scope="module")
def snap24(daily: pd.DataFrame):
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    return snapshot_arrays(daily, "2026-10-24", provinces)


@pytest.fixture(scope="module")
def origin24(snap24):
    return build_origin(snap24, Settings(), errors=True)


def _weeks(origin, loc: str) -> list[float]:
    """Nowcast of the four newest weeks, newest first."""
    row = origin.point[origin.snap.locations.index(loc)]
    return [round(float(x), 1) for x in row[-4:][::-1]]


def test_10_24_chains_by_8_days_with_no_tail(origin24) -> None:
    fit = origin24.fit
    # The 8 and 9 day pairs are in window, the 50 day pair is not.
    assert fit.pairs.tolist() == [[1, 2], [2, 3]]
    assert fit.steps.tolist() == [8, 9]
    snap = origin24.snap
    day = int(snap.cut[-1])
    # A day of age 0 is chained over ages 0, 8, ..., 40 only.
    f = fit.factors[0, 0]
    assert origin24.growth[0, day] == pytest.approx(
        float(np.prod(f[0:42:8])), rel=1e-5
    )
    # A day of age 41 has one step, and age 42 or more none.
    assert origin24.growth[0, day - 41] == pytest.approx(f[41], rel=1e-5)
    assert origin24.growth[0, day - 42] == 1.0


def test_10_17_chains_by_9_days(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT",))
    origin = build_origin(snap, Settings(), errors=False)
    assert origin.fit.steps.tolist() == [9]
    day = int(snap.cut[-1])
    f = origin.fit.factors[0, 0]
    assert origin.growth[0, day] == pytest.approx(
        float(np.prod(f[0:42:9])), rel=1e-5
    )


def test_10_09_raises_with_no_delay_pair(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-09", ("CD-IT",))
    with pytest.raises(NoDelayPair):
        build_origin(snap, Settings(), errors=False)


def test_the_50_day_pair_gives_no_factor_but_its_counts_stay(
    daily: pd.DataFrame,
) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT",))
    fit = fit_delay(snap, 2, Settings())
    assert fit is not None
    assert fit.pairs.tolist() == [[1, 2]]
    assert 0 not in fit.pairs
    first = weekly_sums(snap, 0)
    assert first[0].sum() > 0
    assert first[0].sum() == snap.counts[0, 0].sum()


def test_10_24_point_nowcasts_pool_every_pair(origin24) -> None:
    # Weeks 10-24, 10-17, 10-10, 10-03. The R script chained the 8 day
    # pair alone and filled zero denominators from the parent, so it gave 439.2,
    # 418.1, 539.7 and 544.1 for the country.
    assert _weeks(origin24, "CD") == pytest.approx(
        [387.5, 376.6, 493.7, 511.9], abs=0.06
    )
    # Tshopo has under 10 cases a week, so uses the national delay.
    assert _weeks(origin24, "CD-TO") == pytest.approx(
        [20.7, 12.0, 21.6, 18.4], abs=0.06
    )


def test_provinces_use_own_factors_where_they_have_10_cases(origin24) -> None:
    assert _weeks(origin24, "CD-IT") == pytest.approx(
        [259.4, 182.4, 288.2, 254.1], abs=0.06
    )
    assert _weeks(origin24, "CD-NK") == pytest.approx(
        [85.7, 150.4, 154.8, 202.8], abs=0.06
    )
    assert _weeks(origin24, "CD-HU") == pytest.approx(
        [11.1, 31.1, 27.4, 29.4], abs=0.06
    )


def test_every_day_with_data_is_nowcast(origin24) -> None:
    snap = origin24.snap
    counts = snap.counts[0, -1]
    assert (origin24.growth[0][snap.observed[-1]] >= 0).all()
    newest = int(snap.cut[-1])
    # The newest day is nowcast too, with no handover to the walk.
    assert origin24.growth[0, newest] > 1.0
    assert counts[newest] >= 0


def test_ages_count_from_the_cutoff_not_as_of(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT",))
    r = 2
    # Release 10-17 has cut-off 10-16, so 10-16 has age 0.
    assert snap.days[snap.cut[r]].date().isoformat() == "2026-10-16"
    assert snap.cut[r] - snap.cut[r] == 0


def test_expected_plan_steps_match_the_r_script(snap24) -> None:
    plans = pd.read_csv(Path(__file__).parent / "data" / "expected_plans.csv")
    row = plans.loc[
        (plans["from"] == "2026-10-17") & (plans["to"] == "2026-10-24")
    ]
    assert row["gap_days"].iloc[0] == 8
    assert row["steps"].iloc[0] == "10-07->10-16 (9d)"
    fit = fit_delay(snap24, 2, Settings())
    assert fit is not None
    plan, mismatch = retro_plan(snap24, 2, 3, fit, Settings())
    assert mismatch == 1
    assert np.asarray(plan.valid).shape[1] == 1
    # 10-09 has no in-window pair, so gives no retrospective.
    assert fit_delay(snap24, 1, Settings()) is None


def test_only_the_10_17_to_10_24_retrospective_is_kept(snap24) -> None:
    errors = retro_errors(snap24, 3, Settings())
    assert errors.releases == ["2026-10-17"]
    strict = retro_errors(snap24, 3, Settings(max_mismatch=0))
    assert strict.releases == []
    assert strict.n.sum() == 0


def test_the_spread_pools_four_errors_at_10_24(origin24) -> None:
    loc = origin24.snap.locations.index("CD")
    assert origin24.rho is not None
    assert origin24.rho_n is not None
    assert origin24.rho[loc, :4] == pytest.approx(
        [0.7006, 8.4005, 4.8158, 106.1023], rel=1e-3
    )
    # CD, Haut-Uele, Ituri and North Kivu give an error each.
    assert origin24.rho_n[loc, :4].tolist() == [4, 4, 4, 4]


def test_no_errors_is_poisson_and_says_so(
    daily: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT", "CD-NK"))
    with caplog.at_level(logging.INFO):
        origin = build_origin(snap, Settings(), errors=True)
    assert origin.rho is not None
    assert origin.rho_n is not None
    assert (origin.rho == 0).all()
    assert (origin.rho_n == 0).all()
    assert "Poisson-Skellam" in caplog.text


def test_a_factor_is_own_by_age_with_10_cases_else_the_parent(
    daily: pd.DataFrame, daily_zones: pd.DataFrame
) -> None:
    both = pd.concat([daily, daily_zones])
    zones = tuple(
        sorted({z for z in daily_zones["location"] if z.startswith("CD-")})
    )
    snap = snapshot_arrays(both, "2026-10-24", zones)
    fit = fit_delay(snap, 3, Settings())
    assert fit is not None
    num, den = pair_sums(
        snap.counts, snap.observed, snap.cut, fit.pairs[:, 0], 42
    )
    every = np.ones((len(fit.pairs), len(fit.steps)))
    own_f = np.asarray(pair_factors(num, den, every))
    n_cases = np.asarray(den).sum(axis=1)
    seen = {"own": 0, "parent": 0}
    for i in range(len(snap.locations)):
        p = int(snap.parent[i])
        if p < 0:
            # The country has no parent, so it uses its own factor.
            np.testing.assert_allclose(fit.factors[i], own_f[i])
            continue
        use = n_cases[i] >= 10
        seen["own"] += int(use.sum())
        seen["parent"] += int((~use).sum())
        np.testing.assert_allclose(
            fit.factors[i][:, use], own_f[i][:, use], err_msg=snap.locations[i]
        )
        np.testing.assert_allclose(
            fit.factors[i][:, ~use],
            fit.factors[p][:, ~use],
            err_msg=snap.locations[i],
        )
    assert seen["own"] > 0
    assert seen["parent"] > 0


def test_the_final_factors_pool_every_in_window_pair(snap24) -> None:
    fit = fit_delay(snap24, 3, Settings())
    assert fit is not None
    assert fit.steps.tolist() == [8, 9]
    # Every step pools both pairs, so the steps share a factor per age.
    np.testing.assert_allclose(fit.factors[:, 0], fit.factors[:, 1])
    # At 10-17 only the 9 day pair is in window, so it gives the factor.
    early = fit_delay(snap24, 2, Settings())
    assert early is not None
    assert early.pairs.tolist() == [[1, 2]]
    assert not np.allclose(early.factors[:, 0], fit.factors[:, 0])


def test_weeks_back_count_from_the_cutoff_week(daily: pd.DataFrame) -> None:
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    snap = snapshot_arrays(daily, "2026-11-01", provinces)
    o = len(snap.releases) - 1
    # The as-of day is a Sunday, one week after the cut-off Saturday.
    assert snap.week[snap.asof[o]] == snap.week[snap.cut[o]] + 1
    assert snap.cut_week(o) == snap.n_weeks - 2
    # The newest data week (ages 0-6) is k = 0, with no empty week at k = 0.
    assert weeks_back(snap, o, 42) == 6
    errors = retro_errors(snap, o, Settings())
    assert errors.n[:, 0].sum() > 0
    origin = build_origin(snap, Settings(), errors=True)
    assert origin.rho is not None
    assert origin.rho[:, 0].max() > 0


def test_a_later_clock_ages_the_origin_days_and_lowers_the_nowcast(
    snap24,
) -> None:
    base = build_origin(snap24, Settings(), errors=False)
    late = build_origin(snap24, Settings(clock_shift=1), errors=False)
    early = build_origin(snap24, Settings(clock_shift=-1), errors=False)
    cd, day = 0, int(snap24.cut[-1])
    f = base.fit.factors[cd, 0]
    # One day later the newest day is age 1, chained over 1, 9, ..., 41.
    assert late.growth[cd, day] == pytest.approx(
        float(np.prod(f[1:42:8])), rel=1e-5
    )
    # One day earlier no age is below 0, so the newest day stays at 0.
    assert early.growth[cd, day] == pytest.approx(
        float(np.prod(f[0:42:8])), rel=1e-5
    )
    assert early.growth[cd, day - 1] == pytest.approx(
        float(np.prod(f[0:42:8])), rel=1e-5
    )
    newest = base.point[cd, -1]
    assert late.point[cd, -1] < newest < early.point[cd, -1]
    # The delay factors come from the release pairs and do not move.
    np.testing.assert_allclose(late.fit.factors, base.fit.factors)
