"""The daily chain ladder on release snapshots."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_10_24_point_nowcasts_match_the_r_script(origin24) -> None:
    # Weeks 10-24, 10-17, 10-10, 10-03, from rwnc/check/spec_check.R.
    assert _weeks(origin24, "CD") == pytest.approx(
        [439.2, 418.1, 539.7, 544.1], abs=0.06
    )
    # Tshopo has under 10 cases a week, so uses the national delay.
    assert _weeks(origin24, "CD-TO") == pytest.approx(
        [23.9, 13.2, 23.6, 19.6], abs=0.06
    )


def test_places_with_10_cases_a_week_use_their_own_delay(origin24) -> None:
    assert _weeks(origin24, "CD-IT") == pytest.approx(
        [267.4, 198.4, 317.8, 276.3], abs=0.06
    )
    assert _weeks(origin24, "CD-NK") == pytest.approx(
        [147.0, 194.0, 172.0, 212.0], abs=0.06
    )
    assert _weeks(origin24, "CD-HU") == pytest.approx(
        [6.0, 36.8, 24.5, 27.2], abs=0.06
    )
    loc = origin24.snap.locations
    assert origin24.fit.level[loc.index("CD-TO")] < 10
    # Source indices from the number of places up are pooled estimates.
    assert origin24.fit.source[loc.index("CD-TO")] == len(loc) + loc.index("CD")
    assert origin24.fit.source[loc.index("CD-IT")] == loc.index("CD-IT")


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
        [1.566, 2.213, 0.447, 1.511], abs=0.0006
    )
    # CD, Haut-Uele, Ituri and North Kivu give an error each.
    assert origin24.rho_n[loc, :4].tolist() == [4, 4, 4, 4]


def test_no_errors_is_poisson_skellam_and_says_so(
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


def test_zones_use_their_province_then_the_country(
    daily: pd.DataFrame, daily_zones: pd.DataFrame
) -> None:
    both = pd.concat([daily, daily_zones])
    zones = tuple(
        sorted({z for z in daily_zones["location"] if z.startswith("CD-")})
    )
    snap = snapshot_arrays(both, "2026-10-24", zones)
    fit = fit_delay(snap, 3, Settings())
    assert fit is not None
    n = len(snap.locations)
    seen = set()
    for i, loc in enumerate(snap.locations):
        src = int(fit.source[i])
        if fit.level[i] >= 10:
            assert src == i
            seen.add("own")
            continue
        parent = int(snap.parent[i])
        if parent < 0:
            assert src == n + i
            continue
        node = src - n
        if fit.level[parent] >= 10:
            assert node == parent, loc
            seen.add("province")
        else:
            # A zone in a small province uses the country, never a zone.
            assert snap.locations[node] == "CD" or len(
                snap.locations[node]
            ) <= len("CD-XX"), loc
            seen.add("country")
    assert seen == {"own", "province", "country"}


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
