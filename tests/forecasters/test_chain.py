"""The daily chain ladder on release snapshots."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.baselinenowcast import log_factors, pair_sums
from bdbvhubrenewal.data.snapshots import snapshot_arrays, weekly_sums
from bdbvhubrenewal.forecasters.chain import (
    Settings,
    _delay_counts,
    _walk_spreads,
    build_origin,
    fit_delay,
    retro_errors,
    weeks_back,
)
from bdbvhubrenewal.forecasters.introductions import introduction_rates
from bdbvhubrenewal.forecasters.obs import MAX_SIZE, PRIOR_SIZE
from bdbvhubrenewal.forecasters.pooling import source_index

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


def test_growth_to_final_sums_the_daily_delay(origin24) -> None:
    fit = origin24.fit
    # The two 8 day pairs are in window, the 51 day pair is not.
    assert fit.pairs.tolist() == [[1, 2], [2, 3]]
    assert (fit.g >= 0).all()
    day = int(origin24.snap.cut[-1])
    g = fit.g[0]
    assert origin24.growth[0, day] == pytest.approx(np.exp(g.sum()))
    assert origin24.growth[0, day - 41] == pytest.approx(np.exp(g[41]))
    assert origin24.growth[0, day - 42] == 1.0


def test_the_start_week_is_the_latest_week_at_95_per_cent(
    snap24, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bdbvhubrenewal.forecasters import chain

    # Week w is 95.01% complete and every later week 94.99%.
    w = snap24.n_weeks - 3
    inv = np.where(snap24.week > w, 0.9499, 1.0)
    inv = np.where(snap24.week == w, 0.9501, inv)
    g = np.broadcast_to(1.0 / inv, (len(snap24.locations), inv.size))
    monkeypatch.setattr(chain, "final_growth", lambda *_: g)
    origin = build_origin(snap24, Settings(), errors=False)
    assert np.allclose(origin.complete[:, w], 0.9501)
    assert (origin.complete[:, w + 1 :] < 0.95).all()
    assert (origin.start == w).all()


def test_10_24_point_nowcasts_use_one_daily_delay(origin24) -> None:
    # Weeks 10-24, 10-17, 10-10, 10-03, from the two 8 day pairs.
    # Downward revisions are moved onto earlier releases before the sums.
    # The 10-09 release is cut at 10-07, its latest onset day.
    assert _weeks(origin24, "CD") == pytest.approx(
        [368.5, 373.4, 488.4, 515.9], abs=0.06
    )
    # Tshopo has under 10 cases a week, so uses the national delay.
    assert _weeks(origin24, "CD-TO") == pytest.approx(
        [10.1, 11.5, 21.3, 18.4], abs=0.06
    )


def test_provinces_use_own_factors_where_they_have_10_cases(origin24) -> None:
    assert _weeks(origin24, "CD-IT") == pytest.approx(
        [212.7, 181.3, 282.6, 253.1], abs=0.06
    )
    assert _weeks(origin24, "CD-NK") == pytest.approx(
        [118.5, 149.5, 154.6, 214.2], abs=0.06
    )
    assert _weeks(origin24, "CD-HU") == pytest.approx(
        [17.7, 30.7, 27.7, 29.9], abs=0.06
    )


def test_every_day_with_data_is_nowcast(origin24) -> None:
    snap = origin24.snap
    counts = snap.counts[0, -1]
    assert (origin24.growth[0][snap.observed[-1]] >= 0).all()
    newest = int(snap.cut[-1])
    # The newest day is nowcast too, with no handover to the walk.
    assert origin24.growth[0, newest] > 1.0
    assert counts[newest] >= 0


def test_every_release_with_a_pair_gives_a_retrospective(
    daily: pd.DataFrame,
) -> None:
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    snap = snapshot_arrays(daily, "2026-11-01", provinces)
    # 10-09 has no in-window pair. 10-17 and 10-24 chain to 11-01 over
    # gaps of 15 and 7 days with their own daily delay.
    errors = retro_errors(snap, 4, Settings())
    assert errors.releases == ["2026-10-17", "2026-10-24"]


def test_new_reports_minus_removals_is_the_realised_change(snap24) -> None:
    errors = retro_errors(snap24, 3, Settings())
    later = weekly_sums(snap24, 3, upto=int(snap24.cut[2]))
    cols = snap24.cut_week(2) - np.arange(errors.valid.shape[2])
    net = errors.added[0] - errors.removed[0]
    want = later[:, cols] - errors.reported[0]
    assert net[errors.valid[0]] == pytest.approx(want[errors.valid[0]])
    assert (errors.added >= 0).all()
    assert (errors.removed >= 0).all()


def test_the_fit_pools_three_errors_at_10_24(origin24) -> None:
    loc = origin24.snap.locations.index("CD")
    assert origin24.size is not None
    assert origin24.removal is not None
    assert origin24.obs_n is not None
    # Haut-Uele, Ituri and North Kivu give an error each. CD is their sum,
    # so it is not in its own pool.
    assert origin24.obs_n[loc, :4].tolist() == [3, 3, 3, 3]
    # Three errors are enough for a fit.
    assert (origin24.size[loc, :4] >= 0.1).all()
    assert (origin24.size[loc, :4] < MAX_SIZE).all()
    assert (origin24.removal >= 0).all()


def test_no_errors_take_the_prior_size(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT", "CD-NK"))
    origin = build_origin(snap, Settings(), errors=True)
    assert origin.obs_n is not None
    assert (origin.obs_n == 0).all()
    np.testing.assert_allclose(origin.size, PRIOR_SIZE, rtol=1e-6)
    assert (origin.removal == 0).all()


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
    window = np.arange(snap.week.size) > int(snap.cut[3]) - 63
    num, den = pair_sums(
        _delay_counts(snap, 3),
        snap.observed & window[None, :],
        snap.cut,
        fit.pairs[:, 0],
        42,
    )
    gaps = snap.cut[fit.pairs[:, 1]] - snap.cut[fit.pairs[:, 0]]
    own_g = log_factors(num, den, gaps)
    n_cases = np.asarray(den).sum(axis=1)
    seen = {"own": 0, "parent": 0}
    for i in range(len(snap.locations)):
        p = int(snap.parent[i])
        if p < 0:
            # The country has no parent, so it uses its own factor.
            np.testing.assert_allclose(fit.g[i], own_g[i])
            continue
        use = n_cases[i] >= 10
        seen["own"] += int(use.sum())
        seen["parent"] += int((~use).sum())
        np.testing.assert_allclose(
            fit.g[i][use], own_g[i][use], err_msg=snap.locations[i]
        )
        np.testing.assert_allclose(
            fit.g[i][~use],
            fit.g[p][~use],
            err_msg=snap.locations[i],
        )
    assert seen["own"] > 0
    assert seen["parent"] > 0


def test_the_delay_uses_every_in_window_pair(snap24) -> None:
    fit = fit_delay(snap24, 3, Settings())
    assert fit is not None
    # At 10-17 only the first 8 day pair is in window.
    early = fit_delay(snap24, 2, Settings())
    assert early is not None
    assert early.pairs.tolist() == [[1, 2]]
    assert not np.allclose(early.g, fit.g)


def test_weeks_back_count_from_the_cutoff_week(daily: pd.DataFrame) -> None:
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    snap = snapshot_arrays(daily, "2026-11-01", provinces)
    o = len(snap.releases) - 1
    # The cut-off is the Saturday 10-31, and the grid ends with its week.
    assert snap.days[snap.cut[o]].date().isoformat() == "2026-10-31"
    assert snap.cut_week(o) == snap.n_weeks - 1
    # The newest data week (ages 0-6) is k = 0, with no empty week at k = 0.
    assert weeks_back(snap, o, 42) == 6
    errors = retro_errors(snap, o, Settings())
    assert errors.valid[:, :, 0].sum() > 0
    origin = build_origin(snap, Settings(), errors=True)
    assert origin.size is not None
    assert origin.size[:, 0].min() < MAX_SIZE


def test_a_later_clock_ages_the_origin_days_and_lowers_the_nowcast(
    snap24,
) -> None:
    base = build_origin(snap24, Settings(), errors=False)
    late = build_origin(snap24, Settings(clock_shift=1), errors=False)
    early = build_origin(snap24, Settings(clock_shift=-1), errors=False)
    cd, day = 0, int(snap24.cut[-1])
    g = base.fit.g[cd]
    # One day later the newest day is age 1.
    assert late.growth[cd, day] == pytest.approx(np.exp(g[1:].sum()))
    # One day earlier no age is below 0, so the newest day stays at 0.
    assert early.growth[cd, day] == pytest.approx(np.exp(g.sum()))
    assert early.growth[cd, day - 1] == pytest.approx(np.exp(g.sum()))
    newest = base.point[cd, -1]
    assert late.point[cd, -1] < newest < early.point[cd, -1]
    # The delay comes from the release pairs and does not move.
    np.testing.assert_allclose(late.fit.g, base.fit.g)


def test_the_delay_uses_the_baselinenowcast_window_of_onset_days(
    snap24,
) -> None:
    assert Settings().history_delay == 63
    assert Settings(max_delay=10).history_delay == 15
    settings = Settings(max_delay=10)
    r = 3
    base = fit_delay(snap24, r, settings)
    assert base is not None
    day = np.arange(snap24.week.size)
    cut = int(snap24.cut[r])

    def scaled(old: bool):
        sel = day <= cut - 15 if old else (day > cut - 15) & (day <= cut)
        counts = snap24.counts.copy()
        counts[:, 1:, sel] *= 3.0
        return dataclasses.replace(snap24, counts=counts)

    outside = fit_delay(scaled(True), r, settings)
    inside = fit_delay(scaled(False), r, settings)
    assert outside is not None
    assert inside is not None
    np.testing.assert_allclose(outside.g, base.g)
    assert not np.allclose(inside.g, base.g)


def test_a_retrospective_fit_uses_its_own_cutoff_for_the_window(
    snap24,
) -> None:
    settings = Settings(max_delay=10)
    r = 2
    cut = int(snap24.cut[r])
    base = fit_delay(snap24, r, settings)
    counts = snap24.counts.copy()
    counts[:, 1:, : cut - 15 + 1] *= 3.0
    other = fit_delay(dataclasses.replace(snap24, counts=counts), r, settings)
    assert base is not None
    assert other is not None
    np.testing.assert_allclose(other.g, base.g)


def test_downward_revisions_move_onto_earlier_releases(snap24) -> None:
    counts = snap24.counts.copy()
    day = int(snap24.cut[1]) - 3
    counts[0, :, day] = [0, 10, 7, 9, 0][: counts.shape[1]]
    snap = dataclasses.replace(snap24, counts=counts)
    got = _delay_counts(snap, 3)[0, :4, day]
    # The 10 to 7 revision is taken from the day's first observed release.
    np.testing.assert_allclose(got, [0, 7, 7, 9])
    cum = _delay_counts(snap, 3)[:, :4]
    seen = snap.observed[:4]
    diff = np.diff(np.where(seen[None], cum, 0.0), axis=1)
    both = seen[1:] & seen[:-1]
    assert (diff[:, both] >= 0).all()
    f = fit_delay(snap, 3, Settings())
    assert f is not None
    assert (f.g >= 0).all()


def test_no_nowcast_is_below_the_reported_count_at_11_01(
    daily: pd.DataFrame, daily_zones: pd.DataFrame
) -> None:
    from bdbvhubrenewal.data import load_health_zones  # noqa: F401
    from bdbvhubrenewal.data.zones import hub_zones

    data = Path(__file__).resolve().parents[2] / "data"
    zones = hub_zones(data / "hub-config" / "tasks.json")
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    cases = {
        "provinces": (daily, provinces),
        "zones": (pd.concat([daily, daily_zones]), zones),
    }
    for name, (frame, places) in cases.items():
        snap = snapshot_arrays(frame, "2026-11-01", places)
        origin = build_origin(snap, Settings(), errors=False)
        short = origin.reported - origin.point
        assert short.max() <= 1e-3, (name, float(short.max()))


def test_introductions_pool_siblings_after_k_silent_weeks() -> None:
    # Country, then three provinces; k = 3 silent weeks before a week at risk.
    reported = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    p = introduction_rates(reported, np.full(4, 8), np.array([-1, 0, 0, 0]), 3)
    # Provinces: 6 + 0 + 3 weeks at risk and 1 first case.
    np.testing.assert_allclose(p[1:], 1 / 9)
    # The country is its own group: 3 weeks at risk, 1 first case.
    assert p[0] == pytest.approx(1 / 3)
    # Weeks after the start week do not count.
    late = introduction_rates(
        reported, np.array([8, 8, 8, 2]), np.array([-1, 0, 0, 0]), 3
    )
    np.testing.assert_allclose(late[1:], 0.0)


def test_the_walk_spread_is_net_of_poisson_noise(origin24) -> None:
    settings = origin24.settings
    n_loc = len(origin24.snap.locations)
    own = np.flatnonzero(origin24.sigma_source < n_loc)
    assert own.size
    for i in own:
        lo = max(int(origin24.start[i]) - settings.walk_weeks + 1, 0)
        y = origin24.reported[i, lo : int(origin24.start[i]) + 1]
        v = y / (1 + y) ** 2
        net = np.diff(np.log1p(y)).var(ddof=1) - (v[1:] + v[:-1]).mean()
        assert net > 0
        assert origin24.sigma[i] == pytest.approx(np.sqrt(net))


def test_a_place_with_no_walk_variance_takes_its_parents(origin24) -> None:
    snap = origin24.snap
    reported = origin24.reported.copy()
    # Make one province with its own spread flat, so its changes are noise.
    i = int(np.flatnonzero(origin24.sigma_source < len(snap.locations))[0])
    reported[i] = np.where(reported[i] > 0, reported[i].mean(), 0.0)
    own, pooled = _walk_spreads(
        reported, origin24.start, snap, origin24.fit.level, origin24.settings
    )
    assert own[i] <= 0
    source = source_index(
        origin24.fit.level,
        snap.parent,
        (own > 0).astype(int)[:, None],
        (pooled > 0).astype(int)[:, None],
        origin24.settings.min_level,
        1,
    )[:, 0]
    assert source[i] >= len(snap.locations)


def test_retrospective_nowcasts_reach_back_as_far_as_baselinenowcasts(
    daily: pd.DataFrame,
) -> None:
    # baselinenowcast uses round(3 D) reference times, half for the delay
    # and the rest for its retrospective nowcasts, so 63 days for D = 42.
    assert Settings().retro_delay == 63
    assert Settings(max_delay=10).retro_delay == 15
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    snap = snapshot_arrays(daily, "2026-11-08", provinces)
    o = len(snap.releases) - 1
    settings = Settings(max_delay=10)
    errors = retro_errors(snap, o, settings)
    for r, release in enumerate(snap.releases[:o]):
        near = snap.cut[o] - snap.cut[r] <= settings.retro_delay
        if not near:
            assert release not in errors.releases
    assert errors.releases
    # With D = 42 every earlier release with a pair is within 63 days.
    assert retro_errors(snap, o, Settings()).releases == [
        "2026-10-17",
        "2026-10-24",
        "2026-11-01",
    ]
