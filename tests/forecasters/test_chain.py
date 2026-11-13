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
    levels,
    retro_errors,
    weeks_back,
)
from bdbvhubrenewal.forecasters.obs import MAX_SIZE, PRIOR_SIZE
from bdbvhubrenewal.forecasters.pooling import pool_matrix, source_index

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
    # Tshopo has under 10 cases a week, so uses the delay pooled over the
    # provinces that have them.
    assert _weeks(origin24, "CD-TO") == pytest.approx(
        [9.9, 11.5, 21.3, 18.5], abs=0.06
    )


def test_provinces_use_own_delay_with_10_cases_a_week(origin24) -> None:
    assert _weeks(origin24, "CD-IT") == pytest.approx(
        [206.7, 181.3, 282.6, 253.1], abs=0.06
    )
    assert _weeks(origin24, "CD-NK") == pytest.approx(
        [105.7, 149.5, 154.6, 214.2], abs=0.06
    )
    assert _weeks(origin24, "CD-HU") == pytest.approx(
        [10.3, 33.0, 26.5, 29.7], abs=0.06
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


def test_new_reports_are_the_change_after_downward_revisions_move(
    snap24,
) -> None:
    errors = retro_errors(snap24, 3, Settings())
    assert not hasattr(errors, "removed")
    # As baselinenowcast preprocesses its triangle, a fall is taken from
    # the earlier releases' counts, so each change is a rise.
    pre = dataclasses.replace(snap24, counts=_delay_counts(snap24, 3))
    later = weekly_sums(pre, 3, upto=int(snap24.cut[2]))
    cols = snap24.cut_week(2) - np.arange(errors.valid.shape[2])
    want = later[:, cols] - errors.reported[0]
    ok = errors.valid[0]
    assert errors.added[0][ok] == pytest.approx(want[ok])
    assert errors.reported[0][ok] == pytest.approx(
        weekly_sums(pre, 2)[:, cols][ok]
    )
    assert (errors.added >= 0).all()


def test_a_place_with_10_cases_fits_its_own_errors_at_10_24(origin24) -> None:
    loc = origin24.snap.locations.index("CD")
    assert origin24.size is not None
    assert not hasattr(origin24, "removal")
    assert origin24.obs_n is not None
    # 10-17 is the one earlier release, so the country has one error at
    # each week back, and it is enough for its own fit.
    assert origin24.obs_n[loc, :4].tolist() == [1, 1, 1, 1]
    assert (origin24.size[loc, :4] >= 0.1).all()
    assert (origin24.size[loc, :4] < MAX_SIZE).all()


def test_no_errors_take_the_prior_size(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT", "CD-NK"))
    origin = build_origin(snap, Settings(), errors=True)
    assert origin.obs_n is not None
    assert (origin.obs_n == 0).all()
    np.testing.assert_allclose(origin.size, PRIOR_SIZE, rtol=1e-6)


def test_a_delay_is_own_with_10_cases_a_week_else_the_pooled_parents(
    daily: pd.DataFrame, daily_zones: pd.DataFrame
) -> None:
    both = pd.concat([daily, daily_zones])
    zones = tuple(
        sorted({z for z in daily_zones["location"] if z.startswith("CD-")})
    )
    snap = snapshot_arrays(both, "2026-10-24", zones)
    settings = Settings()
    fit = fit_delay(snap, 3, settings)
    assert fit is not None
    window = np.arange(snap.week.size) > int(snap.cut[3]) - 63
    num, den = pair_sums(
        _delay_counts(snap, 3),
        snap.observed & window[None, :],
        snap.cut,
        fit.pairs[:, 0],
        42,
    )
    num, den = np.asarray(num), np.asarray(den)
    gaps = snap.cut[fit.pairs[:, 1]] - snap.cut[fit.pairs[:, 0]]
    level = levels(snap, 3, settings)
    pool = pool_matrix(level, snap.parent)
    own_g = log_factors(num, den, gaps)
    pooled_g = log_factors(
        np.einsum("nm,mpa->npa", pool, num),
        np.einsum("nm,mpa->npa", pool, den),
        gaps,
    )
    # The same rule as the walk spread: own with 10 cases a week, else
    # the pooled estimate of the nearest parent that has them. Zones have
    # their own threshold
    # (`test_a_zone_below_the_threshold_takes_its_provinces_delay`).
    src = source_index(level, snap.parent)
    kept = np.array([c.count("-") < 2 for c in snap.locations])
    want = np.concatenate([own_g, pooled_g])[src]
    np.testing.assert_allclose(fit.g[kept], want[kept])
    assert (src[kept] < len(level)).any()
    assert (src[kept] >= len(level)).any()


@pytest.mark.parametrize("threshold", [5.0, 10.0, 30.0, np.inf])
def test_a_zone_below_the_threshold_takes_its_provinces_delay(
    daily: pd.DataFrame, daily_zones: pd.DataFrame, threshold: float
) -> None:
    both = pd.concat([daily, daily_zones])
    zones = tuple(
        sorted({z for z in daily_zones["location"] if z.startswith("CD-")})
    )
    snap = snapshot_arrays(both, "2026-10-24", zones)
    settings = Settings(zone_delay_min_level=threshold)
    fit = fit_delay(snap, 3, settings)
    assert fit is not None
    level = levels(snap, 3, settings)
    zone = np.array([c.count("-") == 2 for c in snap.locations])
    rule = source_index(level, snap.parent)
    # A zone with `threshold` cases a week uses its own delay, and
    # otherwise its province's.
    big = np.flatnonzero(zone & (level >= threshold))
    small = np.flatnonzero(zone & (level < threshold))
    assert small.size
    np.testing.assert_array_equal(fit.source[big], big)
    assert fit.own[big].all()
    assert not fit.own[small].any()
    np.testing.assert_array_equal(
        fit.source[small], fit.source[snap.parent[small]]
    )
    # Provinces and the country keep the rule.
    np.testing.assert_array_equal(fit.source[~zone], rule[~zone])


def test_one_rule_shares_the_size_by_level_only(origin24) -> None:
    assert not hasattr(Settings(), "min_errors")
    assert origin24.obs_source is not None
    src = source_index(origin24.fit.level, origin24.snap.parent)
    for k in range(origin24.obs_source.shape[1]):
        np.testing.assert_array_equal(origin24.obs_source[:, k], src)


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


def test_there_is_no_clock_shift() -> None:
    assert not hasattr(Settings(), "clock_shift")


def test_no_walk_spread_anywhere_raises(snap24, monkeypatch) -> None:
    # If no place, pool or country has a walk variance above 0, there is
    # no spread to fall back on, so the model gives no forecast.
    from bdbvhubrenewal.forecasters import ModelUnavailableError, chain

    def flat(reported, *_):
        n = reported.shape[0]
        return np.zeros(n), np.full(n, np.nan)

    monkeypatch.setattr(chain, "_walk_spreads", flat)
    with pytest.raises(ModelUnavailableError, match="walk spread"):
        build_origin(snap24, Settings(), errors=False)


def test_the_delay_uses_the_baselinenowcast_window_of_onset_days(
    snap24,
) -> None:
    assert Settings().history_delay == 63
    assert Settings(max_delay=10).history_delay == 15
    # Every place fits its own delay, so the level, which reads older
    # weeks, does not change which delay a place uses.
    settings = Settings(max_delay=10, min_level=0.0)
    r = 3
    base = fit_delay(snap24, r, settings)
    assert base is not None
    day = np.arange(snap24.week.size)
    cut = int(snap24.cut[r])

    def scaled(old: bool):
        sel = day <= cut - 15 if old else (day > cut - 15) & (day <= cut)
        counts = snap24.counts.copy()
        # Tripling the latest release only changes its ratios to the one
        # before.
        counts[:, r, sel] *= 3.0
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
    # Every place fits its own delay, so the level, which reads older
    # weeks, does not change which delay a place uses.
    settings = Settings(max_delay=10, min_level=0.0)
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

    data = Path(__file__).resolve().parents[1] / "fixtures" / "data"
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
