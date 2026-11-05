"""Snapshot extension and hub weeks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from bnc_helpers import read, read_ints, read_vec

from bdbvhubrenewal.baselinenowcast import (
    age_blocks,
    fill_row,
    growth,
    hub_week_matrix,
    hub_weeks,
    log_factors,
    pair_sums,
    window_pairs,
)


def test_hub_week_matrix_uses_sunday_to_saturday_weeks() -> None:
    first = "2026-07-05"  # a Sunday
    w, sats = hub_week_matrix(first, 17)
    assert w.shape == (3, 17)
    assert [str(s) for s in sats] == ["2026-07-11", "2026-07-18", "2026-07-25"]
    np.testing.assert_array_equal(w.sum(1), [7, 7, 3])
    w2, sats2 = hub_week_matrix("2026-07-08", 17)
    assert str(sats2[0]) == "2026-07-11"
    assert w2[0].sum() == 4
    assert w2.sum() == 17
    np.testing.assert_array_equal(w2.sum(0), np.ones(17))


def test_hub_weeks_give_12_horizons_at_a_saturday_and_13_otherwise() -> None:
    d = 84
    n_ref = 250
    for cutoff, expected in (("2026-10-24", 12), ("2026-10-16", 13)):
        first = pd.Timestamp(cutoff) - pd.Timedelta(days=n_ref - 1)
        t = np.arange(n_ref)[:, None]
        dl = np.arange(d + 1)[None, :]
        mask = (t + dl) <= n_ref - 1
        agg = hub_weeks(first.date())
        _, m = agg(np.where(mask, 1.0, 0.0), mask)
        assert int((~m.all(axis=1)).sum()) == expected
        # The cut-off week holds only days up to the cut-off.
        w, _ = hub_week_matrix(first.date(), n_ref)
        assert w[-1].sum() == {"2026-10-24": 7, "2026-10-16": 6}[cutoff]


def test_hub_weeks_sum_days_and_propagate_unobserved() -> None:
    first = "2026-07-05"
    n = 16
    v = np.ones((n, 2))
    m = np.ones((n, 2), bool)
    m[-1, 1] = False
    out, ok = hub_weeks(first)(v, m)
    np.testing.assert_allclose(np.asarray(out)[:, 0], [7, 7, 2])
    np.testing.assert_array_equal(ok[:, 1], [True, True, False])
    assert ok[:, 0].all()


# ---- E1: skipped cells in the chain ladder ratio of sums ---------------


def _recon() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """A right-truncated triangle as daily release snapshots."""
    tri = read("recon", "tri")
    d, n, n_use = read_ints("recon", "settings")
    t = np.arange(n)
    counts = np.zeros((1, n, n))
    for r in range(n):
        for day in range(r + 1):
            counts[0, r, day] = np.nansum(tri[day, : r - day + 1])
    observed = (t[None, :] <= t[:, None]) & (t[None, :] >= n - n_use)
    return counts, observed, t, d, n, n_use


def test_daily_snapshots_reduce_to_estimate_delay() -> None:
    counts, observed, cut, d, _, _ = _recon()
    pairs, steps = window_pairs(cut, d)
    assert set(steps) == {1}
    num, den = pair_sums(counts, observed, cut, pairs, d)
    assert bool(jnp.all(jnp.sum(den, axis=1) > 0))
    g = growth(log_factors(num, den, steps), np.arange(d + 1), d)[0]
    np.testing.assert_allclose(g[d], 1.0)
    np.testing.assert_allclose(
        np.cumsum(read_vec("recon", "pmf"))[:d], 1.0 / g[:d], rtol=1e-12
    )


def test_daily_snapshots_reproduce_the_filled_row_totals() -> None:
    counts, observed, cut, d, n, n_use = _recon()
    pairs, steps = window_pairs(cut, d)
    num, den = pair_sums(counts, observed, cut, pairs, d)
    rows = np.arange(n - n_use, n)
    g = growth(log_factors(num, den, steps), n - 1 - rows, d)[0]
    got = counts[0, n - 1, rows] * g
    np.testing.assert_allclose(got, read_vec("recon", "row_totals"), rtol=1e-12)


def _skip() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    d, n_loc, _ = read_ints("skip", "settings")
    counts = np.stack(
        [np.nan_to_num(read("skip", f"counts_{i + 1}")) for i in range(n_loc)]
    )
    observed = read("skip", "observed").astype(bool)
    cut = np.array(read_ints("skip", "cut"))
    return counts, observed, cut, d


def test_pair_sums_skip_unobserved_cells_as_in_r() -> None:
    counts, observed, cut, d = _skip()
    n_loc, n_rel, _ = counts.shape
    pairs = np.arange(n_rel - 1)
    # Garbage under False must never enter.
    junk = np.where(observed[None], counts, 12345.0)
    num, den = pair_sums(junk, observed, cut, pairs, d)
    r = read("skip", "pair_sums").reshape(n_loc, n_rel - 1, 2, d)
    np.testing.assert_allclose(num, r[:, :, 0])
    np.testing.assert_allclose(den, r[:, :, 1])


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nan = 99999.0  # unobserved cells hold junk
    r0 = [10, 8, 5] + [nan] * 5
    r1 = [12, 10, 9, 7, 0] + [nan] * 3
    r2 = [13, 11, 10, 9, 2, 3, 4, 1]
    counts = np.array([[r0, r1, r2]], float)
    cut = np.array([2, 4, 7])
    observed = np.arange(8)[None, :] <= cut[:, None]
    return counts, observed, cut


def test_pairs_of_equal_step_are_summed_before_the_ratio() -> None:
    counts = np.array([[[4.0, 2.0, 0, 0], [5, 3, 0, 0], [6, 4, 1, 0]]])
    observed = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], bool)
    cut = np.array([1, 2, 3])
    pairs, _ = window_pairs(cut, 3)
    num, den = pair_sums(counts, observed, cut, pairs, 3)
    g = log_factors(num, den, [1, 1])
    # Age 0 has the cell of pair 0 only, the other is unobserved.
    np.testing.assert_allclose(np.exp(g[0]), [3 / 2, 9 / 7, 6 / 5])


def test_the_longest_gap_gives_no_pair_but_keeps_its_counts() -> None:
    counts = np.ones((1, 2, 60))
    observed = np.ones((2, 60), bool)
    cut = np.array([0, 50])
    pairs, _ = window_pairs(cut, 42)
    assert len(pairs) == 0
    _, den = pair_sums(counts, observed, cut, [0], 42)
    assert float(den.sum()) == 1.0
    assert counts.sum() == 120


def test_chain_vmaps_over_locations() -> None:
    counts, observed, cut = _toy()
    both = np.concatenate([counts, 2 * counts])
    pairs, _ = window_pairs(cut, 4)
    num, den = pair_sums(both, observed, cut, pairs, 4)
    g = log_factors(num, den, [2, 3])
    # Doubling every count leaves every ratio, so the delay, the same.
    np.testing.assert_allclose(g[0], g[1], atol=1e-12)
    single = jax.vmap(
        lambda c: pair_sums(c[None], observed, cut, pairs, 4)[0][0]
    )(jnp.asarray(both))
    np.testing.assert_allclose(single, np.asarray(num))


def test_gaps_7_8_and_9_recover_one_daily_delay() -> None:
    d = 12
    true = np.array([0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.04, 0, 0.02, 0, 0])
    total = np.concatenate([[0.0], np.cumsum(true)])
    cut = np.array([9, 16, 24, 33])
    n_day = 34
    counts = np.zeros((1, len(cut), n_day))
    for r, c in enumerate(cut):
        age = np.clip(c - np.arange(n_day), 0, d)
        share = np.exp(total[age] - total[d])
        counts[0, r] = np.where(c >= np.arange(n_day), 1e4 * share, 0)
    observed = np.arange(n_day)[None, :] <= cut[:, None]
    pairs, gaps = window_pairs(cut, d)
    np.testing.assert_array_equal(gaps, [7, 8, 9])
    num, den = pair_sums(counts, observed, cut, pairs, d)
    g = log_factors(num, den, gaps)
    np.testing.assert_allclose(g[0], true, atol=1e-9)
    # Growth to final is the sum of the delay from the age on.
    np.testing.assert_allclose(
        growth(g, np.arange(d), d)[0], np.exp(total[d] - total[:d])
    )


def test_the_delay_is_never_negative() -> None:
    num = np.array([[[10.0, 4.0, 5.0]]])
    den = np.array([[[5.0, 5.0, 5.0]]])
    g = log_factors(num, den, [1])
    # Age 1 fell from 5 to 4, which is no removal in the delay.
    np.testing.assert_allclose(g[0], [np.log(2), 0, 0])


def test_fill_row_matches_r_apply_delay_on_zero_rows() -> None:
    tri, pmf = read("zeros", "tri"), read_vec("zeros", "pmf")
    want = np.cumsum(read("zeros", "apply"), axis=1)
    cdf = np.cumsum(pmf)
    for row in np.flatnonzero(np.isnan(tri).any(axis=1)):
        a = int(np.flatnonzero(~np.isnan(tri[row])).max())
        got = [
            fill_row(np.nansum(tri[row]), cdf[a : b + 1])
            for b in range(a, cdf.size)
        ]
        np.testing.assert_allclose(got, want[row, a:], rtol=1e-12)


def test_age_blocks_pool_ages_until_they_reach_the_count() -> None:
    den = np.array([[0.0, 3, 5, 16, 10, 2, 1]])
    np.testing.assert_array_equal(age_blocks(den, 10), [0, 0, 0, 0, 1, 1, 1])
    # 0 gives one block per age.
    np.testing.assert_array_equal(age_blocks(den, 0), np.arange(7))


def test_ages_in_a_block_share_one_factor() -> None:
    # Ages 0 and 1 have 2 and 3 cases, so they share a factor.
    den = np.array([[[2.0, 3.0, 20.0, 20.0]]])
    num = den * np.exp([[[0.6, 0.2, 0.1, 0.0]]])
    g = log_factors(num, den, [1], min_count=5)[0]
    assert g[0] == g[1]
    # The shared factor is the denominator-weighted mean of the log ratios.
    assert g[0] == pytest.approx((2 * 0.6 + 3 * 0.2) / 5)
