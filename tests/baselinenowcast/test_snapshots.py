"""Snapshot extension and hub weeks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from bnc_helpers import read, read_ints, read_vec

from bdbvhubrenewal.baselinenowcast import (
    chain_growth,
    exact_plan,
    hub_week_matrix,
    hub_weeks,
    pair_factors,
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
    f, ok = pair_factors(num, den, np.zeros(len(pairs), int), 1)
    assert bool(ok.all())
    plan = exact_plan(np.arange(d + 1), 1, 0, d)
    g = np.asarray(chain_growth(f, plan))[0]
    np.testing.assert_allclose(g[d], 1.0)
    np.testing.assert_allclose(
        np.cumsum(read_vec("recon", "pmf"))[:d], 1.0 / g[:d], rtol=1e-12
    )


def test_daily_snapshots_reproduce_the_filled_row_totals() -> None:
    counts, observed, cut, d, n, n_use = _recon()
    pairs, _ = window_pairs(cut, d)
    num, den = pair_sums(counts, observed, cut, pairs, d)
    f, _ = pair_factors(num, den, np.zeros(len(pairs), int), 1)
    rows = np.arange(n - n_use, n)
    plan = exact_plan(n - 1 - rows, 1, 0, d)
    g = np.asarray(chain_growth(f, plan))[0]
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


def test_pair_factors_match_r_and_leave_zero_denominators_unidentified() -> (
    None
):
    counts, observed, cut, d = _skip()
    pairs, steps = window_pairs(cut, d)
    assert 9 not in steps
    assert len(cut) - 1 == len(steps) + 1
    uniq = read_ints("skip", "uniq_steps")
    assert list(uniq) == sorted(set(steps.tolist()))
    num, den = pair_sums(counts, observed, cut, pairs, d)
    slot = np.searchsorted(uniq, steps)
    f, ok = pair_factors(num, den, slot, len(uniq))
    expected = read("skip", "factors").reshape(counts.shape[0], len(uniq), d)
    np.testing.assert_array_equal(~np.isnan(expected), np.asarray(ok))
    assert not bool(ok.all())
    np.testing.assert_allclose(
        np.asarray(f)[ok], expected[~np.isnan(expected)], rtol=1e-12
    )


def test_exact_chain_matches_r() -> None:
    counts, observed, cut, d = _skip()
    pairs, steps = window_pairs(cut, d)
    num, den = pair_sums(counts, observed, cut, pairs, d)
    uniq = read_ints("skip", "uniq_steps")
    f, _ = pair_factors(num, den, np.searchsorted(uniq, steps), len(uniq))
    s_o = read_ints("skip", "chain_step")[0]
    assert s_o == steps[-1]
    plan = exact_plan(np.arange(d), s_o, uniq.index(s_o), d)
    g = np.asarray(chain_growth(f, plan))
    expected = read("skip", "chain")
    np.testing.assert_array_equal(np.isnan(expected), np.isnan(g))
    np.testing.assert_allclose(
        g[~np.isnan(g)], expected[~np.isnan(expected)], rtol=1e-12
    )


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nan = 99999.0  # unobserved cells hold junk
    r0 = [10, 8, 5] + [nan] * 5
    r1 = [12, 10, 9, 7, 0] + [nan] * 3
    r2 = [13, 11, 10, 9, 2, 3, 4, 1]
    counts = np.array([[r0, r1, r2]], float)
    cut = np.array([2, 4, 7])
    observed = np.arange(8)[None, :] <= cut[:, None]
    return counts, observed, cut


def test_hand_derived_toy_with_two_steps_and_a_zero_denominator() -> None:
    counts, observed, cut = _toy()
    pairs, steps = window_pairs(cut, 4)
    np.testing.assert_array_equal(steps, [2, 3])
    num, den = pair_sums(counts, observed, cut, pairs, 4)
    # Pair 0 (step 2) has ages 2, 1, 0 at release 0; age 3 has no cell.
    np.testing.assert_array_equal(num[0, 0], [9, 10, 12, 0])
    np.testing.assert_array_equal(den[0, 0], [5, 8, 10, 0])
    # Pair 1 (step 3) has ages 3..0 at release 1; age 4 is outside.
    np.testing.assert_array_equal(num[0, 1], [2, 11 - 2, 10, 11])
    np.testing.assert_array_equal(den[0, 1], [0, 7, 9, 10])
    f, ok = pair_factors(num, den, [0, 1], 2)
    np.testing.assert_array_equal(
        np.asarray(ok)[0], [[1, 1, 1, 0], [0, 1, 1, 1]]
    )
    np.testing.assert_allclose(
        np.asarray(f)[0, 0, :3], [9 / 5, 10 / 8, 12 / 10]
    )
    np.testing.assert_allclose(
        np.asarray(f)[0, 1, 1:], [9 / 7, 10 / 9, 11 / 10]
    )
    # The origin is release 2 with step 3.
    plan = exact_plan(np.arange(6), 3, 1, 4)
    g = np.asarray(chain_growth(f, plan))[0]
    assert np.isnan(g[0])  # needs F3(0), which is unidentified
    np.testing.assert_allclose(g[1:4], [9 / 7, 10 / 9, 11 / 10])
    np.testing.assert_allclose(g[4:], 1.0)


def test_pairs_of_equal_step_are_summed_before_the_ratio() -> None:
    counts = np.array([[[4.0, 2.0, 0, 0], [5, 3, 0, 0], [6, 4, 1, 0]]])
    observed = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], bool)
    cut = np.array([1, 2, 3])
    pairs, _ = window_pairs(cut, 3)
    num, den = pair_sums(counts, observed, cut, pairs, 3)
    f, _ = pair_factors(num, den, [0, 0], 1)
    # Age 0 has the cell of pair 0 only, the other is unobserved.
    np.testing.assert_allclose(np.asarray(f)[0, 0], [3 / 2, 9 / 7, 6 / 5])


def test_exact_plan_has_no_tail_and_none_from_the_window() -> None:
    slot, age, mask = exact_plan([0, 1, 5, 7, 8, 12], 3, 2, 8)
    np.testing.assert_array_equal(mask.sum(1), [3, 3, 1, 1, 0, 0])
    np.testing.assert_array_equal(age[0][mask[0]], [0, 3, 6])
    np.testing.assert_array_equal(age[3][mask[3]], [7])
    assert np.all(slot[mask] == 2)
    # Fewest steps with a + M s >= D, so the last age is below D.
    assert int(age[mask].max()) < 8
    with pytest.raises(ValueError):
        exact_plan([-1], 3, 0, 8)
    with pytest.raises(ValueError):
        exact_plan([0], 0, 0, 8)


def test_the_longest_step_gives_no_factor_but_keeps_its_counts() -> None:
    counts = np.ones((1, 2, 60))
    observed = np.ones((2, 60), bool)
    cut = np.array([0, 50])
    pairs, _ = window_pairs(cut, 42)
    assert len(pairs) == 0
    num, den = pair_sums(counts, observed, cut, [0], 42)
    assert float(den.sum()) == 1.0
    _, ok = pair_factors(num[:, :0], den[:, :0], np.zeros(0, int), 1)
    assert not bool(ok.any())
    assert counts.sum() == 120


def test_chain_vmaps_over_locations() -> None:
    counts, observed, cut = _toy()
    both = np.concatenate([counts, 2 * counts])
    pairs, _ = window_pairs(cut, 4)
    num, den = pair_sums(both, observed, cut, pairs, 4)
    f, _ = pair_factors(num, den, [0, 1], 2)
    plan = exact_plan(np.arange(4), 3, 1, 4)
    g = np.asarray(chain_growth(f, plan))
    np.testing.assert_allclose(g[0], g[1])
    single = jax.vmap(
        lambda c: pair_sums(c[None], observed, cut, pairs, 4)[0][0]
    )(jnp.asarray(both))
    np.testing.assert_allclose(single, np.asarray(num))
