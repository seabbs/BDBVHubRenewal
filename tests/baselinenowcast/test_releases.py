"""Release data extension (E1) and hub weeks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bnc_helpers import from_na, read, read_ints, read_vec

from bdbvhubrenewal.baselinenowcast import (
    estimate_delay,
    estimate_delay_releases,
    expand_release_triangle,
    hub_week_matrix,
    hub_weeks,
    preprocess_negative_values,
    release_increments,
)


def _rel() -> tuple[np.ndarray, np.ndarray, list[int], int]:
    cum_na = read("rel", "cum")
    d, n_ref, *cutoffs = read_ints("rel", "settings")
    cum, rmask = from_na(cum_na)
    assert cum.shape[0] == n_ref
    return cum, rmask, cutoffs, d


def test_release_increments_match_r() -> None:
    cum, rmask, cutoffs, d = _rel()
    inc, lo, hi, mask = release_increments(cum, rmask, cutoffs, d)
    r_inc = read("rel", "increments")
    np.testing.assert_array_equal(mask, ~np.isnan(r_inc))
    np.testing.assert_allclose(np.asarray(inc)[mask], r_inc[mask], atol=1e-8)
    np.testing.assert_array_equal(lo[mask], read("rel", "lo")[mask])
    np.testing.assert_array_equal(hi[mask], read("rel", "hi")[mask])


def test_increments_drop_beyond_max_delay_and_clip_the_first() -> None:
    cum, rmask, cutoffs, d = _rel()
    _, _, hi, mask = release_increments(cum, rmask, cutoffs, d)
    assert hi[mask].max() == d
    # Release 2 for onset day 0 has c_1 - t = 8 >= D so is dropped.
    assert not mask[0, 1]
    assert mask[0, 0]
    assert hi[0, 0] == d


def test_increments_are_missing_not_zero_for_unreleased_days() -> None:
    cum, rmask, cutoffs, d = _rel()
    inc, _, _, mask = release_increments(cum, rmask, cutoffs, d)
    assert not mask[12, 0]
    assert not mask[13, 1]
    assert float(np.asarray(inc)[~mask].sum()) == 0.0


def test_values_under_false_never_change_the_extension() -> None:
    cum, rmask, cutoffs, d = _rel()
    dirty = np.where(rmask, cum, 1e6)
    a = release_increments(dirty, rmask, cutoffs, d)
    b = release_increments(cum, rmask, cutoffs, d)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_expand_with_uniform_pmf_matches_r_even_spread() -> None:
    cum, rmask, cutoffs, d = _rel()
    inc, lo, hi, mask = release_increments(cum, rmask, cutoffs, d)
    pre = preprocess_negative_values(inc, mask, validate=False)
    p0 = np.full(d + 1, 1 / (d + 1))
    x, xm = expand_release_triangle(pre, lo, hi, mask, p0)
    r = read("rel", "expanded_uniform")
    np.testing.assert_array_equal(xm, ~np.isnan(r))
    np.testing.assert_allclose(np.asarray(x)[xm], r[xm], atol=1e-8)


def test_one_em_iteration_matches_r_estimate_delay() -> None:
    cum, rmask, cutoffs, d = _rel()
    inc, lo, hi, mask = release_increments(cum, rmask, cutoffs, d)
    n = read_ints("rel", "n")[0]
    pmf, n_iter, _ = estimate_delay_releases(
        inc, lo, hi, mask, n=n, tol=0.0, max_iter=1
    )
    assert n_iter == 1
    np.testing.assert_allclose(pmf, read_vec("rel", "pmf_iter1"), atol=1e-8)


def test_fifty_em_iterations_match_r_em() -> None:
    cum, rmask, cutoffs, d = _rel()
    inc, lo, hi, mask = release_increments(cum, rmask, cutoffs, d)
    n = read_ints("rel", "n")[0]
    pmf, n_iter, _ = estimate_delay_releases(
        inc, lo, hi, mask, n=n, tol=0.0, max_iter=50
    )
    assert n_iter == 50
    np.testing.assert_allclose(pmf, read_vec("rel", "pmf_iter50"), atol=1e-8)
    pre = preprocess_negative_values(inc, mask, validate=False)
    x, xm = expand_release_triangle(pre, lo, hi, mask, pmf)
    r = read("rel", "expanded_iter50")
    np.testing.assert_allclose(np.asarray(x)[xm], r[xm], atol=1e-8)


def test_expanded_row_sums_equal_preprocessed_increment_row_sums() -> None:
    cum, rmask, cutoffs, d = _rel()
    inc, lo, hi, mask = release_increments(cum, rmask, cutoffs, d)
    pre = np.asarray(preprocess_negative_values(inc, mask, validate=False))
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(d + 1))
    x, _ = expand_release_triangle(pre, lo, hi, mask, p)
    np.testing.assert_allclose(np.asarray(x).sum(1), pre.sum(1), atol=1e-9)


def test_even_split_when_the_pmf_is_zero_on_an_interval() -> None:
    inc = np.array([[6.0]])
    lo = np.array([[1]])
    hi = np.array([[3]])
    mask = np.array([[True]])
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    x, _ = expand_release_triangle(inc, lo, hi, mask, p, cutoff=3)
    np.testing.assert_allclose(np.asarray(x)[0, :4], [0.0, 2.0, 2.0, 2.0])


def _daily_releases(
    tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    """One release per day so every interval is a single day."""
    n_ref, n_delay = tri.shape
    d = n_delay - 1
    cum = np.zeros((n_ref, n_ref))
    rm = np.zeros((n_ref, n_ref), bool)
    for k in range(n_ref):
        for t in range(k + 1):
            upto = min(k - t, d)
            cum[t, k] = tri[t, : upto + 1].sum()
            rm[t, k] = True
    return cum, rm, list(range(n_ref)), d


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_day_intervals_reduce_to_estimate_delay(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_ref, n_delay = 16, 5
    tri = rng.poisson(rng.uniform(1, 30, n_delay), (n_ref, n_delay)).astype(
        float
    )
    tri[3, 2] = -4.0
    tri[8, 4] = -2.0
    t = np.arange(n_ref)[:, None]
    d = np.arange(n_delay)[None, :]
    tmask = (t + d) <= n_ref - 1
    tri = np.where(tmask, tri, 0.0)
    pre = np.asarray(preprocess_negative_values(tri, tmask))
    cum, rm, cutoffs, dmax = _daily_releases(tri)
    inc, lo, hi, mask = release_increments(cum, rm, cutoffs, dmax)
    n = 10
    expected = estimate_delay(pre, tmask, n=n)
    pmf, n_iter, change = estimate_delay_releases(inc, lo, hi, mask, n=n)
    np.testing.assert_allclose(pmf, expected, atol=1e-12)
    assert n_iter <= 2
    assert change < 1e-6
    pre_inc = preprocess_negative_values(inc, mask, validate=False)
    x, xm = expand_release_triangle(pre_inc, lo, hi, mask, np.asarray(expected))
    np.testing.assert_array_equal(xm, tmask)
    np.testing.assert_allclose(np.asarray(x), pre, atol=1e-9)


def _simulate_sparse(seed: int = 4) -> tuple[np.ndarray, ...]:
    """Large counts, a known pmf and releases every 8 days."""
    rng = np.random.default_rng(seed)
    d = 20
    pmf = np.exp(-np.arange(d + 1) / 4.0)
    pmf /= pmf.sum()
    n_ref = 120
    level = 20000 * np.ones(n_ref)
    full = rng.poisson(level[:, None] * pmf[None, :])
    cutoffs = [60, 70, 78, 86, 94, 102, 110, 119]
    cum = np.zeros((n_ref, len(cutoffs)))
    rm = np.zeros((n_ref, len(cutoffs)), bool)
    for k, c in enumerate(cutoffs):
        for t in range(c + 1):
            cum[t, k] = full[t, : min(c - t, d) + 1].sum()
            rm[t, k] = True
    return cum, rm, np.asarray(cutoffs), d, pmf  # type: ignore[return-value]


def test_em_recovers_the_pmf_from_sparse_releases() -> None:
    cum, rm, cutoffs, d, pmf = _simulate_sparse()
    inc, lo, hi, mask = release_increments(cum, rm, cutoffs, d)
    est, n_iter, change = estimate_delay_releases(inc, lo, hi, mask, n=60)
    assert change < 1e-6
    assert n_iter < 5000
    np.testing.assert_allclose(est, pmf, atol=0.01)


def test_em_converges_with_the_default_tolerance() -> None:
    cum, rm, cutoffs, d, _ = _simulate_sparse(seed=9)
    inc, lo, hi, mask = release_increments(cum, rm, cutoffs, d)
    _, n_iter, change = estimate_delay_releases(inc, lo, hi, mask, n=60)
    assert change < 1e-6
    assert 1 < n_iter < 5000


def test_release_aborts() -> None:
    cum, rm, cutoffs, d = _rel()
    with pytest.raises(ValueError):
        release_increments(cum, rm, cutoffs[:2], d)
    with pytest.raises(ValueError):
        release_increments(cum, rm, cutoffs[::-1], d)
    inc, lo, hi, mask = release_increments(cum, rm, cutoffs, d)
    with pytest.raises(ValueError):
        estimate_delay_releases(inc, lo, hi, mask, n=1000)


def test_hub_week_matrix_uses_sunday_to_saturday_weeks() -> None:
    first = "2026-07-05"  # a Sunday
    w, sats = hub_week_matrix(first, 17)
    assert w.shape == (3, 17)
    assert [str(s) for s in sats] == ["2026-07-11", "2026-07-18", "2026-07-25"]
    np.testing.assert_array_equal(w.sum(1), [7, 7, 3])
    w2, sats2 = hub_week_matrix("2026-07-08", 17)
    assert str(sats2[0]) == "2026-07-18"
    assert w2[0, :4].sum() == 0
    assert w2[0].sum() == 7


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
