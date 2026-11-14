"""Delay estimation and point nowcasts against R baselinenowcast 0.2.2."""

from __future__ import annotations

import numpy as np
import pytest
from bnc_helpers import Case, from_na, read, read_vec

from bdbvhubrenewal.baselinenowcast import (
    apply_delay,
    estimate_and_apply_delay,
    estimate_delay,
    preprocess_negative_values,
)


def test_estimate_delay_matches_r(case: Case) -> None:
    raw = estimate_delay(case.values, case.mask, n=case.n_hist)
    np.testing.assert_allclose(raw, read_vec(case.name, "pmf_raw"), atol=1e-8)
    pre = preprocess_negative_values(case.values, case.mask)
    got = estimate_delay(pre, case.mask, n=case.n_hist)
    np.testing.assert_allclose(got, read_vec(case.name, "pmf"), atol=1e-8)
    full = estimate_delay(pre, case.mask)
    np.testing.assert_allclose(full, read_vec(case.name, "pmf_all"), atol=1e-8)


def test_apply_delay_matches_r(case: Case) -> None:
    pre = preprocess_negative_values(case.values, case.mask)
    pmf = read_vec(case.name, "pmf")
    got = apply_delay(pre, case.mask, pmf)
    np.testing.assert_allclose(
        got, read(case.name, "apply"), rtol=1e-10, atol=1e-8
    )


def test_estimate_and_apply_delay_matches_r(case: Case) -> None:
    pre = preprocess_negative_values(case.values, case.mask)
    got = estimate_and_apply_delay(pre, case.mask, n=case.n_hist)
    np.testing.assert_allclose(
        got, read(case.name, "eaa"), rtol=1e-10, atol=1e-8
    )
    given = estimate_and_apply_delay(
        pre, case.mask, n=case.n_hist, delay_pmf=read_vec(case.name, "pmf")
    )
    np.testing.assert_allclose(given, got, atol=1e-8)


def test_observed_cells_are_kept(small: Case) -> None:
    got = np.asarray(estimate_and_apply_delay(small.values, small.mask))
    np.testing.assert_allclose(got[small.mask], small.values[small.mask])


def test_values_under_false_never_change_outputs(case: Case) -> None:
    dirty = np.where(case.mask, case.values, 1e6)
    a = estimate_delay(dirty, case.mask, n=case.n_hist)
    b = estimate_delay(case.values, case.mask, n=case.n_hist)
    np.testing.assert_array_equal(a, b)
    pa = apply_delay(dirty, case.mask, b)
    pb = apply_delay(case.values, case.mask, b)
    np.testing.assert_array_equal(pa, pb)


def test_jit_kernels_run_under_vmap_over_locations(small: Case) -> None:
    import jax

    stacked = np.stack([small.values, 2 * small.values])
    out = jax.vmap(
        lambda v: estimate_and_apply_delay(v, small.mask, validate=False)
    )(stacked)
    assert out.shape == (2, *small.values.shape)


def _toy() -> tuple[np.ndarray, np.ndarray]:
    return from_na(read("toy", "tri"))


def _no_complete_row() -> tuple[np.ndarray, np.ndarray, int | None]:
    v, m = _toy()
    m = m.copy()
    m[-1, :] = False
    return v, m, None


def _zeros_only() -> tuple[np.ndarray, np.ndarray, int | None]:
    v = np.zeros((6, 3))
    m = np.ones((6, 3), bool)
    m[-1, 1:] = False
    m[-2, 2:] = False
    v[:4] = 1.0
    return v, m, 2


def _empty_column() -> tuple[np.ndarray, np.ndarray, int | None]:
    m = np.ones((3, 3), bool)
    m[:, 2] = False
    return np.ones((3, 3)), m, None


@pytest.mark.parametrize(
    "inputs",
    [
        lambda: (*_toy(), 99),
        lambda: (*_toy(), 0),
        lambda: (*_toy(), 2),
        lambda: (*_toy(), 1.5),
        _no_complete_row,
        _zeros_only,
        _empty_column,
    ],
    ids=[
        "n-too-big",
        "n-zero",
        "n-too-small",
        "n-not-integer",
        "no-complete-row",
        "zeros-only",
        "empty-column",
    ],
)
def test_estimate_delay_aborts(inputs) -> None:  # type: ignore[no-untyped-def]
    v, m, n = inputs()
    with pytest.raises(ValueError):
        estimate_delay(v, m) if n is None else estimate_delay(v, m, n=n)


def test_apply_delay_aborts() -> None:
    v, m = _toy()
    pmf = np.asarray(estimate_delay(v, m, n=5))
    with pytest.raises(ValueError):
        apply_delay(v, m, pmf[1:])
    with pytest.raises(ValueError):
        apply_delay(v[:4], m[:4], pmf)
    with pytest.raises(ValueError):
        apply_delay(v, m, np.array([-0.1, 0.5, 0.4, 0.2]))
    short = m.copy()
    short[-1, 1:] = False
    with pytest.raises(ValueError):
        apply_delay(v, short, np.array([0.0, 0.5, 0.3, 0.2]))


def test_estimate_and_apply_delay_aborts() -> None:
    v, m = _toy()
    with pytest.raises(ValueError):
        estimate_and_apply_delay(v, m, n=99)
    with pytest.raises(ValueError):
        estimate_and_apply_delay(v, m, n=2)
