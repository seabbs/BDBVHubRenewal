"""Triangle helpers against R baselinenowcast 0.2.2."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from bnc_helpers import Case, from_na, read, read_vec

from bdbvhubrenewal.baselinenowcast import (
    allocate_reference_times,
    apply_reporting_structure,
    get_reporting_structure,
    preprocess_negative_values,
    rolling_sum,
    truncate_to_rows,
)


def test_preprocess_matches_r(case: Case) -> None:
    got = preprocess_negative_values(case.values, case.mask)
    expected = np.nan_to_num(read(case.name, "preprocess"))
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_preprocess_ignores_values_under_false(case: Case) -> None:
    dirty = np.where(case.mask, case.values, 1e6)
    got = preprocess_negative_values(dirty, case.mask)
    expected = preprocess_negative_values(case.values, case.mask)
    np.testing.assert_array_equal(got, expected)


def test_truncate_to_rows_matches_r(case: Case) -> None:
    out = truncate_to_rows(case.values, case.mask, n=3)
    assert len(out) == 3
    for i, (v, m) in enumerate(out, start=1):
        expected, expected_m = from_na(read(case.name, f"trunc_{i}"))
        np.testing.assert_array_equal(m, expected_m)
        np.testing.assert_allclose(v, expected, atol=1e-8)


def test_truncate_to_rows_default_n(small: Case) -> None:
    n_complete = int(small.mask.all(axis=1).sum())
    out = truncate_to_rows(small.values, small.mask)
    assert len(out) == n_complete - 1


def test_reporting_structure_matches_r(case: Case) -> None:
    expected = read_vec(case.name, "structure")
    got = np.atleast_1d(get_reporting_structure(case.mask))
    np.testing.assert_allclose(got, expected)


def test_reporting_structure_zero_without_missing() -> None:
    assert get_reporting_structure(np.ones((4, 3), bool)) == 0


def test_apply_reporting_structure_matches_r(case: Case) -> None:
    trunc = truncate_to_rows(case.values, case.mask, n=3)
    structure = get_reporting_structure(case.mask)
    got = apply_reporting_structure(*trunc[0], structure=structure)
    np.testing.assert_array_equal(got, ~np.isnan(read(case.name, "applied_1")))
    got1 = apply_reporting_structure(*trunc[1], structure=1)
    np.testing.assert_array_equal(
        got1, ~np.isnan(read(case.name, "applied_struct1"))
    )


def test_allocate_reference_times_matches_r(case: Case) -> None:
    grid = read(case.name, "alloc")
    for scale, prop, n_min, n_hist, n_retro in grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.isnan(n_hist):
                with pytest.raises(ValueError):
                    allocate_reference_times(case.mask, scale, prop, int(n_min))
                continue
            got = allocate_reference_times(case.mask, scale, prop, int(n_min))
        assert got == (int(n_hist), int(n_retro)), (scale, prop, n_min)


def test_rolling_sum_matches_numpy() -> None:
    x = np.arange(20.0).reshape(10, 2)
    m = np.ones(x.shape, bool)
    m[8:, 1] = False
    v, ok = rolling_sum(x, m, 3)
    assert v.shape == (8, 2)
    np.testing.assert_allclose(
        v[:, 0], np.convolve(x[:, 0], np.ones(3), "valid")
    )
    assert ok[:, 0].all()
    assert not ok[-2:, 1].any()
    assert ok[:-2, 1].all()


@pytest.mark.parametrize("structure", [0, 9, [3, 3]])
def test_structure_aborts(structure: object) -> None:
    c = from_na(read("toy", "tri"))
    with pytest.raises(ValueError):
        apply_reporting_structure(*c, structure=structure)


@pytest.mark.parametrize("n", [-1, 99])
def test_truncate_aborts_on_n_out_of_range(n: int) -> None:
    v, m = from_na(read("toy", "tri"))
    with pytest.raises(ValueError):
        truncate_to_rows(v, m, n=n)


def test_allocation_aborts() -> None:
    _, m = from_na(read("toy", "tri"))
    with pytest.raises(ValueError):
        allocate_reference_times(m, scale_factor=-1)
    with pytest.raises(ValueError):
        allocate_reference_times(m, prop_delay=1.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            allocate_reference_times(m, scale_factor=0.5)


def test_invalid_na_pattern_aborts() -> None:
    v = np.ones((4, 3))
    m = np.ones((4, 3), bool)
    m[1, 1] = False
    with pytest.raises(ValueError):
        preprocess_negative_values(v, m)
    with pytest.raises(ValueError):
        preprocess_negative_values(v, np.zeros((4, 3), bool))
