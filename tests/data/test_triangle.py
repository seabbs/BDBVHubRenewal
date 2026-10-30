from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.triangle import (
    apply_delay,
    chainladder_fill,
    delay_pmf,
    fit_nb_size,
    preprocess_negative_values,
    reporting_triangle,
    retrospective_errors,
    week_shares,
)


def _rows(vintage: str, week: str, delays: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vintage": vintage,
            "location": "CD-IT",
            "onset_week": week,
            "delay": delays,
        }
    )


def test_reporting_triangle_counts_and_masks_unseen_cells() -> None:
    rows = pd.concat(
        [
            _rows("2026-10-24", "2026-10-10", [0, 0, 5, 13]),
            _rows("2026-10-24", "2026-10-17", [1, 6]),
        ]
    )
    weeks, tri = reporting_triangle(rows, "2026-10-24", max_delay=20)
    assert weeks == ["2026-10-10", "2026-10-17"]
    assert tri.shape == (2, 21)
    # A case with onset week W and delay d is published on W + d.
    assert tri[0, 0] == 2
    assert tri[0, 5] == 1
    assert tri[0, 13] == 1
    assert np.isnan(tri[0, 15])
    assert not np.isnan(tri[0, 14])
    assert tri[1, 1] == 1
    assert tri[1, 6] == 1
    assert np.isnan(tri[1, 8])
    assert not np.isnan(tri[1, 7])


def _expected_triangle(
    totals: np.ndarray, pmf: np.ndarray, seen: np.ndarray
) -> np.ndarray:
    full = totals[:, None] * pmf[None, :]
    cols = np.arange(pmf.size)[None, :]
    return np.where(cols <= seen[:, None], full, np.nan)


def test_chainladder_recovers_a_multiplicative_triangle() -> None:
    pmf = np.array([0.5, 0.3, 0.15, 0.05])
    totals = np.array([100.0, 120.0, 90.0, 150.0, 80.0])
    seen = np.array([3, 3, 2, 1, 0])
    tri = _expected_triangle(totals, pmf, seen)
    filled = chainladder_fill(tri)
    assert np.allclose(filled, totals[:, None] * pmf[None, :])
    assert np.allclose(delay_pmf(tri), pmf)


def test_fit_nb_size_recovers_the_overdispersion() -> None:
    rng = np.random.default_rng(1)
    pred = rng.uniform(20, 200, size=(400, 2))
    size = 5.0
    obs = rng.negative_binomial(size, size / (size + pred))
    est = fit_nb_size(obs, pred)
    assert est.shape == (2,)
    assert np.allclose(est, size, rtol=0.25)
    # Poisson data give a large size.
    pois = rng.poisson(pred)
    assert (fit_nb_size(pois, pred) > 50).all()


def test_retrospective_errors_compare_past_nowcasts_with_later_data() -> None:
    rng = np.random.default_rng(3)
    pmf = np.array([0.4, 0.3, 0.2, 0.1])
    n_rows = 20
    totals = rng.poisson(200, n_rows).astype(float)
    full = np.array([rng.multinomial(int(t), pmf) for t in totals], float)
    cols = np.arange(pmf.size)[None, :]
    seen = np.minimum(n_rows - 1 - np.arange(n_rows), pmf.size - 1)
    tri = np.where(cols <= seen[:, None], full, np.nan)
    obs, pred = retrospective_errors(tri, n_retro=5, n_history=8, step=1)
    # One row per past nowcast, one column per horizon (rows still open).
    assert obs.shape == pred.shape == (5, 3)
    # Unbiased: what was later added is close to what was predicted.
    assert obs.sum() == pytest.approx(pred.sum(), rel=0.1)


def test_apply_delay_follows_baselinenowcast() -> None:
    # Missing cells of column d get (x + 1 - F(d - 1)) / F(d - 1) * p(d),
    # with x the row's sum over earlier columns, filled as it goes.
    pmf = np.array([0.5, 0.3, 0.2])
    tri = np.array(
        [[10.0, 6.0, 4.0], [8.0, 5.0, np.nan], [4.0, np.nan, np.nan]]
    )
    out = apply_delay(tri, pmf)
    assert out[1, 2] == pytest.approx((13 + 1 - 0.8) / 0.8 * 0.2)
    first = (4 + 1 - 0.5) / 0.5 * 0.3
    assert out[2, 1] == pytest.approx(first)
    assert out[2, 2] == pytest.approx((4 + first + 1 - 0.8) / 0.8 * 0.2)
    assert out[0].tolist() == [10.0, 6.0, 4.0]


def test_fit_nb_size_is_bounded_as_baselinenowcast() -> None:
    rng = np.random.default_rng(4)
    pred = rng.uniform(20, 200, size=(200, 1))
    assert fit_nb_size(rng.poisson(pred), pred)[0] == pytest.approx(
        1000, rel=0.01
    )


def test_preprocess_negative_values_moves_negatives_to_earlier_delays() -> None:
    # As baselinenowcast: from the longest delay back, a negative cell is
    # set to 0 and taken off the next earlier one.
    tri = np.array([[5.0, -2.0, 3.0, np.nan], [1.0, 2.0, -4.0, 1.0]])
    out = preprocess_negative_values(tri)
    assert out[0, :3].tolist() == [3.0, 0.0, 3.0]
    assert np.isnan(out[0, 3])
    assert out[1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_delay_pmf_needs_cases() -> None:
    tri = np.array([[0.0, 0.0], [0.0, np.nan]])
    with pytest.raises(ValueError, match="no cases"):
        delay_pmf(tri)


def test_week_shares_compare_the_file_with_the_point_nowcast() -> None:
    # Two complete rows set a pmf of 0.5 on each delay. The open row has 4
    # cases at delay 0, so its nowcast is 8; with 6 of its cases in the
    # file, 3/4 of it is published.
    tri = np.array([[5.0, 5.0], [3.0, 3.0], [4.0, np.nan]])
    pmf = np.array([0.5, 0.5])
    shares = week_shares(tri, pmf, np.array([10.0, 6.0, 6.0]))
    assert shares[:2] == pytest.approx([1.0, 1.0])
    assert shares[2] == pytest.approx(6 / (4 + (4 + 1 - 0.5) / 0.5 * 0.5))
    # No cases expected: all published.
    zero = week_shares(np.zeros((2, 2)), pmf, np.zeros(2))
    assert zero == pytest.approx([1.0, 1.0])
