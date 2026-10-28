from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.triangle import (
    chainladder_fill,
    delay_pmf,
    fit_nb_size,
    reporting_triangle,
    retrospective_errors,
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
    obs, pred = retrospective_errors(tri, n_retro=5, n_history=8)
    # One row per past nowcast, one column per horizon (rows still open).
    assert obs.shape == pred.shape == (5, 3)
    # Unbiased: what was later added is close to what was predicted.
    assert obs.sum() == pytest.approx(pred.sum(), rel=0.1)
