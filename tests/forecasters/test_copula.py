"""The Gaussian copula that pairs a model's marginal draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.copula import (
    GaussianCopula,
    estimate_copula,
    normal_scores,
)


def _draws(n: int = 2000, places: int = 4, steps: int = 6) -> np.ndarray:
    """Independent draws whose marginals differ by cell."""
    rng = np.random.default_rng(0)
    scale = np.arange(1, places * steps + 1).reshape(places, steps)
    return rng.gamma(2.0, scale, size=(n, places, steps))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def test_each_cell_keeps_its_draws() -> None:
    draws = _draws()
    copula = GaussianCopula(0.8, 0.5, 0.6)
    out = copula.couple(draws, np.random.default_rng(1))
    assert out.shape == draws.shape
    assert np.array_equal(np.sort(out, axis=0), np.sort(draws, axis=0))


def test_draws_follow_the_correlations() -> None:
    copula = GaussianCopula(0.8, 0.5, 0.3)
    out = copula.couple(_draws(), np.random.default_rng(2))
    # Spearman's rank correlation of a Gaussian copula with correlation r
    # is 6 / pi * arcsin(r / 2).
    rank = {r: 6 / np.pi * np.arcsin(r / 2) for r in (0.8, 0.64, 0.5, 0.3)}
    assert _rank_corr(out[:, 1, 0], out[:, 1, 1]) == pytest.approx(
        rank[0.8], abs=0.05
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 2]) == pytest.approx(
        rank[0.64], abs=0.05
    )
    assert _rank_corr(out[:, 1, 3], out[:, 2, 3]) == pytest.approx(
        rank[0.5], abs=0.05
    )
    assert _rank_corr(out[:, 0, 3], out[:, 3, 3]) == pytest.approx(
        rank[0.3], abs=0.05
    )
    # Across places and weeks the correlations multiply.
    assert _rank_corr(out[:, 1, 0], out[:, 2, 1]) == pytest.approx(
        6 / np.pi * np.arcsin(0.4 / 2), abs=0.05
    )


def test_no_correlation_leaves_the_draws_unpaired() -> None:
    out = GaussianCopula().couple(_draws(), np.random.default_rng(3))
    assert abs(_rank_corr(out[:, 1, 0], out[:, 1, 1])) < 0.06
    assert abs(_rank_corr(out[:, 0, 0], out[:, 1, 0])) < 0.06


def test_daily_steps_take_the_weekly_correlation_over_seven_days() -> None:
    copula = GaussianCopula(0.5, 0.0, 0.0)
    out = copula.couple(_draws(steps=15), np.random.default_rng(4), step_days=1)
    assert _rank_corr(out[:, 1, 0], out[:, 1, 7]) == pytest.approx(
        6 / np.pi * np.arcsin(0.5 / 2), abs=0.05
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 1]) == pytest.approx(
        6 / np.pi * np.arcsin(0.5 ** (1 / 7) / 2), abs=0.05
    )


def test_an_impossible_place_correlation_still_pairs_the_draws() -> None:
    # Strong ties to the country and none between 26 provinces is no
    # correlation matrix, so the nearest one is used.
    draws = _draws(n=500, places=27, steps=2)
    out = GaussianCopula(0.3, 0.0, 0.9).couple(draws, np.random.default_rng(5))
    assert np.array_equal(np.sort(out, axis=0), np.sort(draws, axis=0))
    assert _rank_corr(out[:, 0, 0], out[:, 1, 0]) > 0.05


@pytest.mark.parametrize(
    "rho", [(-0.1, 0, 0), (0, 1.0, 0), (0, 0, -0.5), (1.0, 0, 0)]
)
def test_correlations_must_be_in_zero_to_one(rho) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        GaussianCopula(*rho)


def _forecasts(values: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        pd.DataFrame(
            {
                "model": "m",
                "origin": "2026-10-24",
                "location": loc,
                "target_end_date": week,
                "sample": np.arange(1, len(v) + 1),
                "value": v,
            }
        )
        for (loc, week), v in values.items()
    )


def test_normal_scores_are_the_truths_place_in_the_forecast() -> None:
    forecasts = _forecasts(
        {
            ("CD", "2026-10-17"): np.arange(100.0),
            ("CD", "2026-10-24"): np.arange(100.0),
            ("CD-IT", "2026-10-17"): np.zeros(100),
        }
    )
    truth = pd.DataFrame(
        {
            "location": ["CD", "CD", "CD-IT"],
            "target_end_date": ["2026-10-17", "2026-10-24", "2026-10-17"],
            "truth": [49.5, 1000.0, 0.0],
        }
    )
    out = normal_scores(forecasts, truth, np.random.default_rng(0))
    z = out.set_index(["location", "target_end_date"])["z"]
    assert len(out) == 3
    assert z[("CD", "2026-10-17")] == pytest.approx(0.0, abs=0.03)
    assert z[("CD", "2026-10-24")] > 2.3
    # A truth tied with every draw takes a random place among them.
    assert np.isfinite(z[("CD-IT", "2026-10-17")])
    assert {"model", "origin", "location", "target_end_date"} <= set(out)


def test_estimate_recovers_the_correlations() -> None:
    truth = GaussianCopula(0.6, 0.4, 0.5)
    rng = np.random.default_rng(6)
    places = [NATIONAL, *(f"CD-{i:02d}" for i in range(5))]
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17, 24)]
    rows = []
    for origin in range(400):
        z = truth.normals(rng, 1, len(places), len(weeks))[0]
        for i, loc in enumerate(places):
            for h, week in enumerate(weeks):
                rows.append(("m", str(origin), loc, week, z[i, h]))
    scores = pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )
    fit = estimate_copula(scores)
    assert fit.rho_time == pytest.approx(0.6, abs=0.05)
    assert fit.rho_provinces == pytest.approx(0.4, abs=0.05)
    assert fit.rho_national == pytest.approx(0.5, abs=0.05)


def test_estimate_of_no_correlation_is_floored_at_zero() -> None:
    rng = np.random.default_rng(7)
    weeks = ["2026-10-17", "2026-10-24"]
    rows = [
        ("m", str(o), loc, w, rng.normal())
        for o in range(5)
        for loc in (NATIONAL, "CD-IT", "CD-NK")
        for w in weeks
    ]
    scores = pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )
    fit = estimate_copula(scores)
    for rho in (fit.rho_time, fit.rho_provinces, fit.rho_national):
        assert 0.0 <= rho < 1.0
