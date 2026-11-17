"""The Gaussian copula that pairs a model's marginal draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.copula import (
    PRIOR,
    GaussianCopula,
    error_scores,
    estimate_copula,
    pair_walks,
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
    out = copula.couple(
        _draws(steps=15), np.random.default_rng(4), days=np.arange(15)
    )
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


def test_estimate_recovers_the_correlations() -> None:
    truth = GaussianCopula(0.6, 0.4, 0.5)
    rng = np.random.default_rng(6)
    places = [NATIONAL, *(f"CD-{i:02d}" for i in range(5))]
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17, 24)]
    rows = []
    for origin in range(400):
        z = truth.normals(rng, 1, len(places), 7 * np.arange(len(weeks)))[0]
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


def test_estimate_of_negative_correlation_is_floored_at_zero() -> None:
    # Every pair of scores, across weeks and places, is of opposite sign.
    rows = [
        ("m", str(o), loc, week, z)
        for o in range(5)
        for loc, sign in ((NATIONAL, 1), ("CD-IT", -1))
        for week, z in (
            ("2026-10-17", sign * (o + 1.0)),
            ("2026-10-24", -sign * (o + 1.0)),
        )
    ]
    scores = pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )
    fit = estimate_copula(scores)
    assert fit.rho_time == 0.0
    assert fit.rho_national == 0.0


def test_steps_are_correlated_by_their_days_apart() -> None:
    out = GaussianCopula(0.8, 0.0, 0.0).couple(
        _draws(steps=3), np.random.default_rng(8), days=np.array([0, 7, 21])
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 2]) == pytest.approx(
        6 / np.pi * np.arcsin(0.8**3 / 2), abs=0.05
    )


def test_with_no_country_every_place_is_a_province() -> None:
    out = GaussianCopula(0.0, 0.5, 0.0).couple(
        _draws(), np.random.default_rng(9), national=False
    )
    assert _rank_corr(out[:, 0, 2], out[:, 1, 2]) == pytest.approx(
        6 / np.pi * np.arcsin(0.5 / 2), abs=0.05
    )


def _structured_scores(n_origins: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    places = [NATIONAL, *(f"CD-{i:02d}" for i in range(5))]
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17, 24)]
    copula = GaussianCopula(0.6, 0.4, 0.5)
    rows = []
    for origin in range(n_origins):
        z = copula.normals(rng, 1, len(places), 7 * np.arange(len(weeks)))
        for i, loc in enumerate(places):
            for h, week in enumerate(weeks):
                rows.append(("m", str(origin), loc, week, z[0, i, h]))
    return pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )


def test_estimate_shrinks_towards_the_prior() -> None:
    scores = _structured_scores()
    prior = GaussianCopula(0.1, 0.1, 0.1)
    raw = estimate_copula(scores)
    assert estimate_copula(scores, prior, weight=0.0) == raw
    # Each correlation counts its own pairs: of 6 places by 4 weeks at
    # 400 origins, 7,200 pairs of weeks and 16,000 of provinces.
    time = estimate_copula(scores, prior, weight=7200.0)
    assert time.rho_time == pytest.approx((raw.rho_time + 0.1) / 2)
    places = estimate_copula(scores, prior, weight=16000.0)
    assert places.rho_provinces == pytest.approx((raw.rho_provinces + 0.1) / 2)
    far = estimate_copula(scores, prior, weight=1e12)
    assert far.rho_time == pytest.approx(0.1)


def test_a_correlation_with_no_variation_takes_the_prior() -> None:
    scores = _structured_scores(n_origins=5).assign(z=0.5)
    fit = estimate_copula(scores, GaussianCopula(0.3, 0.2, 0.1))
    assert fit == GaussianCopula(0.3, 0.2, 0.1)


ZONED = GaussianCopula(0.8, 0.3, 0.6, rho_zones=0.7)


def test_zone_correlation_is_estimated_from_zone_pairs() -> None:
    rng = np.random.default_rng(23)
    places = [NATIONAL, "CD-A", "CD-B", "CD-A-01", "CD-A-02", "CD-B-01"]
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17)]
    copula = GaussianCopula(0.6, 0.2, 0.4, rho_zones=0.75)
    rows = []
    for origin in range(1500):
        z = copula.normals(
            rng, 1, len(places), 7 * np.arange(len(weeks)), locations=places
        )
        for i, loc in enumerate(places):
            for h, week in enumerate(weeks):
                rows.append(("m", str(origin), loc, week, z[0, i, h]))
    scores = pd.DataFrame(
        rows, columns=["model", "origin", "location", "target_end_date", "z"]
    )
    got = estimate_copula(scores)
    assert got.rho_zones == pytest.approx(0.75, abs=0.04)
    # Only the two provinces give rho_p, not zones of different provinces.
    assert got.rho_provinces == pytest.approx(0.2, abs=0.06)
    assert got.rho_time == pytest.approx(0.6, abs=0.04)


def test_the_factor_is_the_cholesky_factor_of_the_nearest_correlation() -> None:
    from statsmodels.stats.correlation_tools import corr_nearest

    from bdbvhubrenewal.forecasters.copula import _factor

    valid = GaussianCopula(0.8, 0.5, 0.6).places(5)
    a = _factor(valid)
    # A valid matrix is reproduced, by its lower-triangular Cholesky factor.
    np.testing.assert_allclose(a @ a.T, valid, atol=1e-10)
    assert np.allclose(a, np.tril(a))
    # An invalid one (rho_n far above rho_p) is replaced by the nearest
    # correlation matrix (Higham), as statsmodels gives it.
    bad = GaussianCopula(0.0, 0.1, 0.9).places(5)
    assert np.linalg.eigvalsh(bad).min() < 0
    b = _factor(bad)
    np.testing.assert_allclose(b @ b.T, corr_nearest(bad), atol=1e-6)
    np.testing.assert_allclose(np.diag(b @ b.T), 1.0, atol=1e-8)


def test_seeded_pairing_statistics_are_unchanged() -> None:
    # The draws a seed gives depend on the factor, but the pairing does
    # not: the same seeded run recovers the copula's rank correlations, as
    # before the factor changed (test_draws_follow_the_correlations).
    out = GaussianCopula(0.8, 0.5, 0.3).couple(
        _draws(n=4000), np.random.default_rng(2)
    )
    rank = {r: 6 / np.pi * np.arcsin(r / 2) for r in (0.8, 0.5, 0.3)}
    assert _rank_corr(out[:, 1, 0], out[:, 1, 1]) == pytest.approx(
        rank[0.8], abs=0.03
    )
    assert _rank_corr(out[:, 1, 3], out[:, 2, 3]) == pytest.approx(
        rank[0.5], abs=0.03
    )
    assert _rank_corr(out[:, 0, 3], out[:, 3, 3]) == pytest.approx(
        rank[0.3], abs=0.03
    )
    again = GaussianCopula(0.8, 0.5, 0.3).couple(
        _draws(n=4000), np.random.default_rng(2)
    )
    np.testing.assert_array_equal(out, again)


def test_walks_keep_their_cells_and_move_as_a_whole() -> None:
    rng = np.random.default_rng(40)
    n, h = 3000, 6
    start = rng.normal(size=(n, 2, 1))
    walk = np.cumsum(
        np.concatenate([start, rng.normal(0, 0.3, (n, 2, h - 1))], axis=2),
        axis=2,
    )
    out = pair_walks(
        walk,
        ("CD-A", "CD-B"),
        np.array([2, 2]),
        PRIOR,
        np.random.default_rng(1),
    )
    assert np.array_equal(np.sort(out, axis=0), np.sort(walk, axis=0))
    # From the start week on, each place's trajectories are kept whole.
    for loc in range(2):
        a, b = walk[:, loc, 2:], out[:, loc, 2:]
        np.testing.assert_array_equal(
            a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])]
        )
    # Before it, cells are paired across places by rho_p.
    assert _rank_corr(out[:, 0, 0], out[:, 1, 0]) == pytest.approx(
        6 / np.pi * np.arcsin(0.7 / 2), abs=0.05
    )
    # The walks are paired across places through their totals.
    assert _rank_corr(out[:, 0, -1], out[:, 1, -1]) > 0.3


def test_zones_are_paired_within_and_across_provinces() -> None:
    zones = ("CD-A-01", "CD-A-02", "CD-B-01")
    draws = _draws(n=3000, places=3, steps=2)
    out = pair_walks(
        draws, zones, np.array([9, 9, 9]), ZONED, np.random.default_rng(41)
    )
    rank = {r: 6 / np.pi * np.arcsin(r / 2) for r in (0.7, 0.3)}
    assert _rank_corr(out[:, 0, 0], out[:, 1, 0]) == pytest.approx(
        rank[0.7], abs=0.05
    )
    assert _rank_corr(out[:, 0, 0], out[:, 2, 0]) == pytest.approx(
        rank[0.3], abs=0.05
    )


def test_error_scores_are_pit_normal_scores_of_recent_nowcasts() -> None:
    mu = np.full((3, 2, 2), 10.0)
    added = np.array([[[10, 3], [10, 30]]] * 3, float)
    valid = np.ones_like(mu, bool)
    valid[0, 0, 0] = False
    size = np.full((2, 2), 5.0)
    out = error_scores(
        added, mu, valid, size, ["r1", "r2", "r3"],
        np.array([7, 14, 70]), ("CD-A", "CD-B"), "m", seed=1,
    )  # fmt: skip
    # The release 70 days back is outside the 8-week window, and the
    # invalid cell gives no score.
    assert set(out["origin"]) == {"r1", "r2"}
    assert len(out) == 2 * 4 - 1
    # A count far above the nowcast gives a high score, far below a low one.
    r2 = out.loc[out["origin"] == "r2"]
    assert r2.loc[r2["location"] == "CD-B", "z"].max() > 2.0
    assert r2.loc[r2["location"] == "CD-A", "z"].min() < -0.5
    again = error_scores(
        added, mu, valid, size, ["r1", "r2", "r3"],
        np.array([7, 14, 70]), ("CD-A", "CD-B"), "m", seed=1,
    )  # fmt: skip
    pd.testing.assert_frame_equal(out, again)
