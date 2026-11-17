"""The Gaussian copula that pairs the walk models' province draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.copula import (
    MAX_RHO,
    PRIOR,
    error_scores,
    estimate_rho,
    pair_walks,
)

PROVINCES = ("CD-A", "CD-B", "CD-C", "CD-D")


def _draws(n: int = 3000, places: int = 4, steps: int = 6) -> np.ndarray:
    """Independent draws whose marginals differ by cell."""
    rng = np.random.default_rng(0)
    scale = np.arange(1, places * steps + 1).reshape(places, steps)
    return rng.gamma(2.0, scale, size=(n, places, steps))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def _spearman(rho: float) -> float:
    """Spearman's rank correlation of a Gaussian copula with `rho`."""
    return 6 / np.pi * np.arcsin(rho / 2)


def _pair(draws: np.ndarray, rho: float, seed: int = 1, **kw) -> np.ndarray:
    places = kw.pop("locations", PROVINCES[: draws.shape[1]])
    anchors = kw.pop("anchors", np.full(len(places), draws.shape[2]))
    return pair_walks(
        draws, places, anchors, rho, np.random.default_rng(seed), **kw
    )


def test_each_cell_keeps_its_draws() -> None:
    draws = _draws()
    out = _pair(draws, 0.6)
    assert out.shape == draws.shape
    assert np.array_equal(np.sort(out, axis=0), np.sort(draws, axis=0))


def test_provinces_are_paired_within_a_week_by_rho_p() -> None:
    out = _pair(_draws(), 0.6)
    for a, b in ((0, 1), (1, 3)):
        assert _rank_corr(out[:, a, 2], out[:, b, 2]) == pytest.approx(
            _spearman(0.6), abs=0.05
        )


def test_weeks_are_not_paired() -> None:
    out = _pair(_draws(), 0.6)
    # Neither a province's own weeks nor two provinces in different weeks.
    assert abs(_rank_corr(out[:, 1, 0], out[:, 1, 1])) < 0.06
    assert abs(_rank_corr(out[:, 0, 0], out[:, 1, 1])) < 0.06


def test_no_correlation_leaves_the_draws_unpaired() -> None:
    out = _pair(_draws(), 0.0)
    assert abs(_rank_corr(out[:, 0, 0], out[:, 1, 0])) < 0.06


def test_zones_and_the_country_are_not_paired() -> None:
    places = (NATIONAL, "CD-A", "CD-B", "CD-A-01", "CD-B-01")
    draws = _draws(places=5)
    out = _pair(draws, 0.9, locations=places)
    # The country and the zones keep their draws in their own order.
    for loc in (0, 3, 4):
        np.testing.assert_array_equal(out[:, loc], draws[:, loc])
    assert _rank_corr(out[:, 1, 0], out[:, 2, 0]) == pytest.approx(
        _spearman(0.9), abs=0.05
    )
    # A task of zones only is left as it is.
    zones = ("CD-A-01", "CD-A-02", "CD-B-01")
    only = _draws(places=3)
    np.testing.assert_array_equal(_pair(only, 0.9, locations=zones), only)


@pytest.mark.parametrize("rho", [-0.1, 1.0])
def test_rho_must_be_in_zero_to_one(rho: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _pair(_draws(n=10), rho)


def test_walks_keep_their_cells_and_move_as_a_whole() -> None:
    rng = np.random.default_rng(40)
    n, h = 3000, 6
    start = rng.normal(size=(n, 2, 1))
    walk = np.cumsum(
        np.concatenate([start, rng.normal(0, 0.3, (n, 2, h - 1))], axis=2),
        axis=2,
    )
    # The two walks start in different weeks.
    anchors = np.array([2, 3])
    out = _pair(walk, PRIOR, locations=("CD-A", "CD-B"), anchors=anchors)
    assert np.array_equal(np.sort(out, axis=0), np.sort(walk, axis=0))
    # From its start week on, each place's trajectories are kept whole.
    for loc, a in enumerate(anchors):
        x, y = walk[:, loc, a:], out[:, loc, a:]
        np.testing.assert_array_equal(
            x[np.lexsort(x.T[::-1])], y[np.lexsort(y.T[::-1])]
        )
    # Before either walk, cells are paired within the week by rho_p.
    assert _rank_corr(out[:, 0, 0], out[:, 1, 0]) == pytest.approx(
        _spearman(PRIOR), abs=0.05
    )
    # The walks are paired with each other, though they start apart.
    assert _rank_corr(out[:, 0, -1], out[:, 1, -1]) > 0.3


def _scores(rho: float, n_origins: int = 400, seed: int = 6) -> pd.DataFrame:
    """Normal scores of provinces paired by `rho` within each week.

    The country and the zones score as their province does, so they would
    pull the estimate up if they were used.
    """
    rng = np.random.default_rng(seed)
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17, 24)]
    rows = []
    for origin in range(n_origins):
        for week in weeks:
            common = rng.standard_normal()
            own = rng.standard_normal(len(PROVINCES))
            z = np.sqrt(rho) * common + np.sqrt(1 - rho) * own
            for loc, value in zip(PROVINCES, z, strict=True):
                rows += [
                    ("m", str(origin), loc, week, value),
                    ("m", str(origin), f"{loc}-01", week, value),
                ]
            rows.append(("m", str(origin), NATIONAL, week, z[0]))
    return pd.DataFrame(
        rows, columns=["model", "origin", "location", "target_end_date", "z"]
    )


def test_estimate_recovers_rho_p_from_provinces_only() -> None:
    fit = estimate_rho(_scores(0.4), weight=0.0)
    assert fit["rho_provinces"] == pytest.approx(0.4, abs=0.05)
    # 6 pairs of 4 provinces in each of 4 weeks at 400 origins.
    assert fit["n_pairs"] == 6 * 4 * 400
    assert not fit["fallback"]


def test_estimate_shrinks_towards_the_prior() -> None:
    scores = _scores(0.4)
    raw = estimate_rho(scores, weight=0.0)["rho_provinces"]
    half = estimate_rho(scores, prior=0.1, weight=9600.0)["rho_provinces"]
    assert half == pytest.approx((raw + 0.1) / 2)
    far = estimate_rho(scores, prior=0.1, weight=1e12)["rho_provinces"]
    assert far == pytest.approx(0.1)


def test_estimate_is_floored_at_zero_and_capped() -> None:
    rows = [
        ("m", str(o), loc, "2026-10-17", sign * (o + 1.0))
        for o in range(5)
        for loc, sign in (("CD-A", 1), ("CD-B", -1))
    ]
    scores = pd.DataFrame(
        rows, columns=["model", "origin", "location", "target_end_date", "z"]
    )
    assert estimate_rho(scores, weight=0.0)["rho_provinces"] == 0.0
    same = scores.assign(z=lambda d: d["origin"].astype(float))
    assert estimate_rho(same, weight=0.0)["rho_provinces"] == MAX_RHO


def test_with_no_pairs_or_no_variation_rho_p_is_the_prior() -> None:
    flat = estimate_rho(_scores(0.4, n_origins=5).assign(z=0.5))
    assert flat == {"rho_provinces": PRIOR, "n_pairs": 0, "fallback": True}
    empty = estimate_rho(_scores(0.4).iloc[:0])
    assert empty == {"rho_provinces": PRIOR, "n_pairs": 0, "fallback": True}


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
