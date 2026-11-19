"""Coherent sample trajectories from quantile forecasts (trajectories)."""

import numpy as np
import pytest

from bdbvhubrenewal.trajectories import couple, read_quantiles

LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])


def _quantiles(series=3, steps=10):
    """Quantiles `[series, steps, levels]`: each cell its own scale."""
    base = np.arange(1, series * steps + 1, dtype=float).reshape(series, steps)
    spread = np.array([-2.0, -1.3, -0.7, 0.0, 0.7, 1.3, 2.0])
    return base[..., None] * 10 + spread


def test_quantile_reading_is_linear_between_levels_and_flat_beyond():
    q = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    u = np.array([0.5, 0.01, 0.99, (0.25 + 0.5) / 2])
    got = read_quantiles(LEVELS, np.repeat(q, 4, axis=0), u)
    assert got == pytest.approx([3.0, 0.0, 6.0, 2.5])


@pytest.mark.parametrize("method", ["independent", "comonotone", "gaussian"])
def test_each_cell_keeps_its_marginal(method):
    q = _quantiles()
    fit = {"rho": 0.9, "s": 0.3} if method == "gaussian" else None
    draws = couple(q, LEVELS, method, fit, n_samples=4000, seed=1)
    assert draws.shape == (4000, 3, 10)
    # Each cell's draws have the cell's own median and quartiles.
    for i, j in [(0, 0), (2, 9), (1, 4)]:
        got = np.quantile(draws[:, i, j], [0.25, 0.5, 0.75])
        assert got == pytest.approx(q[i, j, 2:5], abs=0.15)


def test_the_methods_differ_only_in_how_cells_move_together():
    q = _quantiles(series=2, steps=6)
    ind = couple(q, LEVELS, "independent", None, n_samples=3000, seed=2)
    com = couple(q, LEVELS, "comonotone", None, n_samples=3000, seed=2)
    gau = couple(q, LEVELS, "gaussian", {"rho": 0.9, "s": 0.3}, 3000, seed=2)

    def corr(d, a, b):
        return np.corrcoef(d[:, a[0], a[1]], d[:, b[0], b[1]])[0, 1]

    # Over steps within a series.
    assert abs(corr(ind, (0, 0), (0, 5))) < 0.1
    assert corr(com, (0, 0), (0, 5)) > 0.99
    assert 0.4 < corr(gau, (0, 0), (0, 1)) < 0.95
    # Across series: comonotone couples steps, not series.
    assert abs(corr(com, (0, 0), (1, 0))) < 0.1
    assert 0.1 < corr(gau, (0, 0), (1, 0)) < 0.5
    # Sums are wider when cells move together.
    assert ind.sum(axis=2).std() < gau.sum(axis=2).std()


def test_draws_are_reproducible_and_methods_are_checked():
    q = _quantiles()
    a = couple(q, LEVELS, "independent", None, n_samples=10, seed=3)
    b = couple(q, LEVELS, "independent", None, n_samples=10, seed=3)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="unknown"):
        couple(q, LEVELS, "nope", None, n_samples=10, seed=3)
    with pytest.raises(ValueError, match="rho"):
        couple(q, LEVELS, "gaussian", None, n_samples=10, seed=3)
