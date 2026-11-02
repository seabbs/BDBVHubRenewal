"""The Skellam observation model on the remaining count."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bdbvhubrenewal.baselinenowcast import fit_by_horizon
from bdbvhubrenewal.forecasters.skellam import (
    fit_skellam,
    rho_from_sums,
    rho_sums,
    sample_skellam,
)


def test_rho_is_the_moment_ratio_on_the_remaining_count() -> None:
    pred = np.array([10.0, -4.0, 20.0])
    obs = np.array([16.0, -8.0, 20.0])
    num, den, n = rho_sums(obs, pred)
    assert num == pytest.approx((36 - 10) + (16 - 4) + (0 - 20))
    assert den == pytest.approx(2 * (10 + 4 + 20))
    assert n == 3
    assert rho_from_sums(num, den) == pytest.approx(18 / 68)


def test_rho_is_zero_without_errors_and_never_negative() -> None:
    assert fit_skellam(np.zeros(0), np.zeros(0)) == 0.0
    # Errors at the Poisson variance leave no extra spread.
    assert fit_skellam(np.array([20.0, 20.0]), np.array([16.0, 16.0])) == 0.0
    # A finished week, with nothing left, adds no spread.
    assert fit_skellam(np.array([3.0]), np.array([0.0])) == 0.0


def test_rho_moves_with_the_error_size() -> None:
    pred = np.full(50, 10.0)
    small = fit_skellam(pred + 5, pred)
    large = fit_skellam(pred + 15, pred)
    assert 0 < small < large


def test_it_plugs_into_the_ports_fit_by_horizon() -> None:
    obs = np.array([[12.0, 30.0], [18.0, 10.0], [15.0, 40.0]])
    pred = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
    got = np.asarray(fit_by_horizon(obs, pred, fit_skellam))
    want = [float(fit_skellam(obs[:, i], pred[:, i])) for i in range(2)]
    assert got.tolist() == pytest.approx(want)


def test_the_draw_has_the_remaining_mean_and_spread_proportional_to_it() -> (
    None
):
    key = jax.random.key(1)
    rem = np.full(20000, 50.0, dtype=np.float32)
    plain = np.asarray(sample_skellam(key, rem, np.float32(0)))
    wide = np.asarray(sample_skellam(key, rem, np.float32(1.5)))
    assert plain.mean() == pytest.approx(50.0, abs=0.5)
    assert wide.mean() == pytest.approx(50.0, abs=1.0)
    assert plain.var() == pytest.approx(50.0, rel=0.1)
    assert wide.var() == pytest.approx(50.0 * (1 + 2 * 1.5), rel=0.1)


def test_a_revision_can_go_down_and_a_finished_week_has_no_spread() -> None:
    key = jax.random.key(2)
    down = np.asarray(
        sample_skellam(key, np.full(2000, -30.0, np.float32), np.float32(0.5))
    )
    assert down.mean() == pytest.approx(-30.0, abs=1.0)
    assert (down < 0).all() or down.min() < 0
    done = np.asarray(
        sample_skellam(key, np.zeros(100, np.float32), np.float32(2.0))
    )
    assert (done == 0).all()
