"""The negative binomial observation model on the remaining count."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bdbvhubrenewal.forecasters.negbin import (
    inv_phi_from_sums,
    inv_phi_sums,
    sample_nowcast,
)


def test_inv_phi_is_the_moment_ratio_with_the_remainder_clamped() -> None:
    obs = np.array([10.0, 10.0, 30.0])
    pred = np.array([20.0, 30.0, 20.0])
    target = np.array([25.0, 20.0, 40.0])
    num, den, n = inv_phi_sums(obs, pred, target)
    # rem+ is 10, 20 and 0, and e is 5, -10 and 20.
    assert num == pytest.approx((25 - 10) + (100 - 20) + (400 - 0))
    assert den == pytest.approx(100 + 400 + 0)
    assert n == 3
    assert inv_phi_from_sums(num, den) == pytest.approx(495 / 500)


def test_inv_phi_is_zero_without_errors_and_never_negative() -> None:
    num, den, n = inv_phi_sums(np.zeros(0), np.zeros(0), np.zeros(0))
    assert n == 0
    assert inv_phi_from_sums(num, den) == 0.0
    # Errors at the Poisson variance leave no extra spread.
    obs, pred = np.array([10.0, 10.0]), np.array([20.0, 20.0])
    num, den, _ = inv_phi_sums(obs, pred, pred)
    assert inv_phi_from_sums(num, den) == 0.0


def test_valid_drops_errors() -> None:
    obs, pred, target = np.full(4, 5.0), np.full(4, 15.0), np.full(4, 30.0)
    keep = np.array([True, True, False, False])
    assert inv_phi_sums(obs, pred, target, keep)[2] == 2


def test_nb_has_the_nowcast_mean_and_the_dispersion() -> None:
    key = jax.random.key(1)
    mean, obs = np.float32(200.0), np.float32(150.0)
    plain = np.asarray(sample_nowcast(key, mean, obs, np.float32(0), 20000))
    wide = np.asarray(sample_nowcast(key, mean, obs, np.float32(0.5), 20000))
    assert plain.mean() == pytest.approx(200.0, abs=1.0)
    assert wide.mean() == pytest.approx(200.0, abs=2.0)
    assert plain.var() == pytest.approx(50.0, rel=0.1)
    assert wide.var() == pytest.approx(50.0 + 0.5 * 50.0**2, rel=0.1)


def test_counts_never_fall_below_the_reported_count() -> None:
    key = jax.random.key(2)
    down = np.asarray(
        sample_nowcast(
            key, np.float32(10.0), np.float32(40.0), np.float32(1.0), 2000
        )
    )
    assert (down == 40.0).all()
    zero = np.asarray(
        sample_nowcast(
            key, np.float32(0.0), np.float32(0.0), np.float32(2.0), 100
        )
    )
    assert (zero == 0).all()
