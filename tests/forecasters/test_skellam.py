"""The Skellam observation model with a spread fitted by moments."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.forecasters.skellam import (
    fit_rho_moments,
    sample_nowcast,
    sample_skellam_moments,
    spread_sums,
)

DATA = Path(__file__).parent / "data"
# national rho for k = 0..3 at origin 10-24, from the R script's errors
R_RHO = [0.157600172794278, 1.08837151684058, 0.309268449825425, 1.525723753449]


def test_rho_reproduces_the_r_script_on_its_errors() -> None:
    err = pd.read_csv(DATA / "retro_errors_1024.csv")
    for k, want in enumerate(R_RHO):
        rows = err.loc[err["k"] == k]
        got = fit_rho_moments(
            rows["obs"].to_numpy(),
            rows["pred"].to_numpy(),
            rows["target"].to_numpy(),
        )
        assert got == pytest.approx(want, rel=1e-9)


def test_rho_is_zero_without_errors_and_never_negative() -> None:
    assert fit_rho_moments(np.zeros(0), np.zeros(0), np.zeros(0)) == 0.0
    # No spread beyond Poisson: errors equal to the Poisson variance.
    obs, pred = np.array([10.0, 10.0]), np.array([20.0, 20.0])
    target = np.array([20.0, 20.0])
    assert fit_rho_moments(obs, pred, target) == 0.0


def test_rho_moves_with_the_error_size() -> None:
    obs, pred = np.full(50, 10.0), np.full(50, 20.0)
    small = fit_rho_moments(obs, pred, pred + 5)
    large = fit_rho_moments(obs, pred, pred + 15)
    assert 0 < small < large


def test_skellam_has_the_nowcast_mean_and_the_extra_spread() -> None:
    key = jax.random.key(1)
    mean, obs = np.float32(200.0), np.float32(150.0)
    plain = np.asarray(
        sample_skellam_moments(key, mean, obs, np.float32(0), 20000)
    )
    wide = np.asarray(
        sample_skellam_moments(key, mean, obs, np.float32(1.5), 20000)
    )
    assert plain.mean() == pytest.approx(200.0, abs=1.0)
    assert wide.mean() == pytest.approx(200.0, abs=2.0)
    assert plain.var() == pytest.approx(50.0, rel=0.1)
    assert wide.var() == pytest.approx(50.0 + 2 * 1.5 * 200.0, rel=0.1)


def test_a_revision_can_go_down_and_counts_stay_non_negative() -> None:
    key = jax.random.key(2)
    down = np.asarray(
        sample_skellam_moments(
            key, np.float32(10.0), np.float32(40.0), np.float32(0.0), 2000
        )
    )
    assert down.mean() == pytest.approx(10.0, abs=1.0)
    assert down.min() >= 0
    zero = np.asarray(
        sample_skellam_moments(
            key, np.float32(0.0), np.float32(0.0), np.float32(2.0), 100
        )
    )
    assert (zero == 0).all()


def test_spread_sums_scale_with_the_quantity_each_variant_names() -> None:
    obs, pred = np.array([10.0, 30.0]), np.array([20.0, 40.0])
    target = np.array([25.0, 32.0])
    err2 = np.array([25.0, 64.0])
    rem = np.array([10.0, 10.0])
    num, den, n = spread_sums(obs, pred, target, model="skellam")
    assert num == pytest.approx((err2 - rem).sum())
    assert den == pytest.approx((2 * pred).sum())
    num, den, _ = spread_sums(obs, pred, target, model="skellam_rem")
    assert num == pytest.approx((err2 - rem).sum())
    assert den == pytest.approx((2 * rem).sum())
    num, den, _ = spread_sums(obs, pred, target, model="nb")
    assert num == pytest.approx((err2 - rem).sum())
    assert den == pytest.approx((rem**2).sum())
    assert n == 2
    with pytest.raises(ValueError, match="observation model"):
        spread_sums(obs, pred, target, model="poisson")


@pytest.mark.parametrize(
    ("model", "spread", "var"),
    [
        ("skellam", 1.5, 50.0 + 2 * 1.5 * 200.0),
        ("skellam_rem", 1.5, 50.0 + 2 * 1.5 * 50.0),
        ("nb", 0.1, 50.0 + 50.0**2 * 0.1),
    ],
)
def test_each_observation_model_has_the_nowcast_mean_and_its_variance(
    model: str, spread: float, var: float
) -> None:
    key = jax.random.key(3)
    draws = np.asarray(
        sample_nowcast(
            key,
            np.float32(200.0),
            np.float32(150.0),
            np.float32(spread),
            40000,
            model,
        )
    )
    assert draws.mean() == pytest.approx(200.0, abs=2.0)
    assert draws.var() == pytest.approx(var, rel=0.1)


def test_a_complete_week_gets_no_spread_under_the_remainder_models() -> None:
    key = jax.random.key(4)
    for model in ("skellam_rem", "nb"):
        draws = np.asarray(
            sample_nowcast(
                key,
                np.float32(80.0),
                np.float32(80.0),
                np.float32(2.0),
                500,
                model,
            )
        )
        assert (draws == 80).all()
