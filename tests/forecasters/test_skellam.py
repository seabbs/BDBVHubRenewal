"""The Skellam observation model with a spread fitted by moments."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.forecasters.skellam import (
    fit_rho_moments,
    sample_skellam_moments,
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
