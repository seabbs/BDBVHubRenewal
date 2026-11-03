"""The two-part Skellam observation model on the remaining count."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bdbvhubrenewal.baselinenowcast import fit_by_horizon
from bdbvhubrenewal.forecasters.skellam import (
    NO_FIT,
    fit_skellam,
    n_errors,
    sample_skellam,
)


def _rows(add, rem, rhat, reported):
    obs = np.stack([add, rem], axis=-1).astype(float)
    pred = np.stack([rhat, reported], axis=-1).astype(float)
    return obs, pred


def test_each_part_is_the_poisson_rate_of_its_own_revisions() -> None:
    obs, pred = _rows([16, 8, 20], [1, 0, 2], [10, 4, 20], [100, 50, 150])
    a, b = np.asarray(fit_skellam(obs, pred)).tolist()
    assert a == pytest.approx(44 / 34)
    assert b == pytest.approx(3 / 300)


def test_rows_marked_missing_are_left_out() -> None:
    obs, pred = _rows([16, np.nan], [1, np.nan], [10, np.nan], [100, np.nan])
    a, b = np.asarray(fit_skellam(obs, pred)).tolist()
    assert a == pytest.approx(1.6)
    assert b == pytest.approx(0.01)
    assert int(n_errors(obs)) == 1


def test_no_errors_gives_the_poisson_nowcast() -> None:
    obs, pred = _rows([], [], [], [])
    assert np.asarray(fit_skellam(obs, pred)).tolist() == list(NO_FIT)
    # Nothing left to report gives no rate for the additions.
    obs, pred = _rows([3], [0], [0], [10])
    assert np.asarray(fit_skellam(obs, pred))[0] == NO_FIT[0]


def test_the_rates_are_recovered_from_simulated_revisions() -> None:
    rng = np.random.default_rng(1)
    rhat = rng.uniform(5, 200, 400)
    reported = rng.uniform(20, 500, 400)
    add = rng.poisson(1.4 * rhat)
    rem = rng.poisson(0.01 * reported)
    a, b = np.asarray(fit_skellam(*_rows(add, rem, rhat, reported))).tolist()
    assert a == pytest.approx(1.4, rel=0.02)
    assert b == pytest.approx(0.01, rel=0.15)


def test_it_plugs_into_the_ports_fit_by_horizon() -> None:
    # Rows are retrospective nowcasts, columns horizons, and the last axis
    # the additions and removals.
    add = np.array([[12.0, 30.0], [18.0, 10.0], [15.0, 40.0]])
    rem = np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    rhat = np.full((3, 2), 10.0)
    reported = np.full((3, 2), 100.0)
    obs, pred = _rows(add, rem, rhat, reported)
    got = np.asarray(fit_by_horizon(obs, pred, fit_skellam))
    assert got.shape == (2, 2)
    for k in range(2):
        want = np.asarray(fit_skellam(obs[:, k], pred[:, k]))
        assert got[k].tolist() == pytest.approx(want.tolist())


def test_the_draw_has_the_two_part_mean_and_variance() -> None:
    key = jax.random.key(1)
    n = 20000
    pred = np.stack([np.full(n, 50.0), np.full(n, 400.0)], -1)
    params = np.array([1.5, 0.01])
    x = np.asarray(sample_skellam(key, pred.astype(np.float32), params))
    assert x.shape == (n,)
    assert x.mean() == pytest.approx(1.5 * 50 - 4, abs=0.5)
    assert x.var() == pytest.approx(1.5 * 50 + 4, rel=0.1)


def test_the_downward_side_is_set_by_the_removals_alone() -> None:
    key = jax.random.key(2)
    n = 20000
    pred = np.stack([np.full(n, 10.0), np.full(n, 20.0)], -1)
    x = np.asarray(
        sample_skellam(key, pred.astype(np.float32), np.array([3.0, 0.01]))
    )
    # A large rate for the additions leaves the chance of a fall small.
    assert (x < 0).mean() < 0.001
    none = np.asarray(
        sample_skellam(key, pred.astype(np.float32), np.array([3.0, 0.0]))
    )
    assert (none >= 0).all()
    done = np.asarray(
        sample_skellam(
            key,
            np.zeros((100, 2), np.float32),
            np.array([3.0, 0.5]),
        )
    )
    assert (done == 0).all()
