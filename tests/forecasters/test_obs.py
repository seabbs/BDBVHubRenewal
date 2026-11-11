"""Observation model: negative binomial new reports on the count to come."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdbvhubrenewal.baselinenowcast import fit_nb
from bdbvhubrenewal.forecasters.obs import (
    PRIOR_SIZE,
    fit_sizes,
    pooled_sizes,
    sample_count,
)


def test_each_week_back_gets_the_ports_nb_fit() -> None:
    rng = np.random.default_rng(1)
    remaining = rng.uniform(5, 50, size=(40, 2))
    added = rng.poisson(remaining).astype(float)
    got = fit_sizes(added, remaining)
    want = [float(fit_nb(added[:, i], remaining[:, i])) for i in range(2)]
    assert got == pytest.approx(want)


def test_empty_cells_pad_without_changing_the_fit() -> None:
    rng = np.random.default_rng(2)
    remaining = rng.uniform(5, 50, size=(30, 1))
    added = rng.negative_binomial(3, 3 / (3 + remaining)).astype(float)
    padded = fit_sizes(
        np.vstack([added, np.zeros((10, 1))]),
        np.vstack([remaining, np.zeros((10, 1))]),
    )
    # Equal up to Brent's float32 tolerance.
    assert padded == pytest.approx(fit_sizes(added, remaining), rel=1e-2)


def test_overdispersed_reports_give_a_smaller_size() -> None:
    rng = np.random.default_rng(3)
    remaining = np.full((200, 1), 30.0)
    wide = rng.negative_binomial(2, 2 / 32, size=(200, 1)).astype(float)
    narrow = rng.poisson(30, size=(200, 1)).astype(float)
    assert fit_sizes(wide, remaining) < fit_sizes(narrow, remaining)


def test_no_errors_give_the_prior_median() -> None:
    added = np.array([[4.0, 0.0], [0.0, 0.0]])
    remaining = np.array([[3.0, 0.0], [0.0, 0.0]])
    size = fit_sizes(added, remaining)
    # One error is enough for a fit, and none gives the prior median.
    assert size[0] != pytest.approx(PRIOR_SIZE)
    np.testing.assert_allclose(size[1], PRIOR_SIZE, rtol=1e-6)
    assert pytest.approx(2.2, abs=0.01) == PRIOR_SIZE


def test_the_mean_is_reported_plus_remaining() -> None:
    n = 100_000
    x = sample_count(
        jax.random.key(0),
        jnp.full(n, 200.0),
        jnp.full(n, 50.0),
        jnp.full(n, 2.0),
    )
    assert float(x.mean()) == pytest.approx(250, rel=0.01)
    # New reports only add, as in baselinenowcast.
    assert float(x.min()) >= 200


def test_a_finished_week_is_its_count() -> None:
    x = sample_count(
        jax.random.key(0),
        jnp.full(100, 12.0),
        jnp.zeros(100),
        jnp.full(100, 2.0),
    )
    assert (np.asarray(x) == 12).all()


def _nb(rng, size, mu):
    return rng.negative_binomial(size, size / (size + mu)).astype(float)


def test_a_place_with_few_errors_is_pooled_towards_its_siblings() -> None:
    rng = np.random.default_rng(4)
    n = 200
    remaining = np.zeros((n, 3, 1))
    added = np.zeros((n, 3, 1))
    remaining[:, :2, 0] = rng.uniform(20, 60, size=(n, 2))
    added[:, :2, 0] = _nb(rng, 5.0, remaining[:, :2, 0])
    # Two errors, both far above their nowcast: a tiny size of its own.
    remaining[:2, 2, 0] = [30.0, 40.0]
    added[:2, 2, 0] = [120.0, 150.0]
    own = fit_sizes(added[:2, 2], remaining[:2, 2])[0]
    got = pooled_sizes(added, remaining)
    assert own < 1.5
    assert got.shape == (3, 1)
    np.testing.assert_allclose(got[:2, 0], 5.0, rtol=0.4)
    assert got[2, 0] > 2 * own


def test_places_that_differ_with_many_errors_keep_their_own_sizes() -> None:
    rng = np.random.default_rng(5)
    n = 300
    remaining = rng.uniform(20, 60, size=(n, 2, 1))
    added = np.stack(
        [_nb(rng, 1.0, remaining[:, 0]), _nb(rng, 30.0, remaining[:, 1])], 1
    )
    got = pooled_sizes(added, remaining)
    own = [fit_sizes(added[:, i], remaining[:, i])[0] for i in range(2)]
    np.testing.assert_allclose(got[:, 0], own, rtol=0.25)


def test_a_place_with_no_errors_takes_the_pooled_size() -> None:
    rng = np.random.default_rng(6)
    remaining = np.zeros((100, 2, 1))
    added = np.zeros((100, 2, 1))
    remaining[:, 0, 0] = rng.uniform(20, 60, size=100)
    added[:, 0, 0] = _nb(rng, 4.0, remaining[:, 0, 0])
    got = pooled_sizes(added, remaining)
    np.testing.assert_allclose(got[1, 0], got[0, 0], rtol=0.05)


def test_no_errors_anywhere_give_the_prior_size() -> None:
    got = pooled_sizes(np.zeros((5, 2, 3)), np.zeros((5, 2, 3)))
    np.testing.assert_allclose(got, PRIOR_SIZE)
