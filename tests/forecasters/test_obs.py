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
