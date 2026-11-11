"""Observation model: new reports on the count to come, less removals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdbvhubrenewal.baselinenowcast import fit_nb, sample_nb
from bdbvhubrenewal.forecasters.obs import (
    PRIOR_SIZE,
    fit_sizes,
    removal_rates,
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
        jnp.zeros(n),
    )
    assert float(x.mean()) == pytest.approx(250, rel=0.01)
    # With no removals new reports only add, as in baselinenowcast.
    assert float(x.min()) >= 200


def test_a_finished_week_is_its_count() -> None:
    x = sample_count(
        jax.random.key(0),
        jnp.full(100, 12.0),
        jnp.zeros(100),
        jnp.full(100, 2.0),
        jnp.zeros(100),
    )
    assert (np.asarray(x) == 12).all()


def test_removals_thin_the_count_reported() -> None:
    n = 100_000
    x = sample_count(
        jax.random.key(0),
        jnp.full(n, 200.0),
        jnp.full(n, 50.0),
        jnp.full(n, 2.0),
        jnp.full(n, 0.1),
    )
    # 200 + 50 new reports less a tenth of the 200 reported.
    assert float(x.mean()) == pytest.approx(230, rel=0.01)
    assert float((x < 200).mean()) > 0
    assert float(x.min()) >= 0


def test_removals_leave_the_new_reports_as_they_were() -> None:
    key = jax.random.key(3)
    y, rem, size = (
        jnp.full(1000, 40.0),
        jnp.full(1000, 10.0),
        jnp.full(1000, 1.5),
    )
    x = sample_count(key, y, rem, size, jnp.zeros(1000))
    np.testing.assert_array_equal(
        np.asarray(x), np.asarray(y + sample_nb(key, rem, size))
    )


def test_the_removal_rate_is_pooled_over_each_depth() -> None:
    # The country, two provinces and two zones of the first province.
    parent = np.array([-1, 0, 0, 1, 1])
    reported = np.array(
        [[[100.0, 50.0], [60, 30], [40, 20], [50, 25], [10, 5]]]
    )
    fell = np.array([[[0.0, 0.0], [3, 0], [1, 2], [2, 1], [0, 0]]])
    valid = np.ones_like(reported, dtype=bool)
    valid[0, 4, 1] = False  # Not an error, so not counted.
    rate = removal_rates(fell, reported, valid, parent)
    np.testing.assert_allclose(rate[0], [0.0, 0.0])
    np.testing.assert_allclose(rate[1], [4 / 100, 2 / 50])
    np.testing.assert_allclose(rate[2], rate[1])
    np.testing.assert_allclose(rate[3], [2 / 60, 1 / 25])
    np.testing.assert_allclose(rate[4], rate[3])
    # A depth with no errors has no removals.
    valid[0, 3:] = False
    rate = removal_rates(fell, reported, valid, parent)
    np.testing.assert_allclose(rate[3:], 0.0)
    np.testing.assert_allclose(rate[1], [4 / 100, 2 / 50])
