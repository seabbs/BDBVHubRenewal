"""The shared random walk."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bdbvhubrenewal.forecasters.walk import random_walk


def test_steps_are_cumulative_within_a_sample() -> None:
    key = jax.random.key(0)
    lam, y = random_walk(key, np.full(4000, 99.0, np.float32), 0.3, 3)
    lam = np.asarray(lam)
    assert lam.shape == np.asarray(y).shape == (4000, 3)
    level = np.log1p(lam)
    steps = np.diff(
        np.column_stack([np.full(4000, np.log(100.0)), level]), axis=1
    )
    assert steps.std(axis=0) == pytest.approx([0.3] * 3, rel=0.1)
    # Sample variance grows with the horizon, as steps accumulate.
    sd = level.std(axis=0)
    assert sd[0] < sd[1] < sd[2]
    assert sd[2] == pytest.approx(0.3 * np.sqrt(3), rel=0.1)
    assert abs(np.corrcoef(steps[:, 0], steps[:, 1])[0, 1]) < 0.1


def test_the_level_is_floored_at_zero_on_the_log_scale() -> None:
    key = jax.random.key(1)
    lam, y = random_walk(key, np.zeros(500, np.float32), 1.0, 4)
    assert (np.asarray(lam) >= 0).all()
    assert (np.asarray(y) >= 0).all()
    # Starting at 0 the walk is floored at 0 about half the time.
    assert 0.35 < (np.asarray(lam)[:, 0] == 0).mean() < 0.65


def test_no_spread_gives_a_flat_rate() -> None:
    lam, _ = random_walk(
        jax.random.key(2), np.full(10, 7.0, np.float32), 0.0, 3
    )
    assert np.allclose(np.asarray(lam), 7.0)


def test_one_key_gives_one_draw() -> None:
    a = random_walk(jax.random.key(3), np.full(5, 3.0, np.float32), 0.2, 2)
    b = random_walk(jax.random.key(3), np.full(5, 3.0, np.float32), 0.2, 2)
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
