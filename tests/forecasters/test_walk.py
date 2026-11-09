"""The shared random walk."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bdbvhubrenewal.forecasters.walk import (
    log1p_poisson_var,
    random_walk,
    walk_noise,
    walk_spread,
)


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


def test_log1p_poisson_var_matches_simulation() -> None:
    rng = np.random.default_rng(0)
    for lam in (0.0, 1.0, 4.0, 12.0, 300.0):
        y = rng.poisson(lam, 200_000)
        assert log1p_poisson_var(np.array([lam]))[0] == pytest.approx(
            np.log1p(y).var(), abs=2e-3
        )


def test_poisson_noise_alone_gives_the_floor() -> None:
    rng = np.random.default_rng(1)
    for lam in (8.0, 20.0):
        y = rng.poisson(lam, 5000).astype(float)
        d = np.diff(np.log1p(y))
        # Noise alone looks like a walk with a spread of about 0.5 to 0.3.
        assert d.std() > 0.3
        assert walk_spread(d, walk_noise(y), 0.1) == 0.1


def test_a_random_walk_on_large_counts_is_recovered() -> None:
    rng = np.random.default_rng(2)
    level = np.log(500.0) + np.cumsum(rng.normal(0, 0.3, (300, 9)), axis=1)
    y = rng.poisson(np.expm1(level)).astype(float)
    d = np.concatenate([np.diff(np.log1p(row)) for row in y])
    noise = np.concatenate([walk_noise(row) for row in y])
    assert walk_spread(d, noise, 0.1) == pytest.approx(0.3, rel=0.05)


def test_a_walk_on_small_counts_loses_only_the_poisson_noise() -> None:
    rng = np.random.default_rng(3)
    level = np.log(10.0) + np.cumsum(rng.normal(0, 0.3, (2000, 9)), axis=1)
    y = rng.poisson(np.expm1(np.clip(level, 0, None))).astype(float)
    d = np.concatenate([np.diff(np.log1p(row)) for row in y])
    noise = np.concatenate([walk_noise(row) for row in y])
    assert d.std() > 0.45
    assert walk_spread(d, noise, 0.1) == pytest.approx(0.3, abs=0.06)


def test_fewer_than_two_changes_give_the_floor() -> None:
    assert walk_spread(np.array([0.7]), np.array([0.1]), 0.1) == 0.1
    assert walk_spread(np.array([]), np.array([]), 0.05) == 0.05
