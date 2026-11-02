"""The snapshot extension (E1): pair sums, factors and chains."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bdbvhubrenewal.baselinenowcast import (
    ChainPlan,
    chain_growth,
    pair_factors,
    pair_sums,
)

# One place, three releases, eight onset days.
# Release cut-offs are days 4, 6 and 7.
COUNTS = np.array(
    [
        [
            [5.0, 3, 4, 2, 1, 0, 0, 0],
            [5.0, 4, 6, 3, 2, 3, 1, 0],
            [5.0, 4, 6, 4, 2, 3, 2, 2],
        ]
    ]
)
CUT = np.array([4, 6, 7])
OBSERVED = np.arange(8)[None, :] <= CUT[:, None]
COUNTS = COUNTS * OBSERVED[None]


def test_pair_sums_skip_cells_a_release_does_not_observe() -> None:
    pairs = np.array([[0, 1], [1, 2]])
    num, den = pair_sums(
        jnp.asarray(COUNTS), jnp.asarray(OBSERVED), jnp.asarray(CUT),
        jnp.asarray(pairs), 4,
    )  # fmt: skip
    assert num.shape == den.shape == (1, 2, 4)
    # Pair (0, 1): ages 0..3 are days 4, 3, 2, 1.
    assert np.allclose(den[0, 0], [1, 2, 4, 3])
    assert np.allclose(num[0, 0], [2, 3, 6, 4])
    # Pair (1, 2): ages 0..3 are days 6, 5, 4, 3.
    assert np.allclose(den[0, 1], [1, 3, 2, 3])
    assert np.allclose(num[0, 1], [2, 3, 2, 4])


def test_ages_beyond_the_window_are_dropped() -> None:
    pairs = np.array([[0, 1]])
    _, den = pair_sums(
        jnp.asarray(COUNTS), jnp.asarray(OBSERVED), jnp.asarray(CUT),
        jnp.asarray(pairs), 2,
    )  # fmt: skip
    assert den.shape == (1, 1, 2)
    assert np.allclose(den[0, 0], [1, 2])


def test_pairs_of_one_step_are_summed_and_zero_denominators_unidentified() -> (
    None
):
    num = jnp.asarray([[[2.0, 3, 0], [4.0, 6, 0], [9.0, 9, 9]]])
    den = jnp.asarray([[[1.0, 3, 0], [3.0, 3, 0], [3.0, 3, 3]]])
    f, ident = pair_factors(num, den, jnp.asarray([0, 0, 1]), 2)
    assert f.shape == ident.shape == (1, 2, 3)
    assert np.allclose(f[0, 0, :2], [6 / 4, 9 / 6])
    assert np.allclose(f[0, 1], [3, 3, 3])
    assert ident[0].tolist() == [[True, True, False], [True, True, True]]
    assert float(f[0, 0, 2]) == 1.0


def test_chain_multiplies_the_factors_along_each_plan() -> None:
    f = jnp.asarray([[[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]])
    plan = ChainPlan(
        step=jnp.asarray([[0, 1], [1, 1], [0, 0]]),
        age=jnp.asarray([[0, 1], [2, 5], [1, 2]]),
        valid=jnp.asarray([[True, True], [True, False], [False, False]]),
    )
    g = chain_growth(f, plan)
    # Row 2: age 5 is invalid, so only factor (1, 2) counts.
    assert np.allclose(g[0], [2.0 * 11.0, 13.0, 1.0])
