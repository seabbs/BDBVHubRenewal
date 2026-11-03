"""The one own-or-pooled rule."""

from __future__ import annotations

import numpy as np

from bdbvhubrenewal.forecasters.pooling import pool_matrix, source_index

# Nodes: 0 country, 1 and 2 provinces, 3 and 4 zones of province 1.
PARENT = np.array([-1, 0, 0, 1, 1])


def test_ten_cases_a_week_is_the_threshold() -> None:
    level = np.array([60.0, 9.9, 10.0, 0.0, 0.0])
    src = source_index(level, PARENT)
    # Province 1 pools to the country, province 2 uses its own.
    assert src[1] == 5 + 0
    assert src[2] == 2
    assert src[0] == 0


def test_a_zone_uses_its_province_then_the_country() -> None:
    level = np.array([60.0, 30.0, 40.0, 12.0, 3.0])
    src = source_index(level, PARENT)
    assert src[3] == 3
    assert src[4] == 5 + 1
    small = np.array([60.0, 5.0, 40.0, 3.0, 3.0])
    src = source_index(small, PARENT)
    assert src[3] == 5 + 0
    assert src[4] == 5 + 0


def test_a_country_below_the_threshold_uses_its_own_pooled_estimate() -> None:
    src = source_index(np.zeros(5), PARENT)
    assert (src[:3] == 5).all()
    assert src[3] == 5


def test_one_error_is_not_enough_for_a_spread() -> None:
    level = np.array([60.0, 30.0, 40.0, 0.0, 0.0])
    n_own = np.array([[5, 1], [1, 2], [2, 2], [0, 0], [0, 0]])
    n_pooled = np.array([[5, 5], [6, 6], [6, 6], [0, 0], [0, 0]])
    src = source_index(level, PARENT, n_own, n_pooled, min_err=2)
    assert src.shape == (5, 2)
    # Province 1 has 1 own error at k = 0 and 2 at k = 1.
    assert src[1].tolist() == [5 + 0, 1]
    assert src[2].tolist() == [2, 2]
    # The country has 1 own error at k = 1 and 5 pooled.
    assert src[0].tolist() == [0, 5 + 0]


def test_the_delay_and_the_walk_spread_share_a_source() -> None:
    level = np.array([60.0, 30.0, 4.0, 0.0, 0.0])
    own = source_index(level, PARENT)
    with_errors = source_index(
        level,
        PARENT,
        np.full((5, 1), 9),
        np.full((5, 1), 9),
    )
    assert (own == with_errors[:, 0]).all()


def test_the_spread_moves_up_alone_when_errors_are_short() -> None:
    level = np.array([60.0, 30.0, 40.0, 0.0, 0.0])
    own = source_index(level, PARENT)
    short = source_index(
        level, PARENT, np.array([[9], [1], [9], [0], [0]]), np.full((5, 1), 9)
    )
    assert own[1] == 1
    assert short[1, 0] == 5 + 0


def test_a_pool_holds_the_qualifying_children_not_the_parent() -> None:
    level = np.array([60.0, 30.0, 4.0, 0.0, 0.0])
    m = pool_matrix(level, PARENT)
    # The country is the sum of its provinces, so it is not counted twice.
    assert m[0].tolist() == [0, 1, 0, 0, 0]
    # With no qualifying child, a node pools alone.
    assert m[1].tolist() == [0, 1, 0, 0, 0]
    assert m[3].tolist() == [0, 0, 0, 1, 0]
