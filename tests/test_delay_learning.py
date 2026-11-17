"""The delay-learning SI analysis (`scripts/delay_learning.py`)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import delay_learning as dl

LAG, CONTEXT, WIDTH = 3, 4, 3


def toy_block() -> dict:
    """One place, three lags: final counts 10, 20, 30 on the nowcast days.

    Lag row d holds the count reported at lag d, known only up to the
    cut-off, which is the last nowcast day.
    """
    final = [1, 2, 3, 4, 10, 20, 30]
    lags = np.full((LAG, CONTEXT + WIDTH), np.nan)
    for d in range(LAG):
        for t in range(CONTEXT + WIDTH):
            if t + d <= CONTEXT + LAG - 1:
                lags[d, t] = final[t] * (d + 1) / 4
    return {
        "context": np.vstack([final, lags]),
        "n_targets": 1,
        "n_covariates": LAG,
        "n_future_covariates": LAG,
        "prediction_length": WIDTH,
        "places": ("CD",),
        "target_places": [0],
        "meta": {"options": {"max_lag": LAG}},
    }


def test_reported_counts_are_each_days_latest_lag_at_the_cutoff():
    block = toy_block()
    # Nowcast day j is seen at lag D - 1 - j by the cut-off.
    np.testing.assert_allclose(
        dl.reported(block), [[10 * 3 / 4, 20 * 2 / 4, 30 / 4]]
    )
    np.testing.assert_allclose(dl.truth(block), [[10, 20, 30]])


def test_a_slower_delay_moves_each_lag_down_by_k_days():
    block = toy_block()
    moved = dl.slower(block, 1)
    seen = dl.reported(moved)
    # Lag d now holds what lag d - 1 held, and lag 0 nothing.
    np.testing.assert_allclose(seen, [[10 * 2 / 4, 20 / 4, 0.0]])
    # The targets, and which cells are known, are unchanged.
    np.testing.assert_array_equal(moved["context"][0], block["context"][0])
    np.testing.assert_array_equal(
        np.isnan(moved["context"][1:]), np.isnan(block["context"][1:])
    )


def test_without_lags_keeps_only_the_targets():
    bare = dl.without_lags(toy_block())
    assert bare["context"].shape == (1, CONTEXT + WIDTH)
    assert bare["n_covariates"] == 0


def test_attention_shares_split_each_count_rows_attention():
    torch = pytest.importorskip("torch")
    # Two places, CD and its child CD-NK, each with a totals row and two
    # lags: rows CD, CD-NK, CD total, CD-NK total, CD lags, CD-NK lags.
    block = {
        "places": ("CD", "CD-NK"),
        "target_places": [0, 1, 0, 1],
        "n_targets": 4,
        "n_covariates": 4,
    }
    n_rows = 8
    weights = [torch.full((2, 1, n_rows, n_rows), 1 / n_rows)]
    layers, lags = dl.shares(block, weights)
    total = layers.groupby("layer")["share"].sum()
    assert np.allclose(total, 1.0)
    got = layers.set_index("kind")["share"]
    # Each count row: itself, its own totals row and two lags, then the
    # other place's count, totals and two lags (parent or child).
    assert got["self"] == pytest.approx(1 / 8)
    assert got["own totals"] == pytest.approx(1 / 8)
    assert got["own lags"] == pytest.approx(2 / 8)
    assert got["parent"] + got["children"] == pytest.approx(4 / 8)
    assert got["other places"] == pytest.approx(0.0)
    assert set(lags["d"]) == {0, 1}
