"""One vectorised scorer: every forecast's draws scored at once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.scoring import (
    SCORE_COLUMNS,
    score_cells,
    score_forecasts,
    score_sample,
)


def _loop(obs, draws):
    return pd.DataFrame(
        [score_sample(y, x) for y, x in zip(obs, draws, strict=True)]
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_score_cells_equals_scoring_each_forecast(seed):
    rng = np.random.default_rng(seed)
    draws = rng.poisson(rng.uniform(0, 50, (40, 1)), (40, 300)).astype(float)
    obs = rng.poisson(20, 40).astype(float)
    got = pd.DataFrame(score_cells(obs, draws))
    want = _loop(obs, draws)
    pd.testing.assert_frame_equal(got[list(SCORE_COLUMNS)], want, rtol=1e-10)


def test_score_cells_adds_coverage_levels():
    draws = np.arange(100, dtype=float)[None, :]
    got = score_cells(np.array([96.0]), draws, coverage=(50, 90, 95))
    assert got["coverage_90"][0] == 0.0
    assert got["coverage_95"][0] == 1.0


def test_score_forecasts_with_ragged_draws_equals_the_loop():
    rng = np.random.default_rng(3)
    rows = []
    for i, m in enumerate([50, 50, 80, 20]):
        rows.append(
            pd.DataFrame(
                {
                    "model": f"m{i % 2}",
                    "location": f"L{i}",
                    "target_end_date": "2026-11-01",
                    "horizon": 1,
                    "sample": np.arange(m),
                    "value": rng.poisson(10, m).astype(float),
                }
            )
        )
    forecasts = pd.concat(rows, ignore_index=True)
    truth = pd.DataFrame(
        {
            "location": [f"L{i}" for i in range(4)],
            "target_end_date": "2026-11-01",
            "truth": [9.0, 12.0, 3.0, 10.0],
        }
    )
    keys = ["model", "location", "target_end_date"]
    got = score_forecasts(forecasts, truth, keys=keys)
    merged = forecasts.merge(truth)
    want = []
    for key, grp in merged.groupby([*keys, "horizon"], sort=True):
        y = float(grp["truth"].iloc[0])
        want.append(
            {
                **dict(zip([*keys, "horizon"], key, strict=True)),
                "truth": y,
                "n_samples": len(grp),
                **score_sample(y, grp["value"].to_numpy()),
            }
        )
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), pd.DataFrame(want), check_dtype=False
    )
