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


WIS_PARTS = ("dispersion", "overprediction", "underprediction")


def test_wis_components_of_sample_forecasts_sum_to_wis():
    rng = np.random.default_rng(5)
    draws = rng.poisson(rng.uniform(1, 40, (60, 1)), (60, 500)).astype(float)
    obs = rng.poisson(15, 60).astype(float)
    got = score_cells(obs, draws)
    total = sum(got[c] for c in WIS_PARTS)
    np.testing.assert_allclose(total, got["wis"], rtol=1e-12)
    assert all((got[c] >= 0).all() for c in WIS_PARTS)


def test_wis_components_of_quantile_forecasts_sum_to_wis():
    from bdbvhubrenewal.evaluation.scoring import (
        QUANTILE_LEVELS,
        score_quantiles,
    )

    rng = np.random.default_rng(6)
    rows = []
    for loc in ("A", "B", "C"):
        q = np.sort(rng.gamma(2, 5, len(QUANTILE_LEVELS)))
        rows.append(
            pd.DataFrame(
                {
                    "model": "m",
                    "location": loc,
                    "target_end_date": "2026-11-01",
                    "horizon": 1,
                    "quantile": QUANTILE_LEVELS,
                    "value": q,
                }
            )
        )
    truth = pd.DataFrame(
        {"location": ["A", "B", "C"], "target_end_date": "2026-11-01",
         "truth": [3.0, 12.0, 40.0]}
    )  # fmt: skip
    got = score_quantiles(
        pd.concat(rows),
        truth,
        ("model", "location", "target_end_date"),
        QUANTILE_LEVELS,
    )
    total = got[list(WIS_PARTS)].sum(axis=1)
    np.testing.assert_allclose(total, got["wis"], rtol=1e-12)
