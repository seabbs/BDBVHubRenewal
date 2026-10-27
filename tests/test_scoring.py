from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scoringrules as sr

from bdbv_hub_renewal.scoring import (
    QUANTILE_LEVELS,
    bias_sample,
    coverage_sample,
    crps_sample,
    relative_skill,
    score_forecasts,
    wis_sample,
)


def _interval_wis(obs: float, samples: np.ndarray) -> float:
    """The usual interval form of the WIS, for comparison."""
    alphas = 2 * QUANTILE_LEVELS[QUANTILE_LEVELS < 0.5]
    med = np.quantile(samples, 0.5)
    total = 0.5 * abs(obs - med)
    for a in alphas:
        lo, hi = np.quantile(samples, [a / 2, 1 - a / 2])
        score = (hi - lo) + 2 / a * max(lo - obs, 0) + 2 / a * max(obs - hi, 0)
        total += a / 2 * score
    return total / (len(alphas) + 0.5)


@pytest.mark.parametrize("obs", [0.0, 12.0, 30.0, 80.0])
def test_wis_matches_interval_form(obs: float) -> None:
    samples = np.random.default_rng(1).poisson(25, 100).astype(float)
    got = wis_sample(obs, samples)
    assert got["wis"] == pytest.approx(_interval_wis(obs, samples))
    parts = got["dispersion"] + got["overprediction"] + got["underprediction"]
    assert parts == pytest.approx(got["wis"])


def test_wis_of_point_mass_is_absolute_error() -> None:
    assert wis_sample(7.0, np.full(10, 3.0))["wis"] == pytest.approx(4.0)


def test_crps_matches_scoringrules() -> None:
    samples = np.random.default_rng(2).gamma(2, 10, 100)
    for obs in (5.0, 20.0, 60.0):
        want = float(sr.crps_ensemble(obs, samples, estimator="nrg"))
        assert crps_sample(obs, samples) == pytest.approx(want)


def test_wis_is_close_to_crps_for_many_samples() -> None:
    samples = np.random.default_rng(3).normal(50, 10, 5000)
    assert wis_sample(55.0, samples)["wis"] == pytest.approx(
        crps_sample(55.0, samples), rel=0.1
    )


def test_bias_and_coverage() -> None:
    samples = np.arange(1, 101, dtype=float)
    assert bias_sample(0.0, samples) == 1.0
    assert bias_sample(1000.0, samples) == -1.0
    assert bias_sample(50.0, samples) == pytest.approx(0.01)
    assert coverage_sample(50.0, samples, 0.5)
    assert not coverage_sample(95.0, samples, 0.5)
    assert coverage_sample(95.0, samples, 0.9)


def test_score_forecasts_and_relative_skill() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for model, centre in (("persistence", 30), ("better", 20)):
        for week in ("2026-09-05", "2026-09-12"):
            for i, v in enumerate(rng.poisson(centre, 100)):
                rows.append(
                    {
                        "model": model,
                        "kind": "real",
                        "vintage": "2026-08-18",
                        "reference_date": "2026-08-22",
                        "location": "CD-IT",
                        "target_end_date": week,
                        "horizon": 2,
                        "sample": i + 1,
                        "value": v,
                    }
                )
    forecasts = pd.DataFrame(rows)
    truth = pd.DataFrame(
        {
            "location": ["CD-IT", "CD-IT"],
            "target_end_date": ["2026-09-05", "2026-09-12"],
            "truth": [20, 20],
        }
    )
    scores = score_forecasts(forecasts, truth)
    assert len(scores) == 4
    rel = relative_skill(scores).set_index("model")
    assert rel.loc["persistence", "relative_wis"] == pytest.approx(1.0)
    assert rel.loc["better", "relative_wis"] < 1
