from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scoringrules as sr

from bdbvhubrenewal.evaluation.scoring import (
    QUANTILE_LEVELS,
    bias_sample,
    coverage_sample,
    crps_sample,
    energy_sample,
    energy_scores,
    relative_skill,
    score_forecasts,
    wis_quantile,
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


def test_wis_of_quantiles_is_the_mean_quantile_score() -> None:
    # The German hub's levels, scored as its paper does: the mean over
    # levels of 2 (1{y < q} - tau) (q - y).
    tau = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    q = np.array([[2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 20.0]] * 3)
    obs = np.array([1.0, 10.0, 30.0])
    want = np.mean(2 * ((obs[:, None] < q) - tau) * (q - obs[:, None]), 1)
    got = wis_quantile(obs, q, tau)
    np.testing.assert_allclose(got["wis"], want)
    parts = got["dispersion"] + got["overprediction"] + got["underprediction"]
    np.testing.assert_allclose(parts, want)


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
    for model, centre in (("randomwalk", 30), ("better", 20)):
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
    assert rel.loc["randomwalk", "relative_wis"] == pytest.approx(1.0)
    assert rel.loc["better", "relative_wis"] < 1


def test_energy_matches_hand_case() -> None:
    # ||X - y||: 4 and 3, mean 3.5. Fair spread: (5 + 5) / (2 * 2 * 1) = 2.5.
    samples = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert energy_sample(np.array([0.0, 4.0]), samples) == pytest.approx(1.0)


def test_energy_matches_scoringrules_fair() -> None:
    rng = np.random.default_rng(5)
    samples = rng.poisson(20, (50, 6)).astype(float)
    obs = rng.poisson(20, 6).astype(float)
    want = float(sr.es_ensemble(obs, samples, estimator="fair"))
    assert energy_sample(obs, samples) == pytest.approx(want)


def test_energy_scores_share_cells_and_scale() -> None:
    rng = np.random.default_rng(6)
    rows = []
    places = {"randomwalk": ("CD", "CD-IT", "CD-NK"), "better": ("CD", "CD-IT")}
    for model, locs in places.items():
        for loc in locs:
            for week in ("2026-09-05", "2026-09-12"):
                for i, v in enumerate(rng.poisson(30, 40)):
                    rows.append(
                        {
                            "model": model,
                            "origin": "2026-08-18",
                            "location": loc,
                            "target_end_date": week,
                            "sample": i + 1,
                            "value": v,
                        }
                    )
    truth = pd.DataFrame(
        {
            "location": ["CD", "CD", "CD-IT", "CD-IT", "CD-NK", "CD-NK"],
            "target_end_date": ["2026-09-05", "2026-09-12"] * 3,
            "truth": [30, 30, 20, 20, 10, 10],
        }
    )
    es = energy_scores(pd.DataFrame(rows), truth, unit=("origin",))
    es = es.set_index(["scale", "model"])
    # CD-NK is only in one model, so provinces score CD-IT's two weeks.
    assert es.loc[("provinces", "better"), "n_cells"] == 2
    assert es.loc[("provinces", "randomwalk"), "n_cells"] == 2
    assert es.loc[("national", "better"), "n_cells"] == 2
    assert set(es.columns) >= {"energy", "log_energy", "n_samples"}
    assert (es["log_energy"] < es["energy"]).all()


def test_relative_skill_is_mean_over_mean_on_the_matched_set() -> None:
    # The model has no forecast for week 3, so it is left out of both
    # means. Mean over mean is (1 + 9) / (2 + 3) = 2, not the mean of the
    # ratios (0.5 + 3) / 2.
    keys = {"kind": "real", "vintage": "v", "reference_date": "r"}
    scores = pd.DataFrame(
        [
            {
                **keys,
                "model": m,
                "location": "CD",
                "target_end_date": w,
                "wis": v,
                "log_wis": v,
                "bias": b,
            }
            for m, w, v, b in (
                ("randomwalk", "1", 2.0, 0.0),
                ("randomwalk", "2", 3.0, 0.0),
                ("randomwalk", "3", 100.0, 0.0),
                ("a", "1", 1.0, 0.2),
                ("a", "2", 9.0, 0.4),
            )
        ]
    )
    rel = relative_skill(scores, means=("wis", "bias")).set_index("model")
    assert rel.loc["a", "relative_wis"] == pytest.approx(2.0)
    assert rel.loc["a", "n"] == 2
    assert rel.loc["a", "wis"] == pytest.approx(5.0)
    assert rel.loc["a", "base_wis"] == pytest.approx(2.5)
    assert rel.loc["a", "bias"] == pytest.approx(0.3)
