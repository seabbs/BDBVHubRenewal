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
    score_quantiles,
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


HUB_LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
"""The German hub's quantile levels."""


def test_wis_of_quantiles_is_the_mean_quantile_score() -> None:
    # As the German hub's paper scores its seven levels: the mean over
    # levels of 2 (1{y < q} - tau) (q - y).
    q = np.array([[2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 20.0]] * 3)
    obs = np.array([1.0, 10.0, 30.0])
    want = np.mean(
        2 * ((obs[:, None] < q) - HUB_LEVELS) * (q - obs[:, None]), axis=1
    )
    got = wis_quantile(obs, q, HUB_LEVELS)
    np.testing.assert_allclose(got["wis"], want)
    parts = got["dispersion"] + got["overprediction"] + got["underprediction"]
    np.testing.assert_allclose(parts, want)


def test_wis_of_samples_is_wis_of_their_quantiles() -> None:
    samples = np.random.default_rng(4).poisson(25, 100).astype(float)
    q = np.quantile(samples, QUANTILE_LEVELS)[None, :]
    got = wis_quantile(np.array([18.0]), q, QUANTILE_LEVELS)
    assert got["wis"][0] == pytest.approx(wis_sample(18.0, samples)["wis"])


def _quantile_rows(model: str, place: str, values: list[float]) -> list[dict]:
    return [
        {
            "model": model,
            "location": place,
            "target_end_date": "2021-12-04",
            "horizon": 0,
            "quantile": tau,
            "value": v,
        }
        for tau, v in zip(HUB_LEVELS, values, strict=True)
    ]


def test_score_quantiles_gives_wis_median_error_and_coverage() -> None:
    q = [2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 20.0]
    forecasts = pd.DataFrame(
        _quantile_rows("a", "DE", q) + _quantile_rows("a", "DE-BY", q)
    )
    truth = pd.DataFrame(
        {
            "location": ["DE", "DE-BY"],
            "target_end_date": "2021-12-04",
            "truth": [10.0, 30.0],
        }
    )
    got = score_quantiles(
        forecasts, truth, ("model", "location", "target_end_date"), HUB_LEVELS
    ).set_index("location")
    want = wis_quantile(np.array([10.0, 30.0]), np.array([q, q]), HUB_LEVELS)
    np.testing.assert_allclose(got.loc[["DE", "DE-BY"], "wis"], want["wis"])
    assert got.loc["DE", "ae_median"] == pytest.approx(1.0)
    assert got.loc["DE", "coverage_50"] == 1.0
    assert got.loc["DE-BY", "coverage_95"] == 0.0
    assert got.loc["DE", "horizon"] == 0
    assert set(got.columns) >= {"truth", "coverage_80", "dispersion"}


def test_score_quantiles_refuses_a_forecast_missing_a_level() -> None:
    rows = _quantile_rows("a", "DE", [2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 20.0])
    forecasts = pd.DataFrame(rows[1:])
    truth = pd.DataFrame(
        {"location": ["DE"], "target_end_date": "2021-12-04", "truth": [1.0]}
    )
    with pytest.raises(ValueError, match="quantile levels"):
        score_quantiles(
            forecasts, truth, ("model", "location", "target_end_date"),
            HUB_LEVELS,
        )  # fmt: skip


def test_wis_of_point_mass_is_absolute_error() -> None:
    assert wis_sample(7.0, np.full(10, 3.0))["wis"] == pytest.approx(4.0)


@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("m", [10, 100, 1000])
def test_crps_is_scoringrules_fair_estimator(seed: int, m: int) -> None:
    # The fair form (Ferro 2014), as the paper scores after #281, so
    # scores at 100 and 1,000 draws compare in expectation.
    samples = np.random.default_rng(seed).gamma(2, 10, m)
    for obs in (5.0, 20.0, 60.0):
        want = float(sr.crps_ensemble(obs, samples, estimator="fair"))
        assert crps_sample(obs, samples) == pytest.approx(want)
        nrg = float(sr.crps_ensemble(obs, samples, estimator="nrg"))
        got = crps_sample(obs, samples, estimator="nrg")
        assert got == pytest.approx(nrg)


@pytest.mark.parametrize("seed", [5, 6])
def test_energy_is_scoringrules_fair_estimator(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.poisson(20, (200, 6)).astype(float)
    y = rng.poisson(20, 6).astype(float)
    want = float(sr.es_ensemble(y, x, estimator="fair"))
    assert energy_sample(y, x) == pytest.approx(want)


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


def test_log_crps_is_the_crps_of_log1p_counts() -> None:
    # As the paper scores Ebola: sample CRPS of log(1 + y) against
    # log(1 + x), with scoringrules' fair estimator.
    from bdbvhubrenewal.evaluation.scoring import SCORE_COLUMNS, score_sample

    assert "log_crps" in SCORE_COLUMNS
    samples = np.random.default_rng(5).poisson(25, 1000).astype(float)
    for obs in (0.0, 12.0, 80.0):
        got = score_sample(obs, samples)
        want = crps_sample(float(np.log1p(obs)), np.log1p(samples))
        assert got["log_crps"] == pytest.approx(want)
        fair = sr.crps_ensemble(
            np.log1p(obs), np.log1p(samples), estimator="fair"
        )
        assert got["log_crps"] == pytest.approx(float(fair))
        assert got["crps"] == pytest.approx(crps_sample(obs, samples))


def test_score_forecasts_gives_log_crps() -> None:
    rows = [
        {
            "model": "m",
            "kind": "real",
            "vintage": "2026-08-18",
            "reference_date": "2026-08-22",
            "location": "CD-IT",
            "target_end_date": "2026-09-05",
            "horizon": 2,
            "sample": i + 1,
            "value": float(v),
        }
        for i, v in enumerate([3, 8, 15, 22, 40])
    ]
    truth = pd.DataFrame(
        {"location": ["CD-IT"], "target_end_date": ["2026-09-05"], "truth": [9]}
    )
    scores = score_forecasts(pd.DataFrame(rows), truth)
    want = crps_sample(float(np.log1p(9)), np.log1p([3, 8, 15, 22, 40]))
    assert scores["log_crps"].iloc[0] == pytest.approx(want)


def test_german_quantile_scores_stay_wis() -> None:
    # The German benchmark scores quantile forecasts, so it keeps WIS.
    from bdbvhubrenewal.evaluation import germany

    assert "wis" in germany.SCORE_COLUMNS
    assert "crps" not in germany.SCORE_COLUMNS
    quantiles = pd.DataFrame(
        {
            "id": [0] * 3,
            "quantile": [0.25, 0.5, 0.75],
            "value": [4.0, 6.0, 9.0],
        }
    )
    assert "wis" in wis_quantile(
        np.array([5.0]),
        quantiles["value"].to_numpy()[None, :],
        quantiles["quantile"].to_numpy(),
    )


def _two_models_same_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(8)
    draws = rng.poisson(30, (100, 2, 2)).astype(float)
    rows = [
        {
            "model": model,
            "kind": "real",
            "vintage": "2026-10-17",
            "reference_date": "2026-10-24",
            "origin": "2026-10-17",
            "location": loc,
            "target_end_date": week,
            "horizon": 0,
            "sample": s + 1,
            "value": draws[s, i, j],
        }
        for model in ("renewal", "teamA-hub")
        for s in range(100)
        for i, loc in enumerate(("CD-IT", "CD-NK"))
        for j, week in enumerate(("2026-10-17", "2026-10-24"))
    ]
    truth = pd.DataFrame(
        {
            "location": ["CD-IT", "CD-IT", "CD-NK", "CD-NK"],
            "target_end_date": ["2026-10-17", "2026-10-24"] * 2,
            "truth": [25, 31, 40, 12],
        }
    )
    return pd.DataFrame(rows), truth


def test_same_samples_score_the_same_whoever_made_them() -> None:
    # Ours or a hub team's: one estimator for every model.
    forecasts, truth = _two_models_same_samples()
    scores = score_forecasts(forecasts, truth).set_index(
        ["model", "location", "target_end_date"]
    )
    for col in ("crps", "log_crps"):
        a = scores.loc["renewal", col].sort_index()
        b = scores.loc["teamA-hub", col].sort_index()
        assert np.allclose(a.to_numpy(), b.to_numpy())
    energy = energy_scores(forecasts, truth, unit=["origin"])
    by = energy.set_index("model")
    for col in ("energy", "log_energy"):
        assert by.loc["renewal", col] == pytest.approx(by.loc["teamA-hub", col])


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list:
    """Record each call of scoringrules' `name`, still computing it."""
    from bdbvhubrenewal.evaluation import scoring

    real = getattr(sr, name)
    calls: list = []

    def wrapped(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(scoring.sr, name, wrapped)
    return calls


def test_wis_is_scoringrules_weighted_interval_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The package gives the WIS; only its split is our own, and the split
    # sums to it.
    calls = _spy(monkeypatch, "weighted_interval_score")
    rng = np.random.default_rng(3)
    y = rng.poisson(20, 40).astype(float)
    q = np.sort(rng.gamma(4, 5, (40, QUANTILE_LEVELS.size)), axis=1)
    got = wis_quantile(y, q, QUANTILE_LEVELS)
    assert calls
    parts = got["dispersion"] + got["overprediction"] + got["underprediction"]
    np.testing.assert_allclose(parts, got["wis"], rtol=1e-12)
    # The interval form (Bracher et al. 2021), from the quantiles.
    tau = QUANTILE_LEVELS
    lo, hi = q[:, tau < 0.5], q[:, tau > 0.5][:, ::-1]
    a = 2 * tau[tau < 0.5]
    med = q[:, np.isclose(tau, 0.5)][:, 0]
    width = (
        hi
        - lo
        + 2
        / a
        * (np.maximum(lo - y[:, None], 0) + np.maximum(y[:, None] - hi, 0))
    )
    want = (0.5 * np.abs(y - med) + (a / 2 * width).sum(axis=1)) / (
        a.size + 0.5
    )
    np.testing.assert_allclose(got["wis"], want, rtol=1e-12)


def test_crps_and_energy_scores_are_scoringrules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crps = _spy(monkeypatch, "crps_ensemble")
    es = _spy(monkeypatch, "es_ensemble")
    rng = np.random.default_rng(4)
    x = rng.normal(10, 3, (50, 4))
    crps_sample(9.0, x[:, 0])
    energy_sample(np.zeros(4), x)
    assert crps
    assert es
    assert es[0]["estimator"] == "fair"


def test_the_variogram_score_is_gone() -> None:
    from bdbvhubrenewal.evaluation import scoring

    assert not hasattr(scoring, "variogram_sample")


def test_scoringrules_wis_adds_the_median_not_its_error() -> None:
    # scoringrules 0.11 and 0.12 add w_median * median, not
    # w_median * |y - median|, so `wis_quantile` passes it the error.
    # If this fails the package is fixed: pass the median itself.
    got = sr.weighted_interval_score(
        np.array([5.0]), np.array([6.0]), np.array([[1.0, 4.0]]),
        np.array([[15.0, 9.0]]), np.array([0.1, 0.5]), backend="numpy",
    )  # fmt: skip
    assert float(np.ravel(got)[0]) == pytest.approx((0.7 + 1.25 + 3) / 2.5)
