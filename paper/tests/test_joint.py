import numpy as np
import pandas as pd
import pytest
import scoringrules as sr

from paperfigs.joint import joint_scores, relative_joint_scores


def _trajectories(model: str, draws: np.ndarray, truth) -> pd.DataFrame:
    """Sample forecasts with `draws[sample, target]`."""
    targets = pd.date_range("2026-01-03", periods=draws.shape[1], freq="7D")
    rows = [
        {
            "study": "ebola",
            "model": model,
            "location": "CD",
            "stratum": "all",
            "origin_date": pd.Timestamp("2026-01-17"),
            "target_date": t,
            "horizon": j - 2,
            "quantile_level": np.nan,
            "sample_id": i,
            "value": float(draws[i, j]),
            "truth": float(truth[j]),
        }
        for i in range(draws.shape[0])
        for j, t in enumerate(targets)
    ]
    return pd.DataFrame(rows)


def test_joint_scores_are_scoringrules_on_log_counts() -> None:
    rng = np.random.default_rng(1)
    draws = rng.poisson(20, size=(50, 4)).astype(float)
    truth = np.array([18.0, 22, 19, 25])
    out = joint_scores(_trajectories("m", draws, truth))
    row = out.iloc[0]
    assert row["energy_log"] == pytest.approx(
        float(sr.es_ensemble(np.log1p(truth), np.log1p(draws)))
    )
    assert row["variogram_log"] == pytest.approx(
        float(sr.vs_ensemble(np.log1p(truth), np.log1p(draws), p=0.5))
    )
    assert row["n_targets"] == 4


def test_pairing_matters_and_baseline_scores_one() -> None:
    base = np.linspace(5, 50, 40)
    paired = np.column_stack([base, base * 1.1, base * 1.2])
    rng = np.random.default_rng(2)
    unpaired = np.column_stack([rng.permutation(c) for c in paired.T])
    truth = np.array([45.0, 49.5, 54])
    fc = pd.concat(
        [
            _trajectories("BDBVHub-baseline", unpaired, truth),
            _trajectories("paired", paired, truth),
        ]
    )
    rel = relative_joint_scores(joint_scores(fc)).set_index("model")
    assert rel.loc["BDBVHub-baseline", "rel_energy_log"] == 1
    assert rel.loc["paired", "rel_variogram_log"] < 1


def test_forecasts_without_every_truth_are_left_out() -> None:
    draws = np.ones((5, 3))
    out = joint_scores(_trajectories("m", draws, [1.0, np.nan, 1.0]))
    assert out.empty
