"""Scoring forecasts paired by the copula against unpaired ones."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.scoring import variogram_sample
from bdbvhubrenewal.evaluation.trajectories import (
    ARMS,
    RHO_COLUMNS,
    SCORE_COLUMNS,
    copula_arms,
    copula_rho,
    copula_scores,
)
from bdbvhubrenewal.forecasters import NATIONAL

WEEKS = ["2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24"]
PLACES = ["CD-A", "CD-B", "CD-C"]


def test_variogram_of_a_point_forecast_at_the_truth_is_zero() -> None:
    y = np.array([1.0, 4.0, 9.0])
    assert variogram_sample(y, np.tile(y, (5, 1))) == pytest.approx(0.0)
    assert variogram_sample(y, np.tile(y[::-1], (5, 1))) > 0


@pytest.mark.skipif(
    importlib.util.find_spec("scoringrules") is None,
    reason="needs scoringrules",
)
def test_variogram_matches_scoringrules() -> None:
    import scoringrules as sr

    rng = np.random.default_rng(0)
    y, x = rng.gamma(2, 3, size=6), rng.gamma(2, 3, size=(40, 6))
    assert variogram_sample(y, x, p=0.5) == pytest.approx(
        float(sr.variogram_score(y, x, p=0.5))
    )


def _forecasts(n: int = 200) -> pd.DataFrame:
    """Independent draws at three origins, the country as their sum."""
    rng = np.random.default_rng(1)
    rows = []
    for origin in ("2026-10-10", "2026-10-17", "2026-10-24"):
        for place in PLACES:
            for week in WEEKS:
                rows.append(
                    pd.DataFrame(
                        {
                            "model": "m",
                            "origin": origin,
                            "location": place,
                            "target_end_date": week,
                            "sample": np.arange(1, n + 1),
                            "value": rng.gamma(2.0, 10.0, size=n).round(),
                        }
                    )
                )
    out = pd.concat(rows, ignore_index=True)
    keys = ["model", "origin", "target_end_date", "sample"]
    nat = out.groupby(keys, as_index=False)["value"].sum()
    nat["location"] = NATIONAL
    return pd.concat([out, nat[list(out.columns)]], ignore_index=True)


def _truth(high: bool) -> pd.DataFrame:
    """Every place high (or low) together, released by 2026-10-20."""
    value = 60.0 if high else 3.0
    rows = [
        {
            "location": p,
            "target_end_date": w,
            "truth": value,
            "truth_vintage": "2026-10-20",
        }
        for p in PLACES
        for w in WEEKS
    ]
    rows += [
        {
            "location": NATIONAL,
            "target_end_date": w,
            "truth": 3 * value,
            "truth_vintage": "2026-10-20",
        }
        for w in WEEKS
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def arms() -> tuple[pd.DataFrame, pd.DataFrame]:
    return copula_arms(_forecasts(), _truth(high=True), seed=1)


def test_arms_keep_each_places_draws(
    arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    samples, _ = arms
    assert set(samples["arm"]) == set(ARMS)
    cells = ["arm", "origin", "location", "target_end_date"]
    places = samples.loc[samples["location"] != NATIONAL]
    sorted_draws = places.groupby(cells)["value"].apply(
        lambda v: tuple(np.sort(v.to_numpy()))
    )
    by_arm = sorted_draws.unstack("arm")
    for arm in ARMS:
        assert (by_arm[arm] == by_arm["unpaired"]).all()
    # The country stays the sum of its provinces in every arm.
    keys = ["arm", "origin", "target_end_date", "sample"]
    total = places.groupby(keys)["value"].sum()
    nat = samples.loc[samples["location"] == NATIONAL].set_index(keys)
    assert np.allclose(nat["value"], total.reindex(nat.index))


def test_rho_has_a_row_per_estimate(
    arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, fits = arms
    rho = copula_rho(fits, study="ebola", strata="provinces")
    assert list(rho.columns) == list(RHO_COLUMNS)
    assert set(rho["arm"]) == {"fixed", "rolling", "rolling_w25"}
    fixed = rho.loc[rho["arm"] == "fixed"]
    assert (fixed["rho_t"] == 0.9).all()


def test_scores_reward_pairing_when_places_move_together(
    arms: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    samples, _ = arms
    scores = copula_scores(
        samples, _truth(high=True), study="ebola", strata="provinces"
    )
    assert list(scores.columns) == list(SCORE_COLUMNS)
    assert len(scores) == len(ARMS) * 3
    mean = scores.groupby("arm").mean(numeric_only=True)
    # A joint extreme is scored better by paired draws, on the sums and
    # on the joint scores, and each place's WIS is the same.
    assert mean.loc["fixed", "energy_log"] < mean.loc["unpaired", "energy_log"]
    assert (
        mean.loc["fixed", "cover90_places"]
        >= mean.loc["unpaired", "cover90_places"]
    )
    assert mean["wis_strata"].nunique() == 1
    assert (
        mean.loc["fixed", "wis_national"]
        != mean.loc["unpaired", "wis_national"]
    )
