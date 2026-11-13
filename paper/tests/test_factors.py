import numpy as np
import pandas as pd
import pytest

from paperfigs.dummy import dummy_factors, dummy_study
from paperfigs.factors import (
    bin_factors,
    relative_by_factor,
    top_factors,
)
from paperfigs.scores import score_forecasts


def test_numeric_factors_bin_to_tertiles() -> None:
    f = pd.DataFrame(
        {"location": list("abcdef"), "delay_mean": [1, 2, 3, 4, 5, 6.0]}
    )
    out = bin_factors(f)
    assert out["delay_mean"].nunique() == 3
    assert out["delay_mean"].iloc[0] != out["delay_mean"].iloc[-1]


def test_few_valued_factors_stay_as_they_are() -> None:
    f = pd.DataFrame({"location": list("abcd"), "report_days": [7, 5, 7, 5]})
    assert set(bin_factors(f)["report_days"]) == {"5", "7"}


def test_places_become_single_or_multi() -> None:
    f = pd.DataFrame({"location": list("abc"), "n_places": [1, 4, 9]})
    out = bin_factors(f)
    assert "n_places" not in out.columns
    assert out["places"].tolist() == ["single", "multi", "multi"]


@pytest.fixture(scope="module")
def synthetic():
    fc, _ = dummy_study("synthetic", seed=1, n_samples=20)
    scores = score_forecasts(fc)
    return scores, dummy_factors(fc, seed=1)


def test_dummy_factors_cover_every_location(synthetic) -> None:
    scores, factors = synthetic
    assert set(factors["location"]) == set(scores["location"])
    assert factors["location"].is_unique


def test_relative_by_factor_is_long(synthetic) -> None:
    scores, factors = synthetic
    rel = relative_by_factor(scores, bin_factors(factors))
    assert {"factor", "level", "model", "rel_score"} <= set(rel.columns)
    base = rel.loc[rel["model"] == "randomwalknowcast", "rel_score"]
    assert np.allclose(base, 1)


def test_top_factors_have_most_spread() -> None:
    rel = pd.DataFrame(
        {
            "factor": ["a", "a", "b", "b", "c", "c"],
            "level": ["x", "y"] * 3,
            "model": "m",
            "rel_score": [1.0, 1.0, 0.5, 2.0, 0.9, 1.1],
        }
    )
    assert top_factors(rel, n=2) == ["b", "c"]
