import numpy as np
import pandas as pd
import pytest

from paperfigs.scores import relative_wis, score_forecasts

LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def _pinball_wis(q: np.ndarray, levels: np.ndarray, y: float) -> float:
    """Reference WIS: summed pinball loss over K + 1/2."""
    k = (len(levels) - 1) / 2
    loss = np.where(y < q, 1 - levels, levels) * np.abs(y - q)
    return float(loss.sum() / (k + 0.5))


def _rows(model, q, truth, location="a", origin="2024-01-10", h=0):
    origin = pd.Timestamp(origin)
    return pd.DataFrame(
        {
            "study": "synthetic",
            "model": model,
            "location": location,
            "origin_date": origin,
            "target_date": origin + pd.Timedelta(days=h),
            "horizon": h,
            "quantile_level": LEVELS,
            "value": np.asarray(q, dtype=float),
            "truth": float(truth),
        }
    )


Q = np.array([2.0, 4.0, 5.0, 7.0, 12.0])


@pytest.mark.parametrize("y", [0.0, 3.0, 5.0, 6.5, 20.0])
def test_wis_matches_pinball_form(y: float) -> None:
    s = score_forecasts(_rows("m", Q, y))
    assert s["wis"].item() == pytest.approx(_pinball_wis(Q, LEVELS, y))


def test_log_wis_scores_log1p_of_quantiles_and_truth() -> None:
    y = 9.0
    s = score_forecasts(_rows("m", Q, y))
    want = _pinball_wis(np.log1p(Q), LEVELS, np.log1p(y))
    assert s["log_wis"].item() == pytest.approx(want)


@pytest.mark.parametrize(
    ("y", "bias"),
    [
        (5.0, 0.0),  # at the median
        (0.0, 1.0),  # below every quantile: overprediction
        (20.0, -1.0),  # above every quantile: underprediction
        (3.0, 0.9),  # 1 - 2 max{tau : q_tau <= y} = 1 - 2 * 0.05
        (8.0, -0.9),  # 1 - 2 min{tau : q_tau >= y} = 1 - 2 * 0.95
    ],
)
def test_bias_follows_scoringutils(y: float, bias: float) -> None:
    s = score_forecasts(_rows("m", Q, y))
    assert s["bias"].item() == pytest.approx(bias)


@pytest.mark.parametrize(
    ("y", "c50", "c90"),
    [(5.0, 1.0, 1.0), (3.0, 0.0, 1.0), (13.0, 0.0, 0.0), (7.0, 1.0, 1.0)],
)
def test_coverage(y: float, c50: float, c90: float) -> None:
    s = score_forecasts(_rows("m", Q, y))
    assert s["coverage_50"].item() == c50
    assert s["coverage_90"].item() == c90


def test_coverage_is_missing_without_the_levels() -> None:
    df = _rows("m", Q, 5.0)
    df = df.loc[~df["quantile_level"].isin([0.05, 0.95])]
    s = score_forecasts(df)
    assert np.isnan(s["coverage_90"].item())
    assert s["coverage_50"].item() == 1.0


def test_forecasts_without_truth_are_dropped() -> None:
    df = pd.concat([_rows("m", Q, 5.0), _rows("m", Q, np.nan, h=-1)])
    assert len(score_forecasts(df)) == 1


def test_one_row_per_forecast_with_keys() -> None:
    df = pd.concat(
        [_rows("m", Q, 5.0, h=h, location=loc) for h in (-1, 0) for loc in "ab"]
    )
    s = score_forecasts(df)
    assert len(s) == 4
    for col in ("study", "model", "location", "origin_date", "horizon"):
        assert col in s.columns


def test_relative_wis_uses_matched_rows_only() -> None:
    base = pd.concat([_rows("randomwalknowcast", Q, 9.0, h=h) for h in (-1, 0)])
    # The model only forecasts h = 0, so only that row of the baseline counts.
    model = _rows("m", Q * 1.5, 9.0, h=0)
    s = score_forecasts(pd.concat([base, model]))
    rel = relative_wis(s)
    got = rel.set_index("model").loc["m"]
    wis_m = _pinball_wis(Q * 1.5, LEVELS, 9.0)
    wis_b = _pinball_wis(Q, LEVELS, 9.0)
    assert got["rel_wis"] == pytest.approx(wis_m / wis_b)
    assert rel.set_index("model").loc["randomwalknowcast", "rel_wis"] == 1


def test_relative_wis_by_group_and_named_baseline() -> None:
    df = pd.concat(
        [_rows("ref", Q, 9.0, location=loc) for loc in "ab"]
        + [_rows("m", Q * 2, 9.0, location=loc) for loc in "ab"]
    )
    rel = relative_wis(score_forecasts(df), by=["location"], baseline="ref")
    assert set(rel["location"]) == {"a", "b"}
    assert {"rel_wis", "rel_log_wis"} <= set(rel.columns)
    m = rel.loc[rel["model"] == "m"]
    assert (m["rel_wis"] > 1).all()


def test_relative_wis_needs_the_baseline() -> None:
    s = score_forecasts(_rows("m", Q, 5.0))
    with pytest.raises(ValueError, match="randomwalknowcast"):
        relative_wis(s)
