import numpy as np
import pandas as pd
import pytest

from paperfigs.scores import (
    STUDY_BASELINES,
    STUDY_COUNTS,
    STUDY_COVERAGE,
    forecast_share,
    relative_scores,
    score_forecasts,
)

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
            "sample_id": pd.NA,
            "value": np.asarray(q, dtype=float),
            "truth": float(truth),
        }
    )


Q = np.array([2.0, 4.0, 5.0, 7.0, 12.0])


@pytest.mark.parametrize("y", [0.0, 3.0, 5.0, 6.5, 20.0])
def test_wis_matches_pinball_form(y: float) -> None:
    s = score_forecasts(_rows("m", Q, y))
    assert s["score"].item() == pytest.approx(_pinball_wis(Q, LEVELS, y))


def test_log_wis_scores_log1p_of_quantiles_and_truth() -> None:
    y = 9.0
    s = score_forecasts(_rows("m", Q, y))
    want = _pinball_wis(np.log1p(Q), LEVELS, np.log1p(y))
    assert s["log_score"].item() == pytest.approx(want)


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
    rel = relative_scores(s)
    got = rel.set_index("model").loc["m"]
    wis_m = _pinball_wis(Q * 1.5, LEVELS, 9.0)
    wis_b = _pinball_wis(Q, LEVELS, 9.0)
    assert got["rel_score"] == pytest.approx(wis_m / wis_b)
    assert rel.set_index("model").loc["randomwalknowcast", "rel_score"] == 1


def test_relative_wis_by_group_and_named_baseline() -> None:
    df = pd.concat(
        [_rows("ref", Q, 5.0, location=loc) for loc in "ab"]
        + [_rows("m", Q * 2, 5.0, location=loc) for loc in "ab"]
    )
    rel = relative_scores(score_forecasts(df), by=["location"], baseline="ref")
    assert set(rel["location"]) == {"a", "b"}
    assert {"rel_score", "rel_log_score"} <= set(rel.columns)
    m = rel.loc[rel["model"] == "m"]
    assert (m["rel_score"] > 1).all()


def test_relative_scores_need_the_baseline() -> None:
    s = score_forecasts(_rows("m", Q, 5.0))
    with pytest.raises(ValueError, match="randomwalknowcast"):
        relative_scores(s)


def test_baselines_differ_by_study() -> None:
    assert STUDY_BASELINES["synthetic"] == "randomwalknowcast"
    assert STUDY_BASELINES["germany"] == "KIT-frozen_baseline"
    assert STUDY_BASELINES["ebola"] == "BDBVHub-baseline"


def test_quantile_forecasts_are_scored_by_wis() -> None:
    assert score_forecasts(_rows("m", Q, 5.0))["metric"].item() == "wis"


def _samples(model, x, truth, location="a"):
    origin = pd.Timestamp("2024-01-10")
    x = np.asarray(x, dtype=float)
    return pd.DataFrame(
        {
            "study": "ebola",
            "model": model,
            "location": location,
            "origin_date": origin,
            "target_date": origin,
            "horizon": 0,
            "quantile_level": np.nan,
            "sample_id": np.arange(x.size),
            "value": x,
            "truth": float(truth),
        }
    )


X = np.array([0, 1, 1, 2, 3, 3, 3, 5, 8, 13], dtype=float)


def _crps_nrg(x: np.ndarray, y: float) -> float:
    return float(
        np.mean(np.abs(x - y)) - np.abs(x[:, None] - x[None, :]).mean() / 2
    )


@pytest.mark.parametrize("y", [0.0, 3.0, 4.0, 20.0])
def test_sample_crps_and_log_crps(y: float) -> None:
    s = score_forecasts(_samples("m", X, y))
    assert s["metric"].item() == "crps"
    assert s["score"].item() == pytest.approx(_crps_nrg(X, y))
    want = _crps_nrg(np.log1p(X), np.log1p(y))
    assert s["log_score"].item() == pytest.approx(want)


@pytest.mark.parametrize(
    ("y", "bias"),
    [(3.0, 1 - (0.7 + 0.4)), (0.0, 1 - 0.1), (20.0, -1.0), (4.0, 1 - 1.4)],
)
def test_sample_bias_for_counts(y: float, bias: float) -> None:
    s = score_forecasts(_samples("m", X, y))
    assert s["bias"].item() == pytest.approx(bias)


def test_sample_coverage_uses_linear_quantiles() -> None:
    lo, hi = np.quantile(X, [0.25, 0.75])
    for y, inside in ((lo, 1.0), (hi + 0.01, 0.0)):
        s = score_forecasts(_samples("m", X, y))
        assert s["coverage_50"].item() == inside


def test_relative_scores_per_study_baseline() -> None:
    df = pd.concat(
        [_samples("BDBVHub-baseline", X, 4.0), _samples("m", X + 3, 4.0)]
    )
    rel = relative_scores(score_forecasts(df)).set_index("model")
    assert rel.loc["BDBVHub-baseline", "rel_score"] == 1
    assert rel.loc["m", "rel_score"] > 1


def test_wis_pins_the_hand_value() -> None:
    # Q = (2, 4, 5, 7, 12) at levels (0.05, 0.25, 0.5, 0.75, 0.95), y = 3:
    # (0.5 * |3 - 5| + 0.05 * 10 + 0.25 * (3 + 2 * 1 / 0.5)) / 2.5 = 1.3.
    # scoringrules' weighted_interval_score (0.11 and 0.12) adds the median
    # itself rather than |y - median| and gives 1.9, so the WIS is built
    # from its interval_score instead.
    s = score_forecasts(_rows("m", Q, 3.0))
    assert s["score"].item() == pytest.approx(1.3)


def test_coverage_levels_follow_the_study() -> None:
    assert STUDY_COVERAGE["germany"] == (0.5, 0.95)
    assert STUDY_COVERAGE["synthetic"] == (0.5, 0.9)
    q = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    df = _rows("m", Q, 11.0).assign(study="germany", quantile_level=q)
    s = score_forecasts(df)
    assert s["coverage_95"].item() == 1.0
    assert "coverage_90" not in s.columns


def test_coverage_levels_can_be_set() -> None:
    s = score_forecasts(_rows("m", Q, 5.0), coverage=(0.5, 0.9, 0.95))
    assert np.isnan(s["coverage_95"].item())
    assert s["coverage_90"].item() == 1.0


def test_relative_scores_match_on_stratum() -> None:
    df = pd.concat(
        [
            _rows("randomwalknowcast", Q, 5.0).assign(stratum="a"),
            _rows("m", Q * 2, 5.0).assign(stratum="b"),
        ]
    )
    rel = relative_scores(score_forecasts(df))
    assert "m" not in set(rel["model"])


def test_sample_bias_does_not_depend_on_other_models() -> None:
    alone = score_forecasts(_samples("m", X, 0.0))
    mixed = pd.concat([_samples("m", X, 0.0), _samples("other", X + 0.5, 0.0)])
    together = score_forecasts(mixed).set_index("model")
    assert together.loc["m", "bias"] == pytest.approx(alone["bias"].item())
    assert alone["bias"].item() == pytest.approx(1 - 0.1)


def test_count_bias_is_a_study_setting() -> None:
    assert all(STUDY_COUNTS.values())
    s = score_forecasts(_samples("m", X, 3.0), counts=False)
    # Continuous form: 1 - 2 P(X <= y) = 1 - 2 * 0.7.
    assert s["bias"].item() == pytest.approx(1 - 1.4)


def test_forecast_share_counts_missing_forecasts() -> None:
    full = pd.concat([_rows("a", Q, 5.0, h=h) for h in (-1, 0)])
    half = _rows("b", Q, 5.0, h=0)
    share = forecast_share(pd.concat([full, half])).set_index("model")
    assert share.loc["a", "share"] == 1.0
    assert share.loc["b", "share"] == 0.5
    assert share.loc["b", "missing"] == 1
