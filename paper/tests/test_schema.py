import pandas as pd
import pytest

from paperfigs.schema import (
    FORECAST_COLUMNS,
    validate_forecasts,
    validate_observations,
)

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _forecast(**overrides) -> pd.DataFrame:
    rows = {
        "study": "synthetic",
        "model": "randomwalknowcast",
        "location": "sim01",
        "origin_date": pd.Timestamp("2024-01-10"),
        "target_date": pd.Timestamp("2024-01-09"),
        "horizon": -1,
        "quantile_level": list(LEVELS),
        "sample_id": pd.NA,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "truth": 3.0,
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_valid_forecasts_pass_unchanged() -> None:
    df = _forecast()
    out = validate_forecasts(df)
    assert list(out.columns) == list(FORECAST_COLUMNS)
    assert len(out) == len(df)


def test_missing_column_fails() -> None:
    with pytest.raises(ValueError, match="truth"):
        validate_forecasts(_forecast().drop(columns="truth"))


def test_extra_column_fails() -> None:
    with pytest.raises(ValueError, match="extra"):
        validate_forecasts(_forecast(extra=1))


def test_unknown_study_fails() -> None:
    with pytest.raises(ValueError, match="study"):
        validate_forecasts(_forecast(study="mars"))


def test_levels_out_of_range_fail() -> None:
    with pytest.raises(ValueError, match="quantile_level"):
        validate_forecasts(
            _forecast(quantile_level=[0.0, 0.25, 0.5, 0.75, 1.0])
        )


def test_negative_value_fails() -> None:
    with pytest.raises(ValueError, match="value"):
        validate_forecasts(_forecast(value=[-1.0, 2, 3, 4, 5]))


def test_crossing_quantiles_fail() -> None:
    with pytest.raises(ValueError, match="decrease"):
        validate_forecasts(_forecast(value=[1.0, 3, 2, 4, 5]))


def test_missing_median_fails() -> None:
    with pytest.raises(ValueError, match=r"0\.5"):
        validate_forecasts(
            _forecast(
                quantile_level=[0.05, 0.25, 0.6, 0.75, 0.95],
            )
        )


def test_duplicate_levels_fail() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_forecasts(pd.concat([_forecast(), _forecast()]))


def test_truth_must_be_constant_within_forecast() -> None:
    with pytest.raises(ValueError, match="truth"):
        validate_forecasts(_forecast(truth=[3.0, 3, 3, 3, 4]))


def test_missing_truth_is_allowed() -> None:
    validate_forecasts(_forecast(truth=float("nan")))


def test_observations_validate() -> None:
    obs = pd.DataFrame(
        {
            "study": ["ebola"],
            "location": ["loc01"],
            "target_date": [pd.Timestamp("2024-01-01")],
            "as_of_date": [pd.Timestamp("2024-01-08")],
            "value": [4.0],
        }
    )
    assert len(validate_observations(obs)) == 1
    with pytest.raises(ValueError, match="as_of_date"):
        validate_observations(obs.drop(columns="as_of_date"))


def _sample_forecast(**overrides) -> pd.DataFrame:
    rows = {
        "study": "ebola",
        "model": "m",
        "location": "loc01",
        "origin_date": pd.Timestamp("2024-01-10"),
        "target_date": pd.Timestamp("2024-01-17"),
        "horizon": 1,
        "quantile_level": float("nan"),
        "sample_id": [0, 1, 2, 3],
        "value": [3.0, 1.0, 4.0, 1.0],
        "truth": 2.0,
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_sample_forecasts_pass() -> None:
    out = validate_forecasts(_sample_forecast())
    assert out["sample_id"].tolist() == [0, 1, 2, 3]


def test_both_forms_on_one_row_fail() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        validate_forecasts(_sample_forecast(quantile_level=0.5))


def test_neither_form_fails() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        validate_forecasts(_sample_forecast(sample_id=pd.NA))


def test_mixed_forms_in_one_forecast_fail() -> None:
    q = _forecast(
        study="ebola",
        model="m",
        location="loc01",
        target_date=pd.Timestamp("2024-01-17"),
        horizon=1,
    )
    with pytest.raises(ValueError, match="one form"):
        validate_forecasts(pd.concat([q, _sample_forecast()]))


def test_duplicate_sample_ids_fail() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_forecasts(_sample_forecast(sample_id=[0, 0, 1, 2]))
