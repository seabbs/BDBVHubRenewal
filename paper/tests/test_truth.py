import numpy as np
import pandas as pd
import pytest

from paperfigs.truth import attach_truth, rolling_truth


def _obs() -> pd.DataFrame:
    # Two vintages a week apart; the count for 1 Jan grows between them.
    d = pd.Timestamp
    return pd.DataFrame(
        {
            "study": "ebola",
            "location": "loc01",
            "target_date": [d("2024-01-01"), d("2024-01-01"), d("2024-01-08")],
            "as_of_date": [d("2024-01-05"), d("2024-01-12"), d("2024-01-12")],
            "value": [3.0, 5.0, 2.0],
        }
    )


def test_truth_is_latest_release_within_d_days() -> None:
    t = rolling_truth(_obs(), delay_days=6).set_index("target_date")
    # 1 Jan + 6 days = 7 Jan: the 5 Jan release is the latest by then.
    assert t.loc[pd.Timestamp("2024-01-01"), "truth"] == 3.0
    t = rolling_truth(_obs(), delay_days=11).set_index("target_date")
    assert t.loc[pd.Timestamp("2024-01-01"), "truth"] == 5.0


def test_truth_is_missing_until_d_days_have_passed() -> None:
    t = rolling_truth(_obs(), delay_days=6).set_index("target_date")
    # 8 Jan + 6 = 14 Jan is after the last release (12 Jan).
    assert np.isnan(t.loc[pd.Timestamp("2024-01-08"), "truth"])


def test_attach_truth_replaces_the_truth_column() -> None:
    fc = pd.DataFrame(
        {
            "study": "ebola",
            "model": "m",
            "location": "loc01",
            "origin_date": pd.Timestamp("2024-01-08"),
            "target_date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-08"),
            ],
            "horizon": [-1, 0],
            "quantile_level": np.nan,
            "sample_id": [0, 0],
            "value": [1.0, 1.0],
            "truth": np.nan,
        }
    )
    out = attach_truth(fc, rolling_truth(_obs(), delay_days=11))
    assert out["truth"].iloc[0] == 5.0
    assert np.isnan(out["truth"].iloc[1])
    assert list(out.columns) == list(fc.columns)


def test_negative_delay_fails() -> None:
    with pytest.raises(ValueError, match="delay_days"):
        rolling_truth(_obs(), delay_days=-1)
