"""Release lags chosen from the reporting delay (forecaster.delay_lags)."""

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation import forecaster
from nowcastfoundation.forecaster import DELAY_QUANTILE, MAX_LAG, delay_lags
from nowcastutils.base import get_model

# Of each day's 100 cases, 50 are reported on the day, 80 by the next day
# and all 100 three days after it.
REPORTED = {0: 50, 1: 80, 2: 80, 3: 100}


def _releases(days: int, start: str = "2026-01-01") -> pd.DataFrame:
    """Daily releases of one place, each giving every onset day so far."""
    first = dt.date.fromisoformat(start)
    rows = []
    for v in range(days):
        vintage = (first + dt.timedelta(days=v)).isoformat()
        for t in range(v + 1):
            rows.append(
                {
                    "vintage": vintage,
                    "location": "A",
                    "onset_date": (first + dt.timedelta(days=t)).isoformat(),
                    "observed": REPORTED.get(v - t, 100),
                    "imputed": 0,
                    "cutoff": vintage,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("quantile", "lags"), [(0.5, 1), (0.8, 2), (0.99, 4), (1.0, 4)]
)
def test_the_lags_hold_the_quantile_of_the_reports(quantile, lags) -> None:
    rows = _releases(200)
    last = rows["cutoff"].max()
    assert delay_lags(rows, "daily", ("A",), last, quantile) == lags


def test_the_default_is_the_99th_percentile() -> None:
    rows = _releases(200)
    assert DELAY_QUANTILE == 0.99
    assert delay_lags(rows, "daily", ("A",), rows["cutoff"].max()) == 4


def test_only_releases_up_to_the_forecast_date_are_read() -> None:
    rows = _releases(200)
    # Later releases change nothing: the triangle ends at the last one.
    early = rows.loc[rows["vintage"] <= "2026-06-01"]
    assert delay_lags(early, "daily", ("A",), "2026-06-01") == 4


def test_a_short_history_keeps_the_fixed_lags() -> None:
    rows = _releases(20)
    got = delay_lags(rows, "daily", ("A",), rows["cutoff"].max())
    assert got == MAX_LAG["daily"]


def test_a_model_takes_a_fixed_lag_count_or_a_quantile(monkeypatch) -> None:
    seen = []
    real = forecaster.hub_triangle

    def spy(*args, **kwargs):
        seen.append(kwargs.get("max_lag"))
        return real(*args, **kwargs)

    model = get_model("fm_chronos_daily_ragged")
    assert model.max_lag is None
    assert model.delay_quantile is None
    fixed = dataclasses.replace(model, max_lag=7)
    assert fixed.lags(_releases(200), ("A",), "2026-07-19") == 7
    by_q = dataclasses.replace(model, delay_quantile=0.8)
    assert by_q.lags(_releases(200), ("A",), "2026-07-19") == 2
    monkeypatch.setattr(forecaster, "hub_triangle", spy)
    assert model.lags(_releases(200), ("A",), "2026-07-19") == 4
    assert np.all(np.array(seen) > 4)
