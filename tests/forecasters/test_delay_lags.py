"""Release lags chosen from the reporting delay (forecaster.lag_count)."""

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcastfoundation import forecaster
from nowcastfoundation.forecaster import (
    LAG_QUANTILE,
    LAG_WINDOW,
    MAX_LAG,
    lag_count,
)
from nowcastutils.base import get_model

# Of each day's 100 cases, 50 are reported on the day, 80 by the next day
# and all 100 three days after it.
REPORTED = {0: 50, 1: 80, 2: 80, 3: 100}


def _releases(
    days: int, start: str = "2026-01-01", reported=None
) -> pd.DataFrame:
    """Daily releases of one place, each giving every onset day so far.

    `reported(d)` is a day's count reported by lag `d` (`REPORTED` by
    default).
    """
    reported = reported or (lambda d: REPORTED.get(d, 100))
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
                    "observed": reported(v - t),
                    "imputed": 0,
                    "cutoff": vintage,
                }
            )
    return pd.DataFrame(rows)


def delay_lags(rows, source, places, last, quantile=LAG_QUANTILE):
    """The lag count of releases `rows`, as a forecast reads them."""
    tri = forecaster.hub_triangle(
        rows, source, places, last, max_lag=LAG_WINDOW[source][1]
    )
    return lag_count(tri, source, quantile)


@pytest.mark.parametrize(
    ("quantile", "lags"), [(0.5, 1), (0.8, 2), (0.99, 4), (1.0, 4)]
)
def test_the_lags_hold_the_quantile_of_the_reports(quantile, lags) -> None:
    rows = _releases(200)
    last = rows["cutoff"].max()
    assert delay_lags(rows, "daily", ("A",), last, quantile) == lags


def test_the_default_is_the_99th_percentile() -> None:
    rows = _releases(200)
    assert LAG_QUANTILE == 0.99
    assert delay_lags(rows, "daily", ("A",), rows["cutoff"].max()) == 4


def test_only_releases_up_to_the_forecast_date_are_read() -> None:
    rows = _releases(200)
    # Later releases change nothing: the triangle ends at the last one.
    early = rows.loc[rows["vintage"] <= "2026-06-01"]
    assert delay_lags(early, "daily", ("A",), "2026-06-01") == 4


def test_a_long_delay_with_a_recent_cut_is_not_biased_short() -> None:
    # A geometric delay: the share reported by lag d is 1 - 0.8^(d + 1),
    # whose 99th percentile is lag 20, so 21 lags. Recent days have not
    # seen their delays; only complete days are read, so the estimate is
    # not short.
    def reported(d):
        return 10_000 * (1 - 0.8 ** (d + 1))

    rows = _releases(200, reported=reported)
    last = rows["cutoff"].max()
    assert delay_lags(rows, "daily", ("A",), last, 0.99) == 21
    # Naively, from the latest 28 days and their counts so far, it is
    # shorter: those days have not been reported for long enough.
    tri = forecaster.hub_triangle(rows, "daily", ("A",), last, max_lag=28)
    days = list(range(tri.last - 27, tri.last + 1))
    part, final = tri.partial[0, days, :], tri.final[0, days]
    seen = ~np.isnan(part)
    share = [np.nansum(part[:, d]) / final[seen[:, d]].sum() for d in range(28)]
    naive = int(np.flatnonzero(np.array(share) >= 0.99)[0]) + 1
    assert naive < 21


def test_little_history_gives_as_many_lags_as_it_allows() -> None:
    rows = _releases(20)
    got = delay_lags(rows, "daily", ("A",), rows["cutoff"].max())
    tri = forecaster.hub_triangle(
        rows, "daily", ("A",), rows["cutoff"].max(), max_lag=1
    )
    assert got == max(1, tri.last - 1) == 18
    assert got != MAX_LAG["daily"]


def test_the_lags_never_exceed_the_history() -> None:
    # 99% of reports by lag 20 needs 21 lags; 15 days of history allow 13.
    def reported(d):
        return 10_000 * (1 - 0.8 ** (d + 1))

    rows = _releases(15, reported=reported)
    last = rows["cutoff"].max()
    assert delay_lags(rows, "daily", ("A",), last) == 13
    model = dataclasses.replace(get_model("fm_chronos_daily_ragged"), lags=40)
    assert model._triangle(rows, ("A",), last).partial.shape[2] == 13


def test_a_fixed_lag_count_overrides_the_rule() -> None:
    rows = _releases(200)
    last = rows["cutoff"].max()
    model = get_model("fm_chronos_daily_ragged")
    assert model.lags is None
    assert model._triangle(rows, ("A",), last).partial.shape[2] == 4
    fixed = dataclasses.replace(model, lags=7)
    assert fixed._triangle(rows, ("A",), last).partial.shape[2] == 7


def test_the_block_takes_the_largest_place_delay() -> None:
    # Place A reports within 4 days, place B within 21: the block takes
    # B's, so neither loses its late reports.
    a = _releases(200)

    def slow(d):
        return 10_000 * (1 - 0.8 ** (d + 1))

    b = _releases(200, reported=slow).assign(location="B")
    rows = pd.concat([a, b], ignore_index=True)
    last = rows["cutoff"].max()
    assert delay_lags(a, "daily", ("A",), last) == 4
    assert delay_lags(rows, "daily", ("A", "B"), last) == 21
