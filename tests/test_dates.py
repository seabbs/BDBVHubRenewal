from __future__ import annotations

import datetime as dt

import pytest

from bdbv_forecasts.dates import (
    HUB_TIMEZONE,
    epiweek_end,
    horizon_weeks,
    reference_date_for_vintage,
    submission_window,
    weeks_between,
)


@pytest.mark.parametrize(
    ("day", "end"),
    [
        ("2026-10-24", "2026-10-24"),  # Saturday
        ("2026-10-25", "2026-10-31"),  # Sunday starts a new week
        ("2026-10-09", "2026-10-10"),  # Friday
    ],
)
def test_epiweek_end(day: str, end: str) -> None:
    assert epiweek_end(day).isoformat() == end


@pytest.mark.parametrize(
    ("vintage", "ref"),
    [
        ("2026-10-24", "2026-10-31"),
        ("2026-10-17", "2026-10-24"),
        ("2026-10-09", "2026-10-17"),
        ("2026-08-18", "2026-08-22"),
    ],
)
def test_reference_date_for_vintage(vintage: str, ref: str) -> None:
    assert reference_date_for_vintage(vintage).isoformat() == ref


def test_submission_window_is_sunday_to_tuesday_in_kinshasa() -> None:
    start, end = submission_window("2026-10-31")
    assert start == dt.datetime(2026, 10, 25, 0, 0, tzinfo=HUB_TIMEZONE)
    assert end.date() == dt.date(2026, 10, 27)
    assert (end.hour, end.minute) == (23, 59)


def test_horizons() -> None:
    assert horizon_weeks("2026-10-31", "2026-11-21") == 3
    assert horizon_weeks("2026-10-31", "2026-10-03") == -4
    with pytest.raises(ValueError, match="Saturdays"):
        weeks_between("2026-10-30", "2026-10-31")
