# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Epi weeks, reference dates and submission windows.

The hub uses epi weeks that run Sunday to Saturday.
A week is labelled by its last day, a Saturday.
A round is labelled by its `reference_date`, the Saturday ending the epi
week in which a forecast is made.
Submissions open on the Sunday and close at the end of the Tuesday of
that week, in Kinshasa time (`submissions_due` from -6 to -4 days).
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

__all__ = [
    "HUB_TIMEZONE",
    "SUBMISSION_END_OFFSET",
    "SUBMISSION_START_OFFSET",
    "as_date",
    "epiweek_end",
    "horizon_weeks",
    "is_saturday",
    "reference_date_for_vintage",
    "submission_window",
    "weeks_between",
]

HUB_TIMEZONE = ZoneInfo("Africa/Kinshasa")
"""The hub's time zone, from `hub-config/admin.json`."""

SUBMISSION_START_OFFSET = -6
"""First day of the submission window, relative to the reference date."""

SUBMISSION_END_OFFSET = -4
"""Last day of the submission window, relative to the reference date."""

_SATURDAY = 5


def as_date(value: dt.date | str) -> dt.date:
    """Convert an ISO date string or a date to a `datetime.date`.

    Parameters
    ----------
    value : datetime.date or str
        A date, or an ISO 8601 string such as `"2026-10-24"`.

    Returns
    -------
    datetime.date
        The date. A `datetime.datetime` is reduced to its date.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


def is_saturday(value: dt.date | str) -> bool:
    """Whether a date is a Saturday, the last day of an epi week.

    Parameters
    ----------
    value : datetime.date or str
        The date to check.

    Returns
    -------
    bool
        True if the date is a Saturday.
    """
    return as_date(value).weekday() == _SATURDAY


def epiweek_end(value: dt.date | str) -> dt.date:
    """Return the Saturday that ends the epi week containing a date.

    Parameters
    ----------
    value : datetime.date or str
        Any date.

    Returns
    -------
    datetime.date
        The Saturday on or after `value`.
    """
    day = as_date(value)
    return day + dt.timedelta(days=(_SATURDAY - day.weekday()) % 7)


def weeks_between(start: dt.date | str, end: dt.date | str) -> int:
    """Count whole weeks from one Saturday to another.

    Parameters
    ----------
    start, end : datetime.date or str
        Saturdays.

    Returns
    -------
    int
        `(end - start) / 7`. Negative if `end` is before `start`.

    Raises
    ------
    ValueError
        If either date is not a Saturday.
    """
    first, last = as_date(start), as_date(end)
    if not (is_saturday(first) and is_saturday(last)):
        raise ValueError("Both dates must be Saturdays.")
    return (last - first).days // 7


def horizon_weeks(
    reference_date: dt.date | str, target_end_date: dt.date | str
) -> int:
    """Return the horizon of a target week in weeks from the reference date.

    Parameters
    ----------
    reference_date : datetime.date or str
        The round's reference date, a Saturday.
    target_end_date : datetime.date or str
        The target week's last day, a Saturday.

    Returns
    -------
    int
        Zero for the reference week, negative for earlier weeks (nowcasts)
        and positive for later weeks (forecasts).
    """
    return weeks_between(reference_date, target_end_date)


def submission_window(
    reference_date: dt.date | str,
) -> tuple[dt.datetime, dt.datetime]:
    """Return the submission window of a round, in Kinshasa time.

    Parameters
    ----------
    reference_date : datetime.date or str
        The round's reference date, a Saturday.

    Returns
    -------
    tuple of datetime.datetime
        The opening time (00:00 on the Sunday) and closing time (23:59:59
        on the Tuesday), both aware datetimes in `HUB_TIMEZONE`.
    """
    ref = as_date(reference_date)
    start_day = ref + dt.timedelta(days=SUBMISSION_START_OFFSET)
    end_day = ref + dt.timedelta(days=SUBMISSION_END_OFFSET)
    start = dt.datetime.combine(start_day, dt.time(0, 0), HUB_TIMEZONE)
    end = dt.datetime.combine(end_day, dt.time(23, 59, 59), HUB_TIMEZONE)
    return start, end


def reference_date_for_vintage(vintage: dt.date | str) -> dt.date:
    """Return the first reference date a data vintage can be submitted to.

    This is the earliest Saturday whose submission window closes on or
    after the vintage date.
    A vintage released on a Saturday serves the round ending a week later.
    One released on a Friday also serves the round ending eight days later,
    because its own round's window closed on the Tuesday.

    Parameters
    ----------
    vintage : datetime.date or str
        The `as_of` date of the data.

    Returns
    -------
    datetime.date
        The reference date, a Saturday.

    Examples
    --------
    >>> reference_date_for_vintage("2026-10-24")
    datetime.date(2026, 10, 31)
    >>> reference_date_for_vintage("2026-08-18")
    datetime.date(2026, 8, 22)
    """
    day = as_date(vintage)
    ref = epiweek_end(day)
    while ref + dt.timedelta(days=SUBMISSION_END_OFFSET) < day:
        ref += dt.timedelta(days=7)
    return ref
