"""The hub's submission window, in Kinshasa time.

`hub-config/tasks.json` sets `submissions_due` to -6 to -4 days relative
to the `reference_date` (a Saturday), in the hub's timezone
(`Africa/Kinshasa`, from `admin.json`). That is Sunday 00:00 to Tuesday
23:59 of the reference week. Kinshasa is UTC+1 with no daylight saving,
so a fixed offset is exact and needs no tz database on the runner.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

KINSHASA = dt.timezone(dt.timedelta(hours=1), "WAT")
SATURDAY = 5


@dataclass(frozen=True)
class Window:
    """One round's submission window."""

    reference_date: dt.date
    opens: dt.datetime
    closes: dt.datetime  # exclusive

    def is_open(self, now: dt.datetime) -> bool:
        """Whether `now` falls inside the window."""
        return self.opens <= now < self.closes

    def hours_left(self, now: dt.datetime) -> float:
        """Hours until the window closes (negative once closed)."""
        return (self.closes - now).total_seconds() / 3600


def window_for(reference_date: dt.date) -> Window:
    """Return the window of the round with this reference date."""
    if reference_date.weekday() != SATURDAY:
        raise ValueError(f"{reference_date} is not a Saturday")
    opens = dt.datetime.combine(
        reference_date - dt.timedelta(days=6), dt.time(), KINSHASA
    )
    closes = dt.datetime.combine(
        reference_date - dt.timedelta(days=3), dt.time(), KINSHASA
    )
    return Window(reference_date, opens, closes)


def current_window(now: dt.datetime) -> Window:
    """Return the window open at `now`, or the next one to open."""
    local = now.astimezone(KINSHASA).date()
    ref = local + dt.timedelta(days=(SATURDAY - local.weekday()) % 7)
    win = window_for(ref)
    if now >= win.closes:
        win = window_for(ref + dt.timedelta(days=7))
    return win


def minutes_left(
    reference_date: str, now: dt.datetime, *, ignore_window: bool = False
) -> float | None:
    """Minutes until the round's window closes, or None without a limit.

    There is no limit when no round is given or the window is ignored
    (dry runs on past rounds).
    """
    if not reference_date or ignore_window:
        return None
    win = window_for(dt.date.fromisoformat(reference_date))
    return (win.closes - now).total_seconds() / 60


def parse_now(value: str | None) -> dt.datetime:
    """Parse an ISO timestamp, defaulting to the current UTC time."""
    if not value:
        return dt.datetime.now(dt.UTC)
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed
