"""The submission workflow's wiring of the hub models it may submit."""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_only_the_chronos_job_submits_chronos_and_only_when_on() -> None:
    workflow = ROOT / ".github" / "workflows" / "submit.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    prepare = jobs["prepare"]["outputs"]
    assert prepare["chronos"] == "${{ steps.tracks.outputs.chronos }}"
    assert prepare["chronos_model"] == (
        "${{ steps.tracks.outputs.chronos_model }}"
    )
    job = jobs["chronos"]
    assert "needs.prepare.outputs.chronos == 'true'" in job["if"]
    track = next(s for s in job["steps"] if s.get("id") == "track")
    assert track["run"] == "ci/steps/track.sh"
    assert track["env"]["TRACK_MODEL"] == (
        "${{ needs.prepare.outputs.chronos_model }}"
    )
    # Every other job that runs track.sh is the baseline's.
    runs = [
        name
        for name, j in jobs.items()
        for s in j.get("steps", [])
        if s.get("run") == "ci/steps/track.sh"
    ]
    assert sorted(runs) == ["baseline", "chronos"]
    assert "chronos" in jobs["record"]["needs"]
    assert "chronos" in jobs["outcome"]["needs"]


def _expand(field: str, low: int, high: int) -> set[int]:
    """Expand one cron field (`*`, `*/n`, `a-b`, lists) to its values."""
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, n = part.split("/")
            step = int(n)
        if part == "*":
            a, b = low, high
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
        else:
            a = b = int(part)
        out.update(range(a, b + 1, step))
    return out


def _schedule_minutes() -> list[int]:
    """Minutes of the week (Sunday 00:00 UTC = 0) the schedule fires."""
    workflow = ROOT / ".github" / "workflows" / "submit.yml"
    on = yaml.safe_load(workflow.read_text())[True]
    times = []
    for entry in on["schedule"]:
        minute, hour, dom, month, dow = entry["cron"].split()
        assert (dom, month) == ("*", "*")
        for d in _expand(dow, 0, 6):
            for h in _expand(hour, 0, 23):
                for m in _expand(minute, 0, 59):
                    times.append((d * 24 + h) * 60 + m)
    return sorted(times)


def test_the_window_is_checked_often_and_off_the_hour() -> None:
    # The hub merges a round into its tasks.json at any time in the window
    # (2026-11-21 merged Monday 14:37 UTC, nine minutes after that day's
    # only check), and GitHub delays or drops schedules at the top of the
    # hour by hours. So the window, Saturday 23:00 to Tuesday 22:59 UTC, is
    # checked at least every 3 hours, never on the hour.
    week = 7 * 24 * 60
    opens = (
        6 * 24 + 23
    ) * 60 - week  # Saturday 23:00, as minutes before Sunday
    closes = (2 * 24 + 22) * 60 + 59  # Tuesday 22:59
    times = [t - week if t >= opens + week else t for t in _schedule_minutes()]
    checks = sorted(t for t in times if opens <= t <= closes)
    edges = [opens, *checks, closes]
    gaps = [b - a for a, b in itertools.pairwise(edges)]
    assert max(gaps) <= 3 * 60, gaps
    assert all(t % 60 != 0 for t in checks), checks
