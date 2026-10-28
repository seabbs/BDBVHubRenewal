# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Health zones: the hub's zone codes and weekly zone counts.

Zone codes are `CD-XX-NN`, with `CD-XX` the province.
The hub lists all 519 as optional locations in `hub-config/tasks.json`.
Weekly counts come from the hub's daily zone onsets
(`daily_zones.csv.gz`), summed by epi week, known plus imputed.
They equal the zone rows of the hub's `time-series.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = ["hub_zones", "province_of", "weekly_zone_counts"]


def province_of(zone: str) -> str:
    """Return the province code of a zone code.

    Parameters
    ----------
    zone : str
        A zone code, `CD-XX-NN`.

    Returns
    -------
    str
        `CD-XX`.
    """
    return zone[:5]


def hub_zones(tasks_json: Path | str) -> tuple[str, ...]:
    """Return the zone codes the hub accepts as locations.

    Parameters
    ----------
    tasks_json : pathlib.Path or str
        Path to the hub's `hub-config/tasks.json`.

    Returns
    -------
    tuple of str
        Sorted zone codes from every round's location options.
    """
    config = json.loads(Path(tasks_json).read_text())
    codes: set[str] = set()
    for rnd in config["rounds"]:
        for task in rnd["model_tasks"]:
            loc = task["task_ids"].get("location", {})
            for key in ("required", "optional"):
                codes.update(loc.get(key) or [])
    return tuple(sorted(c for c in codes if len(c.split("-")) == 3))


def weekly_zone_counts(daily_zones: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Sum one vintage's daily zone onsets by epi week.

    Parameters
    ----------
    daily_zones : pandas.DataFrame
        The tidy daily zone table (`load_daily_zones`), with columns
        `vintage`, `location`, `onset_date`, `observed` and `imputed`.
    vintage : str
        The vintage, ISO format.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date` (the Saturday ending the
        onset week) and `total`, for weeks with at least one onset.
    """
    rows = daily_zones.loc[daily_zones["vintage"].astype(str) == vintage]
    onset = pd.to_datetime(rows["onset_date"])
    end = onset + pd.to_timedelta((5 - onset.dt.weekday) % 7, unit="D")
    out = (
        rows.assign(
            target_end_date=end.dt.date.astype(str),
            total=rows["observed"] + rows["imputed"],
        )
        .groupby(["location", "target_end_date"], as_index=False)["total"]
        .sum()
    )
    kept: pd.DataFrame = out.loc[out["total"] > 0]
    return kept.reset_index(drop=True)
