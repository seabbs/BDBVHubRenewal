# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Health zones: the hub's zone codes and zone tables.

Zone codes are `CD-XX-NN`, with `CD-XX` the province
(`bdbvhubrenewal.data.hierarchy`).
The hub lists all 519 as optional locations in `hub-config/tasks.json`.
Weekly zone counts come from the hub's daily zone onsets
(`daily_zones.csv.gz`) through `bdbvhubrenewal.data.snapshots.weekly_counts`,
known plus imputed.
They equal the zone rows of the hub's `time-series.csv`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data.hierarchy import is_zone
from bdbvhubrenewal.data.tidy import load_daily_zones, load_health_zones

__all__ = [
    "ZoneData",
    "hub_zones",
    "load_zone_data",
]


@dataclass(frozen=True)
class ZoneData:
    """The health-zone tables a forecast task carries.

    Attributes
    ----------
    weekly : pandas.DataFrame
        Weekly zone counts by release (`load_health_zones`).
    daily : pandas.DataFrame
        Daily zone onsets by release (`load_daily_zones`).
    codes : tuple of str
        The hub's zone codes (`hub_zones`).
    """

    weekly: pd.DataFrame
    daily: pd.DataFrame
    codes: tuple[str, ...]


def load_zone_data(data_dir: Path | str) -> ZoneData:
    """Load the zone tables and codes of a data directory.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Directory holding the tidy tables and `hub-config/tasks.json`.

    Returns
    -------
    ZoneData
    """
    return ZoneData(
        load_health_zones(data_dir),
        load_daily_zones(data_dir),
        hub_zones(Path(data_dir) / "hub-config" / "tasks.json"),
    )


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
    return tuple(sorted(c for c in codes if is_zone(c)))


def zone_places(zones: tuple[str, ...]) -> pd.DataFrame:
    """Stub."""
    raise NotImplementedError
