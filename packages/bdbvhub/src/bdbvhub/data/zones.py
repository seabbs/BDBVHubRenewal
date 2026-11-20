# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Health zones: the hub's zone codes and zone tables.

Zone codes are `CD-XX-NN`, with `CD-XX` the province
(`nowcastutils.hierarchy`).
The hub lists all 519 as optional locations in `hub-config/tasks.json`.
Weekly zone counts come from the hub's daily zone onsets
(`daily_zones.csv.gz`) through `nowcastutils.snapshots.weekly_counts`,
known plus imputed.
They equal the zone rows of the hub's `time-series.csv`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

import pandas as pd

from bdbvhub.data.tidy import (
    load_daily_zones,
    load_health_zones,
    load_locations,
)
from nowcastutils.hierarchy import is_zone

__all__ = [
    "PLACE_COLUMNS",
    "ZoneData",
    "hub_places",
    "hub_zones",
    "load_zone_data",
    "zone_places",
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


PLACE_COLUMNS: Final = ("location", "population", "latitude", "longitude")
"""The columns of a task's places (`ForecastTask.places`)."""


def zone_places() -> pd.DataFrame:
    """Health zone populations and centres, as places.

    Centroids and WorldPop counts of the hub's mapped health zones, from
    `zone_centres.csv` (see its header). A zone the map lacks has no
    row. The zone model places it in its province
    (`forecasters.zonerenewal`).

    Returns
    -------
    pandas.DataFrame
        Columns `PLACE_COLUMNS`.
    """
    path = files("bdbvhub.data").joinpath("zone_centres.csv")
    with path.open() as f:
        table = pd.read_csv(f, comment="#")
    table = table.rename(columns={"worldpop": "population"})
    return table.loc[:, list(PLACE_COLUMNS)]


def hub_places(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the places of the hub's tasks: country, provinces and zones.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables (`load_locations`).

    Returns
    -------
    pandas.DataFrame
        Columns `PLACE_COLUMNS`: the country and provinces
        (`load_locations`), then the mapped zones (`zone_places`).
    """
    provinces = load_locations(data_dir).loc[:, list(PLACE_COLUMNS)]
    return pd.concat([provinces, zone_places()], ignore_index=True)
