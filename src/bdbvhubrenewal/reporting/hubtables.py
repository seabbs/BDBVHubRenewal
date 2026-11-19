# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Every model's forecast at the latest release in the hub's format.

The results release carries these tables, so every model, submitted or
not, is archived with each hub release. For each model `hub_tables`
gives the hub sample table, with the country, the provinces and the
health zones, as `bdbvhubrenewal nowcast` writes it for a submission,
and the same forecast as hub quantiles.
As in a submission, the zone fits of `renewal` and `renewalbvd` give
their own zone samples, and the zones of every other model come from
`zones_for_model`, which fits no model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.files import write_csv
from bdbvhubrenewal.data.hierarchy import is_zone
from bdbvhubrenewal.data.snapshots import weekly_counts
from bdbvhubrenewal.forecasters import ForecastTask
from bdbvhubrenewal.forecasters.base import NATIONAL, SAMPLE_COLUMNS
from bdbvhubrenewal.forecasters.zones import zones_for_model
from bdbvhubrenewal.submission.hub import (
    own_national,
    to_hub_table,
    zones_need_not_sum,
)

__all__ = [
    "HUB_QUANTILE_LEVELS",
    "hub_quantiles",
    "hub_tables",
    "write_hub_tables",
]

HUB_QUANTILE_LEVELS: Final = (
    0.01,
    0.025,
    *np.round(np.arange(0.05, 0.951, 0.05), 2),
    0.975,
    0.99,
)
"""The 23 quantile levels of hubverse quantile forecasts."""


def hub_quantiles(
    table: pd.DataFrame, levels: Sequence[float] = HUB_QUANTILE_LEVELS
) -> pd.DataFrame:
    """Return a hub sample table as hub quantiles.

    Parameters
    ----------
    table : pandas.DataFrame
        A hub sample table (`to_hub_table`).
    levels : sequence of float
        The quantile levels, by default the hub's.

    Returns
    -------
    pandas.DataFrame
        The same columns, with `output_type` `quantile` and
        `output_type_id` the level, one row per location, week and level.
    """
    keys = [c for c in table.columns if c not in ("output_type_id", "value")]
    keys.remove("output_type")
    out = (
        table.groupby(keys)["value"]
        .quantile(np.array(levels))
        .rename_axis([*keys, "output_type_id"])
        .reset_index()
    )
    out.insert(len(keys), "output_type", "quantile")
    return out[list(table.columns)]


def _unassigned(rows: pd.DataFrame) -> pd.DataFrame | None:
    """Return national minus the province sum, by week and sample."""
    keys = ["target_end_date", "sample"]
    national = rows.loc[rows["location"] == NATIONAL].set_index(keys)["value"]
    if national.empty:
        return None
    provinces = rows.loc[
        (rows["location"] != NATIONAL) & ~rows["location"].map(is_zone)
    ].groupby(keys)["value"]
    gap = (national - provinces.sum()).fillna(national).clip(lower=0)
    return gap.rename("value").reset_index()


def hub_tables(
    forecasts: pd.DataFrame,
    task: ForecastTask,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
    fitted: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Return each model's hub sample table at one task.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `run_forecasts` output of every model at `task`, with the
        national rows.
    task : ForecastTask
        The task.
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    zones : tuple of str
        The hub's zone codes.
    fitted : pandas.DataFrame, optional
        The zone fits' zone samples (`evaluation.store.ZONE_FITS_FILE`),
        with `model` and `vintage`. A model with rows at the task's
        vintage takes them in place of `zones_for_model`.

    Returns
    -------
    dict
        Hub sample table by model.
    """
    counts = weekly_counts(daily_zones, task.vintage)
    rows = forecasts.loc[forecasts["vintage"] == task.vintage]
    if "as_of" in rows:
        rows = rows.loc[rows["as_of"].isna()]
    out = {}
    for name, own in rows.groupby("model"):
        places = own.loc[own["location"] != NATIONAL, list(SAMPLE_COLUMNS)]
        provinces = places.loc[~places["location"].map(is_zone)]
        mine = (
            None
            if fitted is None
            else fitted.loc[
                (fitted["model"] == name)
                & (fitted["vintage"].astype(str) == task.vintage)
                & fitted["location"].isin(zones)
                & fitted["target_end_date"]
                .astype(str)
                .isin(task.target_end_dates)
            ]
        )
        if mine is not None and not mine.empty:
            split = mine.loc[:, list(SAMPLE_COLUMNS)].reset_index(drop=True)
        else:
            split = zones_for_model(
                str(name), task, places, counts, zones, daily_zones
            )
        # A model that forecasts the country keeps its national rows.
        country_own = own_national(str(name))
        country = own.loc[own["location"] == NATIONAL, list(SAMPLE_COLUMNS)]
        out[str(name)] = to_hub_table(
            pd.concat([provinces, country]) if country_own else provinces,
            task.reference_date,
            unassigned=None if country_own else _unassigned(own),
            zones=split,
            forecast_date=task.vintage,
            zones_sum=not zones_need_not_sum(str(name)),
            national="own" if country_own else "sum",
        )
    return out


def write_hub_tables(
    tables: Mapping[str, pd.DataFrame], out: Path | str
) -> list[Path]:
    """Write each model's hub samples and quantiles.

    Parameters
    ----------
    tables : mapping
        `hub_tables`.
    out : pathlib.Path or str
        The directory.

    Returns
    -------
    list of pathlib.Path
        `<model>-samples.csv.gz` and `<model>-quantiles.csv.gz` of each
        model.
    """
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        for kind, frame in (
            ("samples", table),
            ("quantiles", hub_quantiles(table)),
        ):
            path = root / f"{name}-{kind}.csv.gz"
            write_csv(frame, path)
            written.append(path)
    return written
