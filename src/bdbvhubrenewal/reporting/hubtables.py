# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Every model's forecast at the latest release in the hub's format.

The results release carries these tables, so every model, submitted or
not, is archived with each hub release. For each model `hub_tables`
gives the hub sample table, with the country, the provinces and the
health zones, as `bdbvhubrenewal nowcast` writes it for a submission,
and the same forecast as hub quantiles.
The zones come from `zone_samples`, as in a submission, which fits no
model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.forecasters import ForecastTask
from bdbvhubrenewal.forecasters.base import NATIONAL, SAMPLE_COLUMNS
from bdbvhubrenewal.submission.hub import to_hub_table, zone_samples

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


def hub_quantiles(table: pd.DataFrame) -> pd.DataFrame:
    """Return a hub sample table as hub quantiles.

    Parameters
    ----------
    table : pandas.DataFrame
        A hub sample table (`to_hub_table`).

    Returns
    -------
    pandas.DataFrame
        The same columns, with `output_type` `quantile` and
        `output_type_id` the level, one row per location, week and level.
    """
    keys = [c for c in table.columns if c not in ("output_type_id", "value")]
    keys.remove("output_type")
    levels = np.array(HUB_QUANTILE_LEVELS)
    out = (
        table.groupby(keys)["value"]
        .quantile(levels)
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
    provinces = rows.loc[rows["location"] != NATIONAL].groupby(keys)["value"]
    gap = (national - provinces.sum()).fillna(national).clip(lower=0)
    return gap.rename("value").reset_index()


def hub_tables(
    forecasts: pd.DataFrame,
    task: ForecastTask,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
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

    Returns
    -------
    dict
        Hub sample table by model.
    """
    counts = weekly_zone_counts(daily_zones, task.vintage)
    rows = forecasts.loc[forecasts["vintage"] == task.vintage]
    if "as_of" in rows:
        rows = rows.loc[rows["as_of"].isna()]
    out = {}
    for name, own in rows.groupby("model"):
        provinces = own.loc[own["location"] != NATIONAL, list(SAMPLE_COLUMNS)]
        split = zone_samples(
            str(name), task, provinces, counts, zones, daily_zones
        )
        out[str(name)] = to_hub_table(
            provinces,
            task.reference_date,
            unassigned=_unassigned(own),
            zones=split,
            forecast_date=task.vintage,
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
