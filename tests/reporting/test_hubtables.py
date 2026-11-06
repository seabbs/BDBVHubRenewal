from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.reporting.hubtables import (
    HUB_QUANTILE_LEVELS,
    hub_quantiles,
    hub_tables,
    write_hub_tables,
)

ROOT = Path(__file__).resolve().parents[2]


def test_hub_quantiles_give_the_23_levels() -> None:
    table = pd.DataFrame(
        {
            "reference_date": "2026-11-07",
            "target": "wk inc confirmed cases",
            "location": "CD",
            "output_type": "sample",
            "output_type_id": np.arange(1, 101),
            "value": np.arange(100.0),
        }
    )
    out = hub_quantiles(table)
    assert len(HUB_QUANTILE_LEVELS) == 23
    assert list(out.columns) == list(table.columns)
    assert set(out["output_type"]) == {"quantile"}
    assert list(out["output_type_id"]) == pytest.approx(
        list(HUB_QUANTILE_LEVELS)
    )
    assert out["value"].is_monotonic_increasing


def test_hub_tables_cover_country_provinces_and_zones(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    tmp_path: Path,
) -> None:
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    task = bt.make_tasks(
        real.loc[real["vintage"] == latest],
        delays,
        bt.load_config(ROOT / "config" / "backtest.toml"),
        history=real,
        daily=daily,
    )[0]
    forecasts = bt.run_forecasts([get_model("randomwalk")], [task])
    zones = hub_zones(ROOT / "data" / "hub-config" / "tasks.json")
    tables = hub_tables(forecasts, task, load_daily_zones(ROOT / "data"), zones)
    table = tables["randomwalk"]
    locations = set(table["location"])
    assert NATIONAL in locations
    assert locations & set(zones)
    assert set(table["output_type_id"]) == set(range(1, 101))
    # National is at least the province sum, sample by sample.
    keys = ["target_end_date", "output_type_id"]
    provinces = (
        table.loc[table["location"].isin(set(task.locations))]
        .groupby(keys)["value"]
        .sum()
    )
    national = table.loc[table["location"] == NATIONAL].set_index(keys)
    assert (national["value"] >= provinces.loc[national.index]).all()
    written = write_hub_tables(tables, tmp_path)
    assert sorted(p.name for p in written) == [
        "randomwalk-quantiles.csv.gz",
        "randomwalk-samples.csv.gz",
    ]
