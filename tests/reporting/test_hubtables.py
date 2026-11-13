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
DATA = ROOT / "tests" / "fixtures" / "data"


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


def test_hub_quantiles_take_other_levels() -> None:
    table = pd.DataFrame(
        {
            "location": "DE",
            "output_type": "sample",
            "output_type_id": np.arange(1, 101),
            "value": np.arange(100.0),
        }
    )
    levels = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    out = hub_quantiles(table, levels)
    assert list(out["output_type_id"]) == pytest.approx(list(levels))
    want = np.quantile(np.arange(100.0), levels)
    assert list(out["value"]) == pytest.approx(list(want))


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
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    tables = hub_tables(forecasts, task, load_daily_zones(DATA), zones)
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


def test_hub_tables_take_a_zone_fit_s_own_zones(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
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
    forecasts = forecasts.assign(model="renewal")
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    zone = next(z for z in zones if z.startswith(task.locations[0] + "-"))
    # The zone fit puts all of the first province's cases in one zone.
    first = forecasts.loc[
        (forecasts["location"] == task.locations[0]) & forecasts["as_of"].isna()
        if "as_of" in forecasts
        else forecasts["location"] == task.locations[0]
    ]
    fitted = first.assign(location=zone, vintage=task.vintage)
    daily_zones = load_daily_zones(DATA)
    table = hub_tables(forecasts, task, daily_zones, zones, fitted)["renewal"]
    rows = table.loc[table["location"].isin(set(zones))]
    assert set(rows["location"]) == {zone}
    mine = table.loc[table["location"] == task.locations[0]]
    keys = ["target_end_date", "output_type_id"]
    pd.testing.assert_series_equal(
        rows.set_index(keys)["value"].sort_index(),
        mine.set_index(keys)["value"].sort_index(),
        check_names=False,
    )
    # Without its zone fit, a model's provinces are split.
    split = hub_tables(forecasts, task, daily_zones, zones)["renewal"]
    assert len(set(split["location"]) & set(zones)) > 1
