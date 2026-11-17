from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily_zones
from bdbvhubrenewal.data.zones import (
    hub_zones,
    is_zone,
    province_of,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.zones import reconcile_zones
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


def test_a_models_own_zones_stay_apart_from_its_provinces(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
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
    walk = bt.run_forecasts([get_model("randomwalk")], [task])
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    mine = tuple(z for z in zones if province_of(z) in task.locations)
    provinces = walk.loc[walk["location"].isin(set(task.locations))]
    # A model that forecasts each zone of its provinces as 1, summed to
    # its provinces as the Chronos-2 models give them.
    daily_zones = load_daily_zones(DATA)
    ones = pd.concat(
        provinces.loc[provinces["location"] == province_of(z)].assign(
            location=z, value=1.0
        )
        for z in mine
    )
    split = reconcile_zones(
        ones,
        provinces,
        task,
        weekly_zone_counts(daily_zones, task.vintage),
        mine,
    )
    own = pd.concat(
        [walk, split.assign(**walk.iloc[0][["kind", "vintage"]].to_dict())]
    ).assign(model="own")
    tables = hub_tables(pd.concat([walk, own]), task, daily_zones, zones)
    keys = ["location", "target_end_date", "output_type_id"]

    def places(name: str) -> pd.DataFrame:
        table = tables[name]
        rows = table.loc[~table["location"].map(is_zone)]
        return rows.sort_values(keys).reset_index(drop=True)

    # The country and provinces are as without the zones.
    pd.testing.assert_frame_equal(places("own"), places("randomwalk"))
    # The zones are the model's own, as they are.
    table = tables["own"]
    rows = table.loc[table["location"].map(is_zone)]
    got = rows.set_index(["location", "target_end_date", "output_type_id"])
    want = split.set_index(["location", "target_end_date", "sample"])
    assert np.array_equal(
        got["value"].sort_index().to_numpy(),
        want["value"].sort_index().to_numpy(),
    )


def test_a_model_that_forecasts_the_country_keeps_its_national_rows(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Chronos-2 models forecast the country in their joint block, and
    # their national draws are written as they are, not as the sum of the
    # provinces. Other models' national rows are the provinces' sum.
    from bdbvhubrenewal.forecasters.base import SAMPLE_COLUMNS
    from bdbvhubrenewal.reporting import hubtables

    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    task = bt.make_tasks(
        real.loc[real["vintage"] == latest],
        delays,
        bt.load_config(ROOT / "config" / "backtest.toml"),
        history=real,
        daily=daily,
    )[0]
    walk = bt.run_forecasts([get_model("randomwalk")], [task])
    national = walk["location"] == NATIONAL
    own = walk.assign(
        model="fm_chronos_daily_ragged_ft",
        value=walk["value"].where(~national, 7.0),
    )
    monkeypatch.setattr(
        hubtables,
        "zones_for_model",
        lambda *_a, **_k: pd.DataFrame(columns=list(SAMPLE_COLUMNS)),
    )
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    tables = hub_tables(
        pd.concat([walk, own]), task, load_daily_zones(DATA), zones
    )
    chronos = tables["fm_chronos_daily_ragged_ft"]
    assert (chronos.loc[chronos["location"] == NATIONAL, "value"] == 7).all()
    rw = tables["randomwalk"]
    wide = rw.pivot_table(
        index=["output_type_id", "target_end_date"],
        columns="location",
        values="value",
    )
    provinces = [c for c in wide.columns if c.count("-") == 1]
    assert (wide[NATIONAL] == wide[provinces].sum(axis=1)).all()
