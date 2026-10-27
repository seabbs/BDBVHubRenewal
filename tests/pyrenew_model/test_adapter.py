"""The model behind the `bdbv_hub_renewal` interfaces, simulated and real."""

import datetime as dt
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.data import load_delays, load_target
from bdbv_hub_renewal.models.base import ForecastTask, check_samples
from bdbv_hub_renewal.pyrenew import FALLBACK, MAIN
from bdbv_hub_renewal.pyrenew.adapter import (
    PyRenewProvinceModel,
    fit_renewal,
    forecast_renewal,
)

from .conftest import MAIN_REPO_DATA, SIM_LOCATIONS, VINTAGE_DAYS, vintage_date

pytestmark = pytest.mark.slow


def weeks(vintage: str, back: int, ahead: int) -> tuple[str, ...]:
    end = pd.Timestamp(vintage).date()
    end += dt.timedelta(days=(5 - end.weekday()) % 7)
    return tuple(
        (end + dt.timedelta(weeks=i)).isoformat()
        for i in range(-back, ahead + 1)
    )


def sim_task(outbreak, sim_tables):
    day = VINTAGE_DAYS[0]
    target, delays = sim_tables[day]
    vintage = vintage_date(outbreak, day)
    return ForecastTask(
        vintage=vintage,
        reference_date=pd.Timestamp(vintage).date(),
        kind="simulated",
        target=target,
        delays=delays,
        locations=SIM_LOCATIONS,
        target_end_dates=weeks(vintage, 6, 3),
    )


def real_task():
    target = load_target(MAIN_REPO_DATA)
    delays = load_delays(MAIN_REPO_DATA)
    vintage = target["vintage"].max()
    target = target.loc[target["vintage"] == vintage]
    return ForecastTask(
        vintage=vintage,
        reference_date=pd.Timestamp(vintage).date(),
        kind="real",
        target=target,
        delays=delays.loc[delays["vintage"] == vintage],
        locations=tuple(sorted(set(target["location"]) - {"CD"})),
        target_end_dates=weeks(vintage, 6, 3),
    )


def coherent(samples, unassigned, run):
    """Province samples plus unassigned equal the model's national draws."""
    total = samples.groupby(["target_end_date", "sample"])["value"].sum()
    if unassigned is not None:
        extra = unassigned.set_index(["target_end_date", "sample"])["value"]
        total = total.add(extra, fill_value=0)
    data, national = run["data"], run["draws"]["national"]
    first = data.start_date + dt.timedelta(days=6)
    for (week, k), value in total.items():
        i = (pd.Timestamp(week).date() - first).days // 7
        if i >= 0:
            assert value == pytest.approx(national[k - 1, i], abs=1e-9)


@pytest.mark.parametrize(
    ("config", "fallback"),
    [(FALLBACK, None), (MAIN, FALLBACK)],
    ids=["fallback", "main_with_fallback"],
)
def test_simulated_task(outbreak, sim_tables, table, config, fallback):
    task = sim_task(outbreak, sim_tables)
    model = PyRenewProvinceModel(
        config=config,
        table=table,
        fallback=fallback,
        num_warmup=300,
        num_samples=300,
    )
    start = time.perf_counter()
    samples, unassigned = model.forecast_with_unassigned(task)
    print(
        f"\nsimulated {config}: {time.perf_counter() - start:.1f} s, "
        f"failures {model.last_run['failures']}"
    )
    check_samples(samples, task)
    coherent(samples, unassigned, model.last_run)
    assert unassigned is None
    # Provinces with no cases are reported with importation draws.
    idle = samples.loc[samples["location"].isin(["CD-BU", "CD-SK"])]
    assert len(idle) and idle["value"].min() >= 0


def test_unreported_idle_provinces_go_to_unassigned(
    outbreak, sim_tables, table
):
    task = sim_task(outbreak, sim_tables)
    cfg = replace(FALLBACK, report_inactive=False)
    model = PyRenewProvinceModel(
        config=cfg, table=table, num_warmup=200, num_samples=200
    )
    samples, unassigned = model.forecast_with_unassigned(task)
    coherent(samples, unassigned, model.last_run)
    assert set(samples["location"]) == {"CD-IT", "CD-NK", "CD-HU", "CD-TO"}
    assert unassigned is not None
    assert set(unassigned.columns) == {"target_end_date", "sample", "value"}


@pytest.mark.skipif(not MAIN_REPO_DATA.exists(), reason="main repo data absent")
@pytest.mark.parametrize("config", [FALLBACK, MAIN], ids=["fallback", "main"])
def test_real_data_fit_contract(config, table, tmp_path):
    task = real_task()
    sampler = {
        "num_chains": 2,
        "num_warmup": 300,
        "num_samples": 300,
        "max_tree_depth": 10,
        "seed": 1,
    }
    start = time.perf_counter()
    record = fit_renewal(task, sampler, tmp_path, config, "x", table)
    seconds = time.perf_counter() - start
    forecast = forecast_renewal(task, tmp_path)
    check_samples(forecast.samples, task)
    ppc = json.loads((tmp_path / "ppc.json").read_text())
    assert ppc["available"]
    rows = [r for r in ppc["rows"] if r["location"] == "CD"]
    cover = np.mean(
        [r["lower_90"] <= r["observed"] <= r["upper_90"] for r in rows]
    )
    rhat = max(p["rhat"] for p in record["parameters"] if p["rhat"])
    div = sum(c["divergences"] for c in record["chains"])
    print(
        f"\nreal {config}: {seconds:.1f} s total, fit "
        f"{record['runtime_seconds']} s, max R-hat {rhat:.3f}, "
        f"divergences {div}, national PPC90 {cover:.2f}"
    )
    assert cover >= 0.75
