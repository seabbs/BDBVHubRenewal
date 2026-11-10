"""Draw randomwalknowcast province and zone samples at the prototype origins.

Run from the repo root:

    uv run python evidence/coherent-reconciliation-review/draws.py OUT

Writes to OUT:
- `province.csv.gz`: the direct province forecast (100 samples, as in the
  release backtest), national included.
- `zones.csv.gz`: the zone forecast on the zone task with `N_POOL`
  samples, a larger pool for importance weighting.
- `delays.csv.gz`: the province run's delay summary, to map final counts
  to later releases.
- `zone_counts.csv.gz`: the reported zone counts at each origin.
- `zone_delays.csv.gz`: the zone run's delay summary.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import (
    load_daily,
    load_daily_zones,
    load_delays,
    load_target,
)
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.forecasters.base import get_model
from bdbvhubrenewal.forecasters.zones import zone_task

ORIGINS = ("2026-10-17", "2026-10-24", "2026-11-01", "2026-11-08")
MODEL = "randomwalknowcast"
N_POOL = 2000


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = "data"
    target, delays = load_target(data), load_delays(data)
    daily, daily_zones = load_daily(data), load_daily_zones(data)
    zones = hub_zones(Path(data) / "hub-config" / "tasks.json")
    config = bt.load_config()
    tasks = store.origin_tasks(target, delays, config, daily)
    prov, zone, summ, counts_all, zsumm = [], [], [], [], []
    for origin in ORIGINS:
        task = tasks[origin]
        model = get_model(MODEL)
        p = model.forecast(task)
        summ.append(model.delay_summary.assign(vintage=origin))
        counts = weekly_zone_counts(daily_zones, origin)
        active = tuple(z for z in zones if z in set(counts["location"]))
        zt = zone_task(task, counts, active, pd.concat([daily, daily_zones]))
        zt = dataclasses.replace(zt, n_samples=N_POOL)
        zmodel = get_model(MODEL)
        z = zmodel.forecast(zt)
        zsumm.append(zmodel.delay_summary.assign(vintage=origin))
        prov.append(p.assign(vintage=origin))
        zone.append(z.assign(vintage=origin))
        counts_all.append(counts.assign(vintage=origin))
        print(origin, len(p), len(z), flush=True)
    pd.concat(prov).to_csv(out / "province.csv.gz", index=False)
    pd.concat(zone).to_csv(out / "zones.csv.gz", index=False)
    pd.concat(summ).to_csv(out / "delays.csv.gz", index=False)
    pd.concat(counts_all).to_csv(out / "zone_counts.csv.gz", index=False)
    pd.concat(zsumm).to_csv(out / "zone_delays.csv.gz", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
