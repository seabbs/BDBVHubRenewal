import sys, numpy as np, pandas as pd, dataclasses
from pathlib import Path
import variants  # noqa
from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import make_tasks, load_config
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.zones import zone_task
cfg = load_config("config/backtest.toml")
target = load_target(); real = target.loc[target["kind"] == "real"]
daily = load_daily(); dz = load_daily_zones(); delays = load_delays()
zones = hub_zones(Path("data/hub-config/tasks.json"))
v = "2026-11-01"
task = make_tasks(real.loc[real["vintage"] == v], delays, cfg, daily=daily)[0]
counts = weekly_zone_counts(dz, v)
zt = zone_task(task, counts, zones, pd.concat([daily, dz]))
o = get_model("randomwalknowcast").origin_at(zt, errors=True)
L = len(o.snap.locations)
rows = {}
for p in ["CD-HU", "CD-IT", "CD-NK", "CD-TO", "CD-BU"]:
    i = o.snap.locations.index(p)
    # the pooled size its sparse zones use
    zs = [j for j, c in enumerate(o.snap.locations) if c.startswith(p + "-") and o.obs_source[j, 0] == L + i]
    rows[p] = dict(own_k0=round(float(o.size[i, 0]), 2), zone_pool_k0=round(float(o.size[zs[0], 0]), 2) if zs else None,
                   zone_pool_k1=round(float(o.size[zs[0], 1]), 2) if zs else None, n_zones=len(zs))
print(sys.argv[1], rows)
