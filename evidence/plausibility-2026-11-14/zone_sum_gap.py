"""Raw zone draws summed against the province draws, by week age.

Usage: zone_sum_gap.py <data dir> <out csv> [model ...]

At the 2026-11-08 origin, per model, the zone model's own draws
(`zone_task`, before `reconcile_zones`) are summed by province and sample
and compared with the province model's draw of the same sample id. A ratio
above 1 in a data week means the split has to scale the zones down, which
is where the split's falls below the published counts come from.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.cli import _task
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, province_of, weekly_zone_counts
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.zones import zone_task

data, out = Path(sys.argv[1]), sys.argv[2]
models = sys.argv[3:] or ["randomwalk", "randomwalknowcast"]
args = argparse.Namespace(
    data=str(data), vintage="2026-11-08", reference_date="2026-11-14",
    backtest_config="config/backtest.toml",
)
task, _ = _task(args)
daily = load_daily_zones(str(data))
counts = weekly_zone_counts(daily, task.vintage)
zones = hub_zones(data / "hub-config" / "tasks.json")
zt = zone_task(task, counts, zones, pd.concat([task.daily_history, daily]))
cut = pd.Timestamp("2026-11-05")
rows = []
for name in models:
    prov = get_model(name).forecast(task)
    raw = get_model(name).forecast(zt)
    raw["province"] = raw.location.map(province_of)
    zs = raw.groupby(["province", "target_end_date", "sample"]).value.sum()
    pv = prov[prov.location.str.len() == 5].set_index(
        ["location", "target_end_date", "sample"]
    ).value
    pv.index.names = zs.index.names
    d = pd.DataFrame({"zones": zs, "prov": pv}).dropna().reset_index()
    d = d[d.prov > 0]
    d["age"] = ((cut - pd.to_datetime(d.target_end_date)).dt.days + 6) // 7
    d["over"] = d.zones > d.prov
    d["ratio"] = d.zones / d.prov
    g = d.groupby("age").agg(
        cells=("over", "size"), share_over=("over", "mean"),
        median_ratio=("ratio", "median"),
        q90_ratio=("ratio", lambda r: np.quantile(r, 0.9)),
    ).reset_index()
    rows.append(g.assign(model=name))
    print(name)
    print(g[g.age.between(-2, 6)].round(3).to_string(index=False))
pd.concat(rows).to_csv(out, index=False)
