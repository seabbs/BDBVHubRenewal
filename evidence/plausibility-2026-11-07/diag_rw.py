"""Delay, start week and nowcast of the random walk origin at 2026-11-01."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.cli import _task
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.zones import zone_task

data = Path(sys.argv[1])
show = sys.argv[2].split(",")
args = argparse.Namespace(
    data=str(data), vintage="latest", backtest_config="config/backtest.toml"
)
task, _ = _task(args)
daily = load_daily_zones(str(data))
counts = weekly_zone_counts(daily, task.vintage)
zones = hub_zones(data / "hub-config" / "tasks.json")
ztask = zone_task(task, counts, zones, pd.concat([task.daily_history, daily]))
model = get_model("randomwalknowcast")
o = model.origin_at(ztask, errors=False)
snap = o.snap
locs = list(snap.locations)
age = snap.cut[o.o] - np.arange(snap.week.size)
print("releases", snap.releases, "cut", snap.cut)
print("pairs", o.fit.pairs)
wk = list(snap.week_ends)
last = len(wk) - 1
for code in show:
    i = locs.index(code)
    G = np.exp(np.cumsum(o.fit.g[i][::-1])[::-1])
    print(
        f"\n{code} level {o.fit.level[i]:.1f} own ages "
        f"{int(o.fit.own[i].sum())}/{o.fit.own[i].size} "
        f"start {wk[o.start[i]]} sigma {o.sigma[i]:.2f}"
    )
    print("  G(0..13)", np.round(G[:14], 2))
    print("  reported", dict(zip(wk[-4:], o.reported[i, -4:].round(1))))
    print("  point   ", dict(zip(wk[-4:], o.point[i, -4:].round(1))))
    print("  complete", dict(zip(wk[-4:], o.complete[i, -4:].round(2))))
