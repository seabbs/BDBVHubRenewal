"""Walk start, spread and quiet flag of chosen zones at the 2026-11-08 origin.

Usage: diag_start.py <data dir> <zone,zone,...>

`randomwalknowcast` walks each sample from its nowcast of `_start_weeks`
(the latest full week at least 5% reported with at least 5 cases), and
`randomwalk` from `origin.start` (the latest week 95% complete).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.cli import _task
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.randomwalk import _label, _quiet, _start_weeks
from bdbvhubrenewal.forecasters.zones import zone_task

data = Path(sys.argv[1])
show = sys.argv[2].split(",")
# "stale": every zone that walks from a week 4 or more weeks before the cut
args = argparse.Namespace(
    data=str(data), vintage="2026-11-08", reference_date="2026-11-14",
    backtest_config="config/backtest.toml",
)
task, _ = _task(args)
daily = load_daily_zones(str(data))
counts = weekly_zone_counts(daily, task.vintage)
zones = hub_zones(data / "hub-config" / "tasks.json")
ztask = zone_task(task, counts, zones, pd.concat([task.daily_history, daily]))
model = get_model("randomwalknowcast")
o = model.origin_at(ztask, errors=True)
locs = list(o.snap.locations)
wk = list(o.snap.week_ends)
w0, quiet = _start_weeks(o), _quiet(o)
cw = o.snap.cut_week(o.o)
print("cut week", wk[cw], "start_nowcast", wk[o.start_nowcast])
if show == ["stale"]:
    show = [c for i, c in enumerate(locs)
            if c.count("-") == 2 and cw - w0[i] >= 4 and not quiet[i]]
rows = []
for code in show:
    i = locs.index(code)
    rows.append(dict(
        zone=code, rwnc_start=wk[w0[i]], rwnc_steps=int(cw - w0[i]),
        rw_start=wk[o.start[i]], quiet=bool(quiet[i]),
        silent=bool(o.silent[i]), sigma=round(float(o.sigma[i]), 2),
        sigma_source=_label(o, int(o.sigma_source[i])),
        level=round(float(o.fit.level[i]), 1),
        reported=" ".join(str(int(x)) for x in o.reported[i, -10:]),
    ))
pd.set_option("display.width", 250)
print("reported: the last 10 weeks, ending", wk[-1])
out = pd.DataFrame(rows)
print(out.to_string(index=False))
if len(sys.argv) > 3:
    out.to_csv(sys.argv[3], index=False)
