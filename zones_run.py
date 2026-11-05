"""Run the zone random walks at vintages and keep direct, reconciled and fit info."""
import sys, pickle, dataclasses
import numpy as np, pandas as pd
import variants  # noqa
from pathlib import Path
from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import make_tasks, BacktestConfig, load_config
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.zones import zone_task, reconcile_zones

out = sys.argv[1]; vintages = sys.argv[2].split(","); n = int(sys.argv[3])
models = sys.argv[4].split(",") if len(sys.argv) > 4 else ["randomwalk", "randomwalknowcast"]
cfg = dataclasses.replace(load_config("config/backtest.toml"), n_samples=n)
target = load_target(); real = target.loc[target["kind"] == "real"]
daily = load_daily(); dz = load_daily_zones(); delays = load_delays()
zones = hub_zones(Path("data/hub-config/tasks.json"))
res = {}
for v in vintages:
    task = make_tasks(real.loc[real["vintage"] == v], delays, cfg, daily=daily)[0]
    counts = weekly_zone_counts(dz, v)
    active = tuple(z for z in zones if z in set(counts["location"]))
    zt = zone_task(task, counts, active, pd.concat([daily, dz]))
    for name in models:
        m = get_model(name)
        direct = m.forecast(zt)
        diag = pd.DataFrame(m.fit_diagnostics)
        prov = m.forecast(task)
        rec = reconcile_zones(direct, prov, task, counts, active)
        o = m.origin_at(zt, errors=name != "randomwalk")
        res[(v, name)] = dict(direct=direct, rec=rec, prov=prov, diag=diag, counts=counts,
                              ref=task.reference_date.isoformat(), locs=list(o.snap.locations),
                              weeks=list(o.snap.week_ends), sigma=o.sigma, sigma_source=o.sigma_source,
                              level=o.fit.level, reported=o.reported, point=o.point,
                              size=o.size, removal=o.removal, start_nowcast=getattr(o, "start_nowcast", None),
                              parent=o.snap.parent)
        print(v, name, len(active), flush=True)
pickle.dump(res, open(out, "wb"))
