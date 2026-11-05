import dataclasses, sys, pickle
import numpy as np, pandas as pd
from bdbvhubrenewal.data import delays_for_vintage, load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.forecasters import get_model

out = sys.argv[1]
n = int(sys.argv[2])
target = load_target(); real = target.loc[target["kind"] == "real"]
daily = load_daily(); delays = load_delays()
cfg = dataclasses.replace(bt.load_config("config/backtest.toml"), n_samples=n)
res = {}
for v in ["2026-10-17", "2026-10-24"]:
    task = bt.make_tasks(real.loc[real["vintage"] == v], delays_for_vintage(delays, v)[0], cfg, history=real, daily=daily)[0]
    for name in ["randomwalk", "randomwalknowcast"]:
        m = get_model(name)
        s = m.forecast(task)
        o = m.origin_at(task, errors=False)
        info = dict(locs=list(o.snap.locations), sigma=o.sigma.tolist(), start=[o.snap.week_ends[int(i)] for i in o.start],
                    start_nowcast=None, week_ends=list(o.snap.week_ends), reported=o.reported.tolist(),
                    changes=[c.tolist() for c in __import__('bdbvhubrenewal.forecasters.chain', fromlist=['x']).walk_changes(o.reported, o.start, o.settings)])
        if name == "randomwalknowcast":
            o2 = m.origin_at(task, errors=True); info["start_nowcast"] = o2.snap.week_ends[o2.start_nowcast]
        res[(v, name)] = (s, info, task.reference_date)
        print(v, name, task.reference_date, info["start"], info["start_nowcast"], dict(zip(info["locs"], np.round(info["sigma"], 3))), flush=True)
pickle.dump(res, open(out, "wb"))
