"""Fit the ragged models, zero-shot and fine-tuned (#193), into a scratch store.

Usage: ft_fits.py ARM STORE, from the fm-finetune worktree's root. ARM is
d4 (D = 4 weeks / 28 days, the models' own) or dmax (D per origin from
the data, as arm_dmax.py: the age by which 99% of the country's hub
target is reported under randomwalknowcast's delay; weekly D in whole
weeks, rounded up).
"""
import math
import sys
import time

import torch

torch.set_num_threads(2)
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from dmax import max_delay

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.forecasters import foundation as f
from bdbvhubrenewal.forecasters.chain import NoDelayPair

arm, out = sys.argv[1], sys.argv[2]
models = [
    f"fm_chronos_{s}_ragged{ft}" for s in ("weekly", "daily") for ft in ("", "_ft")
]
d = "data"
tasks = store.origin_tasks(load_target(d), load_delays(d), load_config(), load_daily(d))
sampler = store.samplers()
lags = {}
for origin, task in tasks.items():
    if arm == "d4":
        lags[origin] = dict(f.MAX_LAG)
        continue
    try:
        days, _ = max_delay(task)
    except NoDelayPair:
        print(origin, "no delay, skipped", flush=True)
        continue
    lags[origin] = {"weekly": math.ceil(days / 7), "daily": days}
default = dict(f.MAX_LAG)
for name in models:
    code = store.code_digest(name)
    for origin, lag in lags.items():
        f.MAX_LAG.update(lag)
        t0 = time.perf_counter()
        fit = store.fit(name, tasks[origin], sampler, code)
        store.write_fit(out, fit)
        print(name, origin, lag, fit.meta["status"],
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        f.MAX_LAG.update(default)
