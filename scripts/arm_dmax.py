"""Fit the ragged models with D the maximum delay from the data, per origin.

Usage: arm_dmax.py STORE. D in days is the age by which 99% of the hub
target is reported under randomwalknowcast's chain ladder delay for the
country (dmax.py); weekly D is that in whole weeks, rounded up.
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

out = sys.argv[1]
d = "data"
tasks = store.origin_tasks(load_target(d), load_delays(d), load_config(), load_daily(d))
sampler = store.samplers()
for name in ("fm_chronos_weekly_ragged", "fm_chronos_daily_ragged"):
    code = store.code_digest(name)
    for origin, task in tasks.items():
        try:
            days, _ = max_delay(task)
        except NoDelayPair:
            print(name, origin, "no delay, skipped", flush=True)
            continue
        f.MAX_LAG.update(weekly=math.ceil(days / 7), daily=days)
        t0 = time.perf_counter()
        fit = store.fit(name, task, sampler, code)
        secs = time.perf_counter() - t0
        store.write_fit(out, fit)
        print(name, origin, f"D={days}d/{math.ceil(days / 7)}w", fit.meta["status"], f"{secs:.1f}s", flush=True)
