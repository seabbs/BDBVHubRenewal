"""Run the renewal models at the 10-24 release, as the backtest does."""
import sys
import pandas as pd
from bdbvhubrenewal.cli import host_devices
host_devices(2)
from bdbvhubrenewal.data import load_target, load_delays, load_daily, delays_for_vintage
import bdbvhubrenewal.evaluation.backtest as bt
from bdbvhubrenewal.forecasters.base import get_model

out, names = sys.argv[1], sys.argv[2:]
config = bt.load_config("config/backtest.toml")
bt.apply_sampler(config)
target, delays = load_target("data"), load_delays("data")
real = target.loc[target["kind"] == "real"]
vintage = "2026-10-24"
rows = real.loc[real["vintage"] == vintage]
dly, note = delays_for_vintage(delays, vintage)
print("note:", note, flush=True)
task = bt.make_tasks(rows, dly.assign(kind="real"), config, history=real,
                     daily=load_daily("data"))[0]
fits = []
for n in names:
    fc = bt.run_forecasts([get_model(n)], [task], cache=out, fits=fits)
    print(n, len(fc), flush=True)
