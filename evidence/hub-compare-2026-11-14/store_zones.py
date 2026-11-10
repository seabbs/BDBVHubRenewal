"""Turn a stored fit's province samples into national, province and zones.

Usage: store_zones.py <fit json.gz> <vintage> <reference date> <out csv>

The fit store keeps national and province samples only. The zones are made
as `nowcast` makes them for a fitted model: `zones_for_model` splits each
province sample across its zones (`ZoneAllocator`), so nothing is refitted.
"""

import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.cli import _task
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.forecasters.zones import zones_for_model

path, vintage, ref, out = sys.argv[1:5]
fit = json.loads(gzip.open(path).read())
f = fit["forecasts"]
samples = pd.DataFrame(f["data"], columns=f["columns"])
print(fit["meta"])
print("stored reference dates", samples.reference_date.unique())
args = argparse.Namespace(
    data="data", vintage=vintage, reference_date=ref,
    backtest_config="config/backtest.toml",
)
task, notes = _task(args)
print("task", task.reference_date, notes)
# The store keeps weeks from horizon -4, so the zones cover those weeks.
weeks = sorted(samples.target_end_date.unique())
task = dataclasses.replace(task, target_end_dates=tuple(weeks))
daily = load_daily_zones("data")
counts = weekly_zone_counts(daily, task.vintage)
zones = zones_for_model(
    fit["meta"]["model"], task, samples, counts,
    hub_zones(Path("data/hub-config/tasks.json")), daily,
)
cols = ["location", "target_end_date", "sample", "value"]
pd.concat([samples[cols], zones[cols]]).to_csv(out, index=False)
print("wrote", out, samples.location.nunique(), zones.location.nunique())
