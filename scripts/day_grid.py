"""Daily Chronos-2 models: the day-level correlation on the BDBV backtest.

Usage, from the fm-copula-rolling worktree's root: day_grid.py DRAWS
For each correlation between adjacent days, the weekly forecasts of the
daily models from the same draws: median bias against the truth, WIS and
coverage of each place-week, and the gate's low-median rule.
"""

import dataclasses
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.evaluation.nowcast import (
    resolve_truth_delay,
    rolling_truth,
)
from bdbvhubrenewal.evaluation.scoring import score_sample
from bdbvhubrenewal.evaluation.store import origin_tasks
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.copula import PRIOR

draws = Path(sys.argv[1])
config = load_config()
target, daily = load_target("data"), load_daily("data")
tasks = origin_tasks(target, load_delays("data"), config, daily)
real = target.loc[target["kind"] == "real"]
truth = rolling_truth(real, resolve_truth_delay(config, target))
GRID = (0.0, 0.3, 0.5, 0.7, 0.9, round(PRIOR.rho_time ** (1 / 7), 3))


def cached(path, groups, horizon, n, seed):
    return [np.load(path)]


rows = []
for name in (
    "fm_chronos_daily_complete",
    "fm_chronos_daily_ragged",
    "fm_chronos_daily_ragged_ft",
):
    for rho_day in GRID:
        copula = dataclasses.replace(PRIOR, rho_time=rho_day**7)
        for origin, task in tasks.items():
            path = draws / f"{name}_{origin}.npy"
            if not path.exists():
                continue
            model = dataclasses.replace(
                get_model(name), sample=partial(cached, path), copula=copula
            )
            f = model.forecast(task)
            # The gate's low-median rule at each place's last week: median
            # under 1/20 of the place's maximum reported week.
            recent = task.target.groupby("location")["total"].max()
            last = f.loc[f["target_end_date"] == f["target_end_date"].max()]
            med_last = last.groupby("location")["value"].median()
            low = (med_last < recent.reindex(med_last.index) / 20) & (
                recent.reindex(med_last.index) >= 20
            )
            m = f.merge(truth, on=["location", "target_end_date"])
            for (loc, week), g in m.groupby(["location", "target_end_date"]):
                y = float(g["truth"].iloc[0])
                s = score_sample(y, g["value"])
                med = float(g["value"].median())
                rows.append(
                    {
                        "model": name,
                        "rho_day": rho_day,
                        "origin": origin,
                        "location": loc,
                        "week": week,
                        "log_bias": np.log1p(med) - np.log1p(y),
                        "wis": s["wis"],
                        "cov50": s["coverage_50"],
                        "cov90": s["coverage_90"],
                        "national": loc == "CD",
                    }
                )
            rows.append(
                {
                    "model": name,
                    "rho_day": rho_day,
                    "origin": origin,
                    "gate_low": int(low.sum()),
                }
            )
t = pd.DataFrame(rows)
cells = t.dropna(subset=["wis"])
out = cells.groupby(["model", "rho_day"]).agg(
    n=("wis", "size"),
    log_bias=("log_bias", "mean"),
    wis=("wis", "mean"),
    cov50=("cov50", "mean"),
    cov90=("cov90", "mean"),
)
nat = cells.loc[cells["national"]].groupby(["model", "rho_day"])
out["wis_national"] = nat["wis"].mean()
out["log_bias_national"] = nat["log_bias"].mean()
latest = max(tasks)
gate = t.loc[t["origin"] == latest].groupby(["model", "rho_day"])["gate_low"]
out["gate_low_latest"] = gate.sum()
pd.set_option("display.width", 200)
print(out.round(3).to_string())
t.to_csv("day-grid-cells.csv.gz", index=False)
