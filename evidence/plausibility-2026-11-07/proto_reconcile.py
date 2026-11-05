"""Prototype zone reconciliations on the 2026-11-01 randomwalk samples."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from bdbvhubrenewal.cli import _task
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, province_of, weekly_zone_counts
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.zones import zone_task, _apportion

data = Path(sys.argv[1]); name = sys.argv[2]
args = argparse.Namespace(data=str(data), vintage="latest", backtest_config="config/backtest.toml")
task, _ = _task(args)
daily = load_daily_zones(str(data))
counts = weekly_zone_counts(daily, task.vintage)
zones = hub_zones(data / "hub-config" / "tasks.json")
m = get_model(name)
prov = m.forecast(task)
raw = m.forecast(zone_task(task, counts, zones, pd.concat([task.daily_history, daily])))
weeks = list(task.target_end_dates)
out = []
for p in ["CD-HU", "CD-NK", "CD-TO", "CD-IT"]:
    mine = sorted(z for z in raw.location.unique() if province_of(z) == p)
    Y = prov[prov.location == p].pivot(index="sample", columns="target_end_date", values="value")[weeks].clip(lower=0).round().to_numpy(np.int64)
    X = np.stack([raw[raw.location == z].pivot(index="sample", columns="target_end_date", values="value")[weeks].to_numpy() for z in mine], -1)
    # rank-matched: give each zone's draws, week by week, the province's ranks
    rank = np.argsort(np.argsort(Y, axis=0, kind="stable"), axis=0)
    Xs = np.sort(X, axis=0)
    Xr = np.take_along_axis(Xs, np.repeat(rank[..., None], len(mine), -1), axis=0)
    # whole trajectories: pair by the sum over the weeks
    pr = np.argsort(np.argsort(Y.sum(1), kind="stable"))
    Xt = np.stack([X[np.argsort(X[:, :, i].sum(1), kind="stable")][pr][:, :, i] for i in range(len(mine))], -1)
    for lab, W in [("current", X), ("rank", Xr), ("traj", Xt)]:
        W = W.astype(float).copy(); empty = W.sum(-1) <= 0; W[empty] = 1.0
        V = _apportion(Y, W)
        for i, z in enumerate(mine):
            for j, w in enumerate(weeks):
                out.append((lab, z, w, *np.quantile(V[:, j, i], [0.05, 0.5, 0.95])))
    for i, z in enumerate(mine):
        for j, w in enumerate(weeks):
            out.append(("raw", z, w, *np.quantile(X[:, j, i], [0.05, 0.5, 0.95])))
pd.DataFrame(out, columns=["how", "location", "ted", "q05", "q50", "q95"]).to_csv(Path(sys.argv[3]), index=False)
