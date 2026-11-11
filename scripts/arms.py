"""Fit the ragged models under an experimental arm into a scratch store.

Usage: arms.py ARM STORE. ARM is base, carry (lag cells carried forward
from the latest release with cut-off at most t + d), reported (one
reported-so-far covariate per place in place of the lag series) or d6
(weekly D = 6).
"""
import sys

import numpy as np
import torch

torch.set_num_threads(2)
from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.forecasters import foundation as f

arm, out = sys.argv[1], sys.argv[2]
base_triangle = f.hub_triangle


def carry_triangle(rows, source, places, last_time):
    """As hub_triangle, with each lag cell from the latest release whose
    cut-off is at most t + d, for t + d up to the latest cut-off."""
    tri = base_triangle(rows, source, places, last_time)
    width = 7 if source == "weekly" else 1
    time_col = "target_end_date" if source == "weekly" else "onset_date"
    rows = rows.loc[rows["location"].isin(places)]
    value = rows["total"] if source == "weekly" else rows["observed"] + rows["imputed"]
    first = f._step(source, str(rows[time_col].min()))
    n_times = len(tri.times)
    place = rows["location"].map({c: i for i, c in enumerate(places)})
    step = (rows[time_col].map(lambda x: (f._step(source, str(x)) - first).days // width)).to_numpy()
    lags = tri.partial.shape[2]
    partial = np.full_like(tri.partial, np.nan)
    for release in sorted(set(rows["vintage"])):
        mine = (rows["vintage"] == release).to_numpy()
        cutoff = str(rows.loc[mine, "cutoff"].iloc[0])
        at = (f._step(source, cutoff) - first).days // width
        counts = np.zeros((len(places), n_times))
        keep = mine & (step >= 0) & (step < n_times)
        np.add.at(counts, (place.to_numpy()[keep].astype(int), step[keep]), value.to_numpy()[keep])
        for d in range(lags):
            # Cells (t, d) with at <= t + d <= last and t <= at.
            lo, hi = max(at - d, 0), min(at, n_times - 1)
            ts = np.arange(lo, hi + 1)
            ts = ts[ts + d <= tri.last]
            partial[:, ts, d] = counts[:, ts]
    return f.HubTriangle(tri.times, tri.final, partial, tri.last, tri.cutoff)


def reported_group(tri, layout, places, start):
    target = tri.final[:, :start]
    if layout == "complete":
        return {"target": target}
    steps = np.arange(start, len(tri.times))
    future = np.where(steps <= tri.last, tri.final[:, start:], np.nan)
    names = [f"{p}_reported" for p in places]
    return {
        "target": target,
        "past_covariates": dict(zip(names, tri.final[:, :start], strict=True)),
        "future_covariates": dict(zip(names, future, strict=True)),
    }


if arm == "carry":
    f.hub_triangle = carry_triangle
elif arm == "reported":
    f._group = reported_group
elif arm == "d6":
    f.MAX_LAG["weekly"] = 6
models = ["fm_chronos_weekly_ragged"] if arm == "d6" else [
    "fm_chronos_weekly_ragged", "fm_chronos_daily_ragged"]
d = "data"
tasks = store.origin_tasks(load_target(d), load_delays(d), load_config(), load_daily(d))
sampler = store.samplers()
for name in models:
    code = store.code_digest(name)
    for origin, task in tasks.items():
        fit = store.fit(name, task, sampler, code)
        print(name, origin, store.write_fit(out, fit), fit.meta["status"], flush=True)
