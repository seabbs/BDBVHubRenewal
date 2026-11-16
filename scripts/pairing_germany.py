"""German development half: pairing on and off, scored as the paper does.

Usage, from a checkout of the copula branch (its root):
  pairing_germany.py OUT MODEL...

For each model and nowcast date of the development half, the states
task is drawn once with 1,000 draws and the config's seed. Chronos-2's
raw marginal draws, and other models' draws, are cached in OUT/draws
(one per model and date), so a rerun resumes. From the same draws:

- `off`: no pairing (Chronos-2 with `GaussianCopula()`, which leaves the
  days, places and weeks independent, as on fm-chronos).
- `fixed`: the copula's prior. For the daily Chronos-2 model this is its
  own copula, so days are paired by `CHRONOS_DAY_RHO` inside the model.
- `rolling`: the rolling copula on top of `fixed`, as shipped.

Germany is the sum of the states (`hub_forecasts`). Each arm's draws
become quantiles at `QUANTILE_LEVELS` (`hub_quantiles`), get their
truth (`with_truth`) and are scored by `score_germany`, as the paper
and site score Germany. The hub's levels have no 90% interval, so the
50% and 90% intervals' coverage also comes from the draws. Rows go to OUT/<model>-scores.csv.gz (one per
forecast and arm) and OUT/<model>-rho.csv.gz.
"""

import dataclasses
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.germany import (
    GERMANY,
    hub_forecasts,
    hub_target,
    hub_triangle,
    nowcast_tasks,
)
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.evaluation.trajectories import copula_arms
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.copula import GaussianCopula
from bdbvhubrenewal.forecasters.foundation import chronos_sampler
from bdbvhubrenewal.reporting.hubtables import hub_quantiles

N_DRAWS = 1000
out = Path(sys.argv[1])
(out / "draws").mkdir(parents=True, exist_ok=True)
cfg = de.load_germany_config("config/germany.toml")
dates = de.nowcast_dates(cfg)
if os.environ.get("PAIRING_N_DATES"):  # smoke runs
    dates = dates[: int(os.environ["PAIRING_N_DATES"])]
# PAIRING_DRAW_ONLY=1 only fills the draw cache, from the last date back,
# so a second process can share a long run; the main run then scores.
draw_only = bool(os.environ.get("PAIRING_DRAW_ONLY"))
if draw_only:
    dates = dates[::-1]
triangle = hub_triangle(Path("data/raw/germany") / Path(cfg.triangle).name)
cols = ["model", "origin", "location", "target_end_date", "sample", "value"]
keys = ["model", "origin", "location", "target_end_date", "horizon"]


def cached(path, checkpoint, groups, horizon, n, seed):
    if path.exists():
        return [np.load(path)]
    draws = chronos_sampler(groups, horizon, n, seed, checkpoint=checkpoint)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, draws[0])
    tmp.replace(path)
    return draws


def drawn(model, task, date):
    s = run_forecasts([model], [task])
    h = hub_forecasts(s, task, date)
    return h.rename(columns={"forecast_date": "origin"})


def quantiles(rows):
    table = rows.rename(columns={"sample": "output_type_id"}).assign(
        output_type="sample"
    )[[*keys, "output_type", "output_type_id", "value"]]
    q = hub_quantiles(table, de.QUANTILE_LEVELS)
    return q.drop(columns="output_type").rename(
        columns={"output_type_id": "quantile"}
    )[[*keys, "quantile", "value"]]


for name in sys.argv[2:]:
    if (out / f"{name}-scores.csv.gz").exists():
        continue
    base = get_model(name)
    chronos = hasattr(base, "checkpoint")
    off, fixed = [], []
    for date in dates:
        task = nowcast_tasks(triangle, date, N_DRAWS, cfg.seed)[0]
        if chronos:
            path = out / "draws" / f"{name}_{date}.npy"
            sample = partial(cached, path, base.checkpoint)
            for arm, copula in ((off, GaussianCopula()), (fixed, base.copula)):
                model = dataclasses.replace(base, sample=sample, copula=copula)
                arm.append(drawn(model, task, date))
        else:
            # Kept per date, so a rerun resumes.
            path = out / "draws" / f"{name}_{date}.pkl"
            if not path.exists():
                drawn(base, task, date).to_pickle(path)
            off.append(pd.read_pickle(path))
        print(name, date, flush=True)
    if draw_only:
        continue
    unpaired = pd.concat(off, ignore_index=True)
    own = pd.concat(fixed, ignore_index=True)[cols] if chronos else None
    horizon = unpaired[keys].drop_duplicates()
    ends = sorted(set(unpaired["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), cfg.truth_delay)
    samples, fits = copula_arms(
        unpaired[cols], truth, cfg.seed, national=GERMANY, fixed=own
    )
    scores = []
    for arm in ("unpaired", "fixed", "rolling"):
        rows = samples.loc[samples["arm"] == arm, cols].merge(
            horizon, on=keys[:-1]
        )
        q = de.with_truth(quantiles(rows), triangle, cfg.truth_delay)
        scored = de.score_germany(q, cfg.truth_delay)
        # The hub's levels give no 90% interval, so the 50% and 90%
        # intervals are also taken from the draws.
        bounds = (
            rows.groupby(keys)["value"]
            .quantile([0.05, 0.25, 0.75, 0.95])
            .unstack()
            .reset_index()
        )
        scored = scored.merge(bounds, on=keys, how="left")
        for level, lo, hi in ((50, 0.25, 0.75), (90, 0.05, 0.95)):
            inside = scored["truth"].between(scored[lo], scored[hi])
            scored[f"draw_coverage_{level}"] = inside.astype(float)
        scored = scored.drop(columns=[0.05, 0.25, 0.75, 0.95])
        scores.append(scored.assign(arm="off" if arm == "unpaired" else arm))
        print(name, arm, "scored", flush=True)
    write_csv(
        pd.concat(scores, ignore_index=True), out / f"{name}-scores.csv.gz"
    )
    write_csv(fits, out / f"{name}-rho.csv.gz")
    print(name, "done", flush=True)
