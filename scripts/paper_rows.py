"""Per-forecast copula rows for the paper (copula-scores, copula-rho).

Usage, from the fm-copula-rolling worktree's root:
  paper_rows.py OUT GERMANY_SAMPLES [CHRONOS_DRAWS]
GERMANY_SAMPLES holds randomwalknowcast's German samples, one pickle per
nowcast date (rolling.py). CHRONOS_DRAWS, if given, holds the Chronos-2
models' raw marginal draws (evidence.py). Scoring is
`bdbvhubrenewal.evaluation.trajectories`.
"""

import dataclasses
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.germany import GERMANY, hub_target, hub_triangle
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.backtest import load_config, run_forecasts
from bdbvhubrenewal.evaluation.nowcast import (
    resolve_truth_delay,
    rolling_truth,
)
from bdbvhubrenewal.evaluation.store import origin_tasks
from bdbvhubrenewal.evaluation.trajectories import (
    copula_arms,
    copula_rho,
    copula_scores,
)
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.copula import GaussianCopula

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
german = Path(sys.argv[2])
draws = Path(sys.argv[3]) if len(sys.argv) > 3 else None
cols = ["model", "origin", "location", "target_end_date", "sample", "value"]
scores, rhos = [], []

# BDBV release backtest, kept once made.
config = load_config()
ebola = out / "ebola-scores.pkl"
target, daily = load_target("data"), load_daily("data")
tasks = origin_tasks(target, load_delays("data"), config, daily)
real = target.loc[target["kind"] == "real"]
truth = rolling_truth(real, resolve_truth_delay(config, target))
rwnc = run_forecasts([get_model("randomwalknowcast")], tasks.values())
rwnc = rwnc.loc[rwnc["as_of"].isna()].rename(columns={"vintage": "origin"})
groups = [(rwnc[cols], None)]


def cached(path, groups, horizon, n, seed):
    return [np.load(path)]


def chronos(name, copula):
    frames = []
    for origin, task in tasks.items():
        path = draws / f"{name}_{origin}.npy"
        if not path.exists():
            continue
        model = dataclasses.replace(
            get_model(name), sample=partial(cached, path), copula=copula
        )
        frames.append(model.forecast(task).assign(model=name, origin=origin))
    return pd.concat(frames, ignore_index=True)[cols]


if draws is not None:
    names = sorted({p.stem.rsplit("_", 1)[0] for p in draws.glob("*.npy")})
    for name in names:
        own = get_model(name).copula
        groups.append((chronos(name, GaussianCopula()), chronos(name, own)))
for unpaired, fixed in groups if not ebola.exists() else []:
    samples, fits = copula_arms(unpaired, truth, config.seed, fixed=fixed)
    scores.append(copula_scores(samples, truth, "ebola", "provinces"))
    rhos.append(copula_rho(fits, "ebola", "provinces"))
    print("ebola", unpaired["model"].iloc[0], flush=True)

if not ebola.exists():
    pd.to_pickle((scores, rhos), ebola)
scores, rhos = pd.read_pickle(ebola)

# German development half, the states and the age groups apart.
cfg = de.load_germany_config("config/germany.toml")
dates = de.nowcast_dates(cfg)
german_samples = pd.concat(
    [pd.read_pickle(german / f"{d}.pkl") for d in dates], ignore_index=True
)
ends = sorted(set(german_samples["target_end_date"]))
triangle = hub_triangle(Path("data/raw/germany") / Path(cfg.triangle).name)
g_truth = rolling_truth(hub_target(triangle, ends), cfg.truth_delay)
for label, rows in german_samples.groupby("model"):
    strata = str(label).split("|")[1]
    unpaired = rows.assign(model="randomwalknowcast")[cols]
    samples, fits = copula_arms(unpaired, g_truth, cfg.seed, national=GERMANY)
    scores.append(
        copula_scores(samples, g_truth, "germany", strata, national=GERMANY)
    )
    rhos.append(copula_rho(fits, "germany", strata))
    print("germany", strata, flush=True)

write_csv(pd.concat(scores, ignore_index=True), out / "copula-scores.csv.gz")
write_csv(pd.concat(rhos, ignore_index=True), out / "copula-rho.csv.gz")
summary = (
    pd.concat(scores)
    .groupby(["study", "strata", "model", "arm"])
    .mean(numeric_only=True)
    .round(3)
)
print(summary.to_string())
