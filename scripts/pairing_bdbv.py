"""BDBV release backtest: pairing on and off, on summed targets.

Usage, from a checkout of the copula branch (its root):
  pairing_bdbv.py OUT MODEL...

Each real origin of the release backtest (less `skip_origins`) is drawn
once with 1,000 draws and the config's seed. Chronos-2's raw marginal
draws are cached in OUT/draws, so a rerun resumes. From the same draws,
the arms are `off` (no pairing: Chronos-2 with `GaussianCopula()`, as on
main), `fixed` (the copula's prior, for the daily Chronos-2 model its
own copula, which pairs its days) and `rolling` (as shipped). On BDBV
no error matures before a scored origin at D = 35 days, so `rolling` is
the prior.

Scores, against the rolling truth at the backtest's truth delay, only
for weeks with a truth:
- `cell`: each province and week, sample CRPS (pairing leaves it alone
  unless the model pairs its days);
- `national`: the country each week (randomwalknowcast's is the sum of
  the provinces; Chronos-2 forecasts it as its own series);
- `provinces_sum`: the sum of the provinces each week;
- `province_total`: each province's sum over its weeks with a truth;
- `national_total`: the country's sum over its weeks with a truth.
For each: sample CRPS and whether the 50% and 90% central intervals
cover the truth. Also the energy score of the provinces and of the
country by origin (`energy_scores`, raw and on log(x + 1)).
Rows go to OUT/<model>-scores.csv.gz and OUT/<model>-energy.csv.gz.
"""

import dataclasses
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation.backtest import load_config, run_forecasts
from bdbvhubrenewal.evaluation.nowcast import (
    resolve_truth_delay,
    rolling_truth,
)
from bdbvhubrenewal.evaluation.scoring import (
    coverage_sample,
    crps_sample,
    energy_scores,
)
from bdbvhubrenewal.evaluation.store import origin_tasks
from bdbvhubrenewal.evaluation.trajectories import copula_arms
from bdbvhubrenewal.forecasters import NATIONAL, get_model
from bdbvhubrenewal.forecasters.copula import GaussianCopula
from bdbvhubrenewal.forecasters.foundation import chronos_sampler

N_DRAWS = 1000
out = Path(sys.argv[1])
(out / "draws").mkdir(parents=True, exist_ok=True)
config = load_config()
target = load_target("data")
tasks = {
    o: dataclasses.replace(t, n_samples=N_DRAWS, seed=config.seed)
    for o, t in origin_tasks(
        target, load_delays("data"), config, load_daily("data")
    ).items()
    if o not in config.skip_origins
}
real = target.loc[target["kind"] == "real"]
truth = rolling_truth(real, resolve_truth_delay(config, target))
cols = ["model", "origin", "location", "target_end_date", "sample", "value"]
cell = ["location", "target_end_date"]


def cached(path, checkpoint, groups, horizon, n, seed):
    if path.exists():
        return [np.load(path)]
    draws = chronos_sampler(groups, horizon, n, seed, checkpoint=checkpoint)
    np.save(path, draws[0])
    return draws


def drawn(model, task, origin):
    s = run_forecasts([model], [task])
    s = s.loc[s["as_of"].isna()]
    # Province and country rows only, as the release backtest scores.
    s = s.loc[s["location"].str.count("-") <= 1]
    return s.assign(origin=origin)[cols]


def score(rows, target_name, by):
    """CRPS and coverage of the per-sample sums of `rows` over `by`."""
    out_rows = []
    for key, grp in rows.groupby(by):
        draws = grp.groupby("sample")["value"].sum().to_numpy()
        obs = float(grp.drop_duplicates(cell)["truth"].sum())
        out_rows.append(
            {
                "target": target_name,
                "unit": "|".join(map(str, np.atleast_1d(key))),
                "truth": obs,
                "crps": crps_sample(obs, draws),
                "cover50": coverage_sample(obs, draws, 0.5),
                "cover90": coverage_sample(obs, draws, 0.9),
            }
        )
    return out_rows


for name in sys.argv[2:]:
    if (out / f"{name}-scores.csv.gz").exists():
        continue
    base = get_model(name)
    chronos = hasattr(base, "checkpoint")
    off, fixed = [], []
    for origin, task in tasks.items():
        if chronos:
            path = out / "draws" / f"{name}_{origin}.npy"
            sample = partial(cached, path, base.checkpoint)
            for arm, copula in ((off, GaussianCopula()), (fixed, base.copula)):
                model = dataclasses.replace(base, sample=sample, copula=copula)
                arm.append(drawn(model, task, origin))
        else:
            off.append(drawn(base, task, origin))
        print(name, origin, flush=True)
    unpaired = pd.concat(off, ignore_index=True)
    own = pd.concat(fixed, ignore_index=True) if chronos else None
    samples, fits = copula_arms(unpaired, truth, config.seed, fixed=own)
    samples = samples.loc[samples["arm"] != "rolling_w25"]
    rows = samples.merge(truth[[*cell, "truth"]], on=cell)
    scores = []
    for (arm, origin), grp in rows.groupby(["arm", "origin"]):
        nat = grp["location"] == NATIONAL
        provinces, country = grp.loc[~nat], grp.loc[nat]
        found = [
            *score(provinces, "cell", cell),
            *score(country, "national", "target_end_date"),
            *score(provinces, "provinces_sum", "target_end_date"),
            *score(provinces, "province_total", "location"),
            *score(country, "national_total", "location"),
        ]
        scores += [
            {"model": name, "arm": arm, "origin": origin, **r} for r in found
        ]
    energy = energy_scores(
        samples.assign(model=samples["model"] + "|" + samples["arm"]),
        truth[[*cell, "truth"]],
        unit=["origin"],
    )
    arms = energy["model"].str.split("|", expand=True)
    energy = energy.assign(model=arms[0], arm=arms[1])
    scored = pd.DataFrame(scores)
    scored["arm"] = scored["arm"].replace({"unpaired": "off"})
    energy["arm"] = energy["arm"].replace({"unpaired": "off"})
    write_csv(scored, out / f"{name}-scores.csv.gz")
    write_csv(energy, out / f"{name}-energy.csv.gz")
    write_csv(fits, out / f"{name}-rho.csv.gz")
    print(name, "done", flush=True)
