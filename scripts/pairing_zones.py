"""BDBV health zones: pairing on and off, for zones forecast directly.

Usage, from a checkout of the copula branch (its root):
  pairing_zones.py OUT MODEL...

Each model forecasts every active zone from the zone's own releases at
each real origin but the latest (`zone_forecasts`, with no split from
provinces), with 1,000 draws and the backtest's seed. The draws are
cached in OUT/draws. From the same draws, the arms are `off` (as drawn),
`fixed` (the copula's prior: zones of a province by `rho_zones`, weeks
by `rho_t`) and `rolling` (estimated from the zones' own matured errors,
as shipped).

Truth is each zone's count in the first release D days after the week
(`rolling_truth` over the zone counts of every release, D the backtest's
truth delay). Scores, by origin and province, for weeks with a truth:
- `energy`: the energy score of the province's zones and weeks as one
  vector, raw and on log(x + 1);
- `zone_sum`: the sum of the province's zones each week (CRPS, 50% and
  90% coverage);
- `zone_total`: each zone's sum over its weeks;
- `cell`: each zone and week (unchanged by pairing).
Rows go to OUT/<model>-zone-scores.csv.gz and OUT/<model>-zone-rho.csv.gz.
"""

import dataclasses
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import (
    load_daily,
    load_daily_zones,
    load_delays,
    load_target,
)
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import load_config, real_vintages
from bdbvhubrenewal.evaluation.nowcast import (
    resolve_truth_delay,
    rolling_truth,
)
from bdbvhubrenewal.evaluation.scoring import (
    coverage_sample,
    crps_sample,
    energy_sample,
)
from bdbvhubrenewal.evaluation.zones import zone_forecasts
from bdbvhubrenewal.forecasters.copula import PRIOR, RollingCopula, pair_samples

N_DRAWS = 1000
out = Path(sys.argv[1])
(out / "draws").mkdir(parents=True, exist_ok=True)
config = dataclasses.replace(load_config(), n_samples=N_DRAWS)
target = load_target("data")
daily, daily_zones = load_daily("data"), load_daily_zones("data")
codes = hub_zones(Path("data/hub-config/tasks.json"))
real = target.loc[target["kind"] == "real"]
vintages = real_vintages(real)
origins = [v for v in vintages[:-1] if v not in config.skip_origins]
delay = resolve_truth_delay(config, target)
cell = ["location", "target_end_date"]
cols = ["model", "origin", *cell, "sample", "value"]

# Zone counts as every release reports them, for the rolling truth.
counts = {v: weekly_zone_counts(daily_zones, v) for v in vintages}
# Each release's cutoff is the province target's for the same release.
cutoff = real.drop_duplicates("vintage").set_index("vintage")["cutoff"]
zone_target = pd.concat(
    [
        c.assign(kind="real", vintage=v, cutoff=cutoff[v])
        for v, c in counts.items()
    ],
    ignore_index=True,
)
truth = rolling_truth(zone_target, delay)


def province(loc: pd.Series) -> pd.Series:
    return loc.str.rsplit("-", n=1).str[0]


def sums(rows, name, by):
    found = []
    for key, grp in rows.groupby(by):
        draws = grp.groupby("sample")["value"].sum().to_numpy()
        obs = float(grp.drop_duplicates(cell)["truth"].sum())
        found.append(
            {
                "target": name,
                "unit": "|".join(map(str, np.atleast_1d(key))),
                "truth": obs,
                "crps": crps_sample(obs, draws),
                "cover50": coverage_sample(obs, draws, 0.5),
                "cover90": coverage_sample(obs, draws, 0.9),
            }
        )
    return found


for name in sys.argv[2:]:
    if (out / f"{name}-zone-scores.csv.gz").exists():
        continue
    frames = []
    for origin in origins:
        path = out / "draws" / f"{name}_{origin}.pkl"
        if not path.exists():
            run = zone_forecasts(
                target, load_delays("data"), daily, daily_zones, codes,
                models=(name,), config=config, origins=[origin],
            )  # fmt: skip
            got = run.samples
            # An origin the model cannot forecast from gives no rows.
            direct = got.loc[got["key"] == name] if "key" in got else got
            direct.to_pickle(path)
        direct = pd.read_pickle(path)
        if not direct.empty and "sample" in direct:
            frames.append(direct.assign(model=name, origin=origin)[cols])
        print(name, origin, flush=True)
    off = pd.concat(frames, ignore_index=True)
    fixed = []
    for origin, rows in off.groupby("origin"):
        rng = np.random.default_rng(
            [config.seed, zlib.crc32(str(origin).encode())]
        )
        fixed.append(pair_samples(rows, PRIOR, rng))
    rolling, fits = RollingCopula().pair(off, truth, config.seed)
    arms = {
        "off": off,
        "fixed": pd.concat(fixed, ignore_index=True),
        "rolling": rolling,
    }
    scores = []
    for arm, table in arms.items():
        rows = table.merge(truth[[*cell, "truth"]], on=cell)
        rows = rows.assign(province=province(rows["location"]))
        for (origin, prov), grp in rows.groupby(["origin", "province"]):
            base = {"model": name, "arm": arm, "origin": origin,
                    "province": prov}  # fmt: skip
            wide = grp.pivot(index="sample", columns=cell, values="value")
            obs = (
                grp.drop_duplicates(cell)
                .set_index(cell)["truth"]
                .reindex(wide.columns)
                .to_numpy()
            )
            x = wide.to_numpy()
            scores.append(
                {
                    **base,
                    "target": "energy",
                    "unit": prov,
                    "n_cells": x.shape[1],
                    "energy": energy_sample(obs, x),
                    "log_energy": energy_sample(np.log1p(obs), np.log1p(x)),
                }
            )
            for r in (
                *sums(grp, "zone_sum", "target_end_date"),
                *sums(grp, "zone_total", "location"),
                *sums(grp, "cell", cell),
            ):
                scores.append({**base, **r})
        print(name, arm, "scored", flush=True)
    write_csv(pd.DataFrame(scores), out / f"{name}-zone-scores.csv.gz")
    write_csv(fits, out / f"{name}-zone-rho.csv.gz")
    print(name, "done", flush=True)
