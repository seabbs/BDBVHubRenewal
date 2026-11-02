"""Release backtest for rwnc-eval2: provinces, zones, variants, clock."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
import bdbvhubrenewal.evaluation.backtest as bt
from bdbvhubrenewal.evaluation.hubcompare import load_hub_submissions
from bdbvhubrenewal.evaluation.nowcast import (
    clock_sensitivity, release_backtest, weeks_back_from_cutoff,
)
from bdbvhubrenewal.forecasters.base import add_national, ModelUnavailableError
from bdbvhubrenewal.forecasters.chain import Settings
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk, RandomWalkNowcast
from bdbvhubrenewal.forecasters.zones import zone_task

out = Path(sys.argv[1])
hub_clone = Path(sys.argv[2])
renewal_cache = Path(sys.argv[3])
what = set(sys.argv[4:]) or {"main", "variants", "age6", "zones", "clock", "fans"}
out.mkdir(parents=True, exist_ok=True)

config = bt.load_config("config/backtest.toml")
target, delays, daily = load_target("data"), load_delays("data"), load_daily("data")
real = target.loc[target["kind"] == "real"]
tasks = bt.make_tasks(real, delays.assign(kind="real"), config, daily=daily)
by_v = {t.vintage: t for t in tasks}
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def stored_forecasts() -> pd.DataFrame:
    frames = []
    f = pd.read_csv("results/forecasts.csv.gz")
    frames.append(f.loc[f["model"].str.startswith("renewal") & (f["kind"] == "real")])
    for p in glob.glob(str(renewal_cache / "*.csv.gz")):
        frames.append(pd.read_csv(p))
    hub = load_hub_submissions(hub_clone)
    hub = hub.loc[hub["reference_date"] == "2026-10-31"]
    hub = hub.loc[hub["location"].str.count("-") <= 1]
    frames.append(hub.assign(vintage="2026-10-24"))
    cols = ["model", "vintage", "location", "target_end_date", "sample", "value"]
    s = pd.concat([x[cols] for x in frames], ignore_index=True)
    s["vintage"] = s["vintage"].astype(str)
    return s


def save(res, tag):
    for name in ("scores", "quantiles", "origins", "fits"):
        getattr(res, name).to_csv(out / f"{tag}-{name}.csv", index=False)
    print(tag, len(res.scores), flush=True)


if "main" in what:
    stored = stored_forecasts()
    print(stored.groupby(["model", "vintage"]).size().to_string(), flush=True)
    save(release_backtest(tasks, daily, target, stored=stored), "main")
if "age6" in what:
    save(release_backtest(tasks, daily, target, complete_rule="age",
                          complete_after=6, stored=stored_forecasts()), "age6")
if "variants" in what:
    for obs in ("skellam_rem", "nb"):
        m = RandomWalkNowcast(name=f"randomwalknowcast_{obs}",
                              settings=Settings(obs_model=obs))
        save(release_backtest(tasks, daily, target, models=[m]), f"var-{obs}")

dz = load_daily_zones("data")
zones = hub_zones("data/hub-config/tasks.json")
both = pd.concat([daily, dz])


def zone_target() -> pd.DataFrame:
    frames = []
    allz = sorted(set(dz["location"]) & set(zones))
    for v in sorted(real["vintage"].astype(str).unique()):
        c = weekly_zone_counts(dz, v)
        if c.empty:
            continue
        first = pd.Timestamp(c["target_end_date"].min())
        last = pd.Timestamp(v) + pd.Timedelta(days=(5 - pd.Timestamp(v).weekday()) % 7)
        weeks = pd.date_range(first, last, freq="7D").date.astype(str)
        grid = pd.MultiIndex.from_product([allz, weeks], names=["location", "target_end_date"])
        t = (c.set_index(["location", "target_end_date"])["total"]
             .reindex(grid, fill_value=0).reset_index())
        frames.append(t.assign(kind="real", vintage=v))
    return pd.concat(frames, ignore_index=True)


def zone_tasks(vintages):
    zt = []
    for v in vintages:
        counts = weekly_zone_counts(dz, v)
        active = tuple(z for z in zones if z in set(counts["location"]))
        zt.append(zone_task(by_v[v], counts, active, both))
    return zt


if "zones" in what:
    ztarget = zone_target()
    ztarget.to_csv(out / "zone-target.csv", index=False)
    zt = zone_tasks(["2026-10-17", "2026-10-24"])
    save(release_backtest(zt, both, ztarget), "zones")

if "clock" in what:
    ct = [by_v[v] for v in ("2026-10-17", "2026-10-24", "2026-11-01")]
    clock_sensitivity(ct, daily, target, weeks=2).to_csv(out / "clock.csv", index=False)
    print("clock done", flush=True)


def fan(task, models, d):
    rows = []
    for m in models:
        try:
            s = add_national(m.forecast(task))
        except ModelUnavailableError:
            continue
        s["weeks_back"] = weeks_back_from_cutoff(d, task.vintage, s["target_end_date"])
        q = s.groupby(["location", "target_end_date", "weeks_back"])["value"].quantile(np.array(LEVELS)).unstack()
        q.columns = [f"q{int(100 * p)}" for p in q.columns]
        rows.append(q.reset_index().assign(model=m.name, origin=task.vintage))
    return pd.concat(rows, ignore_index=True)


if "fans" in what:
    ms = [RandomWalk(), RandomWalkNowcast(),
          RandomWalkNowcast(name="randomwalknowcast_skellam_rem", settings=Settings(obs_model="skellam_rem")),
          RandomWalkNowcast(name="randomwalknowcast_nb", settings=Settings(obs_model="nb"))]
    vs = ["2026-10-17", "2026-10-24", "2026-11-01"]
    pd.concat([fan(by_v[v], ms, daily) for v in vs]).to_csv(out / "fans.csv", index=False)
    pd.concat([fan(t, ms[:2], both) for t in zone_tasks(vs)]).to_csv(out / "zone-fans.csv", index=False)
    print("fans done", flush=True)
