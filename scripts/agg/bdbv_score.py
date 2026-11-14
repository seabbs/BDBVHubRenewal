"""Score the grouping fits of `bdbv_fits.py` with main's release backtest.

National and provinces: `score_releases` on the complete target, with the
rolling truth at D (`resolve_truth_delay`, 98%) and, for breadth, on the
complete weeks of the latest release (no D). Zones: each zone-week's
count in the first release D days after it, as in the #200 evidence.
randomwalknowcast is run here; its zones are its zone-level run summed to
its provinces, as in its hub file.
Relative WIS and log WIS against randomwalknowcast by level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import (
    load_daily,
    load_daily_zones,
    load_delays,
    load_target,
)
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.zones import (
    hub_zones,
    is_zone,
    load_zone_data,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import (
    ReleaseForecasts,
    forecast_releases,
    relative_wis,
    resolve_truth_delay,
    score_releases,
)
from bdbvhubrenewal.evaluation.scoring import relative_skill, score_forecasts
from bdbvhubrenewal.evaluation.zones import (
    SUMMED,
    ZoneForecasts,
    zone_forecasts,
)
from bdbvhubrenewal.forecasters.base import NATIONAL

BASE = "randomwalknowcast"
KEYS = [
    "model",
    "origin",
    "target",
    "truth_vintage",
    "location",
    "target_end_date",
]


def rolling_zone_truth(daily_zones: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Each zone-week's count in the first release `delay` days after it."""
    cutoffs = (
        daily_zones.groupby("vintage")["cutoff"].max().map(as_date)
    ).sort_values()
    counts = {v: weekly_zone_counts(daily_zones, v) for v in cutoffs.index}
    weeks = sorted({w for c in counts.values() for w in c["target_end_date"]})
    zones = sorted({z for c in counts.values() for z in c["location"]})
    rows = []
    for week in weeks:
        due = as_date(week) + pd.Timedelta(days=delay)
        later = cutoffs[cutoffs >= due]
        if later.empty:
            continue
        v = later.index[0]
        c = counts[v]
        got = c.loc[c["target_end_date"] == week].set_index("location")
        rows.append(
            pd.DataFrame(
                {
                    "location": zones,
                    "target_end_date": week,
                    "truth": got["total"]
                    .reindex(zones, fill_value=0)
                    .to_numpy(),
                    "truth_vintage": v,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def level(location: str) -> str:
    if location == NATIONAL:
        return "national"
    return "zones" if is_zone(location) else "provinces"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--fits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = load_target(args.data)
    delays = load_delays(args.data)
    daily = load_daily(args.data)
    config = bt.load_config("config/backtest.toml")
    tasks = store.origin_tasks(
        target, delays, config, daily, load_zone_data(args.data)
    )
    parts = sorted(
        p for p in Path(args.fits).iterdir() if p.is_dir() and "__" in p.name
    )
    origins = sorted({p.name.split("__")[1] for p in parts})
    run = ReleaseForecasts.read(parts)
    base = forecast_releases([tasks[o] for o in origins], target, [BASE])
    both = ReleaseForecasts(
        *(
            pd.concat([getattr(run, n), getattr(base, n)], ignore_index=True)
            for n in ReleaseForecasts._NAMES
        )
    )
    delay = resolve_truth_delay(config, target)
    tables = []
    for label, d in ((f"rolling D={delay}", delay), ("latest release", None)):
        res = score_releases(
            [tasks[o] for o in origins], daily, target, both, truth_delay=d
        )
        s = res.scores.loc[res.scores["target"] == "complete"].copy()
        s["level"] = s["location"].map(level)
        s.to_csv(out / f"scores-{d or 'latest'}.csv.gz", index=False)
        rel = relative_wis(s, by=("level",), baseline=BASE)
        tables.append(rel.assign(truth=label))
    # Zones on the rolling truth.
    daily_zones = load_daily_zones(args.data)
    codes = hub_zones(Path(args.data) / "hub-config" / "tasks.json")
    rw = out / "rw-zones"
    if not (rw / "zone-forecast-runs.csv.gz").exists():
        zone_forecasts(
            target,
            delays,
            daily,
            daily_zones,
            codes,
            models=[BASE],
            config=config,
            origins=origins,
        ).write(rw)
    zrun = ZoneForecasts.read([rw])
    summed = zrun.samples.loc[zrun.samples["key"] == BASE + SUMMED]
    mine = run.samples.loc[run.samples["location"].map(is_zone)]
    # Only zones with cases at the origin are scored, as main's zone
    # backtest has.
    samples = pd.concat(
        [
            mine.drop(columns=[c for c in mine if c == "as_of"]),
            summed.drop(columns="key").assign(model=BASE),
        ],
        ignore_index=True,
    )
    samples["origin"] = samples["origin"].astype(str)
    samples["target_end_date"] = samples["target_end_date"].astype(str)
    active = {
        o: set(weekly_zone_counts(daily_zones, o)["location"]) for o in origins
    }
    samples = samples.loc[
        [loc in active.get(o, ()) for loc, o in
         zip(samples["location"], samples["origin"], strict=True)]
    ]  # fmt: skip
    truth = rolling_zone_truth(daily_zones, delay)
    samples = samples.merge(
        truth[["location", "target_end_date", "truth_vintage"]],
        on=["location", "target_end_date"],
    )
    samples = samples.loc[samples["truth_vintage"] > samples["origin"]]
    ref = samples["origin"].map(
        lambda v: pd.Timestamp(bt.reference_date_for_vintage(v))
    )
    samples["horizon"] = (
        pd.to_datetime(samples["target_end_date"]) - ref
    ).dt.days // 7
    samples["target"] = "complete"
    zscores = score_forecasts(samples, truth, keys=KEYS)
    zscores.to_csv(out / "zone-scores.csv.gz", index=False)
    zrel = relative_skill(zscores, baseline=BASE, match_on=KEYS[1:])
    zrel = zrel.rename(
        columns={"relative_wis": "rel_wis", "relative_log_wis": "rel_log_wis"}
    )
    tables.append(zrel.assign(level="zones", truth=f"rolling D={delay}"))
    table = pd.concat(tables, ignore_index=True)[
        ["truth", "level", "model", "n", "rel_wis", "rel_log_wis"]
    ]
    table.to_csv(out / "relative.csv", index=False)
    print(f"D = {delay}; origins {origins}")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(np.round(table, 3).to_string(index=False))


if __name__ == "__main__":
    main()
