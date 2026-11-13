"""Score zone forecasts on the rolling complete truth, as main scores.

Each zone and week's truth is its count in the first release whose
cut-off is at least D days after the week ends, D from
`resolve_truth_delay` (config/backtest.toml), 0 where that release has
no onset; weeks no release has reached are not scored.
randomwalknowcast's zones are its zone-level run made to sum to its
provinces, as in a hub file (`zone_forecasts`, `SUMMED`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_daily, load_daily_zones, load_delays, load_target
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.nowcast import resolve_truth_delay
from bdbvhubrenewal.evaluation.scoring import relative_skill, score_forecasts
from bdbvhubrenewal.evaluation.zones import SUMMED, ZoneForecasts, zone_forecasts

KEYS = ["model", "origin", "target", "truth_vintage", "location", "target_end_date"]
BASE = "randomwalknowcast"


def rolling_zone_truth(daily_zones: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Each zone-week's count in the first release `delay` days after it."""
    cutoffs = (
        daily_zones.groupby("vintage")["cutoff"].max().map(as_date).sort_values()
    )
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
        got = c.loc[c["target_end_date"] == week].set_index("location")["total"]
        rows.append(
            pd.DataFrame(
                {
                    "location": zones,
                    "target_end_date": week,
                    "truth": got.reindex(zones, fill_value=0).to_numpy(),
                    "truth_vintage": v,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--fits", nargs="+", required=True)
    parser.add_argument("--rw", default="results/rw-zones")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    target = load_target(args.data)
    delays = load_delays(args.data)
    daily = load_daily(args.data)
    daily_zones = load_daily_zones(args.data)
    codes = hub_zones(Path(args.data) / "hub-config" / "tasks.json")
    config = bt.load_config("config/backtest.toml")
    delay = resolve_truth_delay(config, target)
    files = sorted(f for d in args.fits for f in Path(d).glob("*__*.csv.gz"))
    origins = sorted({f.name.split("__")[1].removesuffix(".csv.gz") for f in files})
    rw = Path(args.rw)
    if not (rw / "zone-forecast-runs.csv.gz").exists():
        zone_forecasts(
            target, delays, daily, daily_zones, codes, models=[BASE],
            config=config, origins=origins,
        ).write(rw)  # fmt: skip
    run = ZoneForecasts.read([rw])
    frames = []
    for f in files:
        name, origin = f.name.removesuffix(".csv.gz").split("__")
        fits = pd.read_csv(f, dtype={"target_end_date": str})
        for variant, rows in fits.groupby("variant"):
            frames.append(
                rows.drop(columns="variant").assign(
                    model=f"{name}:{variant}", origin=origin
                )
            )
    summed = run.samples.loc[run.samples["key"] == BASE + SUMMED]
    frames.append(summed.drop(columns="key").assign(model=BASE))
    samples = pd.concat(frames, ignore_index=True)
    samples["origin"] = samples["origin"].astype(str)
    samples["target_end_date"] = samples["target_end_date"].astype(str)
    samples = samples.loc[samples["origin"].isin(origins)]
    truth = rolling_zone_truth(daily_zones, delay)
    samples = samples.merge(
        truth[["location", "target_end_date", "truth_vintage"]],
        on=["location", "target_end_date"],
    )
    # Only a truth released after the origin is a test of the forecast.
    samples = samples.loc[samples["truth_vintage"] > samples["origin"]]
    ref = samples["origin"].map(
        lambda v: pd.Timestamp(bt.reference_date_for_vintage(v))
    )
    samples["horizon"] = (
        pd.to_datetime(samples["target_end_date"]) - ref
    ).dt.days // 7
    samples["target"] = "complete"
    scores = score_forecasts(samples, truth, keys=KEYS)
    scores.to_csv(out / "zone-scores.csv.gz", index=False)
    skill = relative_skill(scores, baseline=BASE, match_on=KEYS[1:])
    skill.to_csv(out / "zone-relative.csv", index=False)
    print(f"Truth delay {delay} days; origins {origins}")
    cols = ["model", "n", "relative_wis", "relative_log_wis"]
    print(np.round(skill[cols].set_index("model"), 3).to_string())


if __name__ == "__main__":
    main()
