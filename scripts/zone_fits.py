"""Fit the Chronos-2 models at each zone-backtest origin, two ways.

own: the block holds every hub target, zones included (fm-zones), and
the zones are the model's own, which it makes sum to its provinces.
allocator: the block holds the country and provinces only (as on
main), and the zones are split from the provinces by ZoneAllocator.

Writes one gzipped CSV per model and origin to --out, with columns
variant, location, target_end_date, sample, value, and a timing row
per fit to --out/timings.csv.
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.zones import load_zone_data, weekly_zone_counts
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.base import ModelUnavailableError
from bdbvhubrenewal.forecasters.zones import ZoneAllocator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--origins", nargs="*")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = load_target(args.data)
    delays = load_delays(args.data)
    daily = load_daily(args.data)
    zones = load_zone_data(args.data)
    daily_zones, codes = zones.daily, zones.codes
    config = bt.load_config("config/backtest.toml")
    real = target.loc[target["kind"] == "real"]
    vintages = [
        v for v in bt.real_vintages(real)[:-1] if v not in config.skip_origins
    ]
    origins = args.origins or vintages
    for vintage in origins:
        counts = weekly_zone_counts(daily_zones, vintage)
        active = tuple(z for z in codes if z in set(counts["location"]))
        zoned, _ = bt.vintage_task(
            target, delays, config, vintage, daily, zones=zones
        )
        plain, _ = bt.vintage_task(target, delays, config, vintage, daily)
        for name in args.models:
            path = out / f"{name}__{vintage}.csv.gz"
            if path.exists():
                continue
            model = get_model(name)
            t0 = time.time()
            try:
                with_zones = model.forecast(zoned)
                t1 = time.time()
                without = model.forecast(plain)
                t2 = time.time()
            except ModelUnavailableError as err:
                print(f"{name} skips {vintage}: {err}", flush=True)
                continue
            frames = {
                "own": with_zones.loc[with_zones["location"].isin(active)],
                "allocator": ZoneAllocator().allocate(
                    without, plain, counts, active
                ),
            }
            pd.concat(
                [f.assign(variant=k) for k, f in frames.items()],
                ignore_index=True,
            ).to_csv(path, index=False)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row = pd.DataFrame(
                [{
                    "model": name, "origin": vintage,
                    "seconds_zones": round(t1 - t0, 1),
                    "seconds_plain": round(t2 - t1, 1),
                    "n_series_targets": int(with_zones["location"].nunique()),
                    "max_rss_mb_process": round(rss / 1024),
                }]
            )  # fmt: skip
            timings = out / "timings.csv"
            row.to_csv(timings, mode="a", header=not timings.exists(), index=False)
            print(row.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
