"""Time one fit of a Chronos-2 model at a vintage, with or without zones.

Run under `/usr/bin/time -v` for the peak memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from bdbvhubrenewal.data import (
    load_daily,
    load_daily_zones,
    load_delays,
    load_health_zones,
    load_target,
)
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.store import NEXT_ROUND_WEEKS
from bdbvhubrenewal.forecasters import check_samples, get_model
from bdbvhubrenewal.forecasters.foundation import _chronos, chronos_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--vintage")
    parser.add_argument("--data", default="data")
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args()
    target = load_target(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = args.vintage or bt.real_vintages(real)[-1]
    zones = {}
    if not args.plain:
        zones = {
            "zones": load_health_zones(args.data),
            "daily_zones": load_daily_zones(args.data),
            "zone_codes": hub_zones(
                Path(args.data) / "hub-config" / "tasks.json"
            ),
        }
    # As the fit store builds it, one week past the horizon.
    task, _ = bt.vintage_task(
        target, load_delays(args.data), bt.load_config("config/backtest.toml"),
        vintage, load_daily(args.data), NEXT_ROUND_WEEKS, **zones,
    )  # fmt: skip
    model = get_model(args.model)
    sizes = []

    def sample(groups, horizon, n, seed):
        g = groups[0]
        sizes.append(len(g["target"]) + len(g.get("past_covariates", {})))
        return chronos_sampler(groups, horizon, n, seed, model.checkpoint)

    model = dataclasses.replace(model, sample=sample)
    t0 = time.time()
    _chronos(model.checkpoint)
    t1 = time.time()
    out = model.forecast(task)
    t2 = time.time()
    check_samples(out, task)
    import torch

    gpu = ""
    if torch.backends.mps.is_available():
        gpu = f" mps_driver_gb={torch.mps.driver_allocated_memory() / 1e9:.1f}"
    print(
        f"{args.model} {vintage} zones={not args.plain} "
        f"places={out['location'].nunique()} series={sizes[0]} load_s={t1 - t0:.1f} "
        f"fit_s={t2 - t1:.1f}{gpu}"
    )


if __name__ == "__main__":
    main()
