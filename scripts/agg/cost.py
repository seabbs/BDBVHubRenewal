"""Time one fit of one arm, with its series per block and peak memory.

Run once per arm, each in its own process, so the peak resident memory is
that fit's (with Chronos-2 loaded). Appends a row to --out: study, model,
option, series (target and lag series per block, `+` between blocks),
memory_gb and seconds (the fit, Chronos-2 loaded beforehand). BDBV fits
the release backtest task at --origin; Germany fits the nowcast date
--date.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import resource
import sys
import time
from pathlib import Path

if os.environ.get("AGG_CPU"):
    import torch

    _mps = torch.backends.mps.is_available

    def _no_mps() -> bool:
        return False

    _no_mps.__wrapped__ = _mps  # type: ignore[attr-defined]
    torch.backends.mps.is_available = _no_mps

import pandas as pd

from bdbvhubrenewal.evaluation.groupings import (
    grouping_model,
    merged_task,
    split_arm,
)
from bdbvhubrenewal.forecasters.foundation import _chronos, chronos_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["ebola", "germany"])
    parser.add_argument("arm", help="model[@checkpoint]:grouping")
    parser.add_argument("--origin", default="2026-11-01")
    parser.add_argument("--date", default="2022-02-09")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    head, _, grouping = args.arm.rpartition(":")
    model, _, ckpt = head.partition("@")
    built = grouping_model(model, grouping, ckpt or None)
    if args.study == "ebola":
        from bdbvhubrenewal.data import load_daily, load_delays, load_target
        from bdbvhubrenewal.data.zones import load_zone_data
        from bdbvhubrenewal.evaluation import backtest as bt
        from bdbvhubrenewal.evaluation import store

        target = load_target("data")
        tasks = [
            store.origin_tasks(
                target, load_delays("data"),
                bt.load_config("config/backtest.toml"),
                load_daily("data"), load_zone_data("data"),
            )[args.origin]
        ]  # fmt: skip
    else:
        from bdbvhubrenewal.data.germany import hub_triangle, nowcast_tasks
        from bdbvhubrenewal.evaluation import germany as de

        config = de.load_germany_config("config/germany.toml")
        assert args.date < config.split_date
        tri = hub_triangle(
            Path("data/raw/germany") / Path(config.triangle).name
        )
        states, ages = nowcast_tasks(
            tri, args.date, config.n_samples, config.seed
        )
        tasks = (
            [merged_task(states, ages)]
            if grouping == "joint"
            else [states, ages]
        )
    sizes: list[int] = []

    def sample(groups, horizon, n, seed):
        for g in groups:
            sizes.append(len(g["target"]) + len(g.get("past_covariates", {})))
        return chronos_sampler(groups, horizon, n, seed, built.checkpoint)

    built = dataclasses.replace(built, sample=sample)
    _chronos(built.checkpoint)
    t0 = time.time()
    for task in tasks:
        built.forecast(task)
    secs = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
    name, option = split_arm(f"{head}:{grouping}")
    row = pd.DataFrame(
        [{
            "study": args.study, "model": name, "option": option,
            "series": "+".join(str(s) for s in sizes),
            "memory_gb": round(gb, 2), "seconds": round(secs, 1),
        }]
    )  # fmt: skip
    out = Path(args.out)
    row.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(row.to_string(index=False, header=False), flush=True)


if __name__ == "__main__":
    main()
