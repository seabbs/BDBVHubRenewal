"""Fit each Chronos-2 model at each release-backtest origin, per grouping.

joint: the country, provinces and zones in one block.
level: the country, the provinces and the zones each in their own block.
bottom: the zones in one block, summed draw by draw to the provinces and
the country.

Tasks are those of main's release backtest (`store.origin_tasks`).
Writes one `ReleaseForecasts` directory per model, grouping and origin to
--out, named `<model>:<grouping>__<origin>`, and a timing row per fit to
--out/timings.csv.
"""

from __future__ import annotations

import os

if os.environ.get("AGG_CPU"):
    import torch

    # CPU only: the laptop GPU is held by the fine-tune trainer.
    _mps = torch.backends.mps.is_available

    def _no_mps() -> bool:
        return False

    _no_mps.__wrapped__ = _mps  # type: ignore[attr-defined]
    torch.backends.mps.is_available = _no_mps

import argparse
import dataclasses
import time
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.zones import load_zone_data
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import forecast_releases
from bdbvhubrenewal.forecasters import get_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--groupings", nargs="+", default=["joint", "level", "bottom"]
    )
    parser.add_argument("--origins", nargs="*")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = load_target(args.data)
    config = bt.load_config("config/backtest.toml")
    tasks = store.origin_tasks(
        target,
        load_delays(args.data),
        config,
        load_daily(args.data),
        load_zone_data(args.data),
    )
    for origin, task in tasks.items():
        if args.origins and origin not in args.origins:
            continue
        for name in args.models:
            for grouping in args.groupings:
                key = f"{name}:{grouping}"
                path = out / f"{key}__{origin}"
                if (path / "release-samples.csv.gz").exists():
                    continue
                model = dataclasses.replace(get_model(name), grouping=grouping)
                t0 = time.time()
                run = forecast_releases([task], target, [model])
                secs = time.time() - t0
                relabel = {
                    k: v.assign(model=key) if "model" in v else v
                    for k, v in dataclasses.asdict(run).items()
                }
                dataclasses.replace(run, **relabel).write(path)
                row = pd.DataFrame(
                    [{"model": key, "origin": origin, "seconds": round(secs)}]
                )
                timings = out / "timings.csv"
                row.to_csv(
                    timings, mode="a", header=not timings.exists(), index=False
                )
                print(row.to_string(index=False, header=False), flush=True)


if __name__ == "__main__":
    main()
