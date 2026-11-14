"""Nowcast the German benchmark's development half with each grouping.

joint: Germany, the states and the age groups in one block (the states
and age groups tasks merged into one).
level: Germany, the states and the age groups each in their own block.
bottom: the states in one block, summed draw by draw to Germany, and the
age groups in another.
Germany is taken from the states task, as `hub_forecasts` does.

Writes one CSV of quantiles per date to --out, with `randomwalknowcast`.
Dates on or after the held-out split are refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data.germany import hub_forecasts, hub_triangle
from bdbvhubrenewal.data.germany import nowcast_tasks
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.reporting.hubtables import hub_quantiles

KEYS = ["model", "origin", "location", "target_end_date", "horizon"]


def merged(states, ages):
    """One task of Germany, the states and the age groups."""

    def both(a, b):
        return (
            pd.concat([a, b], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return dataclasses.replace(
        states,
        locations=(*states.locations, *ages.locations),
        target=both(states.target, ages.target),
        target_history=both(states.target_history, ages.target_history),
        daily_history=both(states.daily_history, ages.daily_history),
    )


def quantiles(samples, task, date, label):
    drawn = hub_forecasts(samples, task, date).rename(
        columns={"forecast_date": "origin", "sample": "output_type_id"}
    )
    table = drawn.assign(model=label, output_type="sample")[
        [*KEYS, "output_type", "output_type_id", "value"]
    ]
    q = hub_quantiles(table, de.QUANTILE_LEVELS)
    return q.drop(columns="output_type").rename(
        columns={"output_type_id": "quantile"}
    )[[*KEYS, "quantile", "value"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/germany")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--groupings", nargs="+", default=["joint", "level", "bottom"]
    )
    parser.add_argument("--part", default="0/1", help="k/n: every nth date")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = de.load_germany_config("config/germany.toml")
    dates = de.nowcast_dates(config)
    assert max(dates) < config.split_date
    k, n = (int(x) for x in args.part.split("/"))
    dates = dates[k::n]
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for date in dates:
        path = out / f"{date}.csv.gz"
        if path.exists():
            continue
        states, ages = nowcast_tasks(
            triangle, date, config.n_samples, config.seed
        )
        frames = []
        t0 = time.time()
        base = get_model("randomwalknowcast")
        for task in (states, ages):
            frames.append(
                quantiles(run_forecasts([base], [task]), task, date, base.name)
            )
        for name in args.models:
            for grouping in args.groupings:
                model = dataclasses.replace(get_model(name), grouping=grouping)
                label = f"{name}:{grouping}"
                if grouping == "joint":
                    task = merged(states, ages)
                    samples = run_forecasts([model], [task])
                    # The states task keeps Germany in `hub_forecasts`.
                    frames.append(quantiles(samples, states, date, label))
                    continue
                for task in (states, ages):
                    samples = run_forecasts([model], [task])
                    frames.append(quantiles(samples, task, date, label))
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"{date} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
