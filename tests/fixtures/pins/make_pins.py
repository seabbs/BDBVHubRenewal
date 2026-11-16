"""Write the 7-day outputs the daily target must leave unchanged.

Run once from the repository root, on the code before the daily target:
  PYTHONPATH=src python tests/fixtures/pins/make_pins.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.germany import read_triangle
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.evaluation.germany import germany_forecasts
from bdbvhubrenewal.forecasters import foundation, get_model

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "tests" / "fixtures" / "data"
GERMANY = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)


class Pipeline:
    """A stand-in Chronos-2: each level's quantile grows with the step."""

    quantiles = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

    def predict_quantiles(self, groups, prediction_length, quantile_levels):
        import torch

        level = torch.tensor(quantile_levels, dtype=torch.float32)
        step = torch.arange(1, prediction_length + 1, dtype=torch.float32)
        out = []
        for g in groups:
            n = np.atleast_2d(g["target"]).shape[0]
            q = 5.0 * step[:, None] * (0.5 + level[None, :])
            row = torch.arange(1, n + 1, dtype=torch.float32)
            out.append(row[:, None, None] * q[None])
        return out, None


def bdbv_tasks():
    target, delays, daily = (
        load_target(DATA),
        load_delays(DATA),
        load_daily(DATA),
    )
    real = target.loc[target["kind"] == "real"]
    out = {}
    for v in ("2026-10-17", "2026-10-24"):
        rows = target.loc[target["vintage"] == v]
        task = make_tasks(
            rows, delays, BacktestConfig(seed=3), history=real, daily=daily
        )[0]
        out[v] = dataclasses.replace(task, n_samples=40)
    return out


def main() -> None:
    foundation._chronos = lambda *_: Pipeline()
    frames = []
    for v, task in bdbv_tasks().items():
        for name in (
            "randomwalknowcast",
            "fm_chronos_daily_ragged",
            "fm_chronos_weekly_ragged",
        ):
            got = get_model(name).forecast(task)
            frames.append(got.assign(model=name, vintage=v))
    pd.concat(frames).to_csv(OUT / "bdbv-7day.csv.gz", index=False)
    triangle = read_triangle(GERMANY)
    de = germany_forecasts(
        [
            "randomwalknowcast",
            "fm_chronos_daily_ragged",
            "fm_chronos_weekly_ragged",
        ],
        triangle,
        ["2021-10-04"],
        n_samples=40,
        seed=1,
    )
    de.to_csv(OUT / "germany-7day.csv.gz", index=False)


if __name__ == "__main__":
    main()
