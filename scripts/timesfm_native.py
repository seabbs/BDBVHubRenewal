"""Score TimesFM 3 from its own quantiles, beside Chronos-2's native run.

Usage:
  python scripts/timesfm_native.py germany LABEL OUT SHARD/N MODEL...
  python scripts/timesfm_native.py bdbv LABEL OUT CELLS MODEL...

As `chronos_native.py` and `native_run.py` (germany-chronos-dev) do for
Chronos-2:

- Weekly TimesFM models: every hub target is one of TimesFM 3's steps, so
  its quantiles are TimesFM 3's deciles read at the scored levels
  (`timesfm.extend_quantiles`: linear between deciles, a normal through
  the two outer deciles beyond them).
- Daily TimesFM models: a week is a sum of steps, so it is the sample
  quantile of `N_DRAWS` trajectories from the extended deciles, drawn with
  the random stream `chronos_native` uses (`STREAM`, keyed on the task),
  so TimesFM 3 and Chronos-2 share their uniform draws.
- Any other model (randomwalknowcast): the quantiles of its own samples,
  as `germany-backtest` scores it (`germany_forecasts`).

`germany` writes OUT/LABEL/<date>.csv.gz for the development dates of
shard SHARD of N (dates i with i % N == SHARD), skipping dates written.
`bdbv` writes OUT/LABEL/scores.csv and quantiles.csv.gz for the release
backtest's complete-target cells in CELLS (a `nowcast-scores.csv`).
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.cuda.is_available = lambda: False  # type: ignore[assignment]
torch.set_num_threads(1)

from bdbvhubrenewal.evaluation.backtest import run_forecasts  # noqa: E402
from bdbvhubrenewal.forecasters import get_model, timesfm  # noqa: E402
from bdbvhubrenewal.forecasters.base import ForecastTask  # noqa: E402
from bdbvhubrenewal.forecasters.foundation import (  # noqa: E402
    FoundationModel,
    quantile_draws,
)

N_DRAWS = 1000
STREAM = "chronos-common"


def _native(levels: np.ndarray):
    """A sampler returning TimesFM 3's quantiles at `levels` as paths."""

    def sample(groups, horizon, n, seed):  # noqa: ARG001
        out = []
        for q in timesfm.timesfm_quantiles(groups, horizon):
            at = timesfm.extend_quantiles(
                np.asarray(timesfm.TIMESFM_LEVELS), q, levels
            )  # [V, H, K]
            out.append(np.moveaxis(at, -1, 0))
        return out

    return sample


def _common(task: ForecastTask):
    """A sampler drawing from the task's shared random stream."""

    def sample(groups, horizon, n, seed):  # noqa: ARG001
        levels = np.asarray(timesfm.EXTENDED_LEVELS)
        rng = task.rng(STREAM)
        out = []
        for q in timesfm.timesfm_quantiles(groups, horizon):
            q = timesfm.extend_quantiles(
                np.asarray(timesfm.TIMESFM_LEVELS), q, levels
            )
            u = rng.uniform(size=(n, *q.shape[:2]))
            out.append(
                quantile_draws(levels, np.broadcast_to(q, (n, *q.shape)), u)
            )
        return out

    return sample


def timesfm_quantiles_at(
    model: FoundationModel, task: ForecastTask, levels: Sequence[float]
) -> pd.DataFrame:
    """Return one TimesFM model's forecasts of one task at `levels`.

    `run_forecasts` columns with `sample` replaced by `quantile`.
    """
    levels = np.asarray(levels, dtype=float)
    if model.source == "weekly":
        run = dataclasses.replace(model, sample=_native(levels))
        t = dataclasses.replace(task, n_samples=len(levels))
        out = run_forecasts([run], [t])
        out["quantile"] = levels[out["sample"].to_numpy() - 1]
        return out.drop(columns="sample")
    run = dataclasses.replace(model, sample=_common(task))
    t = dataclasses.replace(task, n_samples=N_DRAWS)
    out = run_forecasts([run], [t])
    keys = [c for c in out.columns if c not in ("sample", "value")]
    return (
        out.groupby(keys, dropna=False)["value"]
        .quantile(levels)
        .rename_axis([*keys, "quantile"])
        .reset_index()
    )


def germany(label: str, out: Path, shard: str, models: list[str]) -> None:
    """Nowcast each development date of a shard."""
    from bdbvhubrenewal.data.germany import (
        hub_forecasts,
        hub_triangle,
        nowcast_tasks,
    )
    from bdbvhubrenewal.evaluation import germany as de

    part, parts = (int(x) for x in shard.split("/"))
    config = de.load_germany_config("config/germany.toml")
    triangle = hub_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    fm = [m for m in models if m.startswith("fm_timesfm_")]
    other = [m for m in models if m not in fm]
    built = [get_model(m) for m in fm]
    root = out / label
    root.mkdir(parents=True, exist_ok=True)
    dates = de.nowcast_dates(config, held_out=False)
    for i, date in enumerate(dates):
        path = root / f"{date}.csv.gz"
        if i % parts != part or path.exists():
            continue
        frames = []
        if other:
            frames.append(
                de.germany_forecasts(
                    other, triangle, [date], config.n_samples, config.seed
                )
            )
        for task in nowcast_tasks(
            triangle, date, config.n_samples, config.seed
        ):
            for model in built:
                q = timesfm_quantiles_at(model, task, de.QUANTILE_LEVELS)
                hub = hub_forecasts(
                    q.rename(columns={"quantile": "sample"}), task, date
                )
                frames.append(
                    hub.rename(
                        columns={
                            "forecast_date": "origin",
                            "sample": "quantile",
                        }
                    )
                )
        fc = pd.concat(frames, ignore_index=True)
        fc = de.with_truth(fc, triangle, config.truth_delay)
        tmp = path.with_suffix(".tmp")
        fc.to_csv(tmp, index=False, compression="gzip")
        tmp.rename(path)
        print(date, flush=True)


def bdbv(label: str, out: Path, cells: Path, models: list[str]) -> None:
    """Score the release backtest's complete-target cells from quantiles."""
    from bdbvhubrenewal.data import load_daily
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation import store
    from bdbvhubrenewal.evaluation.scoring import (
        QUANTILE_LEVELS,
        score_quantiles,
    )

    config = bt.load_config()
    data = Path("data")
    tasks = store.origin_tasks(
        pd.read_csv(data / "target.csv", dtype={"kind": str}),
        pd.read_csv(data / "delays.csv.gz"),
        config,
        load_daily(data),
    )
    truth = pd.read_csv(cells)
    truth = truth.loc[truth["target"] == "complete"]
    truth = truth[
        ["origin", "truth_vintage", "location", "target_end_date", "truth"]
    ].drop_duplicates()
    frames = []
    for origin in sorted(set(truth["origin"])):
        if origin not in tasks:
            continue
        for name in models:
            q = timesfm_quantiles_at(
                get_model(name), tasks[origin], QUANTILE_LEVELS
            )
            frames.append(q.assign(origin=origin))
    fc = pd.concat(frames, ignore_index=True)
    fc = fc.loc[fc["as_of"].isna()].drop(columns="as_of")
    cols = ["origin", "truth_vintage", "location", "target_end_date"]
    fc = fc.merge(truth[cols], on=["origin", "location", "target_end_date"])
    scores = score_quantiles(
        fc[[*cols, "model", "horizon", "quantile", "value"]],
        truth,
        ("model", "origin", "truth_vintage", "location", "target_end_date"),
        QUANTILE_LEVELS,
    )
    root = out / label
    root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(root / "scores.csv", index=False)
    fc.to_csv(root / "quantiles.csv.gz", index=False, compression="gzip")
    print(f"{label}: {len(scores)} scored cells")


if __name__ == "__main__":
    study, label, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if study == "germany":
        germany(label, out, sys.argv[4], sys.argv[5:])
    else:
        bdbv(label, out, Path(sys.argv[4]), sys.argv[5:])
