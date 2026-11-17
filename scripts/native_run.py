"""Score Chronos-2 models from native quantiles (`chronos_native`).

Usage:
  python scripts/native_run.py germany LABEL OUT MODEL...
  python scripts/native_run.py bdbv LABEL OUT CELLS MODEL...

`germany` writes OUT/LABEL/<date>.csv.gz per development date (skipping
dates already written): quantiles at the hub's seven levels with the
40-day truth, as `germany-backtest` writes them. `bdbv` writes
OUT/LABEL/scores.csv: the release backtest's complete-target cells and
truth from CELLS (a `nowcast-scores.csv`), scored from quantiles at the
backtest's 23 WIS levels. The fine-tuned models load the checkpoint
`BDBV_CHRONOS_FT` names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chronos  # noqa: F401
import peft  # noqa: F401
import torch

torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
torch.cuda.is_available = lambda: False  # type: ignore[assignment]
torch.set_num_threads(1)

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from chronos_native import chronos_quantiles  # noqa: E402

from bdbvhubrenewal.forecasters import get_model  # noqa: E402


def germany(label: str, out: Path, models: list[str]) -> None:
    """Nowcast each development date from native quantiles."""
    from bdbvhubrenewal.data.germany import (
        hub_forecasts,
        hub_triangle,
        nowcast_tasks,
    )
    from bdbvhubrenewal.evaluation import germany as de

    config = de.load_germany_config("config/germany.toml")
    triangle = hub_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    built = [get_model(m) for m in models]
    root = out / label
    root.mkdir(parents=True, exist_ok=True)
    for date in de.nowcast_dates(config, held_out=False):
        path = root / f"{date}.csv.gz"
        if path.exists():
            continue
        frames = []
        for task in nowcast_tasks(
            triangle, date, config.n_samples, config.seed
        ):
            for model in built:
                q = chronos_quantiles(model, task, de.QUANTILE_LEVELS)
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
            q = chronos_quantiles(
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
        germany(label, out, sys.argv[4:])
    else:
        bdbv(label, out, Path(sys.argv[4]), sys.argv[5:])
