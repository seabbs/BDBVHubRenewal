# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Backtest our models on the German hospitalisation nowcast benchmark.

The benchmark is that of Wolffram et al. (2023): nowcasts of the 7-day
hospitalisation incidence by Meldedatum in Germany, made each day from
2021-11-22 to 2022-04-29, for Germany, its states and its age groups.
`germany_backtest` runs the models on the tasks of each nowcast date
(`bdbvhubrenewal.data.germany.nowcast_tasks`), for the sums ending 0, 7,
14, 21 and 28 days back (`HORIZONS`), and scores them as the backtest does
(`score_forecasts`).
Each sum's truth is its count in the data $D$ days after it ends, the
paper's 40-day truth (`rolling_truth`).
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data.germany import (
    HORIZONS,
    hub_forecasts,
    hub_target,
    level,
    nowcast_tasks,
)
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.evaluation.scoring import score_forecasts
from bdbvhubrenewal.forecasters import get_model

__all__ = [
    "GermanyConfig",
    "fetch_germany",
    "germany_backtest",
    "load_germany_config",
    "nowcast_dates",
]

_KEYS = ("model", "target", "origin", "truth_vintage", "location")
"""Columns that, with the week, identify a scored forecast."""


@dataclass(frozen=True)
class GermanyConfig:
    """The benchmark's source and settings (`config/germany.toml`).

    Attributes
    ----------
    repo, commit, triangle : str
        The hub repository, its commit and the reporting triangle's path.
    first_date, last_date : str
        The first and last nowcast date.
    models : tuple of str
        The models run.
    n_samples, seed : int
        Samples per forecast, and the seed.
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.
    """

    repo: str
    commit: str
    triangle: str
    first_date: str
    last_date: str
    models: tuple[str, ...]
    n_samples: int
    seed: int
    truth_delay: int


def load_germany_config(
    path: Path | str = "config/germany.toml",
) -> GermanyConfig:
    """Read the benchmark settings.

    Parameters
    ----------
    path : pathlib.Path or str
        The TOML file.

    Returns
    -------
    GermanyConfig
        The settings.
    """
    raw = tomllib.loads(Path(path).read_text())
    hub, now = raw["hub"], raw["nowcast"]
    return GermanyConfig(
        repo=hub["repo"],
        commit=hub["commit"],
        triangle=hub["triangle"],
        first_date=now["first_date"],
        last_date=now["last_date"],
        models=tuple(now["models"]),
        n_samples=int(now["n_samples"]),
        seed=int(now["seed"]),
        truth_delay=int(raw["truth"]["delay"]),
    )


def nowcast_dates(config: GermanyConfig) -> list[str]:
    """Return every nowcast date, oldest first.

    Parameters
    ----------
    config : GermanyConfig
        Gives the first and last date.

    Returns
    -------
    list of str
        ISO dates.
    """
    days = pd.date_range(config.first_date, config.last_date, freq="D")
    return [d.date().isoformat() for d in days]


def fetch_germany(  # pragma: no cover - network
    config: GermanyConfig, dest: Path | str
) -> Path:
    """Download the reporting triangle at its commit.

    Parameters
    ----------
    config : GermanyConfig
        The source.
    dest : pathlib.Path or str
        Directory the triangle goes to, by its base name, with
        `sources.json`.

    Returns
    -------
    pathlib.Path
        The triangle.
    """
    from bdbvhubrenewal.data.tidy import _request

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://raw.githubusercontent.com/{config.repo}/{config.commit}/"
        f"{config.triangle}"
    )
    path = out / Path(config.triangle).name
    path.write_bytes(_request(url, None))
    sources = {
        "repo": config.repo,
        "commit": config.commit,
        "path": config.triangle,
        "accessed": dt.date.today().isoformat(),
    }
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    return path


def germany_backtest(
    models: Sequence[str],
    triangle: pd.DataFrame,
    dates: Sequence[str],
    n_samples: int,
    seed: int,
    truth_delay: int,
) -> pd.DataFrame:
    """Nowcast each date with each model and score the nowcasts.

    Parameters
    ----------
    models : sequence of str
        Registered model names.
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date
        (`bdbvhubrenewal.data.germany.read_triangle`).
    dates : sequence of str
        Nowcast dates.
    n_samples, seed : int
        Samples per forecast, and the seed.
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.

    Returns
    -------
    pandas.DataFrame
        `score_forecasts` output, with `target` (`"<D>-day"`), `origin`
        (the nowcast date), `truth_vintage` (the report date of the
        truth), `location` (the place), `horizon` (days) and `pool` (the
        paper's level, `national`, `states` or `age`). Sums with no data
        $D$ days after them are not scored.
    """
    ends = sorted(
        {
            (dt.date.fromisoformat(d) + dt.timedelta(days=h)).isoformat()
            for d in dates
            for h in HORIZONS
        }
    )
    truth = rolling_truth(hub_target(triangle, ends), truth_delay)
    built = [get_model(m) for m in models]
    frames = []
    for date in dates:
        for task in nowcast_tasks(triangle, date, n_samples, seed):
            samples = run_forecasts(built, [task])
            if samples.empty:
                continue
            drawn = (
                hub_forecasts(samples, task, date)
                .rename(columns={"forecast_date": "origin"})
                .merge(
                    truth[["location", "target_end_date", "truth_vintage"]],
                    on=["location", "target_end_date"],
                )
            )
            frames.append(
                score_forecasts(
                    drawn.assign(target=f"{truth_delay}-day"),
                    truth,
                    keys=(*_KEYS, "target_end_date"),
                )
            )
    if not frames:
        return pd.DataFrame(columns=[*_KEYS, "target_end_date", "pool"])
    scores = pd.concat(frames, ignore_index=True)
    scores["pool"] = scores["location"].map(level)
    return scores
