# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The held-out synthetic evaluation of the paper (issue #206).

The held-out outbreaks come from `scripts/finetune_simulate.R` with seeds
the fine-tune never saw. Each outbreak is released weekly, on one weekday
from Wednesday to Saturday drawn by the outbreak, as for the training
blocks (`scripts/finetune_chronos.py`). A release's vintage is its
cut-off.

`origin_cutoffs` picks `n` origins per outbreak, spread evenly over the
releases from the `MIN_RELEASES`-th to the last whose forecast weeks are
final in the simulation. `outbreak_tasks` builds a `ForecastTask` at
each origin, holding the weekly and daily releases up to it. The
reference date is the Saturday ending the cut-off's week, and the
forecast weeks are `HORIZONS` weeks from it: four nowcast weeks, the
cut-off's week and three forecast weeks. The models keep their own
release lag $D$ (`foundation.MAX_LAG`, 4 weeks or 28 days, inside the
fine-tune's 2 to 7 weeks).

The truth is the final simulated count of each place and week, with the
country the sum of the places. `evaluate` runs the models and returns
one long table of `HUB_QUANTILE_LEVELS` quantiles of their samples.
"""

from __future__ import annotations

import datetime as dt
import logging
import zlib
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import epiweek_end
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    get_model,
)
from bdbvhubrenewal.forecasters.finetune import releases, weekly_cutoffs
from bdbvhubrenewal.reporting.hubtables import HUB_QUANTILE_LEVELS

__all__ = [
    "COLUMNS",
    "COMPLETE_WEEKS",
    "HORIZONS",
    "MIN_RELEASES",
    "MODELS",
    "evaluate",
    "models_by_name",
    "origin_cutoffs",
    "outbreak_tasks",
    "quantile_table",
    "series_label",
    "truth_table",
]

log = logging.getLogger(__name__)

MODELS: Final = (
    "randomwalknowcast",
    "fm_chronos_weekly_ragged",
    "fm_chronos_daily_ragged",
    "fm_chronos_weekly_ragged_ft",
    "fm_chronos_daily_ragged_ft",
)
"""The baseline, the zero-shot ragged models and the fine-tuned ones."""

MIN_RELEASES: Final = 6
"""Releases up to the first origin: two weeks of context past the
weekly models' four-week $D$."""

COMPLETE_WEEKS: Final = 13
"""Weeks from an origin to the simulation's last report, so its last
forecast week is final under the longest delay (eight weeks)."""

HORIZONS: Final = tuple(range(-4, 4))
"""Forecast weeks from the reference date."""

COLUMNS: Final = (
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "value",
    "truth",
)
"""The long table's columns (`paper/schema.md`)."""


def series_label(outbreak: str, place: str) -> str:
    """Return the `location` of a place of an outbreak, unique over them."""
    return f"{outbreak}_{place}"


def origin_cutoffs(
    counts: pd.DataFrame, outbreak: str, n_origins: int
) -> tuple[list[str], list[str]]:
    """Return an outbreak's weekly cut-offs and its origins among them.

    Parameters
    ----------
    counts : pandas.DataFrame
        The outbreak's triangle, as `finetune.releases` reads it.
    outbreak : str
        Its id, which seeds the release weekday.
    n_origins : int
        Origins wanted. Fewer are returned if fewer releases qualify.

    Returns
    -------
    tuple of list of str
        Every release's cut-off up to the last origin, and the origins.
    """
    rng = np.random.default_rng(zlib.crc32(outbreak.encode()))
    weekday = int(rng.integers(2, 6))
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = end - dt.timedelta(weeks=COMPLETE_WEEKS)
    cutoffs = weekly_cutoffs(
        str(counts["report_date"].min()), last.isoformat(), weekday
    )
    if len(cutoffs) < MIN_RELEASES:
        return cutoffs, []
    at = np.linspace(MIN_RELEASES - 1, len(cutoffs) - 1, n_origins)
    picks = sorted({round(i) for i in at})
    return cutoffs, [cutoffs[i] for i in picks]


def outbreak_tasks(
    counts: pd.DataFrame,
    outbreak: str,
    n_origins: int,
    n_samples: int = 1000,
    seed: int = 1,
) -> list[ForecastTask]:
    """Return a task at each origin of one outbreak.

    Parameters
    ----------
    counts : pandas.DataFrame
        The outbreak's triangle.
    outbreak : str
        Its id.
    n_origins : int
        Origins wanted (`origin_cutoffs`).
    n_samples : int
        Sample trajectories per forecast.
    seed : int
        Seed of the tasks.

    Returns
    -------
    list of ForecastTask
        One per origin. A single place is the country, with no provinces.
    """
    cutoffs, origins = origin_cutoffs(counts, outbreak, n_origins)
    if not origins:
        return []
    weekly = releases(counts, "weekly", cutoffs)
    daily = releases(counts, "daily", cutoffs)
    weekly.insert(0, "kind", "simulated")
    daily.insert(0, "kind", "simulated")
    provinces = tuple(sorted(set(weekly["location"]) - {NATIONAL}))
    tasks = []
    for origin in origins:
        ref = epiweek_end(origin)
        target = weekly.loc[weekly["vintage"] == origin]
        tasks.append(
            ForecastTask(
                vintage=origin,
                reference_date=ref,
                kind="simulated",
                cutoff=origin,
                target=target.reset_index(drop=True),
                delays=None,
                locations=provinces,
                target_end_dates=tuple(
                    (ref + dt.timedelta(weeks=h)).isoformat() for h in HORIZONS
                ),
                n_samples=n_samples,
                seed=seed,
                target_history=weekly.loc[
                    weekly["vintage"] <= origin
                ].reset_index(drop=True),
                daily_history=daily.loc[daily["vintage"] <= origin].reset_index(
                    drop=True
                ),
            )
        )
    return tasks


def truth_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Return the final count of each place and week of an outbreak.

    Parameters
    ----------
    counts : pandas.DataFrame
        The outbreak's triangle.

    Returns
    -------
    pandas.DataFrame
        `location`, `target_end_date` and `truth`, the country included.
    """
    last = str(pd.Timestamp(counts["report_date"].max()).date())
    rows = releases(counts, "weekly", [last])
    return rows.rename(columns={"total": "truth"})[
        ["location", "target_end_date", "truth"]
    ]


def quantile_table(
    forecasts: pd.DataFrame, truth: pd.DataFrame, outbreak: str
) -> pd.DataFrame:
    """Return the long quantile table of one outbreak's forecasts.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `backtest.run_forecasts`.
    truth : pandas.DataFrame
        Output of `truth_table`. Weeks it lacks have no cases.
    outbreak : str
        The outbreak's id, which labels its places.

    Returns
    -------
    pandas.DataFrame
        Columns `COLUMNS`.
    """
    keys = ["model", "vintage", "location", "target_end_date", "horizon"]
    final = forecasts.loc[forecasts["as_of"].isna()]
    out = (
        final.groupby(keys)["value"]
        .quantile(list(HUB_QUANTILE_LEVELS))
        .rename_axis([*keys, "quantile_level"])
        .reset_index()
    )
    out = out.merge(truth, on=["location", "target_end_date"], how="left")
    out["truth"] = out["truth"].fillna(0.0)
    out["location"] = [series_label(outbreak, p) for p in out["location"]]
    out.insert(0, "study", "synthetic")
    out = out.rename(
        columns={"vintage": "origin_date", "target_end_date": "target_date"}
    )
    return out[list(COLUMNS)]


def models_by_name(
    names: Sequence[str],
) -> tuple[list[ForecastModel], list[str]]:
    """Build the registered models among `names`.

    Returns
    -------
    tuple
        The models, and the names that are not registered.
    """
    models, skipped = [], []
    for name in names:
        try:
            models.append(get_model(name))
        except KeyError:
            skipped.append(name)
    return models, skipped


def evaluate(
    counts: pd.DataFrame,
    models: Sequence[ForecastModel],
    n_origins: int,
    n_samples: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    """Forecast one outbreak at each origin and return the long table.

    A model that raises `ModelUnavailableError` has no rows for that
    task. One without its fine-tuned checkpoint (`FileNotFoundError`) has
    none for the outbreak.

    Parameters
    ----------
    counts : pandas.DataFrame
        One outbreak's triangle, with its `outbreak` column.
    models : sequence of ForecastModel
        The models.
    n_origins, n_samples, seed
        As `outbreak_tasks`.

    Returns
    -------
    pandas.DataFrame
        Columns `COLUMNS`.
    """
    outbreak = str(counts["outbreak"].iloc[0])
    tasks = outbreak_tasks(counts, outbreak, n_origins, n_samples, seed)
    frames = []
    for model in models:
        try:
            frames.append(run_forecasts([model], tasks))
        except FileNotFoundError as err:
            log.warning("%s skipped %s: %s", model.name, outbreak, err)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    forecasts = pd.concat(frames, ignore_index=True)
    return quantile_table(forecasts, truth_table(counts), outbreak)
