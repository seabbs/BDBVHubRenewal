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
cut-off's week and three forecast weeks. The table gives each horizon in
days from the origin, as `paper/schema.md` asks. The models keep their own
release lag $D$ (`foundation.MAX_LAG`, 4 weeks or 28 days, inside the
fine-tune's 2 to 7 weeks).

The truth is the final simulated count of each place and week, with the
country the sum of the places. `evaluate` runs the models and returns
one long table of their samples, `quantile_level` empty.
"""

from __future__ import annotations

import dataclasses
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
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk

__all__ = [
    "COLUMNS",
    "COMPLETE_WEEKS",
    "FACTORS",
    "HORIZONS",
    "MIN_RELEASES",
    "MODELS",
    "OBSERVATION_COLUMNS",
    "delay_quantile",
    "evaluate",
    "factor_table",
    "model_task",
    "models_by_name",
    "observation_table",
    "origin_cutoffs",
    "outbreak_tasks",
    "release_cutoffs",
    "sample_table",
    "series_label",
    "truth_table",
]

log = logging.getLogger(__name__)

MODELS: Final = (
    "randomwalknowcast",
    "randomwalk",
    "fm_chronos_weekly_complete",
    "fm_chronos_daily_complete",
    "fm_chronos_weekly_ragged",
    "fm_chronos_daily_ragged",
    "fm_chronos_weekly_ragged_ft",
    "fm_chronos_daily_ragged_ft",
)
"""The baseline, the random walk, the zero-shot Chronos-2 models on complete
data and on the ragged releases, and the fine-tuned ragged ones."""

MIN_RELEASES: Final = 6
"""Releases up to the first origin: two weeks of context past the
weekly models' four-week $D$."""

COMPLETE_WEEKS: Final = 13
"""Weeks from an origin to the simulation's last report, so its last
forecast week is final under the longest delay (eight weeks)."""

MAX_DELAY_DAYS: Final = 56
"""The simulations' longest reporting delay."""

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
    "sample_id",
    "value",
    "truth",
)
"""The long table's columns (`paper/schema.md`)."""

FACTORS: Final = (
    "delay_mean",
    "delay_sd",
    "delay_drift",
    "hazard_sd",
    "dow_sd",
    "report_days",
    "log_r0",
    "n_steps",
    "gt_mean",
    "n_places",
    "coupling",
    "sqrt_phi",
    "weeks",
    "cases",
)
"""Outbreak settings (`scripts/finetune_simulate.R`) that scores are
broken down by, as `factor_table` gives them."""

OBSERVATION_COLUMNS: Final = (
    "study",
    "location",
    "target_date",
    "as_of_date",
    "value",
)
"""The observations table's columns (`paper/schema.md`)."""


def series_label(outbreak: str, place: str) -> str:
    """Return the `location` of a place of an outbreak, unique over them."""
    return f"{outbreak}_{place}"


def release_cutoffs(counts: pd.DataFrame, outbreak: str) -> list[str]:
    """Return an outbreak's weekly release cut-offs to its last report.

    Releases fall on one weekday from Wednesday to Saturday, drawn from
    the outbreak's id, as the training blocks' do.

    Parameters
    ----------
    counts : pandas.DataFrame
        The outbreak's triangle, as `finetune.releases` reads it.
    outbreak : str
        Its id.

    Returns
    -------
    list of str
        The cut-offs, ISO format.
    """
    rng = np.random.default_rng(zlib.crc32(outbreak.encode()))
    weekday = int(rng.integers(2, 6))
    return weekly_cutoffs(
        str(counts["report_date"].min()),
        str(counts["report_date"].max()),
        weekday,
    )


def origin_cutoffs(
    counts: pd.DataFrame, outbreak: str, n_origins: int
) -> tuple[list[str], list[str]]:
    """Return an outbreak's weekly cut-offs and its origins among them.

    Parameters
    ----------
    counts : pandas.DataFrame
        The outbreak's triangle.
    outbreak : str
        Its id.
    n_origins : int
        Origins wanted. Fewer are returned if fewer releases qualify.

    Returns
    -------
    tuple of list of str
        The release cut-offs up to the last origin, and the origins.
    """
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = (end - dt.timedelta(weeks=COMPLETE_WEEKS)).isoformat()
    cutoffs = [c for c in release_cutoffs(counts, outbreak) if c <= last]
    if len(cutoffs) < MIN_RELEASES:
        return cutoffs, []
    at = np.linspace(MIN_RELEASES - 1, len(cutoffs) - 1, n_origins)
    picks = sorted({round(i) for i in at})
    return cutoffs, [cutoffs[i] for i in picks]


def outbreak_tasks(
    counts: pd.DataFrame,
    outbreak: str,
    n_origins: int,
    n_samples: int = 200,
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


def sample_table(
    forecasts: pd.DataFrame, truth: pd.DataFrame, outbreak: str
) -> pd.DataFrame:
    """Return the long sample table of one outbreak's forecasts.

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
    keep = ["model", "vintage", "location", "target_end_date", "sample"]
    out = forecasts.loc[forecasts["as_of"].isna(), [*keep, "value"]]
    out = out.rename(columns={"sample": "sample_id"})
    out["quantile_level"] = np.nan
    out = out.merge(truth, on=["location", "target_end_date"], how="left")
    out["truth"] = out["truth"].fillna(0.0)
    out["location"] = [series_label(outbreak, p) for p in out["location"]]
    out.insert(0, "study", "synthetic")
    out = out.rename(
        columns={"vintage": "origin_date", "target_end_date": "target_date"}
    )
    out["horizon"] = (
        pd.to_datetime(out["target_date"]) - pd.to_datetime(out["origin_date"])
    ).dt.days
    return out[list(COLUMNS)]


def observation_table(
    counts: pd.DataFrame, outbreak: str, n_origins: int
) -> pd.DataFrame:
    """Return the weekly counts as of each release, and the final counts.

    The releases run to the last report, so `paperfigs.truth.rolling_truth`
    finds the count reported D days after each target.

    Parameters
    ----------
    counts : pandas.DataFrame
        One outbreak's triangle.
    outbreak : str
        Its id.
    n_origins : int
        As `origin_cutoffs`. An outbreak with no origin has no rows.

    Returns
    -------
    pandas.DataFrame
        `paper/schema.md`'s observations: `study`, `location`,
        `target_date`, `as_of_date` and `value`. The last `as_of_date` is
        the simulation's last report, which gives the truth.
    """
    _, origins = origin_cutoffs(counts, outbreak, n_origins)
    if not origins:
        return pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    end = str(pd.Timestamp(counts["report_date"].max()).date())
    cutoffs = release_cutoffs(counts, outbreak)
    rows = releases(counts, "weekly", sorted({*cutoffs, end}))
    out = pd.DataFrame(
        {
            "study": "synthetic",
            "location": [series_label(outbreak, p) for p in rows["location"]],
            "target_date": rows["target_end_date"],
            "as_of_date": rows["vintage"],
            "value": rows["total"].astype(float),
        }
    )
    return out[list(OBSERVATION_COLUMNS)]


def model_task(model: ForecastModel, task: ForecastTask) -> ForecastTask:
    """Return the task as a model takes it.

    The random walk models forecast the task's provinces, so a
    single-place outbreak, which has none, gives them the country.
    """
    if not task.locations and isinstance(model, RandomWalk):
        return dataclasses.replace(task, locations=(NATIONAL,))
    return task


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
    n_samples: int = 200,
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
            mine = [model_task(model, t) for t in tasks]
            frames.append(run_forecasts([model], mine))
        except FileNotFoundError as err:
            log.warning("%s skipped %s: %s", model.name, outbreak, err)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    forecasts = pd.concat(frames, ignore_index=True)
    return sample_table(forecasts, truth_table(counts), outbreak)


def factor_table(counts: pd.DataFrame, settings: pd.DataFrame) -> pd.DataFrame:
    """Return each series of one outbreak with its outbreak's settings.

    Every outbreak has the same aggregation, release schedule rule and
    $D$, so only the simulation settings vary.

    Parameters
    ----------
    counts : pandas.DataFrame
        One outbreak's triangle, with its `outbreak` column.
    settings : pandas.DataFrame
        The simulator's `settings-*.csv` rows, one per outbreak.

    Returns
    -------
    pandas.DataFrame
        `location` (`series_label`), then the `FACTORS` in `settings`, raw,
        and `delay_q98`, the outbreak's 98% reporting delay in days
        (`delay_quantile`), the $D$ of its rolling truth.
    """
    outbreak = str(counts["outbreak"].iloc[0])
    places = sorted(set(truth_table(counts)["location"]))
    factors = [f for f in FACTORS if f in settings]
    row = settings.loc[settings["outbreak"] == outbreak, factors].iloc[0]
    out = pd.DataFrame(
        {"location": [series_label(outbreak, p) for p in places]}
    )
    for f in factors:
        out[f] = row[f]
    out["delay_q98"] = delay_quantile(counts, 0.98)
    return out


def delay_quantile(counts: pd.DataFrame, q: float) -> int:
    """Return a quantile of an outbreak's reporting delay, in whole days.

    The delay is from onset to report over every simulated case, counting
    only onsets at least `MAX_DELAY_DAYS` before the last report, so every
    delay of theirs is seen.

    Parameters
    ----------
    counts : pandas.DataFrame
        One outbreak's triangle.
    q : float
        The quantile, in (0, 1].

    Returns
    -------
    int
        The fewest days by which at least `q` of the cases are reported.
    """
    ref = pd.to_datetime(counts["reference_date"])
    rep = pd.to_datetime(counts["report_date"])
    seen = ref <= rep.max() - pd.Timedelta(days=MAX_DELAY_DAYS)
    delay = (rep - ref).dt.days[seen]
    by = counts.loc[seen, "count"].groupby(delay).sum().sort_index()
    share = by.cumsum() / by.sum()
    return int(share.index[np.argmax(share.to_numpy() >= q - 1e-12)])
