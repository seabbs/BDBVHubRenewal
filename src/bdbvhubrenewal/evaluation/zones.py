# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Zone backtest: allocated zone forecasts against zone persistence.

Each real vintage before the truth vintage is forecast by the province
models, and their samples are split across zones (`ZoneAllocator`).
The baseline, `zonepersistence`, runs persistence on each zone's own
series (`zone_task`), which does not sum to the provinces.
Every zone of a forecast province is scored, including zones with no
cases, against the zone counts as reported at the truth vintage.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.simulate import SimulatedOutbreak
from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    real_vintages,
    score_tasks,
)
from bdbvhubrenewal.evaluation.scoring import relative_skill
from bdbvhubrenewal.forecasters.base import ForecastTask, get_model
from bdbvhubrenewal.forecasters.persistence import Persistence
from bdbvhubrenewal.forecasters.zones import ZoneAllocator, zone_task

__all__ = [
    "ZONE_BASELINE",
    "scenario_zone_scores",
    "zone_backtest",
    "zone_relative_skill",
]

ZONE_BASELINE = "zonepersistence"
"""Name of the zone baseline in the scores."""


def _frame(
    samples: pd.DataFrame, task: ForecastTask, name: str
) -> pd.DataFrame:
    """Add the columns `score_tasks` matches on."""
    out = samples.copy()
    ref = pd.Timestamp(task.reference_date)
    out["horizon"] = (pd.to_datetime(out["target_end_date"]) - ref).dt.days // 7
    out.insert(0, "model", name)
    out.insert(1, "kind", task.kind)
    out.insert(2, "vintage", task.vintage)
    out.insert(3, "reference_date", task.reference_date.isoformat())
    return out


def zone_backtest(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
    models: Sequence[str] = ("persistence",),
    allocator: ZoneAllocator | None = None,
    config: BacktestConfig | None = None,
    truth_date: str | None = None,
) -> pd.DataFrame:
    """Score allocated zone forecasts and zone persistence.

    Parameters
    ----------
    target : pandas.DataFrame
        The province target table (`load_target`).
    delays : pandas.DataFrame or None
        The delay table (`load_delays`).
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    zones : tuple of str
        The hub's zone codes (`hub_zones`).
    models : sequence of str
        Registered province models whose samples are allocated.
    allocator : ZoneAllocator, optional
        The split. Defaults to `ZoneAllocator()`.
    config : BacktestConfig, optional
        Horizons, samples and seed. Defaults to `BacktestConfig()`.
    truth_date : str, optional
        The truth vintage. Defaults to the latest real vintage.

    Returns
    -------
    pandas.DataFrame
        `score_tasks` output for the final counts, one row per model,
        vintage, zone and week, with models named by their province
        model and `ZONE_BASELINE`.
    """
    allocator = allocator or ZoneAllocator()
    config = config or BacktestConfig()
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    truth_date = truth_date or vintages[-1]
    frames = []
    for vintage in (v for v in vintages if v < truth_date):
        task = make_tasks(real.loc[real["vintage"] == vintage], delays, config)[
            0
        ]
        counts = weekly_zone_counts(daily_zones, vintage)
        zt = zone_task(task, counts, zones)
        frames.append(_frame(Persistence().forecast(zt), task, ZONE_BASELINE))
        for name in models:
            prov = get_model(name).forecast(task)
            frames.append(
                _frame(
                    allocator.allocate(prov, task, counts, zones), task, name
                )
            )
    forecasts = pd.concat(frames, ignore_index=True)
    truth = weekly_zone_counts(daily_zones, truth_date)
    weeks = sorted(
        w for w in forecasts["target_end_date"].unique() if w <= truth_date
    )
    grid = pd.MultiIndex.from_product(
        [sorted(forecasts["location"].unique()), weeks],
        names=["location", "target_end_date"],
    )
    truth = (
        truth.set_index(["location", "target_end_date"])["total"]
        .reindex(grid, fill_value=0)
        .rename("truth")
        .reset_index()
    )
    scores = score_tasks(forecasts, truth, truth_date)
    return scores.loc[scores["scoring"] == "as reported"].reset_index(drop=True)


def zone_relative_skill(
    scores: pd.DataFrame, by: Sequence[str] = ()
) -> pd.DataFrame:
    """Return each model's mean WIS and log WIS relative to zone persistence.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `zone_backtest`.
    by : sequence of str
        Extra columns to compare within, such as `("horizon",)`.

    Returns
    -------
    pandas.DataFrame
        See `bdbvhubrenewal.evaluation.scoring.relative_skill`.
    """
    return relative_skill(scores, baseline=ZONE_BASELINE, by=by)


def _zone_truth(
    counts: pd.DataFrame, forecasts: pd.DataFrame, truth_date: str
) -> pd.DataFrame:
    """Zone counts as a truth table over the forecast zones and weeks."""
    weeks = sorted(
        w for w in forecasts["target_end_date"].unique() if w <= truth_date
    )
    grid = pd.MultiIndex.from_product(
        [sorted(forecasts["location"].unique()), weeks],
        names=["location", "target_end_date"],
    )
    return (
        counts.set_index(["location", "target_end_date"])["total"]
        .reindex(grid, fill_value=0)
        .rename("truth")
        .reset_index()
    )


def scenario_zone_scores(
    outbreak: SimulatedOutbreak,
    tasks: Sequence[ForecastTask],
    models: Sequence[str],
    end: str,
    later_weeks: int,
    complete_after_weeks: int = 0,
    allocator: ZoneAllocator | None = None,
) -> pd.DataFrame:
    """Score allocated zone forecasts and zone persistence on a scenario.

    As `zone_backtest`, with the simulated zone counts
    (`SimulatedOutbreak.zone_vintage`) in place of the real ones, scored
    against the final simulated counts and as reported `later_weeks`
    after each forecast date.

    Parameters
    ----------
    outbreak : SimulatedOutbreak
        The scenario.
    tasks : sequence of ForecastTask
        Its province tasks.
    models : sequence of str
        Registered province models whose samples are allocated.
    end : str
        Date of the final simulated data.
    later_weeks : int
        Weeks after each date for the as-of scores.
    complete_after_weeks : int
        As for `score_tasks`.
    allocator : ZoneAllocator, optional
        The split. Defaults to `ZoneAllocator()`.

    Returns
    -------
    pandas.DataFrame
        `score_tasks` output with `truth_view`.
    """
    allocator = allocator or ZoneAllocator()
    zones = tuple(sorted({z for _, z, _ in outbreak.settings.zone_shares}))
    frames = []
    for task in tasks:
        counts = outbreak.zone_vintage(task.vintage)
        zt = zone_task(task, counts, zones)
        frames.append(_frame(Persistence().forecast(zt), task, ZONE_BASELINE))
        for name in models:
            prov = get_model(name).forecast(task)
            frames.append(
                _frame(
                    allocator.allocate(prov, task, counts, zones), task, name
                )
            )
    forecasts = pd.concat(frames, ignore_index=True)
    final = _zone_truth(outbreak.zone_vintage(end), forecasts, end)
    scores = [
        score_tasks(forecasts, final, end, complete_after_weeks).assign(
            truth_view="final"
        )
    ]
    for task in tasks:
        later = (
            as_date(task.vintage) + dt.timedelta(weeks=later_weeks)
        ).isoformat()
        rows = forecasts.loc[forecasts["vintage"] == task.vintage]
        truth = _zone_truth(outbreak.zone_vintage(later), rows, later)
        scores.append(
            score_tasks(rows, truth, later, complete_after_weeks).assign(
                truth_view=f"as of {later_weeks} weeks later"
            )
        )
    out = pd.concat(scores, ignore_index=True)
    return out.loc[out["scoring"] == "as reported"].reset_index(drop=True)
