# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Zone backtest: the random walk models run at zone level.

Each real vintage before the truth vintage is forecast by `randomwalk` and
`randomwalknowcast` on each zone's own release snapshots (`zone_task`),
for every zone with cases at the vintage.
The zone delay, observation model and walk spread are each the zone's own
where it has at least 10 cases a week, else its province's, else
national (`pooling`).
Each model is scored as it runs, and again with its zone samples made to
sum to its province samples (`reconcile_zones`), which is the form
submitted.
Zones are scored on the zone count each later real vintage reports, with
the final-count samples drawn as for provinces
(`bdbvhubrenewal.evaluation.nowcast.draw_reported`) under the zone delays
fitted at the origin.
The energy score takes every zone and week of an origin and release as
one vector, from the same drawn samples.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.simulate import SimulatedOutbreak
from bdbvhubrenewal.data.snapshots import release_cutoff
from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    energy_tasks,
    make_tasks,
    real_vintages,
    score_tasks,
)
from bdbvhubrenewal.evaluation.nowcast import draw_reported, reported_share
from bdbvhubrenewal.evaluation.scoring import (
    energy_scores,
    relative_skill,
    score_forecasts,
)
from bdbvhubrenewal.forecasters.base import (
    ForecastTask,
    ModelUnavailableError,
    get_model,
)
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk
from bdbvhubrenewal.forecasters.zones import (
    ZoneAllocator,
    reconcile_zones,
    zone_task,
)

log = logging.getLogger(__name__)

__all__ = [
    "SUMMED",
    "ZONE_BASELINE",
    "scenario_zone_scores",
    "zone_backtest",
    "zone_relative_skill",
]

ZONE_BASELINE = "randomwalk"
"""Name of the zone baseline in the scores."""

SUMMED = " (summed to provinces)"
"""Suffix of a model whose zones were made to sum to its provinces."""

_KEYS = (
    "model",
    "origin",
    "target",
    "truth_vintage",
    "location",
    "target_end_date",
)
"""Columns that identify one zone forecast scored against one release."""


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
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
    models: Sequence[str] = ("randomwalk", "randomwalknowcast"),
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score the random walk models run at zone level on later releases.

    Parameters
    ----------
    target : pandas.DataFrame
        The province target table (`load_target`).
    delays : pandas.DataFrame or None
        The delay table (`load_delays`).
    daily : pandas.DataFrame
        The daily table of the country and provinces (`load_daily`).
    daily_zones : pandas.DataFrame
        The daily zone table (`load_daily_zones`).
    zones : tuple of str
        The hub's zone codes (`hub_zones`). Those with cases at a vintage
        are forecast from it.
    models : sequence of str
        Registered models that read release snapshots.
    config : BacktestConfig, optional
        Horizons, samples and seed. Defaults to `BacktestConfig()`.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores, `score_forecasts` output with `model`, `origin`,
        `target` (`"reported"`), `truth_vintage` and `horizon` in weeks
        from the reference date, one row per model, origin, later
        vintage, zone and week. Each model appears under its name and
        again under its name with `SUMMED`.
        Then `energy_scores` of every model's zones by `origin`, `target`
        and `truth_vintage`, from the same drawn samples. Then the zone
        forecasts made to sum to the provinces, as submitted, under each
        model's name: `model`, `origin`, `reference_date`, `location`,
        `target_end_date`, `sample` and `value`.
    """
    config = config or BacktestConfig()
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    frames, energy, kept = [], [], []
    for vintage in vintages[:-1]:
        task = make_tasks(
            real.loc[real["vintage"] == vintage], delays, config, daily=daily
        )[0]
        counts = weekly_zone_counts(daily_zones, vintage)
        active = tuple(z for z in zones if z in set(counts["location"]))
        zt = zone_task(task, counts, active, pd.concat([daily, daily_zones]))
        try:
            origin = RandomWalk().origin_at(zt, errors=False)
        except ModelUnavailableError as err:
            log.warning("No zone delay fit at %s: %s", vintage, err)
            continue
        forecasts = {}
        for name in models:
            model = get_model(name)
            direct = model.forecast(zt)
            prov = model.forecast(task)
            forecasts[name] = direct
            forecasts[name + SUMMED] = reconcile_zones(
                direct, prov, task, counts, active
            )
            kept.append(
                forecasts[name + SUMMED].assign(
                    model=name,
                    origin=vintage,
                    reference_date=task.reference_date.isoformat(),
                )
            )
        weeks = sorted(set(direct["target_end_date"]))
        now = reported_share(origin, _cutoff(daily, vintage), weeks)
        reported = counts.rename(columns={"total": "reported"})[
            ["location", "target_end_date", "reported"]
        ]
        # Each model draws from its own stream, as in release_backtest.
        rngs = {name: task.rng(name) for name in forecasts}
        for later in (v for v in vintages if v > vintage):
            then = reported_share(origin, _cutoff(daily, later), weeks)
            truth = _zone_truth(
                weekly_zone_counts(daily_zones, later), direct, later
            )
            drawn = pd.concat(
                [
                    draw_reported(
                        samples, reported, now, then, rngs[name]
                    ).assign(
                        model=name,
                        origin=vintage,
                        target="reported",
                        truth_vintage=later,
                    )
                    for name, samples in forecasts.items()
                ],
                ignore_index=True,
            )
            drawn["horizon"] = _weeks(drawn, task)
            frames.append(score_forecasts(drawn, truth, keys=_KEYS))
            energy.append(energy_scores(drawn, truth, unit=_KEYS[1:4]))
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(energy, ignore_index=True),
        pd.concat(kept, ignore_index=True)[
            [
                "model",
                "origin",
                "reference_date",
                "location",
                "target_end_date",
                "sample",
                "value",
            ]
        ],
    )


def _weeks(samples: pd.DataFrame, task: ForecastTask) -> pd.Series:
    """Weeks from the task's reference date to each sample's week."""
    ref = pd.Timestamp(task.reference_date)
    return (pd.to_datetime(samples["target_end_date"]) - ref).dt.days // 7


def _cutoff(daily: pd.DataFrame, vintage: str) -> dt.date:
    """Return the cut-off of a release."""
    return release_cutoff(daily.loc[daily["vintage"] == vintage], vintage)


def zone_relative_skill(
    scores: pd.DataFrame, by: Sequence[str] = ()
) -> pd.DataFrame:
    """Return each model's mean WIS and log WIS relative to zone randomwalk.

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
    return relative_skill(
        scores, baseline=ZONE_BASELINE, by=by, match_on=_KEYS[1:]
    )


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score allocated zone forecasts on a scenario.

    As `zone_backtest`, with the simulated zone counts
    (`SimulatedOutbreak.zone_vintage`) in place of the real ones, scored
    against the final simulated counts and as reported `later_weeks`
    after each forecast date. Scenarios have no zone snapshots, so there
    is no zone random walk.

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
    tuple of pandas.DataFrame
        `score_tasks` output with `truth_view`, and `energy_tasks` output
        with `truth_view`.
    """
    allocator = allocator or ZoneAllocator()
    zones = tuple(sorted({z for _, z, _ in outbreak.settings.zone_shares}))
    frames = []
    for task in tasks:
        counts = outbreak.zone_vintage(task.vintage)
        for name in models:
            try:
                prov = get_model(name).forecast(task)
            except ModelUnavailableError as err:
                log.warning("%s skipped %s: %s", name, task.vintage, err)
                continue
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
    energy = [
        energy_tasks(forecasts, final, end, complete_after_weeks).assign(
            truth_view="final"
        )
    ]
    view = f"as of {later_weeks} weeks later"
    for task in tasks:
        later = (
            as_date(task.vintage) + dt.timedelta(weeks=later_weeks)
        ).isoformat()
        rows = forecasts.loc[forecasts["vintage"] == task.vintage]
        truth = _zone_truth(outbreak.zone_vintage(later), rows, later)
        scores.append(
            score_tasks(rows, truth, later, complete_after_weeks).assign(
                truth_view=view
            )
        )
        energy.append(energy_tasks(rows, truth, later).assign(truth_view=view))
    out = pd.concat(scores, ignore_index=True)
    return (
        out.loc[out["scoring"] == "as reported"].reset_index(drop=True),
        pd.concat(energy, ignore_index=True),
    )
