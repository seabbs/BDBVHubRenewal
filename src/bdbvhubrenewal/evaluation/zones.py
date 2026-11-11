# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Zone backtest: models that read release snapshots, run at zone level.

Each real vintage before the truth vintage is forecast on each zone's
own release snapshots (`zone_task`), for every zone with cases at the
vintage.
The functions default to the two random walks, and the CLI's
`zone-backtest` adds `pprandomwalknowcast`, which is fitted at each
origin.
The zone delay, observation model and walk spread are each the zone's own
where it has at least 10 cases a week, else its province's, else
national (`pooling`).
Each model is scored as it runs, and again with its zone samples made to
sum to its province samples (`reconcile_zones`), which is the form
submitted.
Zones are scored on the zone count each later real vintage reports.
`randomwalk` is scored on its samples as they are, with no delay.
`randomwalknowcast`'s final-count samples are drawn as for provinces
(`bdbvhubrenewal.evaluation.nowcast.draw_reported`) under its own zone
delay at the origin.
The energy score takes every zone and week of an origin and release as
one vector, from the same drawn samples.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.releases import release_cutoff
from bdbvhubrenewal.data.simulate import SimulatedOutbreak
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    energy_tasks,
    make_tasks,
    real_vintages,
    score_tasks,
)
from bdbvhubrenewal.evaluation.nowcast import complete_weeks, reported_samples
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
from bdbvhubrenewal.forecasters.zones import (
    ZoneAllocator,
    reconcile_zones,
    zone_task,
)

log = logging.getLogger(__name__)

__all__ = [
    "SUMMED",
    "ZONE_BASELINE",
    "ZoneForecasts",
    "scenario_zone_scores",
    "score_zone_allocations",
    "score_zones",
    "zone_allocations",
    "zone_backtest",
    "zone_forecasts",
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
    origins: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score models run at zone level on later releases.

    Two targets are scored, as for the provinces (`release_backtest`): what
    each later release reported, and the complete weeks of the latest
    release, decided by `complete_weeks` from the zone counts of every
    release, a week with no onset counting as 0.

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
    origins : collection of str, optional
        Forecast only from these vintages, so blocks of origins can run
        apart and their outputs be stacked. Every release is still read
        for the truth and the complete weeks. All but the latest if None.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores, `score_forecasts` output with `model`, `origin`,
        `target` (`"reported"` or `"complete"`), `truth_vintage` and
        `horizon` in weeks
        from the reference date, one row per model, origin, later
        vintage, zone and week. Each model appears under its name and
        again under its name with `SUMMED`.
        Then `energy_scores` of every model's zones by `origin`, `target`
        and `truth_vintage`, from the same drawn samples. Then the zone
        forecasts made to sum to the provinces, as submitted, under each
        model's name: `model`, `origin`, `reference_date`, `location`,
        `target_end_date`, `sample` and `value`.
    """
    run = zone_forecasts(
        target, delays, daily, daily_zones, zones, models, config, origins
    )
    return score_zones(target, delays, daily, daily_zones, run, config)


@dataclass(frozen=True)
class ZoneForecasts:
    """The forecasts `zone_backtest` scores, made apart from scoring.

    Every table has `origin`, so the forecasts of models run apart can be
    stacked (`read`) and scored together (`score_zones`).

    Attributes
    ----------
    runs : pandas.DataFrame
        One row per origin and model that forecast there, in the order
        run: `origin` and `model`.
    samples : pandas.DataFrame
        Zone samples under `key`, the model's name for its direct zone
        forecast and the name with `SUMMED` for it made to sum to the
        provinces: `key`, `origin`, `location`, `target_end_date`,
        `sample` and `value`.
    delays : pandas.DataFrame
        Each model's delay summary at each origin, if it has one.
    """

    runs: pd.DataFrame
    samples: pd.DataFrame
    delays: pd.DataFrame

    _NAMES = ("runs", "samples", "delays")

    def write(self, out: Path | str) -> None:
        """Write the tables to `out` as `zone-forecast-<table>.csv.gz`."""
        Path(out).mkdir(parents=True, exist_ok=True)
        for name in self._NAMES:
            write_csv(
                getattr(self, name), Path(out) / f"zone-forecast-{name}.csv.gz"
            )

    @classmethod
    def read(cls, parts: Sequence[Path | str]) -> ZoneForecasts:
        """Stack the tables `write` wrote to each part, in order."""
        tables = {}
        for name in cls._NAMES:
            frames = []
            for part in parts:
                try:
                    frames.append(
                        pd.read_csv(
                            Path(part) / f"zone-forecast-{name}.csv.gz",
                            dtype={"origin": str, "vintage": str},
                        )
                    )
                except pd.errors.EmptyDataError:
                    continue
            # A part with no rows (a block every model skips) adds none.
            frames = [f for f in frames if len(f)] or frames[:1]
            tables[name] = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=["origin", "model"])
            )
        return cls(**tables)


def _zone_task(
    real: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    config: BacktestConfig,
    vintage: str,
) -> ForecastTask:
    """Return the province task at an origin."""
    return make_tasks(
        real.loc[real["vintage"] == vintage], delays, config, daily=daily
    )[0]


def zone_forecasts(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
    models: Sequence[str] = ("randomwalk", "randomwalknowcast"),
    config: BacktestConfig | None = None,
    origins: Collection[str] | None = None,
) -> ZoneForecasts:
    """Forecast the zones from every origin, without scoring.

    The forecasting half of `zone_backtest`.

    Parameters
    ----------
    target, delays, daily, daily_zones, zones, models, config, origins
        As for `zone_backtest`.

    Returns
    -------
    ZoneForecasts
        The direct and summed zone samples and the delay summaries.
    """
    config = config or BacktestConfig()
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    runs: list[dict[str, str]] = []
    samples, summaries = [], []
    for vintage in vintages[:-1]:
        if origins is not None and vintage not in origins:
            continue
        task = _zone_task(real, delays, daily, config, vintage)
        counts = weekly_zone_counts(daily_zones, vintage)
        active = tuple(z for z in zones if z in set(counts["location"]))
        zt = zone_task(task, counts, active, pd.concat([daily, daily_zones]))
        for name in models:
            model = get_model(name)
            try:
                direct = model.forecast(zt)
                summary = getattr(model, "delay_summary", None)
                prov = model.forecast(task)
            except ModelUnavailableError as err:
                log.warning("%s has no zone forecast at %s: %s", name,
                            vintage, err)  # fmt: skip
                continue
            runs.append({"origin": vintage, "model": name})
            summed = reconcile_zones(direct, prov, task, counts, active)
            for key, table in ((name, direct), (name + SUMMED, summed)):
                samples.append(table.assign(key=key, origin=vintage))
            if summary is not None:
                summaries.append(summary.assign(origin=vintage))

    def stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=["origin", "model"])
        return pd.concat(frames, ignore_index=True)

    return ZoneForecasts(
        pd.DataFrame(runs, columns=["origin", "model"]),
        stack(samples),
        stack(summaries),
    )


def score_zones(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    run: ZoneForecasts,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score the forecasts of `zone_forecasts`.

    The scoring half of `zone_backtest`, which it equals given the same
    models and origins.

    Parameters
    ----------
    target, delays, daily, daily_zones, config
        As for `zone_backtest`.
    run : ZoneForecasts
        The forecasts, of models run together or stacked.

    Returns
    -------
    tuple of pandas.DataFrame
        As `zone_backtest`.
    """
    config = config or BacktestConfig()
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    latest = vintages[-1]
    done = _complete_zone_weeks(daily_zones, daily, vintages)
    latest_counts = weekly_zone_counts(daily_zones, latest)
    runs = run.runs.astype({"origin": str})
    frames, energy, kept = [], [], []
    for vintage in vintages[:-1]:
        here = runs.loc[runs["origin"] == vintage, "model"].astype(str)
        if here.empty:
            continue
        task = _zone_task(real, delays, daily, config, vintage)
        counts = weekly_zone_counts(daily_zones, vintage)
        forecasts: dict[str, pd.DataFrame] = {}
        summaries: dict[str, pd.DataFrame | None] = {}
        for name in here:
            for key in (name, name + SUMMED):
                mine = (run.samples["key"] == key) & (
                    run.samples["origin"].astype(str) == vintage
                )
                forecasts[key] = (
                    run.samples.loc[mine]
                    .drop(columns=["key", "origin"])
                    .reset_index(drop=True)
                )
            summary = _summary(run.delays, name, vintage)
            summaries[name] = summaries[name + SUMMED] = summary
            kept.append(
                forecasts[name + SUMMED].assign(
                    model=name,
                    origin=vintage,
                    reference_date=task.reference_date.isoformat(),
                )
            )
        if not forecasts:
            continue
        reported = counts.rename(columns={"total": "reported"})[
            ["location", "target_end_date", "reported"]
        ]
        # Each model draws from its own stream, as in release_backtest.
        rngs = {name: task.rng(name) for name in forecasts}
        later = [v for v in vintages if v > vintage]
        by_model = {
            key: reported_samples(
                samples,
                summaries[key],
                key.removesuffix(SUMMED),
                vintage,
                later,
                daily,
                reported,
                rngs[key],
            )
            for key, samples in forecasts.items()
        }
        for o2 in later:
            drawn = [
                draws[o2].assign(
                    model=key,
                    origin=vintage,
                    target="reported",
                    truth_vintage=o2,
                )
                for key, draws in by_model.items()
                if o2 in draws
            ]
            if not drawn:
                continue
            both = pd.concat(drawn, ignore_index=True)
            truth = _zone_truth(weekly_zone_counts(daily_zones, o2), both, o2)
            both["horizon"] = _weeks(both, task)
            frames.append(score_forecasts(both, truth, keys=_KEYS))
            energy.append(energy_scores(both, truth, unit=_KEYS[1:4]))
        # The final-count samples on the latest release's complete weeks.
        final = pd.concat(
            [
                samples.assign(
                    model=key,
                    origin=vintage,
                    target="complete",
                    truth_vintage=latest,
                )
                for key, samples in forecasts.items()
            ],
            ignore_index=True,
        )
        truth = _zone_truth(latest_counts, final, latest).merge(done)
        final["horizon"] = _weeks(final, task)
        frames.append(score_forecasts(final, truth, keys=_KEYS))
        energy.append(energy_scores(final, truth, unit=_KEYS[1:4]))
    if not frames:
        # No origin forecast, as in a block of origins every model skips.
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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


def _summary(
    table: pd.DataFrame, model: str, origin: str
) -> pd.DataFrame | None:
    """Return a model's delay summary at an origin, or None."""
    if table.empty or "origin" not in table:
        return None
    mine = table.loc[
        (table["model"] == model) & (table["origin"].astype(str) == origin)
    ]
    if mine.empty:
        return None
    return mine.drop(columns="origin").reset_index(drop=True)


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


def _complete_zone_weeks(
    daily_zones: pd.DataFrame, daily: pd.DataFrame, vintages: Sequence[str]
) -> pd.DataFrame:
    """Return the zones and weeks the latest release reports in full.

    `complete_weeks` on the zone counts of every release, over every zone
    and week with an onset in any release, a missing week counting as 0.
    """
    counts = {v: weekly_zone_counts(daily_zones, v) for v in vintages}
    every = pd.concat(counts.values())
    grid = pd.MultiIndex.from_product(
        [sorted(set(every["location"])), sorted(set(every["target_end_date"]))],
        names=["location", "target_end_date"],
    )
    target = pd.concat(
        [
            c.set_index(["location", "target_end_date"])["total"]
            .reindex(grid, fill_value=0)
            .reset_index()
            .assign(kind="real", vintage=v)
            for v, c in counts.items()
        ],
        ignore_index=True,
    )
    weeks = complete_weeks(target, daily, vintages[-1])
    return weeks.loc[weeks["complete"], ["location", "target_end_date"]]


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
    forecasts = zone_allocations(outbreak, tasks, models, allocator)
    return score_zone_allocations(
        outbreak,
        [t.vintage for t in tasks],
        forecasts,
        end,
        later_weeks,
        complete_after_weeks,
    )


def zone_allocations(
    outbreak: SimulatedOutbreak,
    tasks: Sequence[ForecastTask],
    models: Sequence[str],
    allocator: ZoneAllocator | None = None,
) -> pd.DataFrame:
    """Split each model's province forecasts of a scenario across zones.

    The forecasting half of `scenario_zone_scores`, with its parameters.

    Returns
    -------
    pandas.DataFrame
        The zone samples, with the columns `score_tasks` matches on.
    """
    allocator = allocator or ZoneAllocator()
    zones = tuple(sorted({z for _, z, _ in outbreak.settings.zone_shares}))
    frames = []
    for task in tasks:
        lag = (as_date(task.vintage) - as_date(task.cutoff)).days
        counts = outbreak.zone_vintage(task.vintage, cutoff_lag=lag)
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
    return pd.concat(frames, ignore_index=True)


def score_zone_allocations(
    outbreak: SimulatedOutbreak,
    dates: Sequence[str],
    forecasts: pd.DataFrame,
    end: str,
    later_weeks: int,
    complete_after_weeks: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score zone forecasts of a scenario.

    The scoring half of `scenario_zone_scores`.

    Parameters
    ----------
    outbreak : SimulatedOutbreak
        The scenario.
    dates : sequence of str
        Its forecast dates, in order.
    forecasts : pandas.DataFrame
        Output of `zone_allocations`.
    end, later_weeks, complete_after_weeks
        As for `scenario_zone_scores`.

    Returns
    -------
    tuple of pandas.DataFrame
        As `scenario_zone_scores`.
    """
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
    for date in dates:
        later = (as_date(date) + dt.timedelta(weeks=later_weeks)).isoformat()
        rows = forecasts.loc[forecasts["vintage"] == date]
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
