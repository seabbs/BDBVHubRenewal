# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Rolling-origin backtest over every real and simulated vintage.

Each vintage becomes a `ForecastTask`: the data as of that vintage, the
reference date it serves (`reference_date_for_vintage`) and the weeks
from `min_horizon` to `max_horizon`.
Every registered model forecasts every task.
Real forecasts are scored by `bdbvhubrenewal.evaluation.nowcast` on the
count each later release reported.

There are two kinds of vintage:

- **real**: published by the hub.
- **simulated**: from a simulated outbreak with known parameters
  (`bdbvhubrenewal.data.simulate`). Truth is the final simulated count.

Every real vintage but the latest is backtested.
`truth_changes` shows how much the truth moved between vintages.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.bvd import MAX_RHAT
from bdbvhubrenewal.data.dates import (
    as_date,
    epiweek_end,
    reference_date_for_vintage,
)
from bdbvhubrenewal.data.releases import cutoff_lags, release_cutoff
from bdbvhubrenewal.data.simulate import (
    SCENARIO_RT_SCALE,
    SimulatedOutbreak,
    SimulationSettings,
    calibrated_settings,
    release_dates,
    simulate_outbreak,
)
from bdbvhubrenewal.evaluation.scoring import energy_scores, score_forecasts
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastModel,
    ForecastTask,
    ModelUnavailableError,
    add_national,
    check_samples,
)
from bdbvhubrenewal.forecasters.chain import MissingSnapshots

__all__ = [
    "BacktestConfig",
    "apply_sampler",
    "check_distinct",
    "development_tasks",
    "energy_tasks",
    "forecast_locations",
    "forecast_weeks",
    "joint_order",
    "load_config",
    "lopo_tasks",
    "make_tasks",
    "merge_by_model",
    "real_vintages",
    "recovery_forecasts",
    "recovery_tasks",
    "run_development",
    "run_forecasts",
    "run_lopo",
    "run_recovery",
    "run_scenarios",
    "saturdays",
    "scenario_forecasts",
    "score_recovery",
    "score_scenarios",
    "score_tasks",
    "truth_changes",
    "truth_from_vintage",
]

log = logging.getLogger(__name__)

FORECAST_COLUMNS = (
    "model",
    "kind",
    "vintage",
    "reference_date",
    *SAMPLE_COLUMNS,
    "as_of",
    "horizon",
)
"""Columns of `run_forecasts` output, which is empty if no model could run."""

_FORECAST_DTYPES = {
    "sample": "int64",
    "value": "float64",
    "as_of": "float64",
    "horizon": "int64",
}


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings, usually read from `config/backtest.toml`.

    Attributes
    ----------
    min_horizon, max_horizon : int
        Range of horizons in each forecast, in weeks.
    n_samples : int
        Samples per forecast.
    seed : int
        Random seed shared by all tasks.
    simulated_dates : tuple of str
        Vintage dates to publish from the simulated outbreak.
    complete_after_weeks : int
        $K$: a truth week counts as complete if it ended at least this
        many weeks before the truth vintage.
    sampler : dict
        NUTS settings of the fitted models (`apply_sampler`).
    bvd_max_rhat : float
        The limit on the joint fit's worst R-hat of a BVDOutbreakSize
        release the BVD variant may use
        (`bdbvhubrenewal.data.bvd.select_release`).
    skip_origins : tuple of str
        Real releases that are not backtest origins. They stay in the
        histories of later origins.
    """

    min_horizon: int = -4
    max_horizon: int = 3
    n_samples: int = 100
    seed: int = 0
    simulated_dates: tuple[str, ...] = ()
    complete_after_weeks: int = 0
    sampler: dict[str, Any] = field(default_factory=dict)
    bvd_max_rhat: float = MAX_RHAT
    skip_origins: tuple[str, ...] = ()


def apply_sampler(config: BacktestConfig) -> None:
    """Set the fitted models' NUTS settings from the backtest config.

    The settings go in the `SAMPLER_ENV` environment variable, which
    models read when they are built. Settings already there are kept.

    Parameters
    ----------
    config : BacktestConfig
        Gives `sampler`.
    """
    from bdbvhubrenewal.forecasters.renewal import SAMPLER_ENV

    if config.sampler and not os.environ.get(SAMPLER_ENV):
        os.environ[SAMPLER_ENV] = json.dumps(config.sampler)


def saturdays(first: str, last: str) -> tuple[str, ...]:
    """List Saturdays from `first` to `last`, inclusive.

    Parameters
    ----------
    first, last : str
        ISO dates.

    Returns
    -------
    tuple of str
        The Saturdays in order.
    """
    start = epiweek_end(first)
    days = pd.date_range(start, as_date(last), freq="7D")
    return tuple(d.date().isoformat() for d in days)


def load_config(path: Path | str = "config/backtest.toml") -> BacktestConfig:
    """Read the backtest settings.

    Parameters
    ----------
    path : pathlib.Path or str
        The TOML file.

    Returns
    -------
    BacktestConfig
        The settings.
    """
    raw = tomllib.loads(Path(path).read_text())
    fc, sp, sim = raw["forecast"], raw["split"], raw["simulated"]
    return BacktestConfig(
        min_horizon=int(fc["min_horizon"]),
        max_horizon=int(fc["max_horizon"]),
        n_samples=int(fc["n_samples"]),
        seed=int(fc["seed"]),
        complete_after_weeks=int(sp["complete_after_weeks"]),
        sampler=dict(raw["sampler"]),
        bvd_max_rhat=float(raw["bvd"]["max_rhat"]),
        simulated_dates=saturdays(sim["first_vintage"], sim["last_vintage"]),
        skip_origins=tuple(str(d) for d in sp.get("skip_origins", ())),
    )


def forecast_weeks(
    reference_date: dt.date | str, config: BacktestConfig
) -> tuple[str, ...]:
    """Return the weeks a forecast covers.

    Parameters
    ----------
    reference_date : datetime.date or str
        The reference date.
    config : BacktestConfig
        Gives the horizon range.

    Returns
    -------
    tuple of str
        ISO Saturdays from `min_horizon` to `max_horizon`.
    """
    ref = as_date(reference_date)
    return tuple(
        (ref + dt.timedelta(weeks=h)).isoformat()
        for h in range(config.min_horizon, config.max_horizon + 1)
    )


def forecast_locations(vintage_rows: pd.DataFrame) -> tuple[str, ...]:
    """Return the provinces a vintage lists, with or without cases.

    Parameters
    ----------
    vintage_rows : pandas.DataFrame
        Target rows of one vintage.

    Returns
    -------
    tuple of str
        Province codes, sorted.
    """
    prov = vintage_rows.loc[vintage_rows["location"] != NATIONAL]
    return tuple(sorted(set(prov["location"])))


def make_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    config: BacktestConfig,
    report_by: Sequence[str] = (),
    history: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
) -> list[ForecastTask]:
    """Build one task per vintage in a target table.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows, possibly of several kinds and vintages.
    delays : pandas.DataFrame or None
        Delay line data with a `vintage` column, and optionally a `kind`
        column. A task gets the rows of its own vintage (and kind), or
        None if there are none.
    config : BacktestConfig
        Horizons, samples and seed.
    report_by : sequence of str
        Truth vintage dates. Each real task asks for the counts as
        reported by those after its own vintage, at their cut-offs, which
        `history` (or `target`) holds.
    history : pandas.DataFrame, optional
        Target rows to take each task's `target_history` from. Defaults
        to `target`.
    daily : pandas.DataFrame, optional
        Daily onsets by release, with `vintage`, `location`, `onset_date`,
        `observed` and `imputed` columns, and optionally `kind`. Each task
        gets the rows of its kind on or before its vintage as
        `daily_history`.

    Returns
    -------
    list of ForecastTask
        Tasks ordered by kind and vintage.

    Examples
    --------
    >>> import dataclasses
    >>> from bdbvhubrenewal.data import (
    ...     delays_for_vintage, load_daily, load_delays, load_target
    ... )
    >>> from bdbvhubrenewal.evaluation import backtest as bt
    >>> target = load_target()
    >>> real = target.loc[target["kind"] == "real"]
    >>> config = dataclasses.replace(
    ...     bt.load_config("config/backtest.toml"), min_horizon=-6
    ... )
    >>> vintage = "2026-10-17"
    >>> task = bt.make_tasks(
    ...     real.loc[real["vintage"] == vintage],
    ...     delays_for_vintage(load_delays(), vintage)[0],
    ...     config,
    ...     history=real,
    ...     daily=load_daily(),
    ... )[0]
    >>> task.reference_date, task.target_end_dates[-1]
    """
    tasks = []
    source = target if history is None else history
    for (kind, vintage), rows in target.groupby(["kind", "vintage"]):
        mine = source["kind"] == kind
        past_target = source.loc[mine & (source["vintage"] <= vintage)]
        ref = reference_date_for_vintage(str(vintage))
        dly, history = None, None
        if delays is not None:
            mask = delays["vintage"] == vintage
            if "kind" in delays.columns:
                mask &= delays["kind"] == kind
            sel = delays.loc[mask]
            dly = sel if not sel.empty else None
            same = delays["vintage"] <= vintage
            if "kind" in delays.columns:
                same &= delays["kind"] == kind
            past = delays.loc[same]
            history = past if not past.empty else None
        tasks.append(
            ForecastTask(
                vintage=str(vintage),
                reference_date=ref,
                kind=str(kind),
                cutoff=release_cutoff(rows, str(vintage)).isoformat(),
                target=rows.reset_index(drop=True),
                delays=dly,
                delay_history=history,
                target_history=past_target if not past_target.empty else None,
                daily_history=_daily_for(daily, str(kind), str(vintage)),
                locations=forecast_locations(rows),
                target_end_dates=forecast_weeks(ref, config),
                n_samples=config.n_samples,
                seed=config.seed,
                report_by={
                    d: release_cutoff(source.loc[mine], d).isoformat()
                    for d in sorted(report_by)
                    if kind != "simulated" and d > str(vintage)
                },
            )
        )
    return tasks


def _fit_rows(model: ForecastModel, task: ForecastTask) -> pd.DataFrame:
    """Return the model's summary of each fit it made for the task."""
    rows = getattr(model, "fit_diagnostics", None) or []
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.insert(0, "model", model.name)
    table.insert(1, "kind", task.kind)
    table.insert(2, "vintage", task.vintage)
    table.insert(3, "masked", "-".join(task.mask))
    return table


def check_distinct(forecasts: pd.DataFrame) -> None:
    """Stop if two models gave the same samples for a task.

    A model that silently ran another would give the same samples, as
    the daily random walk once did when it had no snapshots.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts` with `model`, `kind`, `vintage` and
        `value`.

    Raises
    ------
    ValueError
        If two models have identical values for the same task.
    """
    for key, rows in forecasts.groupby(["kind", "vintage"]):
        seen: dict[str, np.ndarray] = {}
        for name, grp in rows.groupby("model"):
            values = grp.sort_values(["location", "target_end_date", "sample"])[
                "value"
            ].to_numpy()
            for other, prior in seen.items():
                if prior.shape == values.shape and np.array_equal(
                    prior, values
                ):
                    raise ValueError(
                        f"{name} and {other} gave identical samples for {key}."
                    )
            seen[str(name)] = values


def run_forecasts(
    models: Sequence[ForecastModel],
    tasks: Iterable[ForecastTask],
    fits: list[pd.DataFrame] | None = None,
    learnt: list[pd.DataFrame] | None = None,
    zones: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run every model on every task, adding the national total.

    A model that raises `ModelUnavailableError` for a task, as the random
    walk models do when no pair of releases is within the delay window,
    has no rows for that task.
    A model with `needs_snapshots` set must be given the task's release
    snapshots, and the run stops if it is not.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to run.
    tasks : iterable of ForecastTask
        Tasks to run them on.
    fits : list, optional
        Collects a table of each model's fit summaries per task, for
        models that set `fit_diagnostics`.
    learnt : list, optional
        Collects each fit's learnt delay per task, for models that set
        `delay_summary`.
    zones : list, optional
        Collects each model's own zone samples per task, with `model`,
        `kind` and `vintage`, for models that set `zone_samples` (the
        zone renewal variants).

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `kind`, `vintage`, `reference_date`, `horizon`,
        `location`, `target_end_date`, `sample`, `value` and `as_of`
        (missing for final counts).

    Examples
    --------
    >>> from bdbvhubrenewal.evaluation.backtest import run_forecasts
    >>> from bdbvhubrenewal.forecasters import get_model
    >>> models = [get_model("randomwalk"), get_model("randomwalknowcast")]
    >>> forecasts = run_forecasts(models, [task])  # tasks from `make_tasks`
    >>> forecasts.groupby("model")["sample"].nunique()
    """
    frames = []
    for task in tasks:
        for model in models:
            if getattr(model, "needs_snapshots", False) and (
                task.daily_history is None or task.daily_history.empty
            ):
                raise MissingSnapshots(
                    f"{model.name} needs daily release snapshots, and the "
                    f"{task.kind} task for {task.vintage} has none."
                )
            if hasattr(model, "fit_diagnostics"):
                model.fit_diagnostics = []
            try:
                samples = model.forecast(task)
            except ModelUnavailableError as err:
                log.warning(
                    "%s skipped %s %s: %s", model.name, task.kind,
                    task.vintage, err,
                )  # fmt: skip
                continue
            table = _fit_rows(model, task)
            if fits is not None and not table.empty:
                fits.append(table)
            delay = getattr(model, "delay_summary", None)
            if learnt is not None and delay is not None:
                learnt.append(delay)
            own = getattr(model, "zone_samples", None)
            if zones is not None and own is not None:
                zones.append(
                    own.assign(
                        model=model.name, kind=task.kind, vintage=task.vintage
                    )
                )
            check_samples(samples, task)
            if "as_of" not in samples:
                samples["as_of"] = np.nan
            samples = add_national(samples)
            ref = pd.Timestamp(task.reference_date)
            samples["horizon"] = (
                pd.to_datetime(samples["target_end_date"]) - ref
            ).dt.days // 7
            samples.insert(0, "model", model.name)
            samples.insert(1, "kind", task.kind)
            samples.insert(2, "vintage", task.vintage)
            samples.insert(3, "reference_date", task.reference_date.isoformat())
            frames.append(samples)
    if not frames:
        return pd.DataFrame(columns=FORECAST_COLUMNS).astype(_FORECAST_DTYPES)
    return pd.concat(frames, ignore_index=True)


def merge_by_model(path: Path | str, new: pd.DataFrame) -> pd.DataFrame:
    """Merge a run's rows into a results file, keeping other models' rows.

    The fitted variants are slow, so their rows come from the fit store
    (`bdbvhubrenewal.evaluation.store`). A run replaces the rows of the
    models it ran and keeps the rest, so a run of no model keeps them all.

    Parameters
    ----------
    path : pathlib.Path or str
        A results file with a `model` column. It need not exist.
    new : pandas.DataFrame
        The run's rows.

    Returns
    -------
    pandas.DataFrame
        The kept rows of other models, then `new`.
    """
    path = Path(path)
    if not path.exists():
        return new
    # An empty table still has a gzip header, so the size cannot tell.
    try:
        old = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return new
    # A run of no model replaces nothing.
    if new.empty:
        return old
    if "model" not in old or "model" not in new:
        return new
    kept = old.loc[~old["model"].isin(set(new["model"]))]
    if kept.empty:
        return new
    return pd.concat([kept, new], ignore_index=True)


def truth_from_vintage(target: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Return one vintage's totals as a truth table.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows.
    vintage : str
        The vintage to use as truth.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`.
    """
    rows = target.loc[target["vintage"] == vintage]
    return rows[["location", "target_end_date", "total"]].rename(
        columns={"total": "truth"}
    )


def score_tasks(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    truth_date: str,
    complete_after_weeks: int = 0,
) -> pd.DataFrame:
    """Score forecasts on every week against one truth vintage.

    Forecasts made at or after `truth_date` are dropped.
    Every week the truth covers is scored, as reported at `truth_date`.
    Final-count rows (no `as_of`) are scored `"as reported"`.
    Rows with `as_of` equal to `truth_date` predict the counts as reported
    then, and are scored `"vintage-matched"`. Rows for other dates are
    left out.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`.
    truth_date : str
        The date of the truth data.
    complete_after_weeks : int
        $K$: weeks that ended at least this many weeks before `truth_date`
        are flagged `complete`.

    Returns
    -------
    pandas.DataFrame
        Output of `bdbvhubrenewal.evaluation.scoring.score_forecasts`, with
        `truth_vintage`, `scoring` and `complete` columns.
    """
    keep = forecasts["vintage"] < truth_date
    as_of = (
        forecasts["as_of"]
        if "as_of" in forecasts
        else pd.Series(np.nan, index=forecasts.index)
    )
    frames = []
    for scoring, rows in (
        ("as reported", as_of.isna()),
        ("vintage-matched", as_of == truth_date),
    ):
        sel = forecasts.loc[keep & rows].drop(columns="as_of", errors="ignore")
        if sel.empty:
            continue
        scores = score_forecasts(sel, truth)
        scores["scoring"] = scoring
        frames.append(scores)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["truth_vintage"] = truth_date
    age = pd.Timestamp(truth_date) - pd.to_datetime(out["target_end_date"])
    out["complete"] = age.dt.days >= 7 * complete_after_weeks
    return out


def energy_tasks(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    truth_date: str,
    complete_after_weeks: int = 0,
) -> pd.DataFrame:
    """Score the final-count forecasts of `score_tasks` with the energy score.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`.
    truth_date : str
        The date of the truth data. Forecasts made at or after it are
        dropped.
    complete_after_weeks : int
        $K$: only weeks that ended at least this many weeks before
        `truth_date` are scored, the `complete` weeks of `score_tasks`.

    Returns
    -------
    pandas.DataFrame
        `bdbvhubrenewal.evaluation.scoring.energy_scores` by `kind`,
        `vintage` and `reference_date`, with `truth_vintage`.
    """
    keep = forecasts["vintage"] < truth_date
    if "as_of" in forecasts:
        keep &= forecasts["as_of"].isna()
    age = pd.Timestamp(truth_date) - pd.to_datetime(truth["target_end_date"])
    done = truth.loc[age.dt.days >= 7 * complete_after_weeks]
    out = energy_scores(
        forecasts.loc[keep], done, unit=("kind", "vintage", "reference_date")
    )
    out["truth_vintage"] = truth_date
    return out


def real_vintages(target: pd.DataFrame) -> list[str]:
    """List the real vintages in a target table, oldest first.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows.

    Returns
    -------
    list of str
        The vintage dates.
    """
    return sorted(set(target.loc[target["kind"] == "real", "vintage"]))


def truth_changes(
    target: pd.DataFrame, locations: Sequence[str] | None = None
) -> pd.DataFrame:
    """Show each week's truth in every real vintage and its change since.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows with every real vintage.
    locations : sequence of str, optional
        Locations to keep. All if None.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `target_end_date`, `truth_vintage`, `truth`
        and `change`, this vintage's value minus the latest one. A
        negative change means the week has been revised upwards since.
        Sorted by location, week and vintage.
    """
    real = target.loc[target["kind"] == "real"]
    if locations is not None:
        real = real.loc[real["location"].isin(list(locations))]
    out = real[["location", "target_end_date", "vintage", "total"]].rename(
        columns={"vintage": "truth_vintage", "total": "truth"}
    )
    latest = out.loc[out["truth_vintage"] == out["truth_vintage"].max()]
    out = out.merge(
        latest[["location", "target_end_date", "truth"]],
        on=["location", "target_end_date"],
        suffixes=("", "_latest"),
    )
    out["change"] = out["truth"] - out.pop("truth_latest")
    return out.sort_values(
        ["location", "target_end_date", "truth_vintage"]
    ).reset_index(drop=True)


def _daily_for(
    daily: pd.DataFrame | None, kind: str, vintage: str
) -> pd.DataFrame | None:
    """Return the daily releases of a kind on or before a vintage, or None."""
    if daily is None:
        return None
    rows = daily
    if "kind" in rows.columns:
        rows = rows.loc[rows["kind"] == kind]
    elif kind != "real":
        return None
    rows = rows.loc[rows["vintage"] <= vintage]
    return rows.reset_index(drop=True) if not rows.empty else None


def development_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    daily: pd.DataFrame | None = None,
) -> list[ForecastTask]:
    """Build the development tasks: every real vintage but the latest.

    Vintages in `config.skip_origins` are left out as origins but stay in
    the later tasks' histories.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    daily : pandas.DataFrame, optional
        Daily onsets by release, with `vintage`, `location`, `onset_date`,
        `observed` and `imputed` columns, and optionally `kind`. Each task
        gets the rows of its kind on or before its vintage as
        `daily_history`.

    Returns
    -------
    list of ForecastTask
        Tasks for every real vintage before the latest but the skipped.
    """
    real = target.loc[target["kind"] == "real"]
    latest = real_vintages(real)[-1]
    return [
        t
        for t in make_tasks(
            real,
            delays.assign(kind="real"),
            config,
            real_vintages(real),
            daily=daily,
        )
        if t.vintage < latest and t.vintage not in config.skip_origins
    ]


def run_development(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    fits: list[pd.DataFrame] | None = None,
    learnt: list[pd.DataFrame] | None = None,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast from every real vintage before the latest.

    The forecasts are scored by
    `bdbvhubrenewal.evaluation.nowcast.release_backtest`.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to compare.
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    fits : list, optional
        Collects fit summaries, as for `run_forecasts`.
    learnt : list, optional
        Collects learnt delays, as for `run_forecasts`.
    daily : pandas.DataFrame, optional
        Daily onsets by release, as for `development_tasks`.

    Returns
    -------
    pandas.DataFrame
        Output of `run_forecasts`.
    """
    tasks = development_tasks(target, delays, config, daily=daily)
    return run_forecasts(models, tasks, fits, learnt)


def lopo_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    provinces: Sequence[str] | None = None,
) -> list[ForecastTask]:
    """Build the development tasks with one province masked in each.

    Parameters
    ----------
    target, delays, config
        As for `development_tasks`.
    provinces : sequence of str, optional
        Provinces to mask. Defaults to every province with cases in each
        task.

    Returns
    -------
    list of ForecastTask
        Each with `mask` set to one province.
    """
    return [
        replace(task, mask=(prov,))
        for task in development_tasks(target, delays, config)
        for prov in task.locations
        if provinces is None or prov in provinces
    ]


def run_lopo(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    provinces: Sequence[str] | None = None,
    fits: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave one province out: forecast it with its counts masked.

    For each development task and each province with cases, the model
    gets the task with that province in `ForecastTask.mask`
    (`lopo_tasks`).
    Only the masked province's forecast is kept and scored, against the
    latest real vintage.
    Models without `supports_mask = True` cannot honour a mask and are
    skipped.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to check.
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    provinces : sequence of str, optional
        Provinces to mask, one at a time. Defaults to every province with
        cases in each task.
    fits : list, optional
        Collects fit summaries, as for `run_forecasts`.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts, with a `masked` column, and the scores. Both empty
        if no model supports a mask.
    """
    able = [m for m in models if getattr(m, "supports_mask", False)]
    for m in models:
        if m not in able:
            log.info("Skipping %s in leave-one-province-out: no mask.", m.name)
    if not able:
        return pd.DataFrame(), pd.DataFrame()
    real = target.loc[target["kind"] == "real"]
    latest = real_vintages(real)[-1]
    frames = []
    for task in lopo_tasks(target, delays, config, provinces=provinces):
        prov = task.mask[0]
        samples = run_forecasts(able, [task], fits)
        kept = samples.loc[samples["location"] == prov].copy()
        kept.insert(4, "masked", prov)
        frames.append(kept)
    forecasts = pd.concat(frames, ignore_index=True)
    scores = score_tasks(
        forecasts,
        truth_from_vintage(real, latest),
        latest,
        config.complete_after_weeks,
    )
    return forecasts, scores


def _release_days(outbreak: SimulatedOutbreak, dates: list[str]) -> list[str]:
    """Release days from the outbreak start to the last vintage date."""
    return release_dates(outbreak.settings.start, max(dates))


def recovery_tasks(
    config: BacktestConfig,
    settings: SimulationSettings | None = None,
    lags: Mapping[int, int] | None = None,
) -> list[ForecastTask]:
    """Build the tasks of the simulated vintages.

    Parameters
    ----------
    config : BacktestConfig
        Settings, including `simulated_dates`.
    settings : SimulationSettings, optional
        The simulated scenario. Defaults to `turning_point`.
    lags : mapping of int to int, optional
        Days from release to data cut-off by release weekday
        (`bdbvhubrenewal.data.releases.cutoff_lags`). None for none.

    Returns
    -------
    list of ForecastTask
        One task per simulated vintage, with the daily releases (Tuesday,
        Wednesday, Friday and Saturday) up to it as `daily_history`.
    """
    outbreak = simulate_outbreak(settings)
    dates = list(config.simulated_dates)
    target, delays = outbreak.vintages(dates, lags)
    daily = outbreak.daily_vintages(_release_days(outbreak, dates), lags)
    return make_tasks(target, delays, config, daily=daily)


def run_recovery(
    models: Sequence[ForecastModel],
    config: BacktestConfig,
    settings: SimulationSettings | None = None,
    fits: list[pd.DataFrame] | None = None,
    lags: Mapping[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run models on simulated vintages and score against the known truth.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to check.
    config : BacktestConfig
        Settings, including `simulated_dates`.
    settings : SimulationSettings, optional
        The simulated scenario. Defaults to `turning_point`.
    fits : list, optional
        Collects fit summaries, as for `run_forecasts`.
    lags : mapping of int to int, optional
        Release cut-off lags, as for `recovery_tasks`.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts, the scores and the energy scores (`energy_tasks`).
    """
    forecasts = recovery_forecasts(models, config, settings, fits, lags)
    scores, energy = score_recovery(forecasts, settings)
    return forecasts, scores, energy


def recovery_forecasts(
    models: Sequence[ForecastModel],
    config: BacktestConfig,
    settings: SimulationSettings | None = None,
    fits: list[pd.DataFrame] | None = None,
    lags: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Forecast the simulated vintages, the forecasting half of `run_recovery`.

    Parameters
    ----------
    models, config, settings, fits, lags
        As for `run_recovery`.

    Returns
    -------
    pandas.DataFrame
        Output of `run_forecasts`.
    """
    return run_forecasts(models, recovery_tasks(config, settings, lags), fits)


def joint_order(forecasts: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Put forecasts stacked by model in the order one run makes them.

    `run_forecasts` runs every model on a task before the next task. Rows
    are sorted by vintage, then by model in the order of `models`, and
    otherwise kept in order.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts with `vintage` and `model`.
    models : sequence of str
        The models in the order one run takes them.

    Returns
    -------
    pandas.DataFrame
        The rows reordered, with a fresh index.
    """
    if forecasts.empty:
        return forecasts.reset_index(drop=True)
    rank = forecasts["model"].map({m: i for i, m in enumerate(models)})
    order = np.lexsort((rank.to_numpy(), forecasts["vintage"].astype(str)))
    return forecasts.iloc[order].reset_index(drop=True)


def score_recovery(
    forecasts: pd.DataFrame, settings: SimulationSettings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score recovery forecasts, the scoring half of `run_recovery`.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `recovery_forecasts`, or of several stacked in
        `joint_order`.
    settings : SimulationSettings, optional
        As for `run_recovery`.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores and the energy scores, as `run_recovery`.
    """
    outbreak = simulate_outbreak(settings)
    end = outbreak.settings.start + dt.timedelta(days=outbreak.settings.n_days)
    truth = outbreak.truth()
    scores = score_tasks(forecasts, truth, end.isoformat())
    scores["truth_vintage"] = "simulated"
    energy = energy_tasks(forecasts, truth, end.isoformat())
    energy["truth_vintage"] = "simulated"
    return scores, energy


def _truth(rows: pd.DataFrame) -> pd.DataFrame:
    """Target rows as a truth table: `location`, `target_end_date`, `truth`."""
    out = rows.rename(columns={"total": "truth"})
    return out[["location", "target_end_date", "truth"]]


def run_scenarios(
    models: Sequence[ForecastModel],
    config: BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    zones: pd.DataFrame,
    as_of: str,
    dates: Sequence[str],
    names: Sequence[str] = tuple(SCENARIO_RT_SCALE),
    later_weeks: int = 2,
    daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run models on scenarios calibrated to the real data and score them.

    For each scenario (`calibrated_settings` at `as_of`), the simulated
    vintages at `dates` are forecast, then scored two ways: against the
    final simulated data (`truth_view` `"final"`, with `complete` for
    weeks reported in full, as `complete_after_weeks`) and against the
    data as reported `later_weeks` after each forecast date (`"as of 2
    weeks later"`), as the real backtest does with later vintages.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to check.
    config : BacktestConfig
        Settings: samples, seed and `complete_after_weeks`.
    target, delays, zones : pandas.DataFrame
        Real target, delay and health zone rows.
    as_of : str
        The real vintage the scenarios are calibrated to.
    dates : sequence of str
        Simulated vintage dates to forecast from.
    names : sequence of str
        Scenarios (`SCENARIO_RT_SCALE` keys).
    later_weeks : int
        Weeks after each date for the as-of scores.
    daily : pandas.DataFrame, optional
        Real daily rows, from which the release cut-off lags are taken
        (`bdbvhubrenewal.data.releases.cutoff_lags`). The simulated
        releases then carry cut-offs, and tasks their daily releases.

    Returns
    -------
    tuple of pandas.DataFrame
        The province forecasts and the scores, with `scenario`,
        `truth_view` and `level` (national, province or zone). Zone
        scores split each model's province samples across zones
        (`ZoneAllocator`) and add zone random walk
        (`evaluation.zones.scenario_zone_scores`).
        Then the energy scores (`energy_tasks`) of every scale, with
        `scenario` and `truth_view`. Then the final simulated counts of
        the country and provinces, `location`, `target_end_date`, `truth`,
        `truth_vintage` and `scenario`.
    """
    forecasts, zone_forecasts = scenario_forecasts(
        models, config, target, delays, zones, as_of, dates, names, daily
    )
    scores, energy, truth = score_scenarios(
        forecasts,
        zone_forecasts,
        [m.name for m in models],
        config,
        target,
        delays,
        zones,
        as_of,
        dates,
        names,
        later_weeks,
        daily,
    )
    return forecasts, scores, energy, truth


def _scenario(
    config: BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    zones: pd.DataFrame,
    as_of: str,
    name: str,
) -> tuple[SimulationSettings, SimulatedOutbreak]:
    """Return a calibrated scenario's settings and its outbreak."""
    settings = calibrated_settings(
        target, delays, zones, as_of, name=name, seed=config.seed
    )
    return settings, simulate_outbreak(settings)


def scenario_forecasts(
    models: Sequence[ForecastModel],
    config: BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    zones: pd.DataFrame,
    as_of: str,
    dates: Sequence[str],
    names: Sequence[str] = tuple(SCENARIO_RT_SCALE),
    daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast the calibrated scenarios, without scoring.

    The forecasting half of `run_scenarios`, with the same parameters.

    Returns
    -------
    tuple of pandas.DataFrame
        The province forecasts and the zone forecasts allocated from
        them (`evaluation.zones.zone_allocations`), each with `scenario`.
    """
    # Local import: the zone backtest builds on this module.
    from bdbvhubrenewal.evaluation.zones import zone_allocations

    lags = None if daily is None else cutoff_lags(daily)
    all_forecasts, all_zones = [], []
    for name in names:
        settings, outbreak = _scenario(
            config, target, delays, zones, as_of, name
        )
        sim_target, sim_delays = outbreak.vintages(list(dates), lags)
        sim_daily = outbreak.daily_vintages(
            _release_days(outbreak, list(dates)), lags
        )
        tasks = make_tasks(sim_target, sim_delays, config, daily=sim_daily)
        forecasts = run_forecasts(models, tasks)
        allocated = zone_allocations(outbreak, tasks, [m.name for m in models])
        all_forecasts.append(forecasts.assign(scenario=settings.name))
        all_zones.append(allocated.assign(scenario=settings.name))
    return (
        pd.concat(all_forecasts, ignore_index=True),
        pd.concat(all_zones, ignore_index=True),
    )


def score_scenarios(
    forecasts: pd.DataFrame,
    zone_forecasts: pd.DataFrame,
    models: Sequence[str],
    config: BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    zones: pd.DataFrame,
    as_of: str,
    dates: Sequence[str],
    names: Sequence[str] = tuple(SCENARIO_RT_SCALE),
    later_weeks: int = 2,
    daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score the forecasts of `scenario_forecasts`.

    The scoring half of `run_scenarios`, which it equals given the same
    arguments. Forecasts of models run apart are stacked in the order of
    `models` (`joint_order`).

    Parameters
    ----------
    forecasts, zone_forecasts : pandas.DataFrame
        Output of `scenario_forecasts`, of models run together or stacked.
    models : sequence of str
        The models, in the order one run takes them.
    config, target, delays, zones, as_of, dates, names, later_weeks, daily
        As for `run_scenarios`. The scenarios are simulated again for
        the truth, which takes seconds.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores, the energy scores and the truth, as `run_scenarios`.
    """
    from bdbvhubrenewal.evaluation.zones import score_zone_allocations

    del daily
    k = config.complete_after_weeks
    all_scores, all_energy, all_truth = [], [], []
    for name in names:
        settings, outbreak = _scenario(
            config, target, delays, zones, as_of, name
        )
        mine = forecasts["scenario"] == settings.name
        own = joint_order(forecasts.loc[mine].drop(columns="scenario"), models)
        end = (
            settings.start + dt.timedelta(days=settings.n_days - 1)
        ).isoformat()
        final = _truth(outbreak.vintage(end)[0])
        scores = [score_tasks(own, final, end, k).assign(truth_view="final")]
        energy = [energy_tasks(own, final, end, k).assign(truth_view="final")]
        for date in dates:
            later = (
                as_date(date) + dt.timedelta(weeks=later_weeks)
            ).isoformat()
            truth = _truth(outbreak.vintage(later)[0])
            rows = own.loc[own["vintage"] == date]
            view = f"as of {later_weeks} weeks later"
            scores.append(
                score_tasks(rows, truth, later, k).assign(truth_view=view)
            )
            energy.append(
                energy_tasks(rows, truth, later).assign(truth_view=view)
            )
        scored = pd.concat(scores, ignore_index=True)
        scored["level"] = np.where(
            scored["location"] == NATIONAL, "national", "province"
        )
        zmine = zone_forecasts["scenario"] == settings.name
        zone, zone_energy = score_zone_allocations(
            outbreak,
            dates,
            joint_order(
                zone_forecasts.loc[zmine].drop(columns="scenario"), models
            ),
            end,
            later_weeks,
            k,
        )
        zone["level"] = "zone"
        all_truth.append(
            final.assign(truth_vintage=end, scenario=settings.name)
        )
        all_scores.append(
            pd.concat([scored, zone], ignore_index=True).assign(
                scenario=settings.name
            )
        )
        all_energy.append(
            pd.concat([*energy, zone_energy], ignore_index=True).assign(
                scenario=settings.name
            )
        )
    return (
        pd.concat(all_scores, ignore_index=True),
        pd.concat(all_energy, ignore_index=True),
        pd.concat(all_truth, ignore_index=True),
    )
