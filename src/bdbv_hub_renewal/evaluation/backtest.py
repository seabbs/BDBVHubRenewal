# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Rolling-origin backtest with a development set and a held-out set.

Each vintage becomes a `ForecastTask`: the data as of that vintage, the
reference date it serves (`reference_date_for_vintage`) and the weeks
from `min_horizon` to `max_horizon`.
Every registered model forecasts every task.
Forecasts are scored on every week against a later truth vintage, as
reported then, and each score records that `truth_vintage`.

There are three kinds of vintage:

- **real**: published by the hub. Truth is the latest real vintage.
- **reconstructed**: rebuilt from the latest real data for past dates
  (`bdbv_hub_renewal.data.reconstruct`). Truth is the latest real vintage.
- **simulated**: from a simulated outbreak with known parameters
  (`bdbv_hub_renewal.data.simulate`). Truth is the final simulated count.

Vintages are split by the model choice record (`results/model-choice.yml`,
see `bdbv_hub_renewal.evaluation.heldout`).
Development runs use vintages up to the latest choice date only.
Every later vintage is held out and scored by
`bdbv_hub_renewal.evaluation.heldout`.
`score_by_truth_vintage` re-scores real forecasts against every later real
vintage, and `truth_changes` shows how much the truth moved meanwhile.
"""

from __future__ import annotations

import datetime as dt
import logging
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bdbv_hub_renewal.data.dates import (
    as_date,
    epiweek_end,
    reference_date_for_vintage,
)
from bdbv_hub_renewal.data.reconstruct import (
    publication_lag_pmf,
    reconstruct_delays,
    reconstruct_vintages,
)
from bdbv_hub_renewal.data.simulate import SimulationSettings, simulate_outbreak
from bdbv_hub_renewal.evaluation.scoring import score_forecasts
from bdbv_hub_renewal.forecasters.base import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    add_national,
    check_samples,
)
from bdbv_hub_renewal.model.model import PRIOR_PUBLICATION_LAG

__all__ = [
    "BacktestConfig",
    "development_tasks",
    "forecast_locations",
    "forecast_weeks",
    "load_config",
    "lopo_tasks",
    "make_tasks",
    "merge_by_model",
    "real_vintages",
    "run_development",
    "run_forecasts",
    "run_lopo",
    "run_recovery",
    "saturdays",
    "score_by_truth_vintage",
    "score_tasks",
    "shard_tasks",
    "truth_changes",
    "truth_from_vintage",
    "uses_lag_prior",
]

log = logging.getLogger(__name__)


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
    development_until : str or None
        The latest model choice date. Vintages after it are held out.
        None if no model has been chosen, so every vintage is development.
    model_choice : pathlib.Path
        The model choice record (`bdbv_hub_renewal.evaluation.heldout`).
    reconstructed_dates : tuple of str
        Vintage dates to reconstruct.
    simulated_dates : tuple of str
        Vintage dates to publish from the simulated outbreak.
    complete_after_weeks : int
        $K$: a truth week counts as complete if it ended at least this
        many weeks before the truth vintage
        (`bdbv_hub_renewal.data.completeness.complete_weeks`).
    """

    min_horizon: int = -4
    max_horizon: int = 3
    n_samples: int = 100
    seed: int = 0
    development_until: str | None = None
    model_choice: Path = Path("results/model-choice.yml")
    reconstructed_dates: tuple[str, ...] = ()
    simulated_dates: tuple[str, ...] = ()
    complete_after_weeks: int = 0

    def is_heldout(self, vintage: dt.date | str) -> bool:
        """Whether a vintage is held out from development.

        Parameters
        ----------
        vintage : datetime.date or str
            The vintage date.

        Returns
        -------
        bool
            True if it is after `development_until`.
        """
        if self.development_until is None:
            return False
        return as_date(vintage) > as_date(self.development_until)


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

    `development_until` is the latest date in the model choice record, if
    the record exists.

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
    fc, sp = raw["forecast"], raw["split"]
    rec, sim = raw.get("reconstructed", {}), raw.get("simulated", {})
    choice = Path(sp["model_choice"])
    dates: list[str] = []
    if choice.exists():
        record = yaml.safe_load(choice.read_text()) or {}
        dates = [str(c["date"]) for c in record.get("choices", [])]
    return BacktestConfig(
        min_horizon=int(fc["min_horizon"]),
        max_horizon=int(fc["max_horizon"]),
        n_samples=int(fc["n_samples"]),
        seed=int(fc["seed"]),
        development_until=max(dates, default=None),
        model_choice=choice,
        complete_after_weeks=int(sp.get("complete_after_weeks", 0)),
        reconstructed_dates=(
            saturdays(rec["first_vintage"], rec["last_vintage"]) if rec else ()
        ),
        simulated_dates=(
            saturdays(sim["first_vintage"], sim["last_vintage"]) if sim else ()
        ),
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
    """Return the provinces with at least one case in a vintage.

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
    totals = prov.groupby("location")["total"].sum()
    return tuple(sorted(totals.index[totals > 0]))


def _reconstructed_delays(delays: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Delay rows for a real vintage that has none, from the next one.

    The next real vintage with delays is cut back to what was published
    by `vintage` (`reconstruct_delays`). 2026-08-18 has no delay file.
    """
    real = delays
    if "kind" in delays.columns:
        real = delays.loc[delays["kind"] == "real"]
    later = sorted(str(v) for v in real["vintage"].unique() if v > vintage)
    if not later:
        return real.iloc[:0]
    out = reconstruct_delays(
        real.loc[real["vintage"] == later[0]],
        vintage,
        lag_pmf=PRIOR_PUBLICATION_LAG,
    )
    if "kind" in delays.columns:
        out["kind"] = "real"
    return out


def make_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    config: BacktestConfig,
    report_by: Sequence[str] = (),
) -> list[ForecastTask]:
    """Build one task per vintage in a target table.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows, possibly of several kinds and vintages.
    delays : pandas.DataFrame or None
        Delay line data with a `vintage` column, and optionally a `kind`
        column. A task gets the rows of its own vintage (and kind). A
        real vintage with none gets the next real vintage's rows cut back
        to its date, or None if there is no later one.
    config : BacktestConfig
        Horizons, samples and seed.
    report_by : sequence of str
        Truth vintage dates. Each real or reconstructed task asks for the
        counts as reported by those after its own vintage.

    Returns
    -------
    list of ForecastTask
        Tasks ordered by kind and vintage.
    """
    tasks = []
    for (kind, vintage), rows in target.groupby(["kind", "vintage"]):
        ref = reference_date_for_vintage(str(vintage))
        dly, history = None, None
        if delays is not None:
            mask = delays["vintage"] == vintage
            if "kind" in delays.columns:
                mask &= delays["kind"] == kind
            sel = delays.loc[mask]
            if sel.empty and kind == "real":
                sel = _reconstructed_delays(delays, str(vintage))
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
                target=rows.reset_index(drop=True),
                delays=dly,
                delay_history=history,
                locations=forecast_locations(rows),
                target_end_dates=forecast_weeks(ref, config),
                n_samples=config.n_samples,
                seed=config.seed,
                report_by=(
                    ()
                    if kind == "simulated"
                    else tuple(sorted(d for d in report_by if d > str(vintage)))
                ),
            )
        )
    return tasks


def uses_lag_prior(task: ForecastTask) -> bool:
    """Whether a task has no publication lag estimate of its own.

    The lag is estimated from pairs of delay files on or before the
    vintage (`bdbv_hub_renewal.data.reconstruct.publication_lag_pmf`).
    Without one, the renewal models use a prior whose shape was set from
    later delay files, so these origins see some later data.

    Parameters
    ----------
    task : ForecastTask
        The task.

    Returns
    -------
    bool
        True if no pair of delay files is available.
    """
    if task.delay_history is None:
        return True
    return publication_lag_pmf(task.delay_history, task.vintage) is None


def shard_tasks(
    tasks: Sequence[ForecastTask], index: int, count: int
) -> list[ForecastTask]:
    """Return every `count`-th task from `index`, to split a run.

    Parameters
    ----------
    tasks : sequence of ForecastTask
        All the tasks.
    index : int
        This shard, from 0.
    count : int
        Number of shards.

    Returns
    -------
    list of ForecastTask
        The shard's tasks. The shards cover every task once.
    """
    return list(tasks)[index::count]


def _cache_path(cache: Path, model: str, task: ForecastTask) -> Path:
    mask = f"_mask-{'-'.join(task.mask)}" if task.mask else ""
    return cache / f"{model}_{task.kind}_{task.vintage}{mask}.csv.gz"


def _fits_path(path: Path) -> Path:
    return path.with_name(path.name.replace(".csv.gz", ".fits.csv"))


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


def run_forecasts(
    models: Sequence[ForecastModel],
    tasks: Iterable[ForecastTask],
    cache: Path | str | None = None,
    fits: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run every model on every task, adding the national total.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to run.
    tasks : iterable of ForecastTask
        Tasks to run them on.
    cache : pathlib.Path or str, optional
        A directory with one file per model and task. A task with a file
        is read from it, and every new forecast is written to it at once,
        so a long run can be resumed or split (`shard_tasks`).
    fits : list, optional
        Collects a table of each model's fit summaries per task, for
        models that set `fit_diagnostics` (cached with the forecasts).

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `kind`, `vintage`, `reference_date`, `horizon`,
        `location`, `target_end_date`, `sample`, `value`, `as_of`
        (missing for final counts) and `lag_prior` (`uses_lag_prior`).
    """
    store = None if cache is None else Path(cache)
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)
    frames = []
    for task in tasks:
        for model in models:
            path = (
                None if store is None else _cache_path(store, model.name, task)
            )
            if path is not None and path.exists():
                frames.append(pd.read_csv(path, dtype={"as_of": object}))
                fit_path = path.with_name(
                    path.name.replace(".csv.gz", ".fits.csv")
                )
                if fits is not None and fit_path.exists():
                    fits.append(pd.read_csv(fit_path, dtype={"masked": str}))
                continue
            if hasattr(model, "fit_diagnostics"):
                model.fit_diagnostics = []
            samples = model.forecast(task)
            table = _fit_rows(model, task)
            if fits is not None and not table.empty:
                fits.append(table)
            check_samples(samples, task)
            if "as_of" not in samples:
                samples["as_of"] = np.nan
            samples = add_national(samples)
            samples["lag_prior"] = uses_lag_prior(task)
            ref = pd.Timestamp(task.reference_date)
            samples["horizon"] = (
                pd.to_datetime(samples["target_end_date"]) - ref
            ).dt.days // 7
            samples.insert(0, "model", model.name)
            samples.insert(1, "kind", task.kind)
            samples.insert(2, "vintage", task.vintage)
            samples.insert(3, "reference_date", task.reference_date.isoformat())
            if path is not None:
                if not table.empty:
                    table.to_csv(_fits_path(path), index=False)
                # Written last, so a file means the task is complete.
                samples.to_csv(path, index=False)
            frames.append(samples)
    return pd.concat(frames, ignore_index=True)


def merge_by_model(path: Path | str, new: pd.DataFrame) -> pd.DataFrame:
    """Merge a run's rows into a results file, keeping other models' rows.

    The fitted variants are slow, so they are run apart from the others,
    for example on a bigger machine. A run replaces the rows of the
    models it ran and keeps the rest.

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
    if not path.exists() or path.stat().st_size <= 1 or "model" not in new:
        return new
    old = pd.read_csv(path)
    if "model" not in old:
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
        Output of `bdbv_hub_renewal.evaluation.scoring.score_forecasts`, with
        `truth_vintage`, `scoring` and `complete` columns, and
        `lag_prior` if the forecasts have it.
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
    if "lag_prior" in forecasts:
        flags = forecasts.groupby(["kind", "vintage"])["lag_prior"].first()
        out = out.merge(flags.reset_index(), on=["kind", "vintage"], how="left")
    out["truth_vintage"] = truth_date
    age = pd.Timestamp(truth_date) - pd.to_datetime(out["target_end_date"])
    out["complete"] = age.dt.days >= 7 * complete_after_weeks
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


def score_by_truth_vintage(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    complete_after_weeks: int = 0,
) -> pd.DataFrame:
    """Score real forecasts against every later real vintage.

    Reconstructed vintages are built from the latest data, so only real
    forecasts are re-scored.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts`.
    target : pandas.DataFrame
        Target rows with every real vintage.
    complete_after_weeks : int
        $K$, as for `score_tasks`.

    Returns
    -------
    pandas.DataFrame
        Scores as from `score_tasks`, one set per `truth_vintage`. Empty
        if no real vintage is later than a forecast.
    """
    real = forecasts.loc[forecasts["kind"] == "real"]
    frames = [
        score_tasks(
            real, truth_from_vintage(target, v), v, complete_after_weeks
        )
        for v in real_vintages(target)
        if (real["vintage"] < v).any()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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


def development_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    include_reconstructed: bool = True,
) -> list[ForecastTask]:
    """Build the development tasks: real and reconstructed, not held out.

    Parameters
    ----------
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    include_reconstructed : bool
        Whether to add reconstructed vintages.

    Returns
    -------
    list of ForecastTask
        Tasks for vintages up to `config.development_until` and before the
        latest real vintage.
    """
    real = target.loc[target["kind"] == "real"]
    latest = real_vintages(real)[-1]
    parts = [real]
    dly_parts = [delays.assign(kind="real")]
    if include_reconstructed:
        dates = [
            d for d in config.reconstructed_dates if not config.is_heldout(d)
        ]
        if dates:
            parts.append(reconstruct_vintages(real, delays, dates, latest))
            src = delays.loc[delays["vintage"] == latest]
            dly_parts.extend(
                reconstruct_delays(src, d).assign(kind="reconstructed")
                for d in dates
            )
    data = pd.concat(parts, ignore_index=True)
    all_delays = pd.concat(dly_parts, ignore_index=True)
    return [
        t
        for t in make_tasks(data, all_delays, config, real_vintages(real))
        if not config.is_heldout(t.vintage) and t.vintage < latest
    ]


def run_development(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    include_reconstructed: bool = True,
    cache: Path | str | None = None,
    fits: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the development backtest on real and reconstructed vintages.

    Vintages after `config.development_until` are held out: they are not
    forecast from, and reconstructed vintages after it are not built.
    The truth is the latest real vintage, and every week is scored.

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
    include_reconstructed : bool
        Whether to add reconstructed vintages.
    cache : pathlib.Path or str, optional
        Forecast cache, as for `run_forecasts`.
    fits : list, optional
        Collects fit summaries, as for `run_forecasts`.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts and the scores.
    """
    real = target.loc[target["kind"] == "real"]
    latest = real_vintages(real)[-1]
    tasks = development_tasks(
        target, delays, config, include_reconstructed=include_reconstructed
    )
    forecasts = run_forecasts(models, tasks, cache, fits)
    truth = truth_from_vintage(real, latest)
    scores = score_tasks(forecasts, truth, latest, config.complete_after_weeks)
    return forecasts, scores


def lopo_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    provinces: Sequence[str] | None = None,
    include_reconstructed: bool = True,
) -> list[ForecastTask]:
    """Build the development tasks with one province masked in each.

    Parameters
    ----------
    target, delays, config
        As for `development_tasks`.
    provinces : sequence of str, optional
        Provinces to mask. Defaults to every province with cases in each
        task.
    include_reconstructed : bool
        Whether to add reconstructed vintages.

    Returns
    -------
    list of ForecastTask
        Each with `mask` set to one province.
    """
    return [
        replace(task, mask=(prov,))
        for task in development_tasks(
            target, delays, config, include_reconstructed=include_reconstructed
        )
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
    include_reconstructed: bool = True,
    cache: Path | str | None = None,
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
    include_reconstructed : bool
        Whether to add reconstructed vintages.
    cache : pathlib.Path or str, optional
        Forecast cache, as for `run_forecasts`.
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
    for task in lopo_tasks(
        target,
        delays,
        config,
        provinces=provinces,
        include_reconstructed=include_reconstructed,
    ):
        prov = task.mask[0]
        samples = run_forecasts(able, [task], cache, fits)
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


def run_recovery(
    models: Sequence[ForecastModel],
    config: BacktestConfig,
    settings: SimulationSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run models on simulated vintages and score against the known truth.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to check.
    config : BacktestConfig
        Settings, including `simulated_dates`.
    settings : SimulationSettings, optional
        The simulated scenario. Defaults to `turning_point`.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts and the scores.
    """
    outbreak = simulate_outbreak(settings)
    target, delays = outbreak.vintages(list(config.simulated_dates))
    tasks = make_tasks(target, delays, config)
    forecasts = run_forecasts(models, tasks)
    end = outbreak.settings.start + dt.timedelta(days=outbreak.settings.n_days)
    scores = score_tasks(forecasts, outbreak.truth(), end.isoformat())
    scores["truth_vintage"] = "simulated"
    return forecasts, scores
