# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Rolling-origin backtest with a development set and a held-out set.

Each vintage becomes a `ForecastTask`: the data as of that vintage, the
reference date it serves (`reference_date_for_vintage`) and the weeks
from `min_horizon` to `max_horizon`.
Every registered model forecasts every task.
Forecasts are scored against a later truth.

There are three kinds of vintage:

- **real**: published by the hub. Truth is the latest real vintage.
- **reconstructed**: rebuilt from the latest real data for past dates
  (`bdbv_forecasts.vintages`). Truth is the latest real vintage.
- **simulated**: from a simulated outbreak with known parameters
  (`bdbv_forecasts.simulate`). Truth is the final simulated count.

Reference dates are split into development and held-out sets in
`config/backtest.toml`.
Development runs never score held-out dates.
`run_heldout` refuses to run unless asked explicitly, needs the model
choice record to exist first, uses real vintages only, and logs the run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .dates import as_date, epiweek_end, reference_date_for_vintage
from .models.base import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    add_national,
    check_samples,
)
from .scoring import score_forecasts
from .simulate import SimulationSettings, simulate_outbreak
from .vintages import reconstruct_delays, reconstruct_vintages

__all__ = [
    "BacktestConfig",
    "HeldOutError",
    "forecast_locations",
    "forecast_weeks",
    "load_config",
    "make_tasks",
    "run_development",
    "run_forecasts",
    "run_heldout",
    "run_recovery",
    "saturdays",
    "score_tasks",
    "truth_from_vintage",
]

log = logging.getLogger(__name__)


class HeldOutError(RuntimeError):
    """Raised when held-out dates would be scored without permission."""


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
    min_truth_age_weeks : int
        Only weeks at least this many weeks older than the truth
        vintage's epi week are scored.
    heldout_reference_dates : tuple of str
        Reference dates kept for the held-out evaluation.
    model_choice : pathlib.Path
        The model choice record that must exist before a held-out run.
    reconstructed_dates : tuple of str
        Vintage dates to reconstruct.
    simulated_dates : tuple of str
        Vintage dates to publish from the simulated outbreak.
    """

    min_horizon: int = -4
    max_horizon: int = 3
    n_samples: int = 100
    seed: int = 0
    min_truth_age_weeks: int = 2
    heldout_reference_dates: tuple[str, ...] = ()
    model_choice: Path = Path("results/model-choice.yml")
    reconstructed_dates: tuple[str, ...] = ()
    simulated_dates: tuple[str, ...] = ()

    @property
    def first_heldout(self) -> str | None:
        """The earliest held-out reference date, if any."""
        return min(self.heldout_reference_dates, default=None)

    def is_heldout(self, reference_date: dt.date | str) -> bool:
        """Whether a reference date is held out.

        Parameters
        ----------
        reference_date : datetime.date or str
            The reference date.

        Returns
        -------
        bool
            True if it is listed in `heldout_reference_dates`.
        """
        return as_date(reference_date).isoformat() in set(
            self.heldout_reference_dates
        )


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
    fc, sc, sp = raw["forecast"], raw["scoring"], raw["split"]
    rec, sim = raw.get("reconstructed", {}), raw.get("simulated", {})
    return BacktestConfig(
        min_horizon=int(fc["min_horizon"]),
        max_horizon=int(fc["max_horizon"]),
        n_samples=int(fc["n_samples"]),
        seed=int(fc["seed"]),
        min_truth_age_weeks=int(sc["min_truth_age_weeks"]),
        heldout_reference_dates=tuple(sp["heldout_reference_dates"]),
        model_choice=Path(sp["model_choice"]),
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


def make_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame | None,
    config: BacktestConfig,
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

    Returns
    -------
    list of ForecastTask
        Tasks ordered by kind and vintage.
    """
    tasks = []
    for (kind, vintage), rows in target.groupby(["kind", "vintage"]):
        ref = reference_date_for_vintage(str(vintage))
        dly = None
        if delays is not None:
            mask = delays["vintage"] == vintage
            if "kind" in delays.columns:
                mask &= delays["kind"] == kind
            sel = delays.loc[mask]
            dly = sel if not sel.empty else None
        tasks.append(
            ForecastTask(
                vintage=str(vintage),
                reference_date=ref,
                kind=str(kind),
                target=rows.reset_index(drop=True),
                delays=dly,
                locations=forecast_locations(rows),
                target_end_dates=forecast_weeks(ref, config),
                n_samples=config.n_samples,
                seed=config.seed,
            )
        )
    return tasks


def run_forecasts(
    models: Sequence[ForecastModel], tasks: Iterable[ForecastTask]
) -> pd.DataFrame:
    """Run every model on every task, adding the national total.

    Parameters
    ----------
    models : sequence of ForecastModel
        Models to run.
    tasks : iterable of ForecastTask
        Tasks to run them on.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `kind`, `vintage`, `reference_date`, `horizon`,
        `location`, `target_end_date`, `sample` and `value`.
    """
    frames = []
    for task in tasks:
        for model in models:
            samples = model.forecast(task)
            check_samples(samples, task)
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
    return pd.concat(frames, ignore_index=True)


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
    config: BacktestConfig,
    *,
    allow_heldout: bool = False,
) -> pd.DataFrame:
    """Score forecasts on weeks the truth has had time to fill in.

    Forecasts made at or after `truth_date` are dropped, as are weeks
    less than `min_truth_age_weeks` before the truth's epi week.
    Held-out reference dates are dropped unless `allow_heldout` is True.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `run_forecasts`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`.
    truth_date : str
        The date of the truth data.
    config : BacktestConfig
        Settings.
    allow_heldout : bool
        Whether held-out reference dates may be scored.

    Returns
    -------
    pandas.DataFrame
        Output of `bdbv_forecasts.scoring.score_forecasts`, with a
        `truth_vintage` column.
    """
    held = forecasts["reference_date"].map(config.is_heldout)
    if held.any() and not allow_heldout:
        log.info("Dropping %d held-out forecast rows.", int(held.sum()))
        forecasts = forecasts.loc[~held]
    cutoff = epiweek_end(truth_date) - dt.timedelta(
        weeks=config.min_truth_age_weeks
    )
    keep = (forecasts["vintage"] < truth_date) & (
        forecasts["target_end_date"] <= cutoff.isoformat()
    )
    scores = score_forecasts(forecasts.loc[keep], truth)
    scores["truth_vintage"] = truth_date
    return scores


def _real_vintages(target: pd.DataFrame) -> list[str]:
    return sorted(set(target.loc[target["kind"] == "real", "vintage"]))


def run_development(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    include_reconstructed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the development backtest on real and reconstructed vintages.

    Held-out reference dates are excluded throughout, and reconstructed
    vintages whose reference date is on or after the first held-out date
    are not built.

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

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts and the scores.
    """
    real = target.loc[target["kind"] == "real"]
    latest = _real_vintages(real)[-1]
    parts = [real]
    dly_parts = [delays.assign(kind="real")]
    if include_reconstructed:
        first_held = config.first_heldout
        dates = [
            d
            for d in config.reconstructed_dates
            if first_held is None
            or reference_date_for_vintage(d).isoformat() < first_held
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
    tasks = [
        t
        for t in make_tasks(data, all_delays, config)
        if not config.is_heldout(t.reference_date) and t.vintage < latest
    ]
    forecasts = run_forecasts(models, tasks)
    truth = truth_from_vintage(real, latest)
    scores = score_tasks(forecasts, truth, latest, config)
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
    no_split = BacktestConfig(
        min_horizon=config.min_horizon,
        max_horizon=config.max_horizon,
        n_samples=config.n_samples,
        seed=config.seed,
        min_truth_age_weeks=config.min_truth_age_weeks,
    )
    scores = score_tasks(forecasts, outbreak.truth(), end.isoformat(), no_split)
    scores["truth_vintage"] = "simulated"
    return forecasts, scores


def run_heldout(
    models: Sequence[ForecastModel],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    *,
    evaluate_holdout: bool = False,
    out_dir: Path | str = "results",
) -> pd.DataFrame:
    """Score the held-out reference dates, once, on real vintages only.

    Parameters
    ----------
    models : sequence of ForecastModel
        The chosen model and the baseline.
    target : pandas.DataFrame
        The tidy target table.
    delays : pandas.DataFrame
        The tidy delay table.
    config : BacktestConfig
        Settings.
    evaluate_holdout : bool
        Must be True. Guards against scoring held-out dates by accident.
    out_dir : pathlib.Path or str
        Where to write `heldout-scores.csv` and `heldout-run.yml`.

    Returns
    -------
    pandas.DataFrame
        The held-out scores.

    Raises
    ------
    HeldOutError
        If `evaluate_holdout` is False or the model choice record is
        missing.
    """
    if not evaluate_holdout:
        raise HeldOutError(
            "Held-out dates are only scored with evaluate_holdout=True "
            "(`--evaluate-holdout`)."
        )
    if not config.model_choice.exists():
        raise HeldOutError(
            f"Write and commit the model choice record {config.model_choice} "
            "before the held-out evaluation."
        )
    log.warning(
        "Running the held-out evaluation for %s.",
        ", ".join(config.heldout_reference_dates),
    )
    real = target.loc[target["kind"] == "real"]
    latest = _real_vintages(real)[-1]
    tasks = [
        t
        for t in make_tasks(real, delays, config)
        if config.is_heldout(t.reference_date) and t.vintage < latest
    ]
    forecasts = run_forecasts(models, tasks)
    truth = truth_from_vintage(real, latest)
    scores = score_tasks(forecasts, truth, latest, config, allow_heldout=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "heldout-scores.csv", index=False)
    choice = config.model_choice.read_bytes()
    record = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "heldout_reference_dates": list(config.heldout_reference_dates),
        "truth_vintage": latest,
        "models": [m.name for m in models],
        "model_choice": str(config.model_choice),
        "model_choice_sha256": hashlib.sha256(choice).hexdigest(),
    }
    (out / "heldout-run.yml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )
    return scores
