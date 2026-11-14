# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Pairing a submission's draws by the model's own recent errors.

A model registered as `paired` draws each place and week on its own.
Before its hub table is made, its draws are paired into trajectories by
the rolling copula (`bdbvhubrenewal.forecasters.copula.RollingCopula`),
as the release backtest pairs them, so the scored and submitted
forecasts are the same.
The copula is estimated from the model's forecasts at earlier origins,
against the truth the releases up to the forecast give.
Those forecasts come from the ledger's archive of each round's
candidates (`archived_forecasts`), then from the fit store
(`stored_forecasts`), and, for a model that is not fitted, from
forecasting again at the earlier releases.
A model with none takes the copula's prior.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.evaluation import store as fit_store
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.forecasters.base import (
    ForecastTask,
    ModelUnavailableError,
    add_national,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.copula import WINDOW_WEEKS, RollingCopula

__all__ = [
    "ARCHIVE_FILE",
    "archived_forecasts",
    "pair_submission",
    "past_forecasts",
    "stored_forecasts",
]

ARCHIVE_FILE: Final = "candidates.csv.gz"
"""A round's archive of its candidates' hub tables, in the ledger's
`archive/<reference_date>/<run>/`."""

_COLUMNS = ["model", "origin", "location", "target_end_date", "sample", "value"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_COLUMNS, "source"])


def _run_key(path: Path) -> tuple[str, int, str]:
    """Order archive files by round, then run number."""
    run = path.parent.name
    number = int(run) if re.fullmatch(r"\d+", run) else -1
    return (path.parent.parent.name, number, run)


def archived_forecasts(archive: Path | str | None, model: str) -> pd.DataFrame:
    """Return a model's archived forecasts of the country and provinces.

    Parameters
    ----------
    archive : pathlib.Path or str or None
        The ledger's `archive` directory, with
        `<reference_date>/<run>/candidates.csv.gz`.
    model : str
        The model.

    Returns
    -------
    pandas.DataFrame
        `model`, `origin` (the release forecast from), `location`,
        `target_end_date`, `sample`, `value` and `source`. Of an origin
        archived by more than one run, the latest run's.
    """
    if archive is None or not Path(archive).is_dir():
        return _empty()
    files = sorted(Path(archive).glob(f"*/*/{ARCHIVE_FILE}"), key=_run_key)
    by_origin: dict[str, pd.DataFrame] = {}
    for path in files:
        table = pd.read_csv(path, dtype={"vintage": str})
        rows = table.loc[
            (table["model"] == model)
            & (table["output_type"] == "sample")
            & (table["location"].str.count("-") <= 1)
        ]
        for origin, own in rows.groupby("vintage"):
            by_origin[str(origin)] = own
    if not by_origin:
        return _empty()
    out = pd.concat(by_origin.values(), ignore_index=True).rename(
        columns={"vintage": "origin", "output_type_id": "sample"}
    )
    out["sample"] = out["sample"].astype(int)
    return out[_COLUMNS].assign(source="archive")


def stored_forecasts(store: Path | str | None, model: str) -> pd.DataFrame:
    """Return a model's stored forecasts: each origin's latest good fit.

    Parameters
    ----------
    store : pathlib.Path or str or None
        The fit store root.
    model : str
        The model.

    Returns
    -------
    pandas.DataFrame
        As `archived_forecasts`, of final counts.
    """
    if store is None or not (Path(store) / model).is_dir():
        return _empty()
    latest: dict[str, tuple[str, pd.DataFrame]] = {}
    for path in sorted((Path(store) / model).glob("*/*.json.gz")):
        fit = fit_store.read_fit(path)
        meta = fit.meta
        if meta.get("status") != "ok" or fit.forecasts.empty:
            continue
        origin, at = str(meta["origin"]), str(meta.get("fitted_at", ""))
        if origin not in latest or at > latest[origin][0]:
            latest[origin] = (at, fit.forecasts)
    frames = []
    for origin, (_, rows) in latest.items():
        if "as_of" in rows:
            rows = rows.loc[rows["as_of"].isna()]
        frames.append(rows.assign(model=model, origin=origin))
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    out["target_end_date"] = out["target_end_date"].astype(str)
    return out[_COLUMNS].assign(source="store")


def _recomputed(
    model: str, tasks: Sequence[ForecastTask], skip: set[str]
) -> pd.DataFrame:
    """Forecast again at each task's release not in `skip`."""
    built = get_model(model)
    frames = []
    for task in tasks:
        if task.vintage in skip:
            continue
        try:
            drawn = built.forecast(task)
        except ModelUnavailableError:
            continue
        frames.append(
            add_national(drawn).assign(model=model, origin=task.vintage)
        )
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    return out[_COLUMNS].assign(source="recomputed")


def past_forecasts(
    model: str,
    origin: str,
    archive: Path | str | None = None,
    store: Path | str | None = None,
    past_tasks: Sequence[ForecastTask] = (),
) -> pd.DataFrame:
    """Return a model's forecasts at origins before `origin`.

    Each origin takes the archive's forecast, then the store's. A model
    that is not fitted is forecast again at each task's release with
    neither.

    Parameters
    ----------
    model : str
        The model.
    origin : str
        The forecast's origin. Only earlier origins are returned.
    archive, store : pathlib.Path or str, optional
        As for `archived_forecasts` and `stored_forecasts`.
    past_tasks : sequence of ForecastTask
        Tasks of earlier releases to forecast again from.

    Returns
    -------
    pandas.DataFrame
        As `archived_forecasts`, with each row's `source`.
    """
    frames = []
    have: set[str] = set()
    for table in (
        archived_forecasts(archive, model),
        stored_forecasts(store, model),
    ):
        table = table.loc[
            (table["origin"] < origin) & ~table["origin"].isin(have)
        ]
        have |= set(table["origin"])
        frames.append(table)
    spec = {s.name: s for s in list_models()}.get(model)
    if spec is not None and not spec.fitted:
        tasks = [t for t in past_tasks if t.vintage < origin]
        frames.append(_recomputed(model, tasks, have))
    frames = [f for f in frames if len(f)]
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True)


def pair_submission(
    model: str,
    samples: pd.DataFrame,
    task: ForecastTask,
    truth_delay: int,
    archive: Path | str | None = None,
    store: Path | str | None = None,
    past_tasks: Sequence[ForecastTask] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair a forecast's draws by the copula of the model's recent errors.

    Parameters
    ----------
    model : str
        The model, registered as `paired`.
    samples : pandas.DataFrame
        Its forecast for `task`: `location`, `target_end_date`, `sample`
        and `value`.
    task : ForecastTask
        The forecast's task, whose `target_history` gives the truth.
    truth_delay : int
        $D$: each week's truth is its count in the first release $D$ days
        after it (`rolling_truth`).
    archive, store, past_tasks : optional
        Where the past forecasts come from (`past_forecasts`). Only the
        tasks of releases that can still give errors in the copula's
        window are forecast again.

    Returns
    -------
    tuple
        The paired samples, and a record of the copula: `model`,
        `origin`, `n_scores`, the three correlations and `source`, the
        past forecasts' sources joined by `+`, or `none`.
    """
    origin = task.vintage
    oldest = as_date(origin) - pd.Timedelta(
        days=truth_delay + 7 * (WINDOW_WEEKS + 4)
    )
    recent = [t for t in past_tasks if as_date(t.vintage) >= oldest]
    past = past_forecasts(model, origin, archive, store, recent)
    history = task.target_history
    truth = (
        rolling_truth(history, truth_delay)
        if history is not None and len(history)
        else pd.DataFrame(
            columns=["location", "target_end_date", "truth", "truth_vintage"]
        )
    )
    current = samples.assign(model=model, origin=origin)
    both = pd.concat([past.drop(columns="source"), current], ignore_index=True)
    paired, fits = RollingCopula().pair(both, truth, seed=task.seed)
    out = paired.loc[paired["origin"] == origin, list(samples.columns)]
    out = out.astype(samples.dtypes.to_dict()).reset_index(drop=True)
    row = fits.loc[fits["origin"] == origin].iloc[0]
    record: dict[str, Any] = {str(k): v for k, v in row.items()}
    sources = sorted(set(past["source"]))
    record["source"] = "+".join(sources) if sources else "none"
    return out, record
