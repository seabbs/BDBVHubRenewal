# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Pairing a submission's draws by the model's own recent errors.

A model registered as `paired` has its draws paired into trajectories
by the rolling copula (`forecasters.copula.RollingCopula`), as the
release backtest pairs them, before its hub table is made.
The copula is estimated from the model's forecasts at earlier origins
(`past_forecasts`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.evaluation import store as fit_store
from bdbvhubrenewal.evaluation.backtest import BacktestConfig
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
    "pair_submission",
    "past_forecasts",
    "stored_forecasts",
]

_COLUMNS = ["model", "origin", "location", "target_end_date", "sample", "value"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_COLUMNS, "source"])


def stored_forecasts(
    store: Path | str | None, model: str, settings: str | None = None
) -> pd.DataFrame:
    """Return a model's stored forecasts: each origin's latest good fit.

    Parameters
    ----------
    store : pathlib.Path or str or None
        The fit store root.
    model : str
        The model.
    settings : str, optional
        Keep only fits of these settings (the fit's `meta["settings"]`,
        such as a fine-tuned model's checkpoint).

    Returns
    -------
    pandas.DataFrame
        `model`, `origin` (the release forecast from), `location`,
        `target_end_date`, `sample`, `value` and `source`, of final
        counts.
    """
    if store is None or not (Path(store) / model).is_dir():
        return _empty()
    latest: dict[str, tuple[str, pd.DataFrame]] = {}
    for path in sorted((Path(store) / model).glob("*/*.json.gz")):
        fit = fit_store.read_fit(path)
        meta = fit.meta
        if meta.get("status") != "ok" or fit.forecasts.empty:
            continue
        if settings is not None and meta.get("settings") != settings:
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
) -> tuple[pd.DataFrame, set[str]]:
    """Forecast again at each task's release not in `skip`.

    Also returns the releases the model cannot forecast from, where the
    backtest has no forecast either.
    """
    built = get_model(model)
    frames, unavailable = [], set()
    for task in tasks:
        if task.vintage in skip:
            continue
        try:
            drawn = built.forecast(task)
        except ModelUnavailableError:
            unavailable.add(task.vintage)
            continue
        frames.append(
            add_national(drawn).assign(model=model, origin=task.vintage)
        )
    if not frames:
        return _empty(), unavailable
    out = pd.concat(frames, ignore_index=True)
    return out[_COLUMNS].assign(source="recomputed"), unavailable


def past_forecasts(
    model: str,
    origin: str,
    store: Path | str | None = None,
    past_tasks: Sequence[ForecastTask] = (),
) -> tuple[pd.DataFrame, set[str]]:
    """Return a model's forecasts at origins before `origin`.

    Each origin takes the store's fit of the model's own settings, as the
    release backtest scores, else a forecast made again with the model at
    the task's release.

    Parameters
    ----------
    model : str
        The model.
    origin : str
        The forecast's origin. Only earlier origins are returned.
    store : pathlib.Path or str, optional
        As for `stored_forecasts`.
    past_tasks : sequence of ForecastTask
        Tasks of earlier releases to forecast again from.

    Returns
    -------
    tuple
        The forecasts, as `stored_forecasts` with each row's `source`,
        and the origins a model forecast again cannot forecast from.
    """
    spec = {s.name: s for s in list_models()}.get(model)
    settings = None if spec is None else fit_store._settings(get_model(model))
    frames = []
    have: set[str] = set()

    def add(table: pd.DataFrame) -> None:
        table = table.loc[
            (table["origin"] < origin) & ~table["origin"].isin(have)
        ]
        have.update(table["origin"])
        frames.append(table)

    add(stored_forecasts(store, model, settings))
    unavailable: set[str] = set()
    if spec is not None:
        tasks = [t for t in past_tasks if t.vintage < origin]
        recomputed, unavailable = _recomputed(model, tasks, have)
        add(recomputed)
    frames = [f for f in frames if len(f)]
    if not frames:
        return _empty(), unavailable
    return pd.concat(frames, ignore_index=True), unavailable


def pair_submission(
    model: str,
    samples: pd.DataFrame,
    task: ForecastTask,
    truth_delay: int,
    store: Path | str | None = None,
    past_tasks: Sequence[ForecastTask] = (),
    max_horizon: int = BacktestConfig().max_horizon,
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
    store, past_tasks : optional
        Where the past forecasts come from (`past_forecasts`). Only the
        tasks of releases that can still give errors in the copula's
        window are forecast again.
    max_horizon : int
        The latest week of a forecast, in weeks after its reference week,
        which sets how far back a release can still give such errors.

    Returns
    -------
    tuple
        The paired samples, and a record of the copula: `model`,
        `origin`, `n_scores`, the three correlations, `sources`, each past
        origin's source (`store` or `recomputed`),
        `source`, the sources used joined by `+` (or `none`), and
        `unavailable`, the earlier origins the model cannot forecast from.

    Raises
    ------
    LookupError
        If no source gives a forecast at the earlier origins in the
        window. The copula is its prior only where the backtest has none
        either: with no earlier origin, as at the first, or none the model
        can forecast from.
    """
    origin = task.vintage
    # A forecast at r reaches week r + max_horizon, which is in the window
    # only if it ends after the origin less D and WINDOW_WEEKS weeks.
    oldest = as_date(origin) - pd.Timedelta(
        days=truth_delay + 7 * (WINDOW_WEEKS + max_horizon + 1)
    )
    recent = [t for t in past_tasks if as_date(t.vintage) >= oldest]
    past, unavailable = past_forecasts(model, origin, store, recent)
    earlier = {t.vintage for t in recent if t.vintage < origin} - unavailable
    if past.empty and earlier:
        raise LookupError(
            f"{model} has no past forecasts before {origin} in the fit store "
            f"({store}) or forecast again, so its draws cannot be paired."
        )
    history = task.target_history
    truth = (
        rolling_truth(history, truth_delay)
        if history is not None and len(history)
        else pd.DataFrame(
            columns=["location", "target_end_date", "truth", "truth_vintage"]
        )
    )
    current = samples.assign(model=model, origin=origin)
    both = (
        pd.concat([past.drop(columns="source"), current], ignore_index=True)
        if len(past)
        else current
    )
    paired, fits = RollingCopula().pair(both, truth, seed=task.seed)
    out = paired.loc[paired["origin"] == origin, list(samples.columns)]
    out = out.astype(samples.dtypes.to_dict()).reset_index(drop=True)
    row = fits.loc[fits["origin"] == origin].iloc[0]
    record: dict[str, Any] = {
        "model": str(row["model"]),
        "origin": str(row["origin"]),
        "n_scores": int(row["n_scores"]),
        **{
            name: float(row[name])
            for name in ("rho_time", "rho_provinces", "rho_national")
        },
    }
    by_origin = past.drop_duplicates("origin").sort_values("origin")
    record["sources"] = dict(
        zip(by_origin["origin"], by_origin["source"], strict=True)
    )
    record["source"] = "+".join(sorted(set(past["source"]))) or "none"
    record["unavailable"] = sorted(unavailable)
    return out, record
