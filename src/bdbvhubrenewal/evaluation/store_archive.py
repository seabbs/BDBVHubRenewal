# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Every candidate model's forecasts for a round, for the ledger.

`archive` returns the hub table each stored model and each submission
candidate would submit for the round, health zones included.
A stored model forecasts from the fit the store holds, and any other
candidate is forecast now.
A model that cannot be archived is reported and left out.
"""

from __future__ import annotations

import base64
import datetime as dt
import sys
import tempfile
import tomllib
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, vintage_task
from bdbvhubrenewal.forecasters import ForecastTask
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    check_samples,
)
from bdbvhubrenewal.submission.fitting import Forecast
from bdbvhubrenewal.submission.hub import HUB_COLUMNS

__all__ = [
    "COLUMNS",
    "FILE",
    "Hub",
    "archive",
    "candidate_models",
    "policy_candidates",
    "store_task",
]

FILE: Final = "candidates.csv.gz"
"""The archive's name in a round's ledger folder."""

RANDOM_WALK: Final = "randomwalk"
"""The model of a `randomwalk` candidate that names none, as in
`ci/steps/candidate.sh`."""

COLUMNS: Final = (
    "model",
    "vintage",
    *HUB_COLUMNS,
    "key",
    "commit",
    "fitted_at",
    "current",
)
"""The archive's columns: the model, its hub table, then its fit."""


Hub = Callable[..., pd.DataFrame]
"""Returns the hub table a model submits (`cli._hub_forecast`): called
with the model and `config`, `fit_dir` or `forecast` by keyword."""


def policy_candidates(
    policy: Path | str = "ci/policy.toml",
) -> dict[str, str | None]:
    """Return every submission candidate's model, with its config.

    A candidate's model is its `model`, and a `randomwalk` candidate that
    names none runs `RANDOM_WALK`. A candidate forecasts with its fit
    configuration, except a `randomwalk` one, which runs its model as
    `ci/steps/candidate.sh` does.

    Parameters
    ----------
    policy : pathlib.Path or str
        The CI policy.

    Returns
    -------
    dict
        Configuration (or None) by model name, in the policy's order.
    """
    raw = tomllib.loads(Path(policy).read_text())
    models: dict[str, str | None] = {}
    for c in raw["candidates"]:
        if c["kind"] == "randomwalk":
            models.setdefault(c.get("model") or RANDOM_WALK, None)
        elif "model" in c:
            models.setdefault(c["model"], c.get("config"))
    return models


def candidate_models(
    policy: Path | str = "ci/policy.toml",
) -> dict[str, str | None]:
    """Return every candidate's model and every stored model.

    Parameters
    ----------
    policy : pathlib.Path or str
        The CI policy.

    Returns
    -------
    dict
        `policy_candidates`, then each stored model that is no candidate,
        with no configuration.
    """
    models = policy_candidates(policy)
    for model in store.stored_models():
        models.setdefault(model, None)
    return models


def store_task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    daily: pd.DataFrame,
    origin: str,
) -> ForecastTask | None:
    """Return the stored fits' task at one origin, as `store.origin_tasks`.

    Parameters
    ----------
    target, delays, daily : pandas.DataFrame
        The tidy tables.
    config : BacktestConfig
        The backtest configuration.
    origin : str
        A real vintage.

    Returns
    -------
    ForecastTask or None
        The task, or None for a release the store does not fit
        (`config.skip_origins`).
    """
    if origin in config.skip_origins:
        return None
    weeks = store.NEXT_ROUND_WEEKS
    return vintage_task(target, delays, config, origin, daily, weeks)[0]


def _stored_forecast(rows: pd.DataFrame, task: ForecastTask) -> Forecast:
    """Return a stored fit's forecast for the round's weeks.

    The nowcasts of earlier report dates (`as_of`) are dropped. Stored
    national rows are the model's own, so what they hold beyond the sum
    of the provinces is the national part no province holds
    (`Forecast.unassigned`), as a fit directory gives it.
    """
    if "as_of" in rows:
        rows = rows.loc[rows["as_of"].isna()]
    weeks = set(task.target_end_dates)
    iso = rows["target_end_date"].map(lambda d: as_date(d).isoformat())
    rows = rows.assign(target_end_date=iso)
    rows = rows.loc[rows["target_end_date"].isin(weeks), list(SAMPLE_COLUMNS)]
    national = rows["location"] == NATIONAL
    samples = rows.loc[~national].reset_index(drop=True)
    check_samples(samples, task)
    keys = ["target_end_date", "sample"]
    total = rows.loc[national].set_index(keys)["value"]
    summed = samples.groupby(keys)["value"].sum()
    extra = (total - summed.reindex(total.index)).clip(lower=0)
    unassigned = extra.rename("value").reset_index()
    return Forecast(samples, None if unassigned.empty else unassigned)


def _stored(
    root: Path,
    model: str,
    config: str | None,
    task: ForecastTask,
    origin_task: ForecastTask | None,
    sampler: Mapping[str, Any],
    code: Callable[[str], str],
    hub: Hub,
) -> pd.DataFrame:
    """Return a stored model's hub table from the fit the store holds."""
    if origin_task is None:
        raise LookupError(f"{task.vintage} is not an origin the store fits")
    origin = origin_task.vintage
    tasks = {origin: origin_task}
    key = store.current_keys(tasks, [model], sampler, code(model))
    fit, _ = store._choose(root, model, origin, key[(model, origin)])
    if fit is None:
        raise LookupError(f"the store holds no good fit at {origin}")
    if config and fit.files:
        # As `store.unpack` writes it.
        with tempfile.TemporaryDirectory() as fit_dir:
            for name, body in fit.files.items():
                (Path(fit_dir) / name).write_bytes(base64.b64decode(body))
            table = hub(model, config=config, fit_dir=Path(fit_dir))
    else:
        table = hub(model, forecast=_stored_forecast(fit.forecasts, task))
    meta = fit.meta
    return table.assign(
        key=meta["key"],
        commit=meta["commit"],
        fitted_at=meta["fitted_at"],
        current=meta["key"] == key[(model, origin)],
    )


def archive(
    root: Path | str,
    task: ForecastTask,
    origin_task: ForecastTask | None,
    models: Mapping[str, str | None],
    sampler: Mapping[str, Any],
    hub: Hub,
    code: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Return the hub table each model would submit for a round.

    Parameters
    ----------
    root : pathlib.Path or str
        The store root.
    task : ForecastTask
        The round's task, as `bdbvhubrenewal nowcast` builds it.
    origin_task : ForecastTask or None
        The store's task at the same origin (`store.origin_tasks`), for
        the stored fits' keys; None if the store fits no such origin.
    models : mapping
        Configuration (or None) by model (`candidate_models`).
    sampler : mapping
        `store.samplers`.
    hub : Hub
        Makes each model's hub table.
    code : callable, optional
        A model's code digest, by default `store.code_digest`.

    Returns
    -------
    pandas.DataFrame
        `COLUMNS`: each model's hub table, with its fit's `key`, `commit`
        and `fitted_at`, and whether it is the fit of the current code
        and data (`current`). A model forecast now has the current
        commit and time and no key. A model that cannot be archived,
        such as a stored model with no good fit, has no rows and is
        reported on standard error.
    """
    digest = code or store.code_digest
    stored = set(store.stored_models())
    rows = [pd.DataFrame(columns=COLUMNS)]
    for model, config in models.items():
        try:
            if model in stored:
                table = _stored(
                    Path(root),
                    model,
                    config,
                    task,
                    origin_task,
                    sampler,
                    digest,
                    hub,
                )
            else:
                now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
                table = hub(model, config=config).assign(
                    key=None,
                    commit=store._commit(),
                    fitted_at=now,
                    current=True,
                )
        except Exception as err:
            print(f"{model} not archived: {err!r}.", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            continue
        rows.append(table.assign(model=model, vintage=task.vintage))
    return pd.concat(rows, ignore_index=True)[list(COLUMNS)]
