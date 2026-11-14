# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Every candidate model's forecasts for a round, for the ledger.

The submission ledger keeps the submitted file of each round.
`archive` adds the hub table every stored model and every submission
candidate would submit for the round, health zones included, so each
can be scored prospectively, and not only the one submitted.
Each table is made by `hub`, the path `bdbvhubrenewal nowcast` takes.
A stored model's fit is the one the store holds when the round is
recorded, the fit `store.load` uses: a submission candidate forecasts
from its fit directory, as the submission does, and any other stored
model from its stored samples.
A model the store does not hold, such as `randomwalk`, is forecast then,
as the submission forecasts it, and nothing is stored.
Each row carries its fit's key, commit and `fitted_at`, so an analysis
can compare when each forecast was made with the window's close.
A model that cannot be archived is reported and left out; the others are
archived.
`ci/steps/archive_candidates.sh` writes it to the ledger as
`archive/<reference_date>/<run>/candidates.csv.gz`
(`bdbvhubrenewal fit-store archive`).

This module is apart from `store`, whose source keys every stored fit.
"""

from __future__ import annotations

import base64
import datetime as dt
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.forecasters import ForecastTask
from bdbvhubrenewal.submission.hub import HUB_COLUMNS

__all__ = ["COLUMNS", "FILE", "Hub", "archive", "candidate_models"]

FILE: Final = "candidates.csv.gz"
"""The archive's name in a round's ledger folder."""

RANDOM_WALK: Final = "randomwalk"
"""The model of a `randomwalk` candidate that names none, as in
`ci/steps/candidate.sh`."""

SAMPLE_COLUMNS: Final = ("location", "target_end_date", "sample", "value")
"""The stored forecasts' columns a hub table is made from."""

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
with the model and `config`, `fit_dir` or `samples` by keyword."""


def candidate_models(
    policy: Path | str = "ci/policy.toml",
) -> dict[str, str | None]:
    """Return every stored model and candidate's model, with its config.

    A candidate's model is its `model`, and a `randomwalk` candidate that
    names none runs `RANDOM_WALK`. A candidate forecasts with its fit
    configuration, except a `randomwalk` one, which runs its model as
    `ci/steps/candidate.sh` does. A stored model that is no candidate
    has no configuration.

    Parameters
    ----------
    policy : pathlib.Path or str
        The CI policy.

    Returns
    -------
    dict
        Configuration (or None) by model name, candidates first.
    """
    raw = tomllib.loads(Path(policy).read_text())
    models: dict[str, str | None] = {}
    for c in raw["candidates"]:
        if c["kind"] == "randomwalk":
            models.setdefault(c.get("model") or RANDOM_WALK, None)
        elif "model" in c:
            models.setdefault(c["model"], c.get("config"))
    for model in store.stored_models():
        models.setdefault(model, None)
    return models


def _stored(
    root: Path,
    model: str,
    config: str | None,
    task: ForecastTask,
    store_task: ForecastTask | None,
    sampler: Mapping[str, Any],
    code: Callable[[str], str],
    hub: Hub,
) -> pd.DataFrame:
    """Return a stored model's hub table from the fit the store holds."""
    if store_task is None:
        raise LookupError(f"{task.vintage} is not an origin the store fits")
    origin = store_task.vintage
    tasks = {origin: store_task}
    key = store.current_keys(tasks, [model], sampler, code(model))
    fit, _ = store._choose(root, model, origin, key[(model, origin)])
    if fit is None:
        raise LookupError(f"the store holds no good fit at {origin}")
    if config and fit.files:
        with tempfile.TemporaryDirectory() as fit_dir:
            for name, body in fit.files.items():
                (Path(fit_dir) / name).write_bytes(base64.b64decode(body))
            table = hub(model, config=config, fit_dir=Path(fit_dir))
    else:
        weeks = {as_date(d) for d in task.target_end_dates}
        samples = fit.forecasts.loc[
            fit.forecasts["target_end_date"].map(as_date).isin(weeks),
            list(SAMPLE_COLUMNS),
        ]
        table = hub(model, samples=samples.reset_index(drop=True))
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
    store_task: ForecastTask | None,
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
    store_task : ForecastTask or None
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
                    store_task,
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
            continue
        rows.append(table.assign(model=model, vintage=task.vintage))
    return pd.concat(rows, ignore_index=True)[list(COLUMNS)]
