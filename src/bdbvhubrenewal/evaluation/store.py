# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The fit store: one stored fit per fitted model and real origin.

The fitted models are slow, so the docs never fit them.
The Fit store workflow (`.github/workflows/fit-store.yml`) fits each
(model, origin) pair once and commits the result to the `fit-store`
branch, one file per fit:

```text
<model>/<origin>/<key>.json.gz
```

The key hashes everything the fit depends on (`fit_key`): the model and
forecaster code and settings (`STORE_INPUTS`), the model's sampler table
in `ci/policy.toml`, the model name and the task at the origin, which
holds only the releases up to the origin.
A new release therefore adds one fit per model and leaves the older ones
valid, while a change to the code or sampler refits every origin.

`plan` lists the fits the store lacks, `fit` makes one, and `load`
writes the stored forecasts where the backtest and the docs read them.
Where the store has no fit for the current key, `load` uses the newest
stored fit of that pair and says so (`StoreStatus`), so the docs show
older fits, marked, while newer ones run.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import subprocess
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    merge_by_model,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.heldout import (
    load_model_choices,
    model_for_vintage,
)
from bdbvhubrenewal.forecasters import ForecastTask, get_model, list_models

__all__ = [
    "STATUS_FILE",
    "STORE_INPUTS",
    "StoreStatus",
    "StoredFit",
    "code_digest",
    "failed",
    "fit",
    "fit_key",
    "load",
    "origin_tasks",
    "plan",
    "read_fit",
    "samplers",
    "store_path",
    "stored_models",
    "write_fit",
]

STORE_INPUTS: Final = (
    "src/bdbvhubrenewal/baselinenowcast",
    "src/bdbvhubrenewal/data",
    "src/bdbvhubrenewal/evaluation/backtest.py",
    "src/bdbvhubrenewal/evaluation/store.py",
    "src/bdbvhubrenewal/forecasters",
    "src/bdbvhubrenewal/model",
    "config/backtest.toml",
    "uv.lock",
)
"""Code and settings every stored fit depends on, relative to the repo."""

STATUS_FILE: Final = "fit-store.json"
"""What `load` used, written beside the results for the docs."""


def stored_models() -> list[str]:
    """Return the registered models the store fits: those marked fitted.

    Returns
    -------
    list of str
        Model names.
    """
    return [s.name for s in list_models() if s.fitted]


def samplers(policy: Path | str = "ci/policy.toml") -> dict[str, Any]:
    """Return each model's NUTS settings from the CI policy.

    A stored fit uses the sampler table of the submission candidate that
    runs the same model, so the docs show the fits the submission makes.

    Parameters
    ----------
    policy : pathlib.Path or str
        The CI policy.

    Returns
    -------
    dict
        Sampler table by model name. A fitted model with no candidate
        (`pprandomwalknowcast`) is absent and uses its own settings.
    """
    raw = tomllib.loads(Path(policy).read_text())
    return {
        c["model"]: dict(c["sampler"])
        for c in raw["candidates"]
        if "model" in c and "sampler" in c
    }


def code_digest(root: Path | str = ".") -> str:
    """Hash the files in `STORE_INPUTS`.

    Parameters
    ----------
    root : pathlib.Path or str
        The repository root.

    Returns
    -------
    str
        SHA-256 hex digest of every file's path and bytes, in path order.
    """
    base = Path(root)
    files: list[Path] = []
    for name in STORE_INPUTS:
        path = base / name
        files += sorted(path.rglob("*")) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def origin_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    daily: pd.DataFrame,
) -> dict[str, ForecastTask]:
    """Build the task of every real origin, as the store fits it.

    A stored task is asked for final counts only (no `report_by`), so it
    holds no release after its origin.

    Parameters
    ----------
    target, delays, daily : pandas.DataFrame
        The tidy tables.
    config : BacktestConfig
        Horizons, samples and seed.

    Returns
    -------
    dict
        Task by origin (the vintage date), in date order.
    """
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(real, delays.assign(kind="real"), config, daily=daily)
    return {t.vintage: t for t in tasks}


def fit_key(
    model: str, task: ForecastTask, code: str, sampler: Mapping[str, Any]
) -> str:
    """Hash the inputs of one stored fit.

    Parameters
    ----------
    model : str
        The model name.
    task : ForecastTask
        The task at the origin (`origin_tasks`).
    code : str
        `code_digest`.
    sampler : mapping
        The model's sampler table (`samplers`), empty if it has none.

    Returns
    -------
    str
        The first 16 hex digits of a SHA-256 hash.
    """
    digest = hashlib.sha256()
    head = {"model": model, "code": code, "sampler": dict(sampler)}
    digest.update(json.dumps(head, sort_keys=True).encode())
    for f in fields(task):
        value = getattr(task, f.name)
        digest.update(f.name.encode())
        if isinstance(value, pd.DataFrame):
            digest.update(value.to_csv(index=False).encode())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:16]


def store_path(store: Path | str, model: str, origin: str, key: str) -> Path:
    """Return where a fit is stored.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root (a checkout of the `fit-store` branch).
    model, origin, key : str
        The fit.

    Returns
    -------
    pathlib.Path
        `<store>/<model>/<origin>/<key>.json.gz`.
    """
    return Path(store) / model / origin / f"{key}.json.gz"


@dataclass(frozen=True)
class StoredFit:
    """One stored fit.

    Attributes
    ----------
    meta : dict
        `model`, `origin`, `key`, `commit` (the code fitted), `fitted_at`
        (ISO time), `sampler` and `status` (`"ok"`, or `"failed"` with
        `error` and no tables).
    forecasts : pandas.DataFrame
        `run_forecasts` output.
    fits : pandas.DataFrame
        The model's fit summaries (`fit_diagnostics`).
    delays : pandas.DataFrame
        The learnt delay (`delay_summary`), empty if the model has none.
    """

    meta: dict[str, Any]
    forecasts: pd.DataFrame
    fits: pd.DataFrame
    delays: pd.DataFrame


_TABLES: Final = ("forecasts", "fits", "delays")


def write_fit(store: Path | str, stored: StoredFit) -> Path:
    """Write a fit to the store as gzipped JSON.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    stored : StoredFit
        The fit.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    m = stored.meta
    path = store_path(store, m["model"], m["origin"], m["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "meta": m,
        **{
            name: json.loads(
                getattr(stored, name).to_json(orient="split", index=False)
            )
            for name in _TABLES
        },
    }
    # mtime 0 keeps the bytes the same for the same fit.
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(json.dumps(body).encode())
    return path


def read_fit(path: Path | str) -> StoredFit:
    """Read a stored fit.

    Parameters
    ----------
    path : pathlib.Path or str
        A file written by `write_fit`.

    Returns
    -------
    StoredFit
        The fit.
    """
    body = json.loads(gzip.decompress(Path(path).read_bytes()))

    def table(name: str) -> pd.DataFrame:
        return pd.read_json(
            io.StringIO(json.dumps(body[name])),
            orient="split",
            dtype=False,
        )

    return StoredFit(body["meta"], *(table(n) for n in _TABLES))


def _commit() -> str:
    """Return the commit being fitted: the CI's, else the checkout's."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def plan(
    tasks: Mapping[str, ForecastTask],
    models: Sequence[str],
    code: str,
    sampler: Mapping[str, Any],
    store: Path | str,
) -> list[dict[str, str]]:
    """List the fits the store lacks.

    A fit that failed is stored as failed, so it is not retried until its
    inputs change.

    Parameters
    ----------
    tasks : mapping
        `origin_tasks`.
    models : sequence of str
        The models to store (`stored_models`).
    code : str
        `code_digest`.
    sampler : mapping
        `samplers`.
    store : pathlib.Path or str
        The store root.

    Returns
    -------
    list of dict
        `model`, `origin` and `key` of each missing fit, newest origin
        first, so the Nowcast page's fits come first.
    """
    missing = []
    for origin in sorted(tasks, reverse=True):
        for model in models:
            key = fit_key(model, tasks[origin], code, sampler.get(model, {}))
            if not store_path(store, model, origin, key).exists():
                missing.append({"model": model, "origin": origin, "key": key})
    return missing


def fit(
    model: str,
    task: ForecastTask,
    code: str,
    sampler: Mapping[str, Any],
) -> StoredFit:
    """Fit a model at an origin and return it for the store.

    Parameters
    ----------
    model : str
        The model name.
    task : ForecastTask
        The task at the origin (`origin_tasks`).
    code : str
        `code_digest`.
    sampler : mapping
        The model's sampler table, empty to use the model's own settings.

    Returns
    -------
    StoredFit
        The forecasts, fit summaries and learnt delay.
    """
    from bdbvhubrenewal.forecasters.renewal import SAMPLER_ENV

    if sampler:
        os.environ[SAMPLER_ENV] = json.dumps(dict(sampler))
    fits: list[pd.DataFrame] = []
    learnt: list[pd.DataFrame] = []
    forecasts = run_forecasts([get_model(model)], [task], fits, learnt)
    meta = {
        "model": model,
        "origin": task.vintage,
        "key": fit_key(model, task, code, sampler),
        "commit": _commit(),
        "fitted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sampler": dict(sampler),
        "status": "ok",
    }
    return StoredFit(
        meta,
        forecasts,
        pd.concat(fits, ignore_index=True) if fits else pd.DataFrame(),
        pd.concat(learnt, ignore_index=True) if learnt else pd.DataFrame(),
    )


def failed(
    model: str,
    task: ForecastTask,
    code: str,
    sampler: Mapping[str, Any],
    error: BaseException,
) -> StoredFit:
    """Record a fit that failed, so the store does not retry it.

    Parameters
    ----------
    model : str
        The model name.
    task : ForecastTask
        The task at the origin.
    code : str
        `code_digest`.
    sampler : mapping
        The model's sampler table.
    error : BaseException
        Why it failed.

    Returns
    -------
    StoredFit
        Meta only, with `status` `"failed"` and `error`.
    """
    meta = {
        "model": model,
        "origin": task.vintage,
        "key": fit_key(model, task, code, sampler),
        "commit": _commit(),
        "fitted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sampler": dict(sampler),
        "status": "failed",
        "error": repr(error),
    }
    empty = pd.DataFrame()
    return StoredFit(meta, empty, empty, empty)


@dataclass(frozen=True)
class StoreStatus:
    """What `load` used for one (model, origin) pair.

    Attributes
    ----------
    model, origin : str
        The pair.
    key : str
        The key of the current code and data.
    used : str or None
        The key of the fit loaded, None if the store has none.
    commit : str or None
        The commit the loaded fit was made at.
    current : bool
        The loaded fit is the current key's.
    failed : bool
        The current key's fit failed.
    """

    model: str
    origin: str
    key: str
    used: str | None
    commit: str | None
    current: bool
    failed: bool


def _choose(
    store: Path, model: str, origin: str, key: str
) -> tuple[StoredFit | None, bool]:
    """Return the current key's fit, else the newest good one, and failure.

    Returns
    -------
    tuple
        The fit to load (or None) and whether the current key failed.
    """
    found = [read_fit(p) for p in (store / model / origin).glob("*.json.gz")]
    current = [f for f in found if f.meta["key"] == key]
    failed = bool(current) and current[0].meta["status"] != "ok"
    if current and not failed:
        return current[0], False
    good = [f for f in found if f.meta["status"] == "ok"]
    if not good:
        return None, failed
    return max(good, key=lambda f: f.meta["fitted_at"]), failed


def load(
    store: Path | str,
    tasks: Mapping[str, ForecastTask],
    models: Sequence[str],
    code: str,
    sampler: Mapping[str, Any],
    config: BacktestConfig,
    results: Path | str = "results",
) -> list[StoreStatus]:
    """Write the stored fits where the backtest and the docs read them.

    Development origins go to `forecasts.csv.gz` and held-out ones to
    `heldout-forecasts.csv.gz`, with the model chosen then.
    Fit summaries go to `backtest-fits.csv` and learnt delays to
    `delay-summary.csv`. Each file keeps the rows of models not stored.
    What was used is written to `STATUS_FILE`.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    tasks : mapping
        `origin_tasks`.
    models : sequence of str
        The stored models.
    code : str
        `code_digest`.
    sampler : mapping
        `samplers`.
    config : BacktestConfig
        Gives the model choice record and the held-out split.
    results : pathlib.Path or str
        The results directory.

    Returns
    -------
    list of StoreStatus
        One per (model, origin), newest origin first.
    """
    from bdbvhubrenewal.forecasters.renewal import DELAY_SUMMARY_FILE

    out = Path(results)
    out.mkdir(parents=True, exist_ok=True)
    choices = load_model_choices(config.model_choice)
    status: list[StoreStatus] = []
    loaded: list[StoredFit] = []
    for origin in sorted(tasks, reverse=True):
        for model in models:
            key = fit_key(model, tasks[origin], code, sampler.get(model, {}))
            chosen, failed = _choose(Path(store), model, origin, key)
            if chosen is not None:
                loaded.append(chosen)
            status.append(
                StoreStatus(
                    model,
                    origin,
                    key,
                    None if chosen is None else chosen.meta["key"],
                    None if chosen is None else chosen.meta["commit"],
                    chosen is not None and chosen.meta["key"] == key,
                    failed,
                )
            )
    dev = [
        f.forecasts for f in loaded if not config.is_heldout(f.meta["origin"])
    ]
    held = [
        f.forecasts.assign(
            chosen_model=model_for_vintage(choices, f.meta["origin"])
        )
        for f in loaded
        if config.is_heldout(f.meta["origin"])
    ]
    for name, frames in (
        ("forecasts.csv.gz", dev),
        ("heldout-forecasts.csv.gz", held),
        ("backtest-fits.csv", [f.fits for f in loaded]),
        (DELAY_SUMMARY_FILE, [f.delays for f in loaded]),
    ):
        _merge(out / name, frames)
    (out / STATUS_FILE).write_text(
        json.dumps([vars(s) for s in status], indent=1) + "\n"
    )
    return status


def _merge(path: Path, frames: Iterable[pd.DataFrame]) -> None:
    """Merge stored rows into a results file by model."""
    rows = [f for f in frames if not f.empty]
    if rows:
        table = pd.concat(rows, ignore_index=True)
        merge_by_model(path, table).to_csv(path, index=False)
