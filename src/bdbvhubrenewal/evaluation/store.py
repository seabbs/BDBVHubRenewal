# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The fit store: one stored fit per fitted model and real origin.

The fitted models are slow, so the docs never fit them.
The Fit store workflow (`.github/workflows/fit-store.yml`) fits each
(model, origin) pair once and commits the result to the `fit-store`
branch, one file per fit:

```text
<model>/<origin>/<key>.json.gz
```

The key (`fit_key`) hashes only what reaches a fit: the task at the
origin, which holds only the releases up to the origin; the built model,
whose representation gives its variant and resolved sampler settings;
the BVDOutbreakSize release a BVD model uses; the code of the packages
the fit runs (`CODE_INPUTS`, without docstrings or comments, and
`run_forecasts`); and the runtime dependencies locked in `uv.lock`.
A new release adds one fit per model and leaves the older ones valid,
while a change to the model code or settings refits every origin.

`plan` lists the fits the store lacks, `fit` makes one, and `load`
writes the stored forecasts where the backtest and the docs read them.
A fit with reporting curves (the renewal models) stores them too, so it
is scored on its own counts as reported by releases after its origin
without depending on them (`forecasters.reporting`).
Where the store has no fit for the current key, `load` uses the newest
stored fit of that pair and says so (`StoreStatus`).
`command` runs `bdbvhubrenewal fit-store`.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import gzip
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import textwrap
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    _fit_rows,
    load_config,
    make_tasks,
    merge_by_model,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.nowcast import (
    REPORTING_FILE,
    stored_reporting,
)
from bdbvhubrenewal.forecasters import (
    ForecastModel,
    ForecastTask,
    get_model,
    list_models,
)

__all__ = [
    "CODE_INPUTS",
    "REPORTING_FILE",
    "STATUS_FILE",
    "StoreStatus",
    "StoredFit",
    "build_model",
    "code_digest",
    "command",
    "current_keys",
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
    "stored_reporting",
    "write_fit",
    "write_reporting",
]

CODE_INPUTS: Final = (
    "src/bdbvhubrenewal/baselinenowcast",
    "src/bdbvhubrenewal/data",
    "src/bdbvhubrenewal/evaluation/store.py",
    "src/bdbvhubrenewal/forecasters",
    "src/bdbvhubrenewal/model",
)
"""Code every stored fit runs, relative to the repository root."""

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


def build_model(name: str, sampler: Mapping[str, Any]) -> ForecastModel:
    """Build a registered model with its sampler table.

    Parameters
    ----------
    name : str
        The model name.
    sampler : mapping
        `samplers`. A model absent from it keeps its own settings.

    Returns
    -------
    ForecastModel
        The model. The environment is left as it was.
    """
    from bdbvhubrenewal.forecasters.renewal import SAMPLER_ENV

    old = os.environ.pop(SAMPLER_ENV, None)
    try:
        if name in sampler:
            os.environ[SAMPLER_ENV] = json.dumps(dict(sampler[name]))
        return get_model(name)
    finally:
        os.environ.pop(SAMPLER_ENV, None)
        if old is not None:
            os.environ[SAMPLER_ENV] = old


def _strip(source: str) -> str:
    """Dump Python source as its syntax tree, without docstrings."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        kept = [
            s
            for s in body
            if not (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str)
            )
        ]
        node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    return ast.dump(tree)


def code_digest(root: Path | str = ".") -> str:
    """Hash the code a stored fit runs and its runtime dependencies.

    Python files in `CODE_INPUTS` are hashed as syntax trees without
    docstrings, so comments and docstrings do not change the key; other
    files by their bytes. `run_forecasts` and the backtest helpers it
    calls are hashed the same way, with the Python minor version.
    The dependencies are `uv export --no-dev --all-extras` of `uv.lock`.

    Parameters
    ----------
    root : pathlib.Path or str
        The repository root.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    base = Path(root)
    files: list[Path] = []
    for name in CODE_INPUTS:
        path = base / name
        files += sorted(path.rglob("*")) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(base).as_posix().encode())
        if path.suffix == ".py":
            digest.update(_strip(path.read_text()).encode())
        else:
            digest.update(path.read_bytes())
    for func in (run_forecasts, _fit_rows):
        digest.update(_strip(inspect.getsource(func)).encode())
    digest.update(repr(sys.version_info[:2]).encode())
    deps = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--no-hashes",
            "--no-header",
            "--no-emit-project",
        ],
        cwd=base,
        capture_output=True,
        text=True,
        check=True,
    )
    digest.update(deps.stdout.encode())
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
        Horizons, samples and seed (`[forecast]`).

    Returns
    -------
    dict
        Task by origin (the vintage date), in date order.
    """
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(real, delays.assign(kind="real"), config, daily=daily)
    return {t.vintage: t for t in tasks}


def _bvd_tag(model: ForecastModel, task: ForecastTask) -> str | None:
    """Return the BVDOutbreakSize release a BVD model uses, else None."""
    variant = getattr(model, "variant", None)
    if variant is None:
        return None
    from bdbvhubrenewal.forecasters.renewal import (
        bvd_record,
        bvd_release,
        model_config,
    )

    if not model_config(variant).bvd:
        return None
    tag = bvd_record(bvd_release(task))["tag"]
    return None if tag is None else str(tag)


def _settings(model: ForecastModel) -> str:
    """Return a dataclass model's fields, else its class name."""
    return repr(model) if is_dataclass(model) else type(model).__qualname__


def fit_key(model: ForecastModel, task: ForecastTask, code: str) -> str:
    """Hash the inputs of one stored fit.

    Parameters
    ----------
    model : ForecastModel
        The model, from `build_model`, before it fits anything.
    task : ForecastTask
        The task at the origin (`origin_tasks`).
    code : str
        `code_digest`.

    Returns
    -------
    str
        The first 16 hex digits of a SHA-256 hash.
    """
    digest = hashlib.sha256()
    head = {
        "model": _settings(model),
        "code": code,
        "bvd": _bvd_tag(model, task),
    }
    digest.update(json.dumps(head, sort_keys=True).encode())
    for f in fields(task):
        value = getattr(task, f.name)
        digest.update(f.name.encode())
        if isinstance(value, pd.DataFrame):
            digest.update(value.to_csv(index=False).encode())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:16]


Keys = dict[tuple[str, str], str]
"""Current key by (model, origin)."""


def current_keys(
    tasks: Mapping[str, ForecastTask],
    models: Sequence[str],
    sampler: Mapping[str, Any],
    code: str,
) -> Keys:
    """Return the current key of every stored model at every origin.

    Parameters
    ----------
    tasks : mapping
        `origin_tasks`.
    models : sequence of str
        The stored models.
    sampler : mapping
        `samplers`.
    code : str
        `code_digest`.

    Returns
    -------
    dict
        Key by (model, origin).
    """
    built = {m: build_model(m, sampler) for m in models}
    return {
        (m, origin): fit_key(built[m], task, code)
        for origin, task in tasks.items()
        for m in models
    }


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
        (ISO time), `settings` (the model's representation) and `status`
        (`"ok"`, or `"failed"` with `error` and no tables).
    forecasts : pandas.DataFrame
        `run_forecasts` output.
    fits : pandas.DataFrame
        The model's fit summaries (`fit_diagnostics`).
    delays : pandas.DataFrame
        The learnt delay (`delay_summary`), empty if the model has none.
    reporting : list of dict or None
        The model's reporting curves (`renewal.reporting_block`), which
        give its counts as reported by any later release, or None if it
        has none.
    """

    meta: dict[str, Any]
    forecasts: pd.DataFrame
    fits: pd.DataFrame
    delays: pd.DataFrame
    reporting: list[dict[str, Any]] | None = None


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
    if stored.reporting:
        body["reporting"] = stored.reporting
    # No timestamp in the gzip header, so the bytes depend on the content.
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

    forecasts, fits, delays = (table(n) for n in _TABLES)
    return StoredFit(
        body["meta"], forecasts, fits, delays, body.get("reporting")
    )


def _commit() -> str:
    """Return the commit being fitted: the CI's, else the checkout's."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def _meta(model: ForecastModel, origin: str, key: str) -> dict[str, Any]:
    return {
        "model": model.name,
        "origin": origin,
        "key": key,
        "commit": _commit(),
        "fitted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "settings": _settings(model),
        "status": "ok",
    }


def plan(
    keys: Keys, store: Path | str, retry_failed: bool = False
) -> list[dict[str, str]]:
    """List the fits the store lacks.

    A fit that failed is stored as failed and is not planned again until
    its inputs change, unless `retry_failed`.

    Parameters
    ----------
    keys : dict
        `current_keys`.
    store : pathlib.Path or str
        The store root.
    retry_failed : bool
        Plan the fits stored as failed again.

    Returns
    -------
    list of dict
        `model`, `origin` and `key` of each fit to make, newest origin
        first, so the Nowcast page's fits come first.
    """
    to_fit = []
    for (model, origin), key in sorted(keys.items(), key=lambda x: x[0][1])[
        ::-1
    ]:
        path = store_path(store, model, origin, key)
        redo = retry_failed and path.exists()
        if redo:
            redo = read_fit(path).meta["status"] != "ok"
        if not path.exists() or redo:
            to_fit.append({"model": model, "origin": origin, "key": key})
    return to_fit


def fit(
    name: str, task: ForecastTask, sampler: Mapping[str, Any], code: str
) -> StoredFit:
    """Fit a model at an origin and return it for the store.

    A fit that raises is returned as failed (`failed`).

    Parameters
    ----------
    name : str
        The model name.
    task : ForecastTask
        The task at the origin (`origin_tasks`).
    sampler : mapping
        `samplers`.
    code : str
        `code_digest`.

    Returns
    -------
    StoredFit
        The forecasts, fit summaries and learnt delay.
    """
    model = build_model(name, sampler)
    meta = _meta(model, task.vintage, fit_key(model, task, code))
    fits: list[pd.DataFrame] = []
    learnt: list[pd.DataFrame] = []
    try:
        forecasts = run_forecasts([model], [task], fits, learnt)
    except Exception as err:
        return failed(meta, err)
    return StoredFit(
        meta,
        forecasts,
        pd.concat(fits, ignore_index=True) if fits else pd.DataFrame(),
        pd.concat(learnt, ignore_index=True) if learnt else pd.DataFrame(),
        getattr(model, "reporting", None) or None,
    )


def failed(meta: Mapping[str, Any], error: BaseException) -> StoredFit:
    """Record a fit that failed, so the store does not retry it.

    Parameters
    ----------
    meta : mapping
        The fit's meta.
    error : BaseException
        Why it failed.

    Returns
    -------
    StoredFit
        Meta only, with `status` `"failed"` and `error`.
    """
    empty = pd.DataFrame()
    record = {**meta, "status": "failed", "error": repr(error)}
    return StoredFit(record, empty, empty, empty)


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
    keys: Keys,
    results: Path | str = "results",
) -> list[StoreStatus]:
    """Write the stored fits where the backtest and the docs read them.

    Every origin goes to `forecasts.csv.gz`.
    Fit summaries go to `backtest-fits.csv` and learnt delays to
    `delay-summary.csv`. Each file keeps the rows of models not stored.
    What was used is written to `STATUS_FILE`.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    keys : dict
        `current_keys`.
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
    status: list[StoreStatus] = []
    loaded: list[StoredFit] = []
    for (model, origin), key in sorted(keys.items(), key=lambda x: x[0][1])[
        ::-1
    ]:
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
    for name, frames in (
        ("forecasts.csv.gz", [f.forecasts for f in loaded]),
        ("backtest-fits.csv", [f.fits for f in loaded]),
        (DELAY_SUMMARY_FILE, [f.delays for f in loaded]),
    ):
        _merge(out / name, frames)
    write_reporting(loaded, out)
    (out / STATUS_FILE).write_text(
        json.dumps([vars(s) for s in status], indent=1) + "\n"
    )
    return status


def write_reporting(fits: Iterable[StoredFit], results: Path | str) -> Path:
    """Merge the fits' reporting curves into `REPORTING_FILE`.

    The curves of every model loaded are replaced, as `merge_by_model`
    replaces its rows, so a fit with none leaves none behind. Other
    models' curves are kept.

    Parameters
    ----------
    fits : iterable of StoredFit
        Loaded fits. Those with no curves are skipped.
    results : pathlib.Path or str
        The results directory.

    Returns
    -------
    pathlib.Path
        The file.
    """
    path = Path(results) / REPORTING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fits = list(fits)
    loaded = {f.meta["model"] for f in fits}
    # As `merge_by_model`: a loaded model's curves are all replaced, so a
    # fit with none leaves none behind.
    rows = {
        key: blocks
        for key, blocks in stored_reporting(results).items()
        if key[0] not in loaded
    }
    for f in fits:
        if f.reporting:
            rows[(f.meta["model"], f.meta["origin"])] = f.reporting
    body = [
        {"model": m, "origin": o, "blocks": b}
        for (m, o), b in sorted(rows.items())
    ]
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(json.dumps(body).encode())
    return path


def _merge(path: Path, frames: Iterable[pd.DataFrame]) -> None:
    """Merge stored rows into a results file by model."""
    rows = [f for f in frames if not f.empty]
    if rows:
        table = pd.concat(rows, ignore_index=True)
        write_csv(merge_by_model(path, table), path)


def command(args: argparse.Namespace) -> int:
    """Run `bdbvhubrenewal fit-store plan|fit|load`.

    `plan` prints the fits to make and writes them as a GitHub Actions
    matrix (`--out`), with every current key (`--keys`) for pruning.
    `fit` fits one model at one origin and writes it to `--store`.
    `load` writes the stored fits into `--results`.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed `fit-store` arguments.

    Returns
    -------
    int
        0 on success, 1 if the fit failed or the arguments are wrong.
    """
    sampler, code = samplers(args.policy), code_digest()
    if args.action == "fit":
        from bdbvhubrenewal.cli import host_devices

        # Before anything starts JAX, so the chains run in parallel.
        chains = sampler.get(args.model, {}).get("num_chains", 1)
        host_devices(int(chains))
    config = load_config(args.config)
    tasks = origin_tasks(
        load_target(args.data),
        load_delays(args.data),
        config,
        load_daily(args.data),
    )
    models = args.models or stored_models()
    if args.action == "fit":
        if args.model not in stored_models() or args.origin not in tasks:
            print(
                f"ERROR: fit needs --model in {stored_models()} and "
                f"--origin in {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        stored = fit(args.model, tasks[args.origin], sampler, code)
        path = write_fit(args.store, stored)
        print(f"Wrote {path} ({stored.meta['status']}).")
        if stored.meta["status"] != "ok":
            print(f"ERROR: {stored.meta['error']}", file=sys.stderr)
            return 1
        return 0
    keys = current_keys(tasks, models, sampler, code)
    if args.action == "plan":
        to_fit = plan(keys, args.store, args.retry_failed)
        for t in to_fit:
            print(f"To fit: {t['model']} at {t['origin']} ({t['key']}).")
        print(f"{len(to_fit)} of {len(keys)} fits to make.")
        if args.out:
            Path(args.out).write_text(json.dumps({"include": to_fit}))
        if args.keys:
            Path(args.keys).write_text(
                json.dumps([f"{m}/{o}/{k}" for (m, o), k in keys.items()])
            )
        return 0
    for s in load(args.store, keys, args.results):
        state = "current" if s.current else "not fitted for this code"
        if s.used and not s.current:
            state += f", older fit {s.used} used"
        if s.failed:
            state += ", the fit for this code failed"
        print(f"{s.model} at {s.origin}: {state}.")
    return 0
