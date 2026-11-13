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
the BVDOutbreakSize release a BVD model uses; the code the model's fit
can run (`fit_code`, without docstrings or comments, and
`run_forecasts`), less the zone split, which runs at load; and the
runtime dependencies locked in `uv.lock`.
A new release adds one fit per model and leaves the older ones valid,
while a change to a model's code or settings refits every origin of the
models that run it, and no other.

`plan` lists the fits the store lacks, `fit` makes one, and `load`
writes the stored forecasts where the backtest and the docs read them.
A submission candidate's fit also holds its fit directory (`fit_files`),
which `unpack` writes out for the submission to forecast from, so the
submission uses the stored fit rather than fitting again.
Where the store has no fit for the current key, `load` uses the newest
stored fit of that pair and says so (`StoreStatus`).
`command` runs `bdbvhubrenewal fit-store`.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import functools
import gzip
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    _fit_rows,
    load_config,
    merge_by_model,
    real_vintages,
    run_forecasts,
    vintage_task,
)
from bdbvhubrenewal.forecasters import (
    ForecastModel,
    ForecastTask,
    get_model,
    list_models,
)

__all__ = [
    "STATUS_FILE",
    "STORE_DATA",
    "STORE_ENTRIES",
    "STORE_SKIP",
    "UNPACKED_FILE",
    "StoreStatus",
    "StoredFit",
    "build_model",
    "code_digest",
    "command",
    "current_keys",
    "failed",
    "fit",
    "fit_code",
    "fit_files",
    "fit_key",
    "load",
    "origin_tasks",
    "plan",
    "read_fit",
    "samplers",
    "store_path",
    "stored_models",
    "unpack",
    "write_fit",
]

STORE_ENTRIES: Final = (
    "src/bdbvhubrenewal/evaluation/backtest.py",
    "src/bdbvhubrenewal/evaluation/store.py",
)
"""The shared modules a stored fit calls: `fit` and `run_forecasts`."""

STORE_SKIP: Final = (
    "src/bdbvhubrenewal/evaluation/digest.py",
    "src/bdbvhubrenewal/evaluation/zones.py",
    "src/bdbvhubrenewal/forecasters/zones.py",
    "src/bdbvhubrenewal/reporting/plots.py",
)
"""Modules `STORE_ENTRIES` import but a fit does not call.

The zone split: a stored fit holds national and province samples only,
and its zones are split from them at load
(`forecasters.zones.zones_for_model`). `fit_code`'s own helper
(`evaluation.digest`), and the plots `evaluation.ppc` imports for
`plot_ppc`, which a fit does not draw.
"""

STORE_DATA: Final = "src/bdbvhubrenewal/data/*.csv"
"""The package's data files, which a fit may read (the province centres
of the renewal models' gravity kernel). Every stored fit's key holds
them."""

STATUS_FILE: Final = "fit-store.json"
"""What `load` used, written beside the results for the docs."""

UNPACKED_FILE: Final = "store.json"
"""Which stored fit `unpack` wrote, in the fit directory."""

NEXT_ROUND_WEEKS: Final = 1
"""Weeks a stored fit forecasts past `[forecast] max_horizon`, so a round
with no new release is forecast from the last release's stored fit."""

COMMIT_FILE: Final = ".commit"
"""The store branch commit, written by `scripts/fit-store.sh fetch`."""


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

    Raises
    ------
    ValueError
        If two candidates with a sampler table run the same model, as a
        stored fit has one.
    """
    raw = tomllib.loads(Path(policy).read_text())
    out: dict[str, Any] = {}
    for c in raw["candidates"]:
        if "model" not in c or "sampler" not in c:
            continue
        if c["model"] in out:
            msg = f"Two candidates in {policy} run {c['model']}."
            raise ValueError(msg)
        out[c["model"]] = dict(c["sampler"])
    return out


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


def fit_code(model: str, root: Path | str = ".") -> list[str]:
    """List the source files a stored fit of one model can run.

    The import closure of the model's module and `STORE_ENTRIES`
    (`evaluation.digest.model_code`), without the other models' modules
    and the modules a fit does not call (`STORE_SKIP`), and the package's
    data files (`STORE_DATA`).

    Parameters
    ----------
    model : str
        A registered model.
    root : pathlib.Path or str
        The repository root.

    Returns
    -------
    list of str
        Paths relative to `root`, sorted.
    """
    from bdbvhubrenewal.evaluation.digest import model_code

    code = model_code(model, root, STORE_ENTRIES, cli=False, skip=STORE_SKIP)
    data = [p.relative_to(root).as_posix() for p in Path(root).glob(STORE_DATA)]
    return sorted([*code, *data])


@functools.cache
def _export(root: str, lock: str) -> str:
    """Return `uv export` of the runtime dependencies; `lock` keys the cache."""
    del lock
    return subprocess.run(
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
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def code_digest(model: str, root: Path | str = ".") -> str:
    """Hash the code a stored fit of a model runs and its dependencies.

    Python files in `fit_code` are hashed as syntax trees without
    docstrings, so comments and docstrings do not change the key; other
    files by their bytes. `run_forecasts` and the backtest helpers it
    calls are hashed the same way, with the Python minor version.
    The dependencies are `uv export --no-dev --all-extras` of `uv.lock`.

    Parameters
    ----------
    model : str
        A registered model.
    root : pathlib.Path or str
        The repository root.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    base = Path(root)
    names = fit_code(model, base)
    files: list[Path] = []
    for name in names:
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
    lock = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        lock.update((base / name).read_bytes())
    digest.update(_export(str(base.resolve()), lock.hexdigest()).encode())
    return digest.hexdigest()


def origin_tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    daily: pd.DataFrame,
) -> dict[str, ForecastTask]:
    """Build the task of every real origin, as the store fits it.

    Each is the task the submission fits at that vintage
    (`vintage_task`), with `NEXT_ROUND_WEEKS` more weeks, so the
    submission forecasts its vintage's own round, or the next round when
    no new release came, from the stored fit. A stored task is asked for
    final counts only (no `report_by`), so it holds no release after its
    origin. Releases in `config.skip_origins`
    are not origins, but stay in the later tasks' histories.

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
    return {
        v: vintage_task(target, delays, config, v, daily, NEXT_ROUND_WEEKS)[0]
        for v in real_vintages(target)
        if v not in config.skip_origins
    }


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
    code: str | Mapping[str, str],
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
    code : str or mapping
        `code_digest`, or each model's by name.

    Returns
    -------
    dict
        Key by (model, origin).
    """
    built = {m: build_model(m, sampler) for m in models}
    codes = {m: code if isinstance(code, str) else code[m] for m in models}
    return {
        (m, origin): fit_key(built[m], task, codes[m])
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
    files : dict
        A submission candidate's fit directory (`fit_files`): file name
        to base64 contents. Empty for other models.
    """

    meta: dict[str, Any]
    forecasts: pd.DataFrame
    fits: pd.DataFrame
    delays: pd.DataFrame
    files: dict[str, str] = field(default_factory=dict)


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
        "files": stored.files,
    }
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
        body["meta"], forecasts, fits, delays, body.get("files", {})
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
    start = time.monotonic()
    try:
        forecasts = run_forecasts([model], [task], fits, learnt)
        files = fit_files(model, task, sampler, time.monotonic() - start)
    except Exception as err:
        return failed(meta, err)
    return StoredFit(
        meta,
        forecasts,
        pd.concat(fits, ignore_index=True) if fits else pd.DataFrame(),
        pd.concat(learnt, ignore_index=True) if learnt else pd.DataFrame(),
        files,
    )


def fit_files(
    model: ForecastModel,
    task: ForecastTask,
    sampler: Mapping[str, Any],
    seconds: float = 0.0,
) -> dict[str, str]:
    """Return a submission candidate's fit directory, after its forecast.

    The directory `bdbvhubrenewal fit --config` would write: the model's
    `write_fit_dir` (samples, the unassigned national part,
    `ppc.json`), with `diagnostics.json` and `fit.json`
    (`write_records`). A model that is no candidate's
    (`renewal.FIT_VARIANTS`) has none.

    Parameters
    ----------
    model : ForecastModel
        The model, after it forecast `task`.
    task : ForecastTask
        The task.
    sampler : mapping
        `samplers`.
    seconds : float
        How long the fit took.

    Returns
    -------
    dict
        File name to base64 contents, or empty.
    """
    from bdbvhubrenewal.forecasters.renewal import FIT_VARIANTS
    from bdbvhubrenewal.submission.fitting import get_config, write_records

    write = getattr(model, "write_fit_dir", None)
    names = [c for c, v in FIT_VARIANTS.items() if v == model.name]
    if write is None or not names:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        diag = write(task, out, config_name=names[0])
        write_records(
            get_config(names[0]),
            task,
            sampler.get(model.name, {}),
            out,
            model=model.name,
            diagnostics=diag,
            seconds=seconds,
        )
        return {
            p.name: base64.b64encode(p.read_bytes()).decode()
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "progress.log"
        }


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


def unpack(
    store: Path | str,
    keys: Keys,
    model: str,
    origin: str,
    out: Path | str,
) -> tuple[str, StoredFit | None]:
    """Write a stored candidate fit's directory for the submission.

    Only the fit of the current key is used: same code and data, same
    fit. Writes the fit's files and `UNPACKED_FILE`, which says which
    fit they came from and at which store commit.

    Parameters
    ----------
    store : pathlib.Path or str
        The store root.
    keys : dict
        `current_keys`.
    model, origin : str
        The fit.
    out : pathlib.Path or str
        The fit directory to write.

    Returns
    -------
    tuple
        `"ok"`, `"missing"` (no fit, or one with no fit directory) or
        `"failed"`, and the stored fit if there is one.
    """
    key = keys[(model, origin)]
    path = store_path(store, model, origin, key)
    if not path.exists():
        return "missing", None
    stored = read_fit(path)
    if stored.meta["status"] != "ok":
        return "failed", stored
    if not stored.files:
        return "missing", stored
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    for name, body in stored.files.items():
        (target / name).write_bytes(base64.b64decode(body))
    commit = Path(store) / COMMIT_FILE
    record = {
        "model": model,
        "origin": origin,
        "key": key,
        "commit": stored.meta["commit"],
        "fitted_at": stored.meta["fitted_at"],
        "store_commit": (
            commit.read_text().strip() if commit.exists() else None
        ),
    }
    (target / UNPACKED_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return "ok", stored


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
    min_horizon: int | None = None,
    max_horizon: int | None = None,
) -> list[StoreStatus]:
    """Write the stored fits where the backtest and the docs read them.

    Every origin goes to `forecasts.csv.gz`, from `min_horizon` to
    `max_horizon`: a stored fit holds the submission's weeks, which start
    earlier and end later.
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
    min_horizon, max_horizon : int, optional
        The backtest's weeks (`[forecast]`). None keeps every week.

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
    forecasts = [f.forecasts for f in loaded]
    low = -(10**6) if min_horizon is None else min_horizon
    high = 10**6 if max_horizon is None else max_horizon
    forecasts = [
        f.loc[f["horizon"].between(low, high)] if "horizon" in f else f
        for f in forecasts
    ]
    for name, frames in (
        ("forecasts.csv.gz", forecasts),
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
        write_csv(merge_by_model(path, table), path)


def command(args: argparse.Namespace) -> int:
    """Run `bdbvhubrenewal fit-store plan|fit|load|unpack`.

    `plan` prints the fits to make and writes them as a GitHub Actions
    matrix (`--out`), with every current key (`--keys`) for pruning;
    `--origins` keeps only those origins.
    `fit` fits one model at one origin and writes it to `--store`.
    `load` writes the stored fits into `--results`.
    `unpack` writes the stored fit of `--model` at `--origin` into the
    fit directory `--out`, for the submission.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed `fit-store` arguments.

    Returns
    -------
    int
        0 on success, 1 if the fit failed or the arguments are wrong.
        `unpack` returns 3 if the store has no fit at the current key
        and 4 if that fit failed.
    """
    sampler = samplers(args.policy)
    if args.action == "fit":
        from bdbvhubrenewal.devices import host_devices

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
    one = args.action in ("fit", "unpack")
    names = [args.model] if one else models
    unknown = sorted(set(names) - set(stored_models()))
    if unknown:
        print(
            f"ERROR: {unknown} not in the stored models {stored_models()}.",
            file=sys.stderr,
        )
        return 1
    code = {m: code_digest(m) for m in names}
    if args.action == "fit":
        if args.origin not in tasks:
            print(
                f"ERROR: fit needs --origin in {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        stored = fit(args.model, tasks[args.origin], sampler, code[args.model])
        path = write_fit(args.store, stored)
        print(f"Wrote {path} ({stored.meta['status']}).")
        if stored.meta["status"] != "ok":
            print(f"ERROR: {stored.meta['error']}", file=sys.stderr)
            return 1
        return 0
    if args.action == "unpack":
        if args.origin not in tasks or not args.out:
            print(
                f"ERROR: unpack needs --out and --origin in {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        keys = current_keys(
            {args.origin: tasks[args.origin]}, names, sampler, code
        )
        state, found = unpack(
            args.store, keys, args.model, args.origin, args.out
        )
        key = keys[(args.model, args.origin)]
        print(f"{args.model} at {args.origin} ({key}): {state}.")
        if state == "failed" and found is not None:
            print(f"ERROR: {found.meta.get('error')}", file=sys.stderr)
        return {"ok": 0, "missing": 3, "failed": 4}[state]
    if args.origins:
        unknown = sorted(set(args.origins) - set(tasks))
        if unknown:
            print(
                f"ERROR: {', '.join(unknown)} not among the origins the "
                f"store fits, {list(tasks)}.",
                file=sys.stderr,
            )
            return 1
        tasks = {o: t for o, t in tasks.items() if o in set(args.origins)}
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
    for s in load(
        args.store, keys, args.results, config.min_horizon, config.max_horizon
    ):
        state = "current" if s.current else "not fitted for this code"
        if s.used and not s.current:
            state += f", older fit {s.used} used"
        if s.failed:
            state += ", the fit for this code failed"
        print(f"{s.model} at {s.origin}: {state}.")
    return 0
