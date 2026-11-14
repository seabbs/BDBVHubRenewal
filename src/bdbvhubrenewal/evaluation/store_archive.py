# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Every candidate model's forecasts for a round, for the ledger.

The submission ledger keeps the submitted file of each round.
`archive` adds the forecasts of every stored model and every submission
candidate at the round's origin, so each can be scored prospectively,
and not only the one submitted.
A stored model's forecasts are those the store holds when the round is
recorded, the fit `store.load` uses.
A model the store does not hold, such as `randomwalk`, is forecast then
by `store.fit`, which writes nothing to the store.
Each row carries its fit's key, commit and `fitted_at`, so an analysis
can compare when each forecast was made with the window's close.
`ci/steps/archive_candidates.sh` writes it to the ledger as
`archive/<reference_date>/<run>/candidates.csv.gz`
(`bdbvhubrenewal fit-store archive`).

This module is apart from `store`, whose source keys every stored fit.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import FORECAST_COLUMNS, load_config
from bdbvhubrenewal.forecasters import ForecastTask

__all__ = ["COLUMNS", "FILE", "archive", "candidate_models", "command"]

FILE: Final = "candidates.csv.gz"
"""The archive's name in a round's ledger folder."""

RANDOM_WALK: Final = "randomwalk"
"""The model of a `randomwalk` candidate that names none, as in
`ci/steps/candidate.sh`."""

COLUMNS: Final = (*FORECAST_COLUMNS, "key", "commit", "fitted_at", "current")
"""The archive's columns: the forecasts, then the fit that made them."""


def candidate_models(policy: Path | str = "ci/policy.toml") -> list[str]:
    """Return every stored model and every submission candidate's model.

    A candidate's model is its `model`, and a `randomwalk` candidate that
    names none runs `RANDOM_WALK`.

    Parameters
    ----------
    policy : pathlib.Path or str
        The CI policy.

    Returns
    -------
    list of str
        Model names, stored models first, without repeats.
    """
    raw = tomllib.loads(Path(policy).read_text())
    named = [
        c.get("model") or RANDOM_WALK
        for c in raw["candidates"]
        if "model" in c or c["kind"] == "randomwalk"
    ]
    return list(dict.fromkeys([*store.stored_models(), *named]))


def archive(
    root: Path | str,
    tasks: Mapping[str, ForecastTask],
    models: Sequence[str],
    sampler: Mapping[str, Any],
    code: Mapping[str, str],
) -> pd.DataFrame:
    """Return each model's forecasts at each origin, with their fit.

    Parameters
    ----------
    root : pathlib.Path or str
        The store root.
    tasks : mapping
        `store.origin_tasks`, for the round's origin.
    models : sequence of str
        The models to archive (`candidate_models`).
    sampler : mapping
        `store.samplers`.
    code : mapping
        `store.code_digest` by model.

    Returns
    -------
    pandas.DataFrame
        `COLUMNS`: the forecasts, every week and sample, with the fit's
        `key`, `commit` and `fitted_at`, and whether it is the fit of
        the current code and data (`current`). A stored model with no
        good fit, a model that cannot be built, or a forecast that
        fails, has no rows.
    """
    stored = set(store.stored_models())
    keys: store.Keys = {}
    for model in models:
        try:
            keys |= store.current_keys(tasks, [model], sampler, code)
        except Exception as err:
            print(f"{model} not archived: {err!r}.", file=sys.stderr)
    rows = [pd.DataFrame(columns=COLUMNS)]
    for (model, origin), key in sorted(keys.items()):
        if model in stored:
            fit, _ = store._choose(Path(root), model, origin, key)
        else:
            fit = store.fit(model, tasks[origin], sampler, code[model])
        if fit is None or fit.meta["status"] != "ok":
            continue
        meta = fit.meta
        rows.append(
            fit.forecasts.assign(
                key=meta["key"],
                commit=meta["commit"],
                fitted_at=meta["fitted_at"],
                current=meta["key"] == key,
            )
        )
    return pd.concat(rows, ignore_index=True)[list(COLUMNS)]


def command(args: argparse.Namespace) -> int:
    """Run `bdbvhubrenewal fit-store archive`.

    Writes `archive` at `--origins` to `--out`, for `--models` or else
    `candidate_models`.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed `fit-store` arguments.

    Returns
    -------
    int
        0 on success, 1 if `--out` or `--origins` is missing or no
        origin is a real one.
    """
    if not (args.out and args.origins):
        print("ERROR: archive needs --out and --origins.", file=sys.stderr)
        return 1
    tasks = store.origin_tasks(
        load_target(args.data),
        load_delays(args.data),
        load_config(args.config),
        load_daily(args.data),
    )
    tasks = {o: t for o, t in tasks.items() if o in set(args.origins)}
    if not tasks:
        print(f"ERROR: no real origin in {args.origins}.", file=sys.stderr)
        return 1
    models = args.models or candidate_models(args.policy)
    code = {m: store.code_digest(m) for m in models}
    table = archive(
        args.store, tasks, models, store.samplers(args.policy), code
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(table, out)
    fits = table.drop_duplicates(["model", "vintage"])
    for row in fits.itertuples():
        print(f"{row.model} at {row.vintage}: {row.key}, {row.fitted_at}.")
    n = len(models) * len(tasks)
    print(f"Archived {len(fits)} of {n} forecasts in {out}.")
    return 0
