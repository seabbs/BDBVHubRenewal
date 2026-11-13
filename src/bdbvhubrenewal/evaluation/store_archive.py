# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Each stored model's forecasts for a round, as the fit store held them.

The submission ledger keeps the submitted file of each round.
`archive` adds every stored model's forecasts at the round's origin, so
each candidate can be scored prospectively, and not only the one
submitted.
It takes the fit `store.load` would have used when the round's window
closed, and records when that fit was made and whether it was made by
the close.
`ci/steps/record.sh` writes it to the ledger as
`archive/<reference_date>/<run>/candidates.csv.gz`
(`bdbvhubrenewal fit-store archive`).

This module is apart from `store`, whose source keys every stored fit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import load_config

__all__ = ["FILE", "archive", "command"]

FILE = "candidates.csv.gz"
"""The archive's name in a round's ledger folder."""


def _fitted_at(fit: store.StoredFit) -> dt.datetime:
    """Return when a fit was made."""
    return dt.datetime.fromisoformat(fit.meta["fitted_at"])


def _chosen(
    root: Path, model: str, origin: str, key: str, closes: dt.datetime
) -> store.StoredFit | None:
    """Return the fit `store.load` would have used at `closes`.

    Of the good fits made by `closes`, the current key's, else the newest.
    When none was made by then, the same choice over every good fit.
    """
    paths = (root / model / origin).glob("*.json.gz")
    good = [f for f in map(store.read_fit, paths) if f.meta["status"] == "ok"]
    made = [f for f in good if _fitted_at(f) <= closes] or good
    current = [f for f in made if f.meta["key"] == key]
    if current:
        return current[0]
    return max(made, key=_fitted_at, default=None)


def archive(
    root: Path | str, keys: store.Keys, closes: dt.datetime
) -> pd.DataFrame:
    """Return each stored model's forecasts for a round, with their fit.

    Parameters
    ----------
    root : pathlib.Path or str
        The store root.
    keys : dict
        `store.current_keys` at the round's origin.
    closes : datetime.datetime
        When the round's submission window closed.

    Returns
    -------
    pandas.DataFrame
        The stored forecasts (`run_forecasts` columns), every week and
        sample, with the fit's `key`, `commit` and `fitted_at`, whether
        it is the current key's (`current`), and whether it was made by
        the close (`prospective`). A pair with no good fit has no rows.
    """
    rows = []
    for (model, origin), key in sorted(keys.items()):
        fit = _chosen(Path(root), model, origin, key, closes)
        if fit is None:
            continue
        meta = fit.meta
        rows.append(
            fit.forecasts.assign(
                key=meta["key"],
                commit=meta["commit"],
                fitted_at=meta["fitted_at"],
                current=meta["key"] == key,
                prospective=_fitted_at(fit) <= closes,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def command(args: argparse.Namespace) -> int:
    """Run `bdbvhubrenewal fit-store archive`.

    Writes `archive` at `--origins` and `--closes` to `--out`.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed `fit-store` arguments.

    Returns
    -------
    int
        0 on success, 1 if `--closes`, `--out` or `--origins` is missing
        or no origin is a real one.
    """
    if not (args.closes and args.out and args.origins):
        print(
            "ERROR: archive needs --closes, --out and --origins.",
            file=sys.stderr,
        )
        return 1
    closes = dt.datetime.fromisoformat(args.closes)
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
    models = args.models or store.stored_models()
    code = {m: store.code_digest(m) for m in models}
    keys = store.current_keys(tasks, models, store.samplers(args.policy), code)
    table = archive(args.store, keys, closes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(table, out)
    fits = table.drop_duplicates(["model", "vintage"]) if len(table) else table
    for row in fits.itertuples():
        when = "by the close" if row.prospective else "after the close"
        print(f"{row.model} at {row.vintage}: {row.key}, made {when}.")
    print(f"Archived {len(fits)} of {len(keys)} stored fits in {out}.")
    return 0
