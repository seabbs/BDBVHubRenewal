# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Command line interface: `bdbv-forecasts <command>`.

Commands:

- `data`: download the hub data and write the tidy tables.
- `backtest`: run the development backtest and the recovery check, or
  with `--evaluate-holdout` the one-off held-out evaluation.
- `nowcast`: forecast from a real vintage, write the hub file and check it.
- `metadata`: write and check a model's metadata YAML.
- `report`: make the docs figures and tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import backtest as bt
from .data import build_tidy_data, fetch_hub_data, load_delays, load_target
from .hub import (
    load_hub_round,
    model_id,
    render_model_metadata,
    to_hub_table,
    validate_model_metadata,
    validate_submission,
    write_model_metadata,
    write_submission,
)
from .models.base import NATIONAL, get_model, list_models

__all__ = ["build_parser", "main"]

MODEL_METADATA = {
    "persistence": {
        "model_name": "Persistence baseline",
        "methods": (
            "Last complete onset week carried forward per province, with "
            "log-scale random walk noise and Poisson draws; national is the "
            "sum of provinces."
        ),
        "methods_long": (
            "For each province with cases, the last onset week judged complete "
            "(its midpoint at least the 90th centile of the published "
            "onset-to-report delay before the data date) is carried forward. "
            "Later weeks are drawn on the log scale as a random walk whose "
            "weekly SD is the SD of the last 8 complete weekly log changes "
            "(floored at 0.1), with Poisson noise. Earlier weeks are the "
            "published counts. Each sample is a joint trajectory and the "
            "national value is the sum of province samples. No spatial "
            "correlation beyond that sum. See "
            "https://seabbs.github.io/bdbv-hub-baseline-forecasts/models/."
        ),
        "data_inputs": (
            "Hub target data (weekly confirmed cases by onset week, observed "
            "plus imputed) and the hub's published onset-to-report delays."
        ),
    }
}
"""Metadata text for each model that may be submitted."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(prog="bdbv-forecasts")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Download and tidy the hub data.")
    data.add_argument("--raw", default="data/raw")
    data.add_argument("--out", default="data")
    data.add_argument("--ref", default="main")

    back = sub.add_parser("backtest", help="Run the backtest.")
    back.add_argument("--config", default="config/backtest.toml")
    back.add_argument("--data", default="data")
    back.add_argument("--models", nargs="*", default=None)
    back.add_argument("--results", default="results")
    back.add_argument("--output", default="output/backtest")
    back.add_argument("--no-reconstructed", action="store_true")
    back.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Run the one-off held-out evaluation instead.",
    )

    now = sub.add_parser("nowcast", help="Forecast and write a hub file.")
    now.add_argument("--model", default="persistence")
    now.add_argument("--vintage", default=None)
    now.add_argument("--data", default="data")
    now.add_argument("--config", default="config/backtest.toml")
    now.add_argument("--out", default="output/submission")
    now.add_argument("--hub-config", default="data/hub-config/tasks.json")

    meta = sub.add_parser("metadata", help="Write model metadata YAML.")
    meta.add_argument("--model", default="persistence")
    meta.add_argument("--out", default="hub/model-metadata")
    meta.add_argument(
        "--schema", default="data/hub-config/model-metadata-schema.json"
    )

    rep = sub.add_parser("report", help="Make the docs figures and tables.")
    rep.add_argument("--figures", default="docs/figures")
    rep.add_argument("--generated", default="docs/generated")
    return parser


def _models(names: Sequence[str] | None) -> list[str]:
    return list(names) if names else [s.name for s in list_models()]


def _backtest(args: argparse.Namespace) -> int:
    config = bt.load_config(args.config)
    target, delays = load_target(args.data), load_delays(args.data)
    models = [get_model(n) for n in _models(args.models)]
    results, output = Path(args.results), Path(args.output)
    results.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_holdout:
        try:
            scores = bt.run_heldout(
                models,
                target,
                delays,
                config,
                evaluate_holdout=True,
                out_dir=results,
            )
        except bt.HeldOutError as err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print(f"Held-out scores: {len(scores)} rows in {results}.")
        return 0
    forecasts, scores = bt.run_development(
        models,
        target,
        delays,
        config,
        include_reconstructed=not args.no_reconstructed,
    )
    forecasts.to_csv(output / "forecasts.csv.gz", index=False)
    scores.to_csv(results / "backtest-scores.csv", index=False)
    rforecasts, rscores = bt.run_recovery(models, config)
    rforecasts.to_csv(output / "recovery-forecasts.csv.gz", index=False)
    rscores.to_csv(results / "recovery-scores.csv", index=False)
    print(
        f"Development: {len(scores)} scored forecasts; recovery: "
        f"{len(rscores)}. Written to {results}."
    )
    return 0


def _nowcast(args: argparse.Namespace) -> int:
    config = bt.load_config(args.config)
    target, delays = load_target(args.data), load_delays(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = args.vintage or str(real["vintage"].max())
    rows = real.loc[real["vintage"] == vintage]
    tasks = bt.make_tasks(rows, delays, config)
    forecasts = bt.run_forecasts([get_model(args.model)], tasks)
    task = tasks[0]
    samples = forecasts[["location", "target_end_date", "sample", "value"]]
    table = to_hub_table(samples, task.reference_date)
    mid = model_id(args.model)
    path = write_submission(table, args.out, mid)
    hub_round = load_hub_round(args.hub_config)
    open_round = task.reference_date.isoformat() in hub_round.reference_dates
    errors = validate_submission(
        table,
        hub_round,
        file_name=path.name,
        mid=mid,
        check_round=open_round,
    )
    print(f"Wrote {path} ({len(table)} rows, locations incl. {NATIONAL}).")
    if not open_round:
        print(
            f"Note: reference date {task.reference_date} is not an open round "
            "in tasks.json, so round checks were skipped."
        )
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


def _metadata(args: argparse.Namespace) -> int:
    text = MODEL_METADATA[args.model]
    record = render_model_metadata(
        args.model,
        model_name=text["model_name"],
        methods=text["methods"],
        methods_long=text["methods_long"],
        data_inputs=text["data_inputs"],
    )
    path = write_model_metadata(record, args.out)
    errors = validate_model_metadata(record, args.schema, path.name)
    print(f"Wrote {path}.")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "data":
        fetch_hub_data(args.raw, ref=args.ref)
        build_tidy_data(args.raw, args.out)
        print(f"Tidy data written to {args.out} on {dt.date.today()}.")
        return 0
    if args.command == "backtest":
        return _backtest(args)
    if args.command == "nowcast":
        return _nowcast(args)
    if args.command == "metadata":
        return _metadata(args)
    from .report import make_report

    make_report(figures=args.figures, generated=args.generated)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
