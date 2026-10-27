# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Command line interface: `bdbv-forecasts <command>`.

Commands:

- `data`: download the hub data and write the tidy tables.
- `backtest`: run the development backtest and the recovery check, or
  with `--evaluate-holdout` the rolling held-out evaluation.
- `nowcast`: forecast from a real vintage, write the hub file and check it.
  With `--config` (and `--fit`), forecast from a fit configuration.
- `fit`: fit a configuration and write the fit directory and its
  `diagnostics.json`, for the submission CI.
- `plot-submission`: plot a submitted hub file.
- `plot-ppc`: plot a fit's posterior predictive checks (`ppc.json`).
- `coherence`: compare the national data with the sum of the provinces.
- `validate`: run the hub's checks on files (`hub_validate`).
- `metadata`: write and check a model's metadata YAML.
- `compare-hub`: score our forecasts against the hub's submitted ones.
- `report`: make the docs figures and tables.

`--version` prints the package version.
Paths to the tidy data default to `$BDBV_DATA_DIR` or `data/`, so the
submission CI can point every command at the data it pinned.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from . import backtest as bt
from .data import (
    build_tidy_data,
    coherence_summary,
    delays_for_vintage,
    fetch_hub_data,
    load_delays,
    load_target,
)
from .fitting import Forecast, get_config, run_fit, run_forecast
from .heldout import HeldOutError, run_heldout
from .hub import (
    load_hub_round,
    model_id,
    render_model_metadata,
    to_hub_table,
    validate_model_metadata,
    validate_submission_file,
    write_model_metadata,
    write_submission,
)
from .models.base import (
    NATIONAL,
    ForecastTask,
    check_samples,
    get_model,
    list_models,
)

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
            "https://seabbs.github.io/bdbv-hub-baseline-forecasts/methods/model/."
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
    parser.add_argument(
        "--version", action="store_true", help="Print the version and exit."
    )
    sub = parser.add_subparsers(dest="command")
    data_dir = os.environ.get("BDBV_DATA_DIR", "data")

    data = sub.add_parser("data", help="Download and tidy the hub data.")
    data.add_argument(
        "--raw", default=os.environ.get("BDBV_RAW_DIR", f"{data_dir}/raw")
    )
    data.add_argument("--out", default=data_dir)
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
        help="Run the rolling held-out evaluation instead.",
    )

    now = sub.add_parser("nowcast", help="Forecast and write a hub file.")
    now.add_argument(
        "--model",
        default="persistence",
        help="Model name: the hub model id is `epiforecasts-<model>`. "
        "Without --config, the registered model to run.",
    )
    _task_args(now, data_dir)
    now.add_argument(
        "--config",
        default=None,
        help="Fit configuration to forecast from, such as `main`.",
    )
    now.add_argument(
        "--fit", default=None, help="Fit directory written by `fit`."
    )
    now.add_argument("--out", default="output/submission")
    now.add_argument("--hub-config", default=None)

    fit = sub.add_parser(
        "fit", help="Fit a configuration and write its diagnostics."
    )
    fit.add_argument("--model", default="persistence")
    fit.add_argument("--config", required=True)
    _task_args(fit, data_dir)
    fit.add_argument(
        "--sampler-json",
        default="{}",
        help="Sampler settings as JSON (a candidate's `sampler` table).",
    )
    fit.add_argument("--out", required=True, help="Fit directory.")

    plot = sub.add_parser("plot-submission", help="Plot a submitted hub file.")
    plot.add_argument("--submission", required=True)
    plot.add_argument("--vintage", default="latest")
    plot.add_argument("--data", default=data_dir)
    plot.add_argument("--out", required=True)

    ppc = sub.add_parser(
        "plot-ppc", help="Plot a fit's posterior predictive checks."
    )
    ppc.add_argument("--ppc", required=True, help="A ppc.json file.")
    ppc.add_argument("--out", required=True)
    ppc.add_argument("--stream", default="total")

    coh = sub.add_parser(
        "coherence",
        help="Compare the national data with the sum of the provinces.",
    )
    coh.add_argument("--vintage", default="latest")
    coh.add_argument("--data", default=data_dir)
    coh.add_argument("--out", default=None)

    sub.add_parser(
        "validate",
        help="Check hub files with hub_validate (see `validate --help`).",
        add_help=False,
    )

    meta = sub.add_parser("metadata", help="Write model metadata YAML.")
    meta.add_argument("--model", default="persistence")
    meta.add_argument("--out", default="hub/model-metadata")
    meta.add_argument(
        "--schema", default=f"{data_dir}/hub-config/model-metadata-schema.json"
    )

    cmp = sub.add_parser(
        "compare-hub", help="Compare our forecasts with hub submissions."
    )
    cmp.add_argument("--config", default="config/hub-comparison.toml")
    cmp.add_argument("--backtest-config", default="config/backtest.toml")
    cmp.add_argument("--data", default=data_dir)
    cmp.add_argument("--hub-dir", default=f"{data_dir}/raw/hub-submissions")
    cmp.add_argument("--out", default="output/hub-comparison")
    cmp.add_argument("--models", nargs="*", default=None)
    cmp.add_argument(
        "--fetch",
        action="store_true",
        help="Download the hub files first.",
    )

    rep = sub.add_parser("report", help="Make the docs figures and tables.")
    rep.add_argument("--figures", default="docs/figures")
    rep.add_argument("--generated", default="docs/generated")
    rep.add_argument(
        "--ledger",
        default=None,
        help="Checkout of the submission-ledger branch, for the pages of "
        "submitted rounds (docs/nowcast/rounds/).",
    )
    return parser


def _task_args(p: argparse.ArgumentParser, data_dir: str) -> None:
    p.add_argument(
        "--vintage", default="latest", help="Real vintage, or `latest`."
    )
    p.add_argument("--data", default=data_dir)
    p.add_argument("--backtest-config", default="config/backtest.toml")
    p.add_argument(
        "--min-horizon",
        type=int,
        default=-6,
        help="First week, in weeks from the reference date. The default "
        "submits 7 weeks up to the reference date, which the CI's "
        "`observed` and `coverage` checks need.",
    )


def _version() -> str:
    try:
        return version("bdbv-forecasts")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _vintage(target_vintages: Sequence[str], vintage: str | None) -> str:
    if vintage in (None, "", "latest"):
        return max(target_vintages)
    assert vintage is not None
    if vintage not in target_vintages:
        raise ValueError(
            f"No real data for vintage {vintage}; latest is "
            f"{max(target_vintages)}. Run `bdbv-forecasts data` first."
        )
    return vintage


def _task(args: argparse.Namespace) -> tuple[ForecastTask, list[str]]:
    """Build the forecast task, and notes on any stream that stood in."""
    config = dataclasses.replace(
        bt.load_config(args.backtest_config), min_horizon=args.min_horizon
    )
    target, delays = load_target(args.data), load_delays(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = _vintage([str(v) for v in real["vintage"].unique()], args.vintage)
    rows = real.loc[real["vintage"] == vintage]
    delays, note = delays_for_vintage(delays, vintage)
    return bt.make_tasks(rows, delays, config)[0], [note] if note else []


def _fit(args: argparse.Namespace) -> int:
    try:
        config = get_config(args.config)
        sampler = json.loads(args.sampler_json or "{}")
        task, _ = _task(args)
        result = run_fit(config, task, sampler, args.out, model=args.model)
    except (KeyError, ValueError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    if config.stand_in:
        print(f"Stand-in: {config.stand_in}.")
    diag = result.diagnostics
    print(
        f"Fitted config {config.name!r} on vintage {task.vintage} "
        f"(sampler {diag.get('sampler')}); wrote {result.fit_dir}."
    )
    return 0


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
        # The held-out run uses the models in the model choice record.
        try:
            scores = run_heldout(
                target,
                delays,
                config,
                evaluate_holdout=True,
                out_dir=results,
            )
        except HeldOutError as err:
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
    by_truth = bt.score_by_truth_vintage(forecasts, target)
    by_truth.to_csv(results / "backtest-scores-by-truth.csv", index=False)
    rforecasts, rscores = bt.run_recovery(models, config)
    rforecasts.to_csv(output / "recovery-forecasts.csv.gz", index=False)
    rscores.to_csv(results / "recovery-scores.csv", index=False)
    print(
        f"Development: {len(scores)} scored forecasts; recovery: "
        f"{len(rscores)}. Written to {results}."
    )
    return 0


def _nowcast(args: argparse.Namespace) -> int:
    try:
        task, handled = _task(args)
        if args.config:
            config = get_config(args.config)
            if config.stand_in:
                print(f"Stand-in: {config.stand_in}.")
            result = run_forecast(config, task, args.fit)
        else:
            samples = get_model(args.model).forecast(task)
            check_samples(samples, task)
            result = Forecast(samples)
        table = to_hub_table(
            result.samples, task.reference_date, unassigned=result.unassigned
        )
    except (KeyError, ValueError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    mid = model_id(args.model)
    path = write_submission(table, args.out, mid)
    _write_national_sidecar(Path(args.out), task, result)
    # Problems handled on the way, for the PR's `Handled:` line.
    (Path(args.out) / "handled.json").write_text(json.dumps(handled) + "\n")
    for note in handled:
        print(f"Handled: {note}")
    hub_config = args.hub_config or str(
        Path(args.data) / "hub-config" / "tasks.json"
    )
    hub_round = load_hub_round(hub_config)
    open_round = task.reference_date.isoformat() in hub_round.reference_dates
    errors = validate_submission_file(path, hub_round, check_round=open_round)
    print(f"Wrote {path} ({len(table)} rows, locations incl. {NATIONAL}).")
    if not open_round:
        print(
            f"Note: reference date {task.reference_date} is not an open round "
            "in tasks.json, so round checks were skipped."
        )
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


def _write_national_sidecar(
    out: Path, task: ForecastTask, result: Forecast
) -> Path:
    """Write `national.json`, which the CI's national check reads."""
    unassigned = []
    if result.unassigned is not None:
        unassigned = [
            {
                "target_end_date": str(r["target_end_date"]),
                "sample": int(r["sample"]),
                "value": float(round(float(r["value"]))),
            }
            for r in result.unassigned.to_dict("records")
        ]
    path = out / "national.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "national": NATIONAL,
        "modelled_provinces": list(task.locations),
        "unassigned": unassigned,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _plot_submission(args: argparse.Namespace) -> int:
    from .plots import plot_submission

    target = load_target(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = _vintage([str(v) for v in real["vintage"].unique()], args.vintage)
    fig = plot_submission(args.submission, vintage, target=target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"Wrote {out}.")
    return 0


def _plot_ppc(args: argparse.Namespace) -> int:
    from .ppc import plot_ppc, read_ppc

    ppc = read_ppc(args.ppc)
    if not ppc.get("available"):
        print(
            f"No posterior predictive checks to plot: {ppc.get('reason')}.",
            file=sys.stderr,
        )
        return 1
    fig = plot_ppc(ppc, stream=args.stream)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"Wrote {out}.")
    return 0


def _coherence(args: argparse.Namespace) -> int:
    target = load_target(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = _vintage([str(v) for v in real["vintage"].unique()], args.vintage)
    summary = coherence_summary(target, vintage)
    _print_coherence(summary)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def _print_coherence(summary: dict[str, object]) -> None:
    if summary["coherent"]:
        print(
            f"National coherence ({summary['as_of']}): the national counts "
            "equal the sum of the provinces in every week and stream."
        )
        return
    print(
        f"WARNING: national coherence ({summary['as_of']}): the national "
        "counts differ from the sum of the provinces.",
        file=sys.stderr,
    )
    streams = summary["streams"]
    assert isinstance(streams, dict)
    for name, s in streams.items():
        print(
            f"  {name}: {s['weeks_with_gap']} weeks with a gap, largest "
            f"{s['max_abs_gap']}, net {s['total_gap']}",
            file=sys.stderr,
        )


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


def _compare_hub(args: argparse.Namespace) -> int:
    from . import hubcompare as hc

    cfg = hc.load_comparison_config(args.config)
    hub_dir = Path(args.hub_dir)
    if args.fetch:  # pragma: no cover - network
        hc.fetch_hub_submissions(hub_dir, cfg.models, ref=cfg.ref)
    weekly, other = hc.load_hub_submissions(hub_dir)
    weekly = weekly.loc[weekly["model"].isin(cfg.models)]
    other = other.loc[other["model"].isin(cfg.models)]
    target, delays = load_target(args.data), load_delays(args.data)
    config = bt.load_config(args.backtest_config)
    ours = hc.our_forecasts(
        sorted(set(weekly["reference_date"])),
        target,
        delays,
        config,
        _models(args.models),
    )
    scores, pending = hc.score_comparison(weekly, ours, target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    forecasts = pd.concat(
        [weekly.drop(columns="output_type").assign(source="hub"), ours],
        ignore_index=True,
    )
    forecasts.to_csv(out / "forecasts.csv.gz", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    pending.to_csv(out / "pending.csv", index=False)
    other.to_csv(out / "other-rounds.csv", index=False)
    teams = hc.load_team_metadata(hub_dir)
    teams.loc[teams["model"].isin(cfg.models)].to_csv(
        out / "teams.csv", index=False
    )
    manifest = hub_dir / "manifest.json"
    if manifest.exists():
        (out / "manifest.json").write_text(manifest.read_text())
    print(
        f"Hub comparison: {len(scores)} scored forecasts, "
        f"{len(pending)} rounds pending, {len(other)} files from other "
        f"rounds. Written to {out}."
    )
    return 0


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["validate"]:
        from .hub_validate import main as validate_main

        return validate_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(_version())
        return 0
    if args.command is None:
        parser.error("a command is required")
    if args.command == "data":
        fetch_hub_data(args.raw, ref=args.ref)
        tables = build_tidy_data(args.raw, args.out)
        print(f"Tidy data written to {args.out} on {dt.date.today()}.")
        target = tables["target"]
        latest = str(target.loc[target["kind"] == "real", "vintage"].max())
        _print_coherence(coherence_summary(target, latest))
        return 0
    if args.command == "fit":
        return _fit(args)
    if args.command == "plot-submission":
        return _plot_submission(args)
    if args.command == "coherence":
        return _coherence(args)
    if args.command == "plot-ppc":
        return _plot_ppc(args)
    if args.command == "backtest":
        return _backtest(args)
    if args.command == "nowcast":
        return _nowcast(args)
    if args.command == "metadata":
        return _metadata(args)
    if args.command == "compare-hub":
        return _compare_hub(args)
    from .report import make_report

    make_report(figures=args.figures, generated=args.generated)
    from .rounds import make_round_pages

    done = make_round_pages(args.ledger, generated=args.generated)
    print(f"Round pages: {len(done)} submitted rounds.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
