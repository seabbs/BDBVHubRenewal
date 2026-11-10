# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Command line interface: `bdbvhubrenewal <command>`.

Commands:

- `data`: download the hub data and write the tidy tables.
- `backtest`: run the development backtest and the recovery check, or
  with `--lopo` the leave-one-province-out check.
- `nowcast`: forecast from a real vintage, write the hub file and check it.
  With `--config` (and `--fit`), forecast from a fit configuration.
- `zone-backtest`: score the random walk models run at zone level.
- `digest`: hash what a backtest reads, to key its cached outputs.
- `concat-results`: stack the results files of runs made apart.
- `score`: score the forecasts of `--forecast-only` runs together.
- `fit`: fit a configuration and write the fit directory and its
  `diagnostics.json`, for the submission CI.
- `fit-store`: list the fitted models' fits the store lacks, make one, or
  load the stored fits into the results (`evaluation.store`).
- `plot-submission`: plot a submitted hub file.
- `plot-ppc`: plot a fit's posterior predictive checks (`ppc.json`).
- `coherence`: compare the national data with the sum of the provinces.
- `validate`: run the hub's checks on files (`submission.validate`).
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

from bdbvhubrenewal.data import (
    build_tidy_data,
    coherence_summary,
    delays_for_vintage,
    fetch_hub_data,
    load_daily,
    load_daily_zones,
    load_delays,
    load_health_zones,
    load_target,
)
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.releases import cutoff_lags
from bdbvhubrenewal.data.simulate import SCENARIO_RT_SCALE
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.devices import host_devices
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import (
    ReleaseBacktest,
    ReleaseForecasts,
    forecast_releases,
    release_backtest,
    score_releases,
    stored_delays,
)
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.renewal import DELAY_SUMMARY_FILE
from bdbvhubrenewal.forecasters.zones import zones_for_model
from bdbvhubrenewal.submission.fitting import (
    Forecast,
    get_config,
    run_fit,
    run_forecast,
)
from bdbvhubrenewal.submission.hub import (
    ROUND_NAME,
    SUBMISSION_CONFIG,
    model_abbr_for,
    model_id,
    render_model_metadata,
    to_hub_table,
    write_model_metadata,
    write_submission,
)
from bdbvhubrenewal.submission.metadata import MODEL_METADATA
from bdbvhubrenewal.submission.validate import (
    Check,
    Hub,
    validate_metadata,
    validate_submission,
)
from bdbvhubrenewal.submission.validate import main as validate_main

__all__ = ["build_parser", "chunk", "host_devices", "main"]

MIN_HORIZON = -6
"""First week forecast, in weeks from the reference date: 7 weeks up to
it, which the CI's `observed` and `coverage` checks need."""

_ABBR_HELP = (
    "Hub model abbreviation. Defaults to `randomwalk` for random walk and "
    "otherwise to the submission abbreviation in config/hub.toml."
)


_CHUNK_HELP = (
    "Run only block K of N (`K/N`, from 0) of the {what}, in order, so the "
    "blocks can run apart and their results be stacked (`concat-results`)."
)


def chunk(items: Sequence[str], spec: str) -> list[str]:
    """Return block K of N of `items`, with `spec` as `K/N` (K from 0).

    The blocks are contiguous and in order, sizes differing by at most
    one, so stacking blocks 0 to N - 1 gives `items` again.

    Parameters
    ----------
    items : sequence of str
        Such as forecast dates or origins.
    spec : str
        `K/N`.

    Returns
    -------
    list of str
        The block, empty if N exceeds the number of items.

    Raises
    ------
    ValueError
        If `spec` is not `K/N` with 0 <= K < N.
    """
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise ValueError(f"chunk must be K/N, got {spec!r}") from None
    if not 0 <= k < n:
        raise ValueError(f"chunk {spec!r} needs 0 <= K < N")
    size, extra = divmod(len(items), n)
    start = k * size + min(k, extra)
    return list(items[start : start + size + (k < extra)])


def _abbr(args: argparse.Namespace) -> str:
    if args.abbr:
        return str(args.abbr)
    if args.model == "randomwalk":
        return "randomwalk"
    return model_abbr_for(args.model, args.hub_toml)


def _abbr_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--abbr", default=None, help=_ABBR_HELP)
    p.add_argument("--hub-toml", default=str(SUBMISSION_CONFIG))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(prog="bdbvhubrenewal")
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
    back.add_argument(
        "--output",
        default="results",
        help="Where the forecasts go. The fitted models' rows there come "
        "from the fit store (`fit-store load`) and are kept.",
    )
    back.add_argument(
        "--scenarios-only",
        action="store_true",
        help="Run only the scenarios calibrated to the real data.",
    )
    back.add_argument(
        "--scenario-as-of",
        default=None,
        help="Real vintage the scenarios copy. Defaults to the latest.",
    )
    back.add_argument(
        "--scenario-dates",
        nargs="*",
        default=None,
        help="Simulated vintages to forecast from. Defaults to the four "
        "weeks up to the scenario vintage.",
    )
    back.add_argument(
        "--scenarios",
        nargs="*",
        choices=list(SCENARIO_RT_SCALE),
        default=None,
        help="Scenarios to run, so each can run apart. Defaults to all.",
    )
    back.add_argument(
        "--recovery-only",
        action="store_true",
        help="Run only the recovery check on the simulated outbreak.",
    )
    back.add_argument(
        "--no-recovery",
        action="store_true",
        help="Leave out the recovery check, to run it apart.",
    )
    back.add_argument(
        "--forecast-only",
        action="store_true",
        help="Forecast without scoring, writing the forecasts to --parts, "
        "so each model can run apart; `score` scores them together.",
    )
    back.add_argument(
        "--parts",
        default="output/parts",
        help="Where --forecast-only writes the forecasts.",
    )
    back.add_argument(
        "--chunk",
        default=None,
        help=_CHUNK_HELP.format(
            what="recovery check's dates (with --recovery-only) or the "
            "scenario dates (with --scenarios-only)"
        ),
    )
    back.add_argument(
        "--lopo",
        action="store_true",
        help="Run the leave-one-province-out check instead.",
    )

    now = sub.add_parser("nowcast", help="Forecast and write a hub file.")
    now.add_argument(
        "--model",
        default="randomwalk",
        help="Model name. Without --config, the registered model to run.",
    )
    _abbr_args(now)
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
    now.add_argument(
        "--forecast-date",
        default=None,
        help="The file's forecast_date. Defaults to today (UTC), the day a "
        "submission's PR is opened.",
    )

    zb = sub.add_parser(
        "zone-backtest", help="Score the random walk models at zone level."
    )
    zb.add_argument("--data", default=data_dir)
    zb.add_argument(
        "--models",
        nargs="*",
        default=["randomwalk", "randomwalknowcast", "pprandomwalknowcast"],
    )
    zb.add_argument(
        "--chunk", default=None, help=_CHUNK_HELP.format(what="origins")
    )
    zb.add_argument(
        "--forecast-only",
        action="store_true",
        help="Forecast without scoring, writing to --parts.",
    )
    zb.add_argument("--parts", default="output/parts")
    zb.add_argument("--out", default="results/zone-scores.csv")
    zb.add_argument("--energy-out", default="results/zone-energy.csv")
    zb.add_argument("--quantiles-out", default="results/zone-quantiles.csv")
    zb.add_argument("--draws-out", default="results/zone-draws.csv.gz")

    dig = sub.add_parser(
        "digest",
        help="Print a hash of what a backtest reads, to key its outputs.",
    )
    dig.add_argument("--root", default=".")
    dig.add_argument(
        "--with",
        dest="extra",
        nargs="*",
        default=[],
        help="Further files or directories the run reads.",
    )
    dig.add_argument("--text", default="", help="Such as the commands run.")
    dig.add_argument(
        "--model",
        default=None,
        help="Hash only this model's code (its import closure) with the "
        "data, for a --forecast-only run of it.",
    )

    sc = sub.add_parser(
        "score",
        help="Score the forecasts written by --forecast-only runs, every "
        "model together, and write the results as one run would.",
    )
    sc.add_argument("--config", default="config/backtest.toml")
    sc.add_argument("--data", default=data_dir)
    sc.add_argument("--results", default="results")
    sc.add_argument("--output", default="results")
    sc.add_argument(
        "parts",
        nargs="+",
        help="The runs' --parts directories, in the order one run takes "
        "the models.",
    )

    cat = sub.add_parser(
        "concat-results",
        help="Stack the results files of runs made apart, such as the "
        "scenarios, in the order given.",
    )
    cat.add_argument("--out", default="results")
    cat.add_argument(
        "--dedupe",
        nargs="*",
        default=[],
        help="Files every part writes in full, such as a scenario's truth "
        "when its dates run apart: one copy of each row is kept.",
    )
    cat.add_argument("parts", nargs="+", help="Each run's results directory.")

    fit = sub.add_parser(
        "fit", help="Fit a configuration and write its diagnostics."
    )
    fit.add_argument("--model", default="randomwalk")
    fit.add_argument("--config", required=True)
    _task_args(fit, data_dir)
    fit.add_argument(
        "--sampler-json",
        default="{}",
        help="Sampler settings as JSON (a candidate's `sampler` table).",
    )
    fit.add_argument("--out", required=True, help="Fit directory.")

    fst = sub.add_parser(
        "fit-store", help="Plan, make or load the fitted models' stored fits."
    )
    fst.add_argument("action", choices=["plan", "fit", "load"])
    fst.add_argument("--store", default="output/fit-store")
    fst.add_argument("--data", default=data_dir)
    fst.add_argument("--config", default="config/backtest.toml")
    fst.add_argument("--policy", default="ci/policy.toml")
    fst.add_argument("--model", help="With `fit`: the model to fit.")
    fst.add_argument("--origin", help="With `fit`: the real origin.")
    fst.add_argument(
        "--out",
        default=None,
        help="With `plan`: write the missing fits as a matrix JSON here.",
    )
    fst.add_argument("--results", default="results")
    fst.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="With `plan` or `load`: only these models (default: all fitted).",
    )
    fst.add_argument(
        "--retry-failed",
        action="store_true",
        help="With `plan`: plan the fits stored as failed again.",
    )
    fst.add_argument(
        "--keys",
        default=None,
        help="With `plan`: write every current key here, for pruning.",
    )

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
        help="Check hub files against the hub's rules (see `validate --help`).",
        add_help=False,
    )

    meta = sub.add_parser("metadata", help="Write model metadata YAML.")
    meta.add_argument(
        "--model", default="randomwalk", choices=sorted(MODEL_METADATA)
    )
    _abbr_args(meta)
    meta.add_argument("--out", default="hub/model-metadata")
    meta.add_argument(
        "--data", default=data_dir, help="Data directory with hub-config/."
    )

    bvd = sub.add_parser(
        "bvd",
        help="Cache the BVDOutbreakSize releases each vintage may use.",
    )
    bvd.add_argument("--data", default=data_dir)
    bvd.add_argument("--backtest-config", default="config/backtest.toml")
    bvd.add_argument(
        "--vintages",
        nargs="*",
        default=None,
        help="Defaults to every real backtest vintage.",
    )

    hid = sub.add_parser(
        "hub-id", help="Print the hub model id that submissions use."
    )
    hid.add_argument("--hub-toml", default=str(SUBMISSION_CONFIG))
    hid.add_argument(
        "--model",
        default="",
        help="A model with its own hub id (`[models]` in config/hub.toml).",
    )

    cmp = sub.add_parser(
        "compare-hub", help="Compare our forecasts with hub submissions."
    )
    cmp.add_argument("--config", default="config/hub-comparison.toml")
    cmp.add_argument("--backtest-config", default="config/backtest.toml")
    cmp.add_argument("--data", default=data_dir)
    cmp.add_argument(
        "--hub-dir",
        default="data/hub-submissions",
        help="Hub metadata and openly licensed forecasts.",
    )
    cmp.add_argument(
        "--cache-dir",
        default=f"{data_dir}/raw/hub-submissions",
        help="Uncommitted hub files, for other licences.",
    )
    cmp.add_argument("--out", default="results/hub-comparison")
    cmp.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to run live. Defaults to the unfitted ones.",
    )
    cmp.add_argument(
        "--stored",
        nargs="*",
        default=None,
        help="Forecast files for the other models. Defaults to the config.",
    )
    cmp.add_argument(
        "--fetch",
        action="store_true",
        help="Download the hub files first.",
    )

    frz = sub.add_parser(
        "freeze-round",
        help="Write a round's docs page to the ledger once, after its PR "
        "merged.",
    )
    frz.add_argument("--ledger", required=True)
    frz.add_argument("--round", required=True)

    rep = sub.add_parser("report", help="Make the docs figures and tables.")
    rep.add_argument("--figures", default="docs/figures")
    rep.add_argument("--generated", default="docs/generated")
    rep.add_argument(
        "--ledger",
        default=None,
        help="Checkout of the submission-ledger branch, for the pages of "
        "submitted rounds (docs/submission/rounds/).",
    )
    return parser


def _task_args(p: argparse.ArgumentParser, data_dir: str) -> None:
    p.add_argument(
        "--vintage", default="latest", help="Real vintage, or `latest`."
    )
    p.add_argument(
        "--reference-date",
        default=None,
        help="The round to forecast. Defaults to the vintage's own round; a "
        "later one forecasts that round from the vintage's data.",
    )
    p.add_argument("--data", default=data_dir)
    p.add_argument("--backtest-config", default="config/backtest.toml")


def _version() -> str:
    try:
        return version("BDBVHubRenewal")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _vintage(target_vintages: Sequence[str], vintage: str | None) -> str:
    if vintage in (None, "", "latest"):
        return max(target_vintages)
    assert vintage is not None
    if vintage not in target_vintages:
        raise ValueError(
            f"No real data for vintage {vintage}; latest is "
            f"{max(target_vintages)}. Run `bdbvhubrenewal data` first."
        )
    return vintage


def _task(args: argparse.Namespace) -> tuple[ForecastTask, list[str]]:
    """Build the forecast task, and notes on any stream that stood in."""
    config = dataclasses.replace(
        bt.load_config(args.backtest_config), min_horizon=MIN_HORIZON
    )
    target, delays = load_target(args.data), load_delays(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = _vintage([str(v) for v in real["vintage"].unique()], args.vintage)
    rows = real.loc[real["vintage"] == vintage]
    delays, note = delays_for_vintage(delays, vintage)
    daily = load_daily(args.data)
    task = bt.make_tasks(rows, delays, config, history=real, daily=daily)[0]
    notes = [note] if note else []
    if args.reference_date and as_date(args.reference_date) != (
        task.reference_date
    ):
        ref = as_date(args.reference_date)
        task = dataclasses.replace(
            task,
            reference_date=ref,
            target_end_dates=bt.forecast_weeks(ref, config),
        )
        notes.append(f"round {ref} forecast from the {vintage} release")
    return task, notes


def _fit(args: argparse.Namespace) -> int:
    try:
        config = get_config(args.config)
        sampler = json.loads(args.sampler_json or "{}")
        from bdbvhubrenewal.forecasters.renewal import sampler_settings

        host_devices(int(sampler_settings(sampler)["num_chains"]))
        task, _ = _task(args)
        result = run_fit(config, task, sampler, args.out, model=args.model)
    except (KeyError, ValueError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    diag = result.diagnostics
    print(
        f"Fitted config {config.name!r} on vintage {task.vintage} "
        f"(sampler {diag.get('sampler')}); wrote {result.fit_dir}."
    )
    return 0


def _models(names: Sequence[str] | None) -> list[str]:
    return list(names) if names else [s.name for s in list_models()]


def _backtest(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.forecasters.renewal import sampler_settings

    config = bt.load_config(args.config)
    bt.apply_sampler(config)
    host_devices(int(sampler_settings()["num_chains"]))
    target, delays = load_target(args.data), load_delays(args.data)
    names = _models(args.models)
    if not args.lopo:
        # Fitted models are never run here: their real forecasts come from
        # the fit store. To fit one locally, run `fit-store fit --model M
        # --origin D` for each origin, then `fit-store load`.
        stored = set(store.stored_models())
        names = [n for n in names if n not in stored]
        if not names:
            print(
                "ERROR: fitted models come from the fit store: run "
                "`task fit-store`.",
                file=sys.stderr,
            )
            return 1
    results, output = Path(args.results), Path(args.output)
    results.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.chunk and not (args.scenarios_only or args.recovery_only):
        print(
            "ERROR: --chunk needs --scenarios-only or --recovery-only.",
            file=sys.stderr,
        )
        return 1
    models = [get_model(n) for n in names]
    if args.forecast_only:
        return _forecast_parts(args, config, target, delays, models)
    fits: list[pd.DataFrame] = []
    if args.scenarios_only:
        as_of, dates = _scenario_dates(args, target)
        sforecasts, sscores, senergy, struth = bt.run_scenarios(
            models,
            config,
            target,
            delays,
            load_health_zones(args.data),
            as_of,
            dates,
            names=args.scenarios or list(SCENARIO_RT_SCALE),
            daily=load_daily(args.data),
        )
        for table, path in (
            (sforecasts, output / "scenario-forecasts.csv.gz"),
            (sscores, results / "scenario-scores.csv.gz"),
            (senergy, results / "scenario-energy.csv"),
        ):
            write_csv(bt.merge_by_model(path, table), path)
        write_csv(struth, results / "scenario-truth.csv")
        print(f"Scenarios: {len(sscores)} scored forecasts in {results}.")
        return 0
    if args.lopo:
        lopo_fc, lopo_scores = bt.run_lopo(
            models,
            target,
            delays,
            config,
            fits=fits,
        )
        write_csv(lopo_fc, output / "lopo-forecasts.csv.gz")
        write_csv(lopo_scores, results / "lopo-scores.csv")
        _write_fits(fits, results / "lopo-fits.csv")
        print(f"Leave one province out: {len(lopo_scores)} scored forecasts.")
        return 0
    if args.recovery_only:
        if args.chunk:
            dates = chunk(config.simulated_dates, args.chunk)
            config = dataclasses.replace(config, simulated_dates=tuple(dates))
        rscores = _recovery(args, config, models, results, output)
        print(f"Recovery: {len(rscores)} scored forecasts in {results}.")
        return 0
    learnt: list[pd.DataFrame] = []
    forecasts = bt.run_development(
        models,
        target,
        delays,
        config,
        fits=fits,
        learnt=learnt,
        daily=load_daily(args.data),
    )
    _write_fits(fits, results / "backtest-fits.csv")
    _write_fits(learnt, output / DELAY_SUMMARY_FILE)
    # Rows of models not run here are kept, so the fitted variants can be
    # run apart from the others, and every model is scored again.
    forecasts = bt.merge_by_model(output / "forecasts.csv.gz", forecasts)
    _release_backtest(args, config, target, delays, results, forecasts)
    write_csv(forecasts, output / "forecasts.csv.gz")
    n_recovery = "not run"
    if not args.no_recovery:
        n_recovery = str(len(_recovery(args, config, models, results, output)))
    print(
        f"Development: {len(forecasts)} forecast rows; recovery: "
        f"{n_recovery}. Written to {results}."
    )
    return 0


def _scenario_dates(
    args: argparse.Namespace, target: pd.DataFrame
) -> tuple[str, list[str]]:
    """Return the scenarios' real vintage and their forecast dates."""
    as_of = args.scenario_as_of or bt.real_vintages(target)[-1]
    dates = args.scenario_dates or [
        (as_date(as_of) - dt.timedelta(weeks=k)).isoformat()
        for k in (3, 2, 1, 0)
    ]
    if args.chunk:
        dates = chunk(dates, args.chunk)
    return as_of, dates


def _forecast_parts(
    args: argparse.Namespace,
    config: bt.BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    models: Sequence[ForecastModel],
) -> int:
    """Forecast without scoring and write the forecasts to `args.parts`.

    The forecasting half of each backtest, so each model can run apart
    and be cached on its own code; `score` scores the parts together.
    """
    parts = Path(args.parts)
    parts.mkdir(parents=True, exist_ok=True)
    daily = load_daily(args.data)
    if args.scenarios_only:
        as_of, dates = _scenario_dates(args, target)
        forecasts, zones = bt.scenario_forecasts(
            models,
            config,
            target,
            delays,
            load_health_zones(args.data),
            as_of,
            dates,
            names=args.scenarios or list(SCENARIO_RT_SCALE),
            daily=daily,
        )
        write_csv(forecasts, parts / "scenario-forecasts.csv.gz")
        write_csv(zones, parts / "scenario-zone-forecasts.csv.gz")
    elif args.recovery_only:
        if args.chunk:
            dates = chunk(config.simulated_dates, args.chunk)
            config = dataclasses.replace(config, simulated_dates=tuple(dates))
        forecasts = bt.recovery_forecasts(
            models, config, lags=cutoff_lags(daily)
        )
        write_csv(forecasts, parts / "recovery-forecasts.csv.gz")
    else:
        fits: list[pd.DataFrame] = []
        learnt: list[pd.DataFrame] = []
        forecasts = bt.run_development(
            models, target, delays, config, fits=fits, learnt=learnt,
            daily=daily,
        )  # fmt: skip
        write_csv(forecasts, parts / "forecasts.csv.gz")
        for name, frames in (
            ("backtest-fits.csv", fits),
            (DELAY_SUMMARY_FILE, learnt),
        ):
            table = pd.concat(frames, ignore_index=True) if frames else None
            write_csv(pd.DataFrame() if table is None else table, parts / name)
        tasks = list(store.origin_tasks(target, delays, config, daily).values())
        forecast_releases(tasks, target, models).write(parts)
    print(f"Forecasts of {[m.name for m in models]} written to {parts}.")
    return 0


def _stack(parts: Sequence[Path], name: str) -> pd.DataFrame | None:
    """Stack one file across the parts that have it, or None if none do."""
    tables = [
        t
        for p in parts
        if (p / name).exists() and not (t := _read_or_empty(p / name)).empty
    ]
    return pd.concat(tables, ignore_index=True) if tables else None


def _order(table: pd.DataFrame) -> list[str]:
    """Return the models of a stacked table in the order they appear."""
    return [str(m) for m in pd.unique(table["model"])]


def _score(args: argparse.Namespace) -> int:
    """Score every model's forecasts from --forecast-only runs together.

    Writes the files the backtest, recovery check, scenarios and zone
    backtest write when they run every model at once.
    """
    from bdbvhubrenewal.evaluation.zones import ZoneForecasts, score_zones

    parts = [Path(p) for p in args.parts]
    results, output = Path(args.results), Path(args.output)
    results.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    config = bt.load_config(args.config)
    target, delays = load_target(args.data), load_delays(args.data)
    daily = load_daily(args.data)
    dev = _stack(parts, "forecasts.csv.gz")
    if dev is not None:
        order = _order(dev)
        forecasts = bt.merge_by_model(
            output / "forecasts.csv.gz", bt.joint_order(dev, order)
        )
        for name, path in (
            ("backtest-fits.csv", results / "backtest-fits.csv"),
            (DELAY_SUMMARY_FILE, output / DELAY_SUMMARY_FILE),
        ):
            table = _stack(parts, name)
            if table is not None:
                write_csv(
                    bt.merge_by_model(path, bt.joint_order(table, order)),
                    path,
                )
        release = [p for p in parts if (p / "release-runs.csv.gz").exists()]
        _release_scores(
            args, config, target, delays, results, forecasts,
            ReleaseForecasts.read(release),
        )  # fmt: skip
        write_csv(forecasts, output / "forecasts.csv.gz")
    recovery = _stack(parts, "recovery-forecasts.csv.gz")
    if recovery is not None:
        recovery = bt.joint_order(recovery, _order(recovery))
        rscores, renergy = bt.score_recovery(recovery)
        for table, path in (
            (recovery, output / "recovery-forecasts.csv.gz"),
            (rscores, results / "recovery-scores.csv"),
            (renergy, results / "recovery-energy.csv"),
        ):
            write_csv(bt.merge_by_model(path, table), path)
    scenario = _stack(parts, "scenario-forecasts.csv.gz")
    if scenario is not None:
        zone = _stack(parts, "scenario-zone-forecasts.csv.gz")
        assert zone is not None
        present = set(scenario["scenario"])
        names = [n for n in SCENARIO_RT_SCALE if f"calibrated_{n}" in present]
        sscores, senergy, struth = bt.score_scenarios(
            scenario,
            zone,
            _order(scenario),
            config,
            target,
            delays,
            load_health_zones(args.data),
            bt.real_vintages(target)[-1],
            sorted(set(scenario["vintage"].astype(str))),
            names,
        )
        # In the order one run makes them: by scenario, then task, then
        # model.
        models = _order(scenario)
        scenario = pd.concat(
            [
                bt.joint_order(
                    scenario.loc[scenario["scenario"] == f"calibrated_{n}"],
                    models,
                )
                for n in names
            ],
            ignore_index=True,
        )
        for table, path in (
            (scenario, output / "scenario-forecasts.csv.gz"),
            (sscores, results / "scenario-scores.csv.gz"),
            (senergy, results / "scenario-energy.csv"),
        ):
            write_csv(bt.merge_by_model(path, table), path)
        write_csv(struth, results / "scenario-truth.csv")
    zoned = [p for p in parts if (p / "zone-forecast-runs.csv.gz").exists()]
    if zoned:
        run = ZoneForecasts.read(zoned)
        scores, energy, kept = score_zones(
            target, delays, daily, load_daily_zones(args.data), run
        )
        _write_zone_results(
            scores,
            energy,
            kept,
            results / "zone-scores.csv",
            results / "zone-energy.csv",
            results / "zone-quantiles.csv",
            results / "zone-draws.csv.gz",
        )
    print(f"Scored the forecasts of {len(parts)} parts into {results}.")
    return 0


def _recovery(
    args: argparse.Namespace,
    config: bt.BacktestConfig,
    models: Sequence[ForecastModel],
    results: Path,
    output: Path,
) -> pd.DataFrame:
    """Run the recovery check, write its files and return its scores."""
    rforecasts, rscores, renergy = bt.run_recovery(
        models, config, lags=cutoff_lags(load_daily(args.data))
    )
    for table, path in (
        (rforecasts, output / "recovery-forecasts.csv.gz"),
        (rscores, results / "recovery-scores.csv"),
        (renergy, results / "recovery-energy.csv"),
    ):
        write_csv(bt.merge_by_model(path, table), path)
    return rscores


def _release_backtest(
    args: argparse.Namespace,
    config: bt.BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    results: Path,
    forecasts: pd.DataFrame,
) -> None:
    """Score every model from every real release.

    The fitted models' final-count forecasts are taken from `forecasts`.
    """
    daily = load_daily(args.data)
    tasks = list(store.origin_tasks(target, delays, config, daily).values())
    stored, learnt = _stored(args, target, forecasts)
    out = release_backtest(
        tasks,
        daily,
        target,
        stored=stored,
        delays=learnt,
        compare=[s.name for s in list_models()],
    )
    _write_release(out, results)


def _release_scores(
    args: argparse.Namespace,
    config: bt.BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    results: Path,
    forecasts: pd.DataFrame,
    run: ReleaseForecasts,
) -> None:
    """Score stacked release forecasts with the stored models'."""
    daily = load_daily(args.data)
    tasks = list(store.origin_tasks(target, delays, config, daily).values())
    stored, learnt = _stored(args, target, forecasts)
    out = score_releases(
        tasks,
        daily,
        target,
        run,
        stored=stored,
        delays=learnt,
        compare=[s.name for s in list_models()],
    )
    _write_release(out, results)


def _write_release(out: ReleaseBacktest, results: Path) -> None:
    """Write the release backtest's tables to `results`."""
    for name, table in (
        ("scores", out.scores),
        ("quantiles", out.quantiles),
        ("energy", out.energy),
        ("origins", out.origins),
        ("fits", out.fits),
    ):
        write_csv(table, results / f"nowcast-{name}.csv")
    print(f"Release backtest: {len(out.scores)} scored forecasts.")


def _stored(
    args: argparse.Namespace, target: pd.DataFrame, forecasts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final-count samples and delays of the models not run here.

    The fitted models' samples come from the backtest forecasts and their
    delays from the delay summaries `fit-store load` writes to the
    results, none if it loaded no fits. Hub teams' forecasts are matched
    to the real vintage serving their round, and have no delay.
    """
    from bdbvhubrenewal.evaluation import hubcompare as hc

    cols = [
        "model",
        "vintage",
        "location",
        "target_end_date",
        "sample",
        "value",
    ]
    fitted = [s.name for s in list_models() if s.fitted]
    ours = forecasts.loc[
        forecasts["model"].isin(fitted)
        & (forecasts["kind"] == "real")
        & forecasts["as_of"].isna()
    ]
    data = Path(args.data)
    hub = hc.load_hub_submissions(
        data / "hub-submissions", data / "raw" / "hub-submissions"
    )
    vintages = sorted(set(target.loc[target["kind"] == "real", "vintage"]))
    hub["vintage"] = [
        hc.match_vintage(r, vintages) for r in hub["reference_date"]
    ]
    samples = pd.concat([ours[cols], hub.dropna(subset=["vintage"])[cols]])
    return samples, stored_delays(args.results)


def _read_or_empty(path: Path) -> pd.DataFrame:
    """Read a results file, or an empty table if it has no columns."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _concat_results(args: argparse.Namespace) -> int:
    """Stack each results file across the parts, in the order given.

    A file in one part only is copied, and empty parts (a block with no
    forecast) are skipped. The parts' rows are kept as they are: unlike
    `merge_by_model`, rows of the same model in two parts are both kept,
    as each block has every model.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = [Path(p) for p in args.parts]
    names = sorted({f.name for p in parts for f in p.glob("*.csv*")})
    for name in names:
        tables = [
            t
            for p in parts
            if (p / name).exists() and not (t := _read_or_empty(p / name)).empty
        ]
        table = (
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        )
        if name in args.dedupe:
            table = table.drop_duplicates(ignore_index=True)
        write_csv(table, out / name)
        print(f"{out / name}: {len(tables)} parts.")
    return 0


def _write_fits(fits: list[pd.DataFrame], path: Path) -> None:
    """Merge the fit summaries of a run into a results file."""
    if fits:
        table = pd.concat(fits, ignore_index=True)
        write_csv(bt.merge_by_model(path, table), path)


def _zones(
    args: argparse.Namespace,
    task: ForecastTask,
    samples: pd.DataFrame,
    hub_config: Path,
) -> tuple[pd.DataFrame | None, str]:
    """Forecast the zones to sum to the provinces, or say why not.

    The zones come from `zones_for_model`.
    """
    try:
        daily = load_daily_zones(args.data)
    except OSError:
        return None, "no zone data, so the file has no health zones"
    counts = weekly_zone_counts(daily, task.vintage)
    if counts.empty:
        return None, f"no zone data for {task.vintage}, so no health zones"
    name = get_config(args.config).model if args.config else args.model
    zones = zones_for_model(
        name, task, samples, counts, hub_zones(hub_config), daily
    )
    return zones, ""


def _nowcast(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.forecasters.renewal import sampler_settings

    # A renewal model given by --model is fitted here.
    host_devices(int(sampler_settings()["num_chains"]))
    hub_config = Path(args.data) / "hub-config" / "tasks.json"
    try:
        task, handled = _task(args)
        if args.config:
            config = get_config(args.config)
            result = run_forecast(config, task, args.fit)
        else:
            samples = get_model(args.model).forecast(task)
            check_samples(samples, task)
            result = Forecast(samples)
        zones, zone_note = _zones(args, task, result.samples, hub_config)
        if zone_note:
            handled.append(zone_note)
        table = to_hub_table(
            result.samples,
            task.reference_date,
            unassigned=result.unassigned,
            zones=zones,
            forecast_date=args.forecast_date,
        )
        path = write_submission(table, args.out, model_id(_abbr(args)))
    except (KeyError, ValueError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    _write_national_sidecar(Path(args.out), task, result)
    # Problems handled on the way, for the PR's `Handled:` line.
    (Path(args.out) / "handled.json").write_text(json.dumps(handled) + "\n")
    # Whether the file has health zones, for the CI's zone check.
    n_zones = 0 if zones is None else int(zones["location"].nunique())
    zone_record = {"zones": n_zones > 0, "n_zones": n_zones, "note": zone_note}
    (Path(args.out) / "zones.json").write_text(json.dumps(zone_record) + "\n")
    for note in handled:
        print(f"Handled: {note}")
    print(f"Wrote {path} ({len(table)} rows, locations incl. {NATIONAL}).")
    hub = Hub.load(args.data)
    rnd = hub.find_round(task.reference_date.isoformat())
    if rnd is None or rnd.name != ROUND_NAME:
        print(
            f"Note: reference date {task.reference_date} is not an open round "
            "in tasks.json, so the hub's checks were skipped."
        )
        return 0
    report = validate_submission(path, hub, skip_window=True)
    # `metadata` writes and checks the model metadata.
    return _print_failed(
        [c for c in report.failed if c.check != "metadata_exists"]
    )


def _print_failed(failed: list[Check]) -> int:
    """Print failed hub checks as `ERROR:` lines, and return 1 if any."""
    for c in failed:
        print(f"ERROR: {c.check}: {c.message}", file=sys.stderr)
    return 1 if failed else 0


def _zone_origins(args: argparse.Namespace) -> list[str] | None:
    """Return the origins of `--chunk`, or None for all."""
    if not args.chunk:
        return None
    real = load_target(args.data).query("kind == 'real'")
    return chunk(bt.real_vintages(real)[:-1], args.chunk)


def _zone_backtest(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.evaluation.zones import zone_backtest

    hub_config = Path(args.data) / "hub-config" / "tasks.json"
    if args.forecast_only:
        from bdbvhubrenewal.evaluation.zones import zone_forecasts

        zone_forecasts(
            load_target(args.data),
            load_delays(args.data),
            load_daily(args.data),
            load_daily_zones(args.data),
            hub_zones(hub_config),
            models=args.models,
            origins=_zone_origins(args),
        ).write(args.parts)
        print(f"Zone forecasts of {args.models} written to {args.parts}.")
        return 0
    scores, energy, forecasts = zone_backtest(
        load_target(args.data),
        load_delays(args.data),
        load_daily(args.data),
        load_daily_zones(args.data),
        hub_zones(hub_config),
        models=args.models,
        origins=_zone_origins(args),
    )
    _write_zone_results(
        scores,
        energy,
        forecasts,
        Path(args.out),
        Path(args.energy_out),
        Path(args.quantiles_out),
        Path(args.draws_out),
    )
    return 0


def _write_zone_results(
    scores: pd.DataFrame,
    energy: pd.DataFrame,
    forecasts: pd.DataFrame,
    out: Path,
    energy_out: Path,
    quantiles_out: Path,
    draws_out: Path,
) -> None:
    """Write the zone backtest's tables."""
    from bdbvhubrenewal.evaluation.hubcompare import forecast_quantiles
    from bdbvhubrenewal.evaluation.zones import zone_relative_skill
    from bdbvhubrenewal.reporting.explorer import draw_subset

    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(scores, out)
    write_csv(energy, energy_out)
    if scores.empty:
        # A block of origins every model skips: empty files, which
        # `concat-results` skips when the blocks are stacked.
        for path in (quantiles_out, draws_out):
            write_csv(pd.DataFrame(), path)
        print(f"Wrote {out} (no origin forecast).")
        return
    # The zone forecasts as submitted, for the Backtest explorer: their
    # plotted quantiles and a fixed subset of draws for the trajectories.
    write_csv(
        forecast_quantiles(forecasts.assign(source="ours"), by=("origin",)),
        quantiles_out,
    )
    write_csv(draw_subset(forecasts).drop(columns="draw"), draws_out)
    rel = zone_relative_skill(scores, by=("horizon",))
    print(rel.round(3).to_string(index=False))
    print(f"Wrote {out} ({len(scores)} scores).")


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
        "modelled_provinces": list(
            task.locations if result.modelled is None else result.modelled
        ),
        "unassigned": unassigned,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _plot_submission(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.reporting.plots import plot_submission

    target = load_target(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = _vintage([str(v) for v in real["vintage"].unique()], args.vintage)
    fig = plot_submission(args.submission, vintage, target=target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The light theme is transparent, which is unreadable in dark mode PRs.
    fig.savefig(out, dpi=100, facecolor="white", transparent=False)
    print(f"Wrote {out}.")
    return 0


def _plot_ppc(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.evaluation.ppc import plot_ppc, read_ppc

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
        _abbr(args),
        model_name=text["model_name"],
        methods=text["methods"],
        methods_long=text["methods_long"],
        data_inputs=text["data_inputs"],
    )
    path = write_model_metadata(record, args.out)
    print(f"Wrote {path}.")
    return _print_failed(validate_metadata(path, Hub.load(args.data)).failed)


def _bvd(args: argparse.Namespace) -> int:  # pragma: no cover - network
    from bdbvhubrenewal.data import bvd

    config = bt.load_config(args.backtest_config)
    vintages = args.vintages
    if not vintages:
        target = load_target(args.data)
        real = target.loc[target["kind"] == "real", "vintage"].astype(str)
        vintages = sorted(set(real))
    token = os.environ.get("GITHUB_TOKEN")
    chosen = bvd.refresh_cache(
        Path(args.data) / "bvd",
        vintages,
        bvd.list_releases(token),
        max_rhat=config.bvd_max_rhat,
    )
    for vintage, choice in chosen.items():
        if choice is None:
            print(f"{vintage}: no release")
            continue
        print(
            f"{vintage}: {choice.tag} (joint R-hat "
            f"{choice.joint_max_rhat:.3f}, limit {choice.max_rhat:g})"
        )
        for tag, reason in choice.skipped:
            print(f"  skipped {tag}: {reason}")
    return 0


def _compare_hub(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.evaluation import hubcompare as hc
    from bdbvhubrenewal.reporting.explorer import TRAJECTORY_DRAWS

    cfg = hc.load_comparison_config(args.config)
    hub_dir, cache = Path(args.hub_dir), Path(args.cache_dir)
    if args.fetch:  # pragma: no cover - network
        hc.fetch_hub_submissions(hub_dir, cache, ref=cfg.ref)
    hub = hc.load_hub_submissions(hub_dir, cache)
    target, delays = load_target(args.data), load_delays(args.data)
    daily = load_daily(args.data)
    config = bt.load_config(args.backtest_config)
    rounds = sorted(set(hub["reference_date"]))
    live = args.models or [s.name for s in list_models() if not s.fitted]
    horizons = (
        pd.to_datetime(hub["target_end_date"])
        - pd.to_datetime(hub["reference_date"])
    ).dt.days // 7
    codes = hub_zones(Path(args.data) / "hub-config" / "tasks.json")
    ours = hc.our_forecasts(
        rounds,
        target,
        delays,
        config,
        live,
        daily,
        load_daily_zones(args.data),
        codes,
        min_horizon=min(int(horizons.min()), config.min_horizon)
        if len(hub)
        else None,
    )
    stored = hc.stored_forecasts(
        cfg.stored if args.stored is None else args.stored,
        rounds,
        exclude=live,
    )
    hub_ids = hc.own_hub_ids()
    forecasts = hc.with_submissions(
        pd.concat(
            [f for f in (hub, ours, stored) if not f.empty], ignore_index=True
        ),
        hc.load_hub_submissions(cfg.submitted),
        hub_ids,
    )
    zones = hc.zone_truth(load_health_zones(args.data), codes, target)
    scores, energy, pending = hc.score_comparison(
        forecasts, target, delays, config, daily, zones
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(scores, out / "scores.csv")
    write_csv(energy, out / "energy.csv")
    write_csv(pending, out / "pending.csv")
    write_csv(hc.forecast_quantiles(forecasts), out / "quantiles.csv.gz")
    teams = hc.load_team_metadata(hub_dir)
    write_csv(
        teams.loc[teams["model"].isin(set(hub["model"]))], out / "teams.csv"
    )
    write_csv(
        hc.trajectory_samples(forecasts, teams, TRAJECTORY_DRAWS),
        out / "samples.csv.gz",
    )
    manifest = hub_dir / "manifest.json"
    accessed = dt.date.today().isoformat()
    if manifest.exists():
        accessed = str(json.loads(manifest.read_text())["accessed"])
    real = target.loc[target["kind"] == "real"]
    summary = {
        "truth_vintage": str(real["vintage"].max()),
        "complete_after_weeks": config.complete_after_weeks,
        "baseline": hc.resolve_baseline(cfg.baseline),
        "preferred_baseline": cfg.baseline,
        "locations": list(cfg.locations),
        "hub_ids": hub_ids,
        "accessed": accessed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Hub comparison: {len(scores)} scored forecasts, "
        f"{len(pending)} rounds pending. Written to {out}."
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

    Examples
    --------
    From the shell, download the data, forecast with the random walk
    nowcast and write its hub file under `output/submission/`:

        bdbvhubrenewal data
        bdbvhubrenewal nowcast --model randomwalknowcast --vintage latest
        bdbvhubrenewal <command> --help

    The same from Python:

    >>> from bdbvhubrenewal.cli import main
    >>> main(["nowcast", "--model", "randomwalknowcast", "--vintage", "latest"])
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # JAX logs its backend search at INFO, which is noise here.
    logging.getLogger("jax").setLevel(logging.WARNING)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["validate"]:
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
    if args.command == "fit-store":
        return store.command(args)
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
    if args.command == "zone-backtest":
        return _zone_backtest(args)
    if args.command == "digest":
        from bdbvhubrenewal.evaluation.digest import (
            backtest_digest,
            model_digest,
        )

        if args.model:
            print(model_digest(args.model, args.root, args.extra, args.text))
        else:
            print(backtest_digest(args.root, args.extra, args.text))
        return 0
    if args.command == "concat-results":
        return _concat_results(args)
    if args.command == "score":
        return _score(args)
    if args.command == "metadata":
        return _metadata(args)
    if args.command == "bvd":  # pragma: no cover - network
        return _bvd(args)
    if args.command == "hub-id":
        print(model_id(model_abbr_for(args.model, args.hub_toml)))
        return 0
    if args.command == "compare-hub":
        return _compare_hub(args)
    if args.command == "freeze-round":
        from bdbvhubrenewal.submission.rounds import freeze_round

        wrote = freeze_round(args.ledger, args.round)
        print(
            f"Round {args.round}: "
            + ("page frozen." if wrote else "already frozen.")
        )
        return 0
    from bdbvhubrenewal.reporting.report import make_report

    make_report(figures=args.figures, generated=args.generated)
    from bdbvhubrenewal.submission.rounds import make_round_pages

    done = make_round_pages(args.ledger, generated=args.generated)
    print(f"Round pages: {len(done)} submitted rounds.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
