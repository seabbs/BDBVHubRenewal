# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Command line interface: `bdbvhubrenewal <command>`.

Commands:

- `data`: download the hub data and write the tidy tables.
- `backtest`: run the development backtest and the recovery check, with
  `--evaluate-holdout` the rolling held-out evaluation, or with `--lopo`
  the leave-one-province-out check.
- `nowcast`: forecast from a real vintage, write the hub file and check it.
  With `--config` (and `--fit`), forecast from a fit configuration.
  The province samples are split across the health zones unless
  `--no-zones` is given.
- `zone-backtest`: score zone forecasts against zone persistence.
- `fit`: fit a configuration and write the fit directory and its
  `diagnostics.json`, for the submission CI.
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
    load_daily_zones,
    load_delays,
    load_health_zones,
    load_target,
)
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.heldout import (
    HeldOutError,
    load_model_choices,
    run_heldout,
)
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastTask,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.zones import ZoneAllocator
from bdbvhubrenewal.submission.fitting import (
    Forecast,
    get_config,
    run_fit,
    run_forecast,
)
from bdbvhubrenewal.submission.hub import (
    SUBMISSION_CONFIG,
    load_hub_round,
    model_abbr_for,
    model_id,
    render_model_metadata,
    submission_model_abbr,
    to_hub_table,
    validate_model_metadata,
    validate_submission_file,
    write_model_metadata,
    write_submission,
)

__all__ = ["build_parser", "host_devices", "main"]

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
            "https://samabbott.co.uk/BDBVHubRenewal/methods/baselines/."
        ),
        "data_inputs": (
            "Hub target data (weekly confirmed cases by onset week, observed "
            "plus imputed) and the hub's published onset-to-report delays."
        ),
    },
    # Draft for the PyRenew model. Not submitted until Sam has reviewed
    # the held-out results.
    "renewal": {
        "model_name": "Province renewal model (PyRenew)",
        "methods": (
            "Province renewal model in PyRenew with pooled Rt, gravity "
            "importation and depletion, fitted to known and imputed onsets "
            "with modelled delays; national is the sum of provinces."
        ),
        "methods_long": (
            "A renewal process for each province with cases, with a national "
            "weekly random walk on log Rt plus partially pooled province "
            "deviations, importation between provinces through a gravity "
            "kernel, and susceptible depletion. Provinces with no cases get "
            "importations that can seed local spread. Confirmed onsets follow "
            "a time-varying ascertainment. Known onsets are right truncated "
            "by a time-varying onset-to-report delay, and the hub's imputed "
            "onsets are modelled as the hub makes them, from cases missing "
            "an onset date by report day, back-sampled with the hub's delay. "
            "Both streams are negative binomial. Fitted by NUTS (NumPyro) to "
            "each data vintage. Each sample is one joint posterior "
            "trajectory and the national value is the sum of province "
            "samples. See https://samabbott.co.uk/BDBVHubRenewal/methods/model/."
        ),
        "data_inputs": (
            "Hub target data (weekly confirmed cases by onset week, observed "
            "plus imputed), the hub's published onset-to-report delays, "
            "province populations and population-weighted province centres."
        ),
    },
    # A hub model of its own (seabbs-baseline), submitted every
    # round. The nowcast step is being built; check this text against it.
    "persistencenow": {
        "model_name": "Persistence on a nowcast",
        "methods": (
            "Chain-ladder nowcast of recent onset weeks with negative binomial "
            "uncertainty, then the hub baseline on each draw. Provinces, "
            "health zones and national."
        ),
        "methods_long": (
            "A baseline model building on the ideas of the baselinenowcast "
            "paper and package (Johnson et al. 2025, Wellcome Open Research, "
            "doi:10.12688/wellcomeopenres.25027.2; "
            "https://github.com/epinowcast/baselinenowcast/). Recent onset "
            "weeks are nowcast with a chain ladder on the reporting triangle, "
            "with a negative binomial for each weekly horizon fitted to past "
            "retrospective nowcasts. Each nowcast draw is carried forward "
            "from its own latest week as the COVID-19 Forecast Hub baseline "
            "on the log scale, with steps drawn from past weekly changes and "
            "their negatives. "
            "Sparse provinces borrow the delay and steps of the provinces "
            "with more data. Province samples "
            "are split across health zones, and the national value is the "
            "sum of provinces. See "
            "https://samabbott.co.uk/BDBVHubRenewal/methods/baselines/."
        ),
        "data_inputs": (
            "Hub target data (weekly confirmed cases by onset week, observed "
            "plus imputed), the hub's published onset-to-report delays and "
            "daily health zone counts."
        ),
    },
}
"""Metadata text for each model that may be submitted."""


_ABBR_HELP = (
    "Hub model abbreviation. Defaults to `persistence` for persistence and "
    "otherwise to the submission abbreviation in config/hub.toml."
)


def _abbr(args: argparse.Namespace) -> str:
    if args.abbr:
        return str(args.abbr)
    if args.model == "persistence":
        return "persistence"
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
        help="Where the forecasts go. They are committed in results/, so "
        "the docs can show fitted variants without refitting them.",
    )
    back.add_argument("--no-reconstructed", action="store_true")
    back.add_argument(
        "--no-recovery",
        action="store_true",
        help="Skip the recovery check on simulated vintages.",
    )
    back.add_argument(
        "--recovery-only",
        action="store_true",
        help="Run only the recovery check on simulated vintages.",
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
        "--cache",
        default=None,
        help="Directory with one forecast file per model and task. Tasks "
        "with a file are read from it, so a run can be resumed or split.",
    )
    back.add_argument(
        "--shard",
        default=None,
        metavar="I/N",
        help="Forecast only shard I of N of the development tasks into "
        "--cache, without scoring. Run once more without it to score.",
    )
    back.add_argument(
        "--no-fit",
        action="store_true",
        help="Never run fitted models. Their rows in results/ are kept and "
        "the held-out forecasts are scored again. The docs build uses this.",
    )
    back.add_argument(
        "--lopo",
        action="store_true",
        help="Run the leave-one-province-out check instead.",
    )
    back.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Run the rolling held-out evaluation instead.",
    )

    now = sub.add_parser("nowcast", help="Forecast and write a hub file.")
    now.add_argument(
        "--model",
        default="persistence",
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
    now.add_argument("--hub-config", default=None)
    now.add_argument(
        "--no-zones",
        dest="zones",
        action="store_false",
        help="Leave the health zones out of the hub file.",
    )

    zb = sub.add_parser(
        "zone-backtest", help="Score zone forecasts against zone persistence."
    )
    zb.add_argument("--data", default=data_dir)
    zb.add_argument(
        "--models", nargs="*", default=["persistence", "persistencenow"]
    )
    zb.add_argument("--hub-config", default=None)
    zb.add_argument("--out", default="results/zone-scores.csv")

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
    _abbr_args(meta)
    meta.add_argument("--out", default="hub/model-metadata")
    meta.add_argument(
        "--schema", default=f"{data_dir}/hub-config/model-metadata-schema.json"
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
        help="Defaults to every real and reconstructed backtest vintage.",
    )

    hid = sub.add_parser(
        "hub-id", help="Print the hub model id that submissions use."
    )
    hid.add_argument("--hub-toml", default=str(SUBMISSION_CONFIG))
    hid.add_argument(
        "--model",
        default=None,
        help="A model with its own hub id (`[models]` in config/hub.toml).",
    )
    cho = sub.add_parser(
        "chosen-model",
        help="Print the model chosen in the model choice record.",
    )
    cho.add_argument("--backtest-config", default="config/backtest.toml")

    cmp = sub.add_parser(
        "compare-hub", help="Compare our forecasts with hub submissions."
    )
    cmp.add_argument("--config", default="config/hub-comparison.toml")
    cmp.add_argument("--backtest-config", default="config/backtest.toml")
    cmp.add_argument("--data", default=data_dir)
    cmp.add_argument(
        "--hub-dir",
        default="data/hub-submissions",
        help="Committed hub files: metadata and openly licensed forecasts.",
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
        bt.load_config(args.backtest_config), min_horizon=args.min_horizon
    )
    target, delays = load_target(args.data), load_delays(args.data)
    real = target.loc[target["kind"] == "real"]
    vintage = _vintage([str(v) for v in real["vintage"].unique()], args.vintage)
    rows = real.loc[real["vintage"] == vintage]
    delays, note = delays_for_vintage(delays, vintage)
    task = bt.make_tasks(rows, delays, config, history=real)[0]
    return task, [note] if note else []


def host_devices(n: int) -> None:
    """Give XLA at least `n` host devices, so `n` chains run in parallel.

    NumPyro runs chains in parallel only with as many devices as chains.
    This must run before JAX starts, so the CLI calls it before any fit.
    Other flags in `XLA_FLAGS` are kept, and a larger count is kept.

    Parameters
    ----------
    n : int
        The number of chains.
    """
    key = "--xla_force_host_platform_device_count="
    flags = os.environ.get("XLA_FLAGS", "").split()
    have = [int(f[len(key) :]) for f in flags if f.startswith(key)]
    if have and max(have) >= n:
        return
    rest = [f for f in flags if not f.startswith(key)]
    os.environ["XLA_FLAGS"] = " ".join([*rest, f"{key}{n}"])


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
    from bdbvhubrenewal.forecasters.renewal import sampler_settings

    config = bt.load_config(args.config)
    bt.apply_sampler(config)
    host_devices(int(sampler_settings()["num_chains"]))
    target, delays = load_target(args.data), load_delays(args.data)
    names = _models(args.models)
    if args.no_fit:
        fitted = {s.name for s in list_models() if s.fitted}
        names = [n for n in names if n not in fitted]
    results, output = Path(args.results), Path(args.output)
    results.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_holdout:
        # The held-out run uses the models in the model choice record.
        record = Path(config.model_choice)
        if record.exists() and not load_model_choices(record):
            print("No model choice recorded, so no vintage is held out.")
            return 0
        try:
            scores = run_heldout(
                target,
                delays,
                config,
                evaluate_holdout=True,
                out_dir=results,
                models=names,
                fit=not args.no_fit,
            )
        except HeldOutError as err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print(f"Held-out scores: {len(scores)} rows in {results}.")
        return 0
    models = [get_model(n) for n in names]
    if args.shard:
        if not args.cache:
            print("ERROR: --shard needs --cache.", file=sys.stderr)
            return 1
        index, count = (int(x) for x in args.shard.split("/"))
        if args.recovery_only:
            tasks = bt.recovery_tasks(config)
        elif args.lopo:
            models = [m for m in models if getattr(m, "supports_mask", False)]
            tasks = bt.lopo_tasks(
                target,
                delays,
                config,
                include_reconstructed=not args.no_reconstructed,
            )
        else:
            tasks = bt.development_tasks(
                target,
                delays,
                config,
                include_reconstructed=not args.no_reconstructed,
            )
        mine = bt.shard_tasks(bt.order_tasks(tasks), index, count)
        bt.run_forecasts(models, mine, args.cache)
        print(f"Shard {args.shard}: {len(mine)} tasks in {args.cache}.")
        return 0
    fits: list[pd.DataFrame] = []
    if args.scenarios_only:
        as_of = args.scenario_as_of or bt.real_vintages(target)[-1]
        dates = args.scenario_dates or [
            (as_date(as_of) - dt.timedelta(weeks=k)).isoformat()
            for k in (3, 2, 1, 0)
        ]
        sforecasts, sscores = bt.run_scenarios(
            models,
            config,
            target,
            delays,
            load_health_zones(args.data),
            as_of,
            dates,
        )
        for table, path in (
            (sforecasts, output / "scenario-forecasts.csv.gz"),
            (sscores, results / "scenario-scores.csv.gz"),
        ):
            bt.merge_by_model(path, table).to_csv(path, index=False)
        print(f"Scenarios: {len(sscores)} scored forecasts in {results}.")
        return 0
    if args.recovery_only:
        rforecasts, rscores = bt.run_recovery(
            models, config, cache=args.cache, fits=fits
        )
        for table, path in (
            (rforecasts, output / "recovery-forecasts.csv.gz"),
            (rscores, results / "recovery-scores.csv"),
        ):
            bt.merge_by_model(path, table).to_csv(path, index=False)
        _write_fits(fits, results / "recovery-fits.csv")
        print(f"Recovery: {len(rscores)} scored forecasts in {results}.")
        return 0
    if args.lopo:
        lopo_fc, lopo_scores = bt.run_lopo(
            models,
            target,
            delays,
            config,
            include_reconstructed=not args.no_reconstructed,
            cache=args.cache,
            fits=fits,
        )
        lopo_fc.to_csv(output / "lopo-forecasts.csv.gz", index=False)
        lopo_scores.to_csv(results / "lopo-scores.csv", index=False)
        _write_fits(fits, results / "lopo-fits.csv")
        print(f"Leave one province out: {len(lopo_scores)} scored forecasts.")
        return 0
    forecasts, scores = bt.run_development(
        models,
        target,
        delays,
        config,
        include_reconstructed=not args.no_reconstructed,
        cache=args.cache,
        fits=fits,
    )
    _write_fits(fits, results / "backtest-fits.csv")
    by_truth = bt.score_by_truth_vintage(
        forecasts, target, config.complete_after_weeks
    )
    # Rows of models not run here are kept, so the fitted variants can be
    # run apart from the others.
    tables = [
        (forecasts, output / "forecasts.csv.gz"),
        (scores, results / "backtest-scores.csv"),
        (by_truth, results / "backtest-scores-by-truth.csv"),
    ]
    recovery = ""
    if not args.no_recovery:
        rforecasts, rscores = bt.run_recovery(models, config)
        tables += [
            (rforecasts, output / "recovery-forecasts.csv.gz"),
            (rscores, results / "recovery-scores.csv"),
        ]
        recovery = f"; recovery: {len(rscores)}"
    for table, path in tables:
        bt.merge_by_model(path, table).to_csv(path, index=False)
    print(
        f"Development: {len(scores)} scored forecasts{recovery}. "
        f"Written to {results}."
    )
    return 0


def _write_fits(fits: list[pd.DataFrame], path: Path) -> None:
    """Merge the fit summaries of a run into a results file."""
    if fits:
        table = pd.concat(fits, ignore_index=True)
        bt.merge_by_model(path, table).to_csv(path, index=False)


def _zones(
    args: argparse.Namespace,
    task: ForecastTask,
    samples: pd.DataFrame,
    hub_config: str,
) -> tuple[pd.DataFrame | None, str]:
    """Split province samples across zones, or say why not."""
    if not args.zones:
        return None, ""
    try:
        daily = load_daily_zones(args.data)
    except OSError:
        return None, "no zone data, so the file has no health zones"
    counts = weekly_zone_counts(daily, task.vintage)
    if counts.empty:
        return None, f"no zone data for {task.vintage}, so no health zones"
    zones = hub_zones(hub_config)
    return ZoneAllocator().allocate(samples, task, counts, zones), ""


def _nowcast(args: argparse.Namespace) -> int:
    hub_config = args.hub_config or str(
        Path(args.data) / "hub-config" / "tasks.json"
    )
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
            nat = samples["location"] == NATIONAL
            national = None
            if nat.any():
                national = samples.loc[
                    nat, ["target_end_date", "sample", "value"]
                ]
            result = Forecast(samples.loc[~nat], national=national)
        zones, zone_note = _zones(args, task, result.samples, hub_config)
        if zone_note:
            handled.append(zone_note)
        table = to_hub_table(
            result.samples,
            task.reference_date,
            unassigned=result.unassigned,
            national=result.national,
            zones=zones,
        )
    except (KeyError, ValueError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    mid = model_id(_abbr(args))
    path = write_submission(table, args.out, mid)
    _write_national_sidecar(Path(args.out), task, result)
    # Problems handled on the way, for the PR's `Handled:` line.
    (Path(args.out) / "handled.json").write_text(json.dumps(handled) + "\n")
    # Whether the file has health zones, for the CI's zone check.
    n_zones = 0 if zones is None else int(zones["location"].nunique())
    zone_record = {"zones": n_zones > 0, "n_zones": n_zones, "note": zone_note}
    (Path(args.out) / "zones.json").write_text(json.dumps(zone_record) + "\n")
    for note in handled:
        print(f"Handled: {note}")
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


def _zone_backtest(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.evaluation.zones import (
        zone_backtest,
        zone_relative_skill,
    )

    hub_config = args.hub_config or str(
        Path(args.data) / "hub-config" / "tasks.json"
    )
    scores = zone_backtest(
        load_target(args.data),
        load_delays(args.data),
        load_daily_zones(args.data),
        hub_zones(hub_config),
        models=args.models,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    rel = zone_relative_skill(scores, by=("horizon",))
    print(rel.round(3).to_string(index=False))
    print(f"Wrote {out} ({len(scores)} scores).")
    return 0


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
        # `fitted`: national is the model's own fit, not the sum of the
        # provinces, so the national check reports the difference only.
        "national_source": "sum" if result.national is None else "fitted",
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
    fig.savefig(out, dpi=100)
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
    if args.model not in MODEL_METADATA:
        print(f"ERROR: no metadata text for {args.model!r}.", file=sys.stderr)
        return 1
    text = MODEL_METADATA[args.model]
    record = render_model_metadata(
        _abbr(args),
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


def _bvd(args: argparse.Namespace) -> int:  # pragma: no cover - network
    from bdbvhubrenewal.data import bvd

    config = bt.load_config(args.backtest_config)
    vintages = args.vintages
    if not vintages:
        target = load_target(args.data)
        real = target.loc[target["kind"] == "real", "vintage"].astype(str)
        vintages = sorted({*real, *config.reconstructed_dates})
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


def _chosen_model(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.evaluation.heldout import load_model_choices

    choices = load_model_choices(
        bt.load_config(args.backtest_config).model_choice
    )
    if not choices:
        print("ERROR: no model has been chosen.", file=sys.stderr)
        return 1
    print(choices[-1].model)
    return 0


def _compare_hub(args: argparse.Namespace) -> int:
    from bdbvhubrenewal.evaluation import hubcompare as hc

    cfg = hc.load_comparison_config(args.config)
    hub_dir, cache = Path(args.hub_dir), Path(args.cache_dir)
    if args.fetch:  # pragma: no cover - network
        hc.fetch_hub_submissions(hub_dir, cache, ref=cfg.ref)
    hub = hc.load_hub_submissions(hub_dir, cache)
    target, delays = load_target(args.data), load_delays(args.data)
    config = bt.load_config(args.backtest_config)
    rounds = sorted(set(hub["reference_date"]))
    live = args.models or [s.name for s in list_models() if not s.fitted]
    horizons = (
        pd.to_datetime(hub["target_end_date"])
        - pd.to_datetime(hub["reference_date"])
    ).dt.days // 7
    ours = hc.our_forecasts(
        rounds,
        target,
        delays,
        config,
        live,
        min_horizon=min(int(horizons.min()), config.min_horizon)
        if len(hub)
        else None,
    )
    stored = hc.stored_forecasts(
        cfg.stored if args.stored is None else args.stored,
        rounds,
        exclude=live,
    )
    forecasts = pd.concat(
        [f for f in (hub, ours, stored) if not f.empty], ignore_index=True
    )
    scores, pending = hc.score_comparison(
        forecasts, target, config.complete_after_weeks
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    pending.to_csv(out / "pending.csv", index=False)
    hc.forecast_quantiles(forecasts).to_csv(
        out / "quantiles.csv.gz", index=False
    )
    teams = hc.load_team_metadata(hub_dir)
    teams.loc[teams["model"].isin(set(hub["model"]))].to_csv(
        out / "teams.csv", index=False
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
        "hub_ids": hc.own_hub_ids(),
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
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # JAX logs its backend search at INFO, which is noise here.
    logging.getLogger("jax").setLevel(logging.WARNING)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["validate"]:
        from bdbvhubrenewal.submission.validate import main as validate_main

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
    if args.command == "zone-backtest":
        return _zone_backtest(args)
    if args.command == "metadata":
        return _metadata(args)
    if args.command == "bvd":  # pragma: no cover - network
        return _bvd(args)
    if args.command == "hub-id":
        abbr = (
            model_abbr_for(args.model, args.hub_toml)
            if args.model
            else submission_model_abbr(args.hub_toml)
        )
        print(model_id(abbr))
        return 0
    if args.command == "chosen-model":
        return _chosen_model(args)
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
