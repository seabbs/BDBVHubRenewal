"""Fit, score and export the aggregation arms of one fine-tuned checkpoint.

Steps of `run_finetuned.sh`, each its own process:

- `bdbv-fit`: each fitted arm (`FITTED_GROUPINGS`) of one model at each
  release-backtest origin, as `bdbv_fits.py`, one `ReleaseForecasts`
  directory per arm and origin, named `<arm>__<origin>`.
- `germany-fit`: each fitted arm of one model at each German development
  date, as `germany_samples.py`, one `<date>__<arm>.csv.gz` per date and
  arm.
- `bdbv-score`, `germany-score`: the arms' scores, as `bdbv_score.py`
  with `export.py`, and `germany_score2.py`, with the paired arm made from
  the bottom arm as `bdbv_paired.py` and `germany_score2.py` make it.
  randomwalknowcast is not rerun: `export` keeps the cells it has in the
  existing rows.
- `export`: the paper's rows (`groupings.merge_checkpoint_rows`).
- `sanity`: a zero-shot arm's rows against the existing ones.

Every fit is resumed from its file. Before each origin or date the
process waits for the box's compute budget and free memory, and after it
clears the JAX caches, as the checkpoint scorer does.
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import groupings as gr
from bdbvhubrenewal.forecasters.foundation import resolve_checkpoint

TRUTH_DELAY = 35
"""D of the zero-shot rows: the rolling truth's days after a week."""

BDBV_ORIGINS = ("2026-10-09", "2026-10-17", "2026-10-24", "2026-11-01")
"""The zero-shot rows' release-backtest origins (results/agg/README.md)."""

BUDGET = Path.home() / ".claude" / "hooks" / "compute-budget.sh"
FREE_GB = float(os.environ.get("AGG_FREE_GB", "16"))
"""Memory kept free for the box's other jobs, beside this job's cap."""
CAP_GB = float(os.environ.get("AGG_CAP_GB", "6"))


def log(text: str) -> None:
    """Print a timestamped line."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}", flush=True)


def available_gb() -> float:
    """Return the box's available memory in GB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def wait_for_room() -> None:
    """Wait until the budget is not red and the cap leaves FREE_GB free."""
    while True:
        verdict = "green"
        if BUDGET.exists():
            out = subprocess.run(
                [str(BUDGET)], capture_output=True, text=True, check=False
            ).stdout.split()
            verdict = out[-1] if out else "green"
        room = available_gb() - CAP_GB
        if verdict != "red" and room >= FREE_GB:
            return
        log(f"waiting: budget {verdict}, {room:.1f} GB free after the cap")
        time.sleep(120)


def clear() -> None:
    """Drop JAX's compile caches and collect, so a long run does not grow."""
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def peak_gb() -> float:
    """Return this process's peak resident memory in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def timing(out: Path, **row: object) -> None:
    """Append a timing row to `out`/timings.csv."""
    path = out / "timings.csv"
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def checked(checkpoint: str | None) -> None:
    """Fail unless a checkpoint resolves to itself, not to another."""
    if checkpoint is None:
        return
    got = resolve_checkpoint(checkpoint).hash
    if got != checkpoint:
        raise SystemExit(f"{checkpoint} resolved to {got}")
    env = os.environ.get("BDBV_CHRONOS_FT")
    if env != checkpoint:
        raise SystemExit(f"BDBV_CHRONOS_FT is {env!r}, not {checkpoint}")


def bdbv_inputs(data: str) -> tuple:
    """Return the target, config, origin tasks and D, checked."""
    from bdbvhubrenewal.data import load_daily, load_delays, load_target
    from bdbvhubrenewal.data.zones import load_zone_data
    from bdbvhubrenewal.evaluation import store
    from bdbvhubrenewal.evaluation.nowcast import resolve_truth_delay

    target = load_target(data)
    config = bt.load_config("config/backtest.toml")
    delay = resolve_truth_delay(config, target)
    if delay != TRUTH_DELAY:
        raise SystemExit(f"D is {delay}, not the zero-shot rows' {TRUTH_DELAY}")
    tasks = store.origin_tasks(
        target,
        load_delays(data),
        config,
        load_daily(data),
        load_zone_data(data),
    )
    missing = set(BDBV_ORIGINS) - set(tasks)
    if missing:
        raise SystemExit(f"no task for origins {sorted(missing)}")
    return target, config, tasks, delay


def bdbv_fit(args: argparse.Namespace) -> None:
    """Fit each arm of one model at each BDBV origin."""
    target, _, tasks, _ = bdbv_inputs(args.data)
    checked(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for origin in args.origins:
        for grouping in args.groupings:
            arm = gr.arm_name(args.model, grouping, args.checkpoint)
            path = out / f"{arm}__{origin}"
            if (path / "release-samples.csv.gz").exists():
                continue
            wait_for_room()
            t0 = time.time()
            run = gr.release_grouping_forecasts(
                [tasks[origin]], target, args.model, grouping, args.checkpoint
            )
            secs = time.time() - t0
            if (run.runs["status"] != "forecast").any():
                raise SystemExit(f"{arm} at {origin}: {run.runs.to_dict()}")
            run.write(path)
            timing(out, arm=arm, unit=origin, seconds=round(secs, 1),
                   peak_gb=round(peak_gb(), 2))  # fmt: skip
            log(f"bdbv {arm} {origin} {secs:.0f}s peak {peak_gb():.2f} GB")
        clear()


def germany_fit(args: argparse.Namespace) -> None:
    """Fit each arm of one model at each German development date."""
    from bdbvhubrenewal.data.germany import hub_triangle
    from bdbvhubrenewal.evaluation import germany as de

    config = de.load_germany_config("config/germany.toml")
    dates = de.nowcast_dates(config)
    assert max(dates) < config.split_date
    if args.dates:
        dates = [d for d in dates if d in set(args.dates)]
    checked(args.checkpoint)
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for date in dates:
        for grouping in args.groupings:
            arm = gr.arm_name(args.model, grouping, args.checkpoint)
            path = out / f"{date}__{arm}.csv.gz"
            if path.exists():
                continue
            wait_for_room()
            t0 = time.time()
            samples = gr.germany_grouping_samples(
                args.model, grouping, triangle, date, config.n_samples,
                config.seed, args.checkpoint,
            )  # fmt: skip
            secs = time.time() - t0
            tmp = path.with_suffix(".tmp.gz")
            samples.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(path)
            timing(out, arm=arm, unit=date, seconds=round(secs, 1),
                   peak_gb=round(peak_gb(), 2))  # fmt: skip
            log(f"germany {arm} {date} {secs:.0f}s peak {peak_gb():.2f} GB")
        clear()


def bdbv_scores(fits: list[Path], data: str) -> pd.DataFrame:
    """Score BDBV parts as `bdbv_score.py` and `export.py`, without a base.

    National and provinces: `score_releases` on the complete target with
    the rolling truth at D. Zones: each zone-week's count in the first
    release D days after it, zones with cases at the origin only. The 95%
    interval's coverage from the samples (`sample_coverage`).
    """
    from bdbvhubrenewal.data import load_daily, load_daily_zones
    from bdbvhubrenewal.data.zones import is_zone, weekly_zone_counts
    from bdbvhubrenewal.evaluation.nowcast import (
        ReleaseForecasts,
        rolling_truth,
        score_releases,
    )
    from bdbvhubrenewal.evaluation.scoring import score_forecasts
    from bdbvhubrenewal.forecasters.base import NATIONAL

    target, _, tasks, delay = bdbv_inputs(data)
    daily = load_daily(data)
    run = ReleaseForecasts.read(fits)
    origins = sorted({str(o) for o in run.runs["origin"]})
    res = score_releases(
        [tasks[o] for o in origins], daily, target, run, truth_delay=delay
    )
    s = res.scores.loc[res.scores["target"] == "complete"].copy()
    s["level"] = np.where(
        s["location"] == NATIONAL,
        "national",
        np.where(s["location"].map(is_zone), "zones", "provinces"),
    )
    daily_zones = load_daily_zones(data)
    samples = run.samples
    if "as_of" in samples:
        samples = samples.loc[samples["as_of"].isna()].drop(columns="as_of")
    samples = samples.astype({"origin": str, "target_end_date": str})
    zs = samples.loc[samples["location"].map(is_zone)]
    active = {
        o: set(weekly_zone_counts(daily_zones, o)["location"]) for o in origins
    }
    zs = zs.loc[
        [loc in active.get(o, ()) for loc, o in
         zip(zs["location"], zs["origin"], strict=True)]
    ]  # fmt: skip
    ztruth = gr.rolling_zone_truth(daily_zones, delay)
    zs = zs.merge(
        ztruth[["location", "target_end_date", "truth_vintage"]],
        on=["location", "target_end_date"],
    )
    zs = zs.loc[zs["truth_vintage"] > zs["origin"]]
    # An origin with no matured zone-week leaves no zone rows.
    ref = pd.to_datetime(
        zs["origin"].map(
            lambda v: pd.Timestamp(bt.reference_date_for_vintage(v))
        )
    )
    zs = zs.assign(
        horizon=(pd.to_datetime(zs["target_end_date"]) - ref).dt.days // 7,
        target="complete",
    )
    keys = ["model", "origin", "target", "truth_vintage", "location",
            "target_end_date"]  # fmt: skip
    zscores = score_forecasts(zs, ztruth, keys=keys).assign(level="zones")
    scores = pd.concat([s, zscores], ignore_index=True)
    scores = scores.astype({"origin": str, "target_end_date": str})
    real = target.loc[target["kind"] == "real"]
    truth = pd.concat([rolling_truth(real, delay), ztruth], ignore_index=True)
    cover = gr.sample_coverage(samples, truth, 0.95)
    cell = ["model", "origin", "location", "target_end_date"]
    out = scores.merge(cover, on=cell, how="left").rename(
        columns={"coverage_50": "cover50"}
    )
    return out[[*cell, "level", "wis", "log_wis", "cover50", "cover95"]]


def bdbv_score(args: argparse.Namespace) -> None:
    """Write `bdbv_scores` of every part of `--fits`."""
    fits = sorted(
        p for d in args.fits for p in Path(d).iterdir()
        if p.is_dir() and "__" in p.name
    )  # fmt: skip
    bdbv_scores(fits, args.data).to_csv(args.out, index=False)
    log(f"bdbv scores of {len(fits)} parts to {args.out}")


def germany_score(args: argparse.Namespace) -> None:
    """Score the German arms as `germany_score2.py`, without a base."""
    from bdbvhubrenewal.data.germany import (
        AGE_GROUPS,
        GERMANY,
        hub_target,
        hub_triangle,
    )
    from bdbvhubrenewal.evaluation import germany as de
    from bdbvhubrenewal.evaluation.nowcast import rolling_truth
    from bdbvhubrenewal.reporting.hubtables import hub_quantiles

    keys = ["model", "origin", "location", "target_end_date", "horizon"]
    config = de.load_germany_config("config/germany.toml")
    files = sorted(Path(args.samples).glob("*.csv.gz"))
    samples = pd.concat(
        [pd.read_csv(f, dtype={"origin": str, "target_end_date": str})
         for f in files],
        ignore_index=True,
    )  # fmt: skip
    assert samples["origin"].max() < config.split_date
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    target = hub_target(triangle, sorted(set(samples["target_end_date"])))
    truth = rolling_truth(target, config.truth_delay)
    nat = target.loc[target["location"] == GERMANY]
    floors = nat.rename(columns={"vintage": "origin", "total": "floor"})[
        ["origin", "location", "target_end_date", "floor"]
    ].astype({"origin": str, "target_end_date": str})
    bottom = samples.loc[samples["model"].str.endswith(":bottom")]
    frames = [samples]
    if len(bottom):
        bottom = bottom.assign(
            model=bottom["model"].str.replace(":bottom", ":paired")
        )
        ages = bottom["location"].isin(set(AGE_GROUPS))
        states = ~ages & (bottom["location"] != GERMANY)
        horizon = bottom[
            ["origin", "target_end_date", "horizon"]
        ].drop_duplicates()
        paired_states, _ = gr.pair_and_sum(
            bottom.loc[states], truth, args.seed, floors=floors,
            national=GERMANY,
        )  # fmt: skip
        paired_ages, _ = gr.pair_and_sum(
            bottom.loc[ages], truth, args.seed + 1, national=GERMANY
        )
        paired = pd.concat(
            [paired_states,
             paired_ages.loc[paired_ages["location"] != GERMANY]],
            ignore_index=True,
        ).merge(horizon, on=["origin", "target_end_date"])  # fmt: skip
        frames.append(paired[samples.columns])
    every = pd.concat(frames, ignore_index=True)
    table = every.rename(columns={"sample": "output_type_id"}).assign(
        output_type="sample"
    )
    q = hub_quantiles(
        table[[*keys, "output_type", "output_type_id", "value"]],
        de.QUANTILE_LEVELS,
    )
    q = q.drop(columns="output_type").rename(
        columns={"output_type_id": "quantile"}
    )[[*keys, "quantile", "value"]]
    forecasts = de.with_truth(q, triangle, config.truth_delay)
    natural = de.score_germany(forecasts, config.truth_delay)
    logged = de.score_germany(
        forecasts.assign(
            value=np.log1p(forecasts["value"].clip(lower=0)),
            truth=np.log1p(forecasts["truth"]),
        ),
        config.truth_delay,
    )
    cell = ["model", "origin", "location", "target_end_date"]
    scores = natural.merge(
        logged[[*cell, "wis"]].rename(columns={"wis": "log_wis"}), on=cell
    )
    out = scores.rename(
        columns={"coverage_50": "cover50", "coverage_95": "cover95"}
    ).assign(level=lambda d: d["pool"].replace({"age": "ages"}))
    out[[*cell, "level", "wis", "log_wis", "cover50", "cover95"]].to_csv(
        args.out, index=False
    )
    log(f"germany scores of {len(files)} files to {args.out}")


def export(args: argparse.Namespace) -> None:
    """Write the paper's rows with this checkpoint's rows put in."""
    existing = pd.read_csv(args.existing, dtype=str).astype(
        dict.fromkeys(("wis", "log_wis", "cover50", "cover95"), float)
    )
    new = []
    for study, path in (("ebola", args.bdbv), ("germany", args.germany)):
        if path:
            dtype = {"origin": str, "target_end_date": str}
            scores = pd.read_csv(path, dtype=dtype)
            new.append(
                gr.checkpoint_rows(scores, existing, study, args.checkpoint)
            )
    rows = pd.concat(new, ignore_index=True)
    merged = gr.merge_checkpoint_rows(existing, rows, args.checkpoint)
    check = merged
    if args.subset:
        # A dry run fits some origins and dates: check those cells only.
        cells = set(rows["origin_date"])
        check = merged.loc[merged["origin_date"].isin(cells)]
    gr.check_checkpoint_rows(check)
    merged.to_csv(args.out, index=False)
    log(
        f"{len(rows)} rows of {args.checkpoint} to {args.out}:\n"
        + rows.groupby(["study", "model", "option"]).size().to_string()
    )


def sanity(args: argparse.Namespace) -> None:
    """Compare a zero-shot arm's fresh rows with the existing rows."""
    existing = pd.read_csv(args.existing, dtype=str).astype(
        dict.fromkeys(("wis", "log_wis", "cover50", "cover95"), float)
    )
    dtype = {"origin": str, "target_end_date": str}
    scores = pd.read_csv(args.scores, dtype=dtype)
    fresh = gr.aggregation_rows(
        pd.concat(
            [scores,
             existing.loc[(existing["study"] == "ebola")
                          & (existing["model"] == "randomwalknowcast")]
             .rename(columns={"origin_date": "origin",
                              "target_date": "target_end_date"})],
            ignore_index=True,
        ),
        "ebola",
    )  # fmt: skip
    fresh = fresh.loc[fresh["model"] != "randomwalknowcast"]
    keys = ["study", "model", "option", "level", "location", "origin_date",
            "target_date"]  # fmt: skip
    old = existing.merge(
        fresh[["model", "option", "origin_date"]].drop_duplicates()
    )
    both = old.merge(fresh, on=keys, how="outer", suffixes=("_old", "_new"),
                     indicator=True)  # fmt: skip
    cols = ["wis", "log_wis", "cover50", "cover95"]
    diff = {
        c: float(np.nanmax(np.abs(both[f"{c}_old"] - both[f"{c}_new"])))
        for c in cols
    }
    n = both["_merge"].value_counts().to_dict()
    exact = all(
        np.array_equal(both[f"{c}_old"], both[f"{c}_new"], equal_nan=True)
        for c in cols
    )
    rel = both["wis_new"].sum() / both["wis_old"].sum()
    log(f"sanity: cells {n}; max abs diff {diff}; exact {exact}; "
        f"wis sum new/old {rel:.6f}")  # fmt: skip
    both.to_csv(args.out, index=False)
    if n.get("left_only", 0) or n.get("right_only", 0):
        raise SystemExit("sanity: the cells differ")
    # CSV round trips leave differences of one ulp, so 1e-9 is the same.
    if max(diff.values()) > max(args.tolerance, 1e-9):
        raise SystemExit("sanity: the scores differ beyond the tolerance")


def choose(args: argparse.Namespace) -> None:
    """Print the slots checkpoint to run (or `-`), then mailbox lines.

    Each chronos-slots release's settings.json is read once (cached in
    --settings) for its `series_slots.init` and `n_outbreaks`.
    """
    import json

    cache = Path(args.settings)
    cache.mkdir(parents=True, exist_ok=True)
    tags = subprocess.run(
        ["gh", "release", "list", "-R", "seabbs/BDBVHubRenewal",
         "--limit", "200", "--json", "tagName", "--jq",
         '.[].tagName | select(startswith("chronos-slots-"))'],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    releases = []
    for tag in tags:
        path = cache / f"{tag}.json"
        if not path.exists():
            got = subprocess.run(
                ["gh", "release", "download", tag, "-R",
                 "seabbs/BDBVHubRenewal", "-p", "settings.json", "-O", "-"],
                capture_output=True, text=True, check=False,
            )  # fmt: skip
            if got.returncode != 0:
                continue  # not published in full yet
            path.write_text(got.stdout)
        settings = json.loads(path.read_text())
        releases.append(
            {
                "label": tag.removeprefix("chronos-"),
                "init": (settings.get("series_slots") or {}).get("init"),
                "n_outbreaks": settings.get("n_outbreaks", 0),
            }
        )
    done = {p.name for p in Path(args.done).glob("slots-*")}
    label = None if args.label == "next" else args.label
    chosen, notes = gr.choose_checkpoint(label, releases, done)
    print(chosen or "-")
    for note in notes:
        print(note)


def main() -> None:
    """Run one step."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    for name, groups in (("bdbv-fit", True), ("germany-fit", True)):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--checkpoint")
        p.add_argument(
            "--groupings", nargs="+", default=list(gr.FITTED_GROUPINGS)
        )
        p.add_argument("--out", required=True)
        if name == "bdbv-fit":
            p.add_argument("--origins", nargs="+", default=list(BDBV_ORIGINS))
        else:
            p.add_argument("--dates", nargs="*")
        assert groups
    p = sub.add_parser("bdbv-score")
    p.add_argument("--data", required=True)
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("germany-score")
    p.add_argument("--data", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export")
    p.add_argument("--existing", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bdbv")
    p.add_argument("--germany")
    p.add_argument("--subset", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("sanity")
    p.add_argument("--existing", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--label", required=True, help="a slots label, or next")
    p.add_argument("--done", required=True)
    p.add_argument("--settings", required=True)
    args = parser.parse_args()
    step = {
        "choose": choose,
        "bdbv-fit": bdbv_fit,
        "germany-fit": germany_fit,
        "bdbv-score": bdbv_score,
        "germany-score": germany_score,
        "export": export,
        "sanity": sanity,
    }[args.step]
    step(args)


if __name__ == "__main__":
    main()
