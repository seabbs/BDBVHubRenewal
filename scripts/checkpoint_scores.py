"""Score published Chronos-2 checkpoints under Sam's rule of 2026-11-16.

Each simulation-trained checkpoint (release `chronos-ft-<hash>`) gets
rows in `evidence/checkpoint-scores.csv` with `rule` = `RULE`:

- `germany`: the development half of the German benchmark
  (config/germany.toml), quantile WIS and log WIS on each model's own
  quantiles (`bdbvhubrenewal.evaluation.germany.germany_forecasts`:
  native quantiles for the weekly Chronos-2 models, quantiles of 1,000
  common draws otherwise), by level (national, states, age).
- `ebola`: the BDBV release backtest's origins (config/backtest.toml,
  data from `bdbvhubrenewal data`), sample CRPS and log CRPS (CRPS of
  log1p) on 1,000 draws per forecast, one seed for every model, and one
  shared random stream for the Chronos-2 models (common random
  numbers), against each week's count in the first release D days
  after it (`rolling_truth`), by level (national, provinces).
- `held_out`: the held-out simulation loss the trainer recorded in the
  release's settings.json (`held_out_loss`, by level, and
  `real_validation_loss` where given); `loss_zero_shot` is the base
  model's loss on the same set (`held_out_loss_start`).

Each study is scored on two targets (`target`): `7day`, the hub's
target, as above, and `daily`, each day's count by onset date (Germany:
by Meldedatum) from 28 days back to the nowcast date, straight from the
daily models' daily steps, not summed. The daily target is scored the
same way, on a daily truth (each day's count in the first report D days
after it), against `randomwalknowcast`'s daily forecasts, for our daily
models only: the hub teams and the weekly models have no daily
forecasts. The reference and the checkpoints of the current run take it
(`targets_for`); other checkpoints only when asked.

Relative scores divide by `randomwalknowcast` on the cells every model
and checkpoint of a level and target made. Each run of the models is a "label":
`reference` (randomwalknowcast and the zero-shot models) or a checkpoint
hash (the `_ft` models with `BDBV_CHRONOS_FT` set to it). Forecasts are
cached per date (Germany) and origin (BDBV), so a rerun resumes.

Usage:
  python scripts/checkpoint_scores.py germany CACHE LABEL [TARGETS]
  python scripts/checkpoint_scores.py bdbv CACHE LABEL [TARGETS]
  python scripts/checkpoint_scores.py heldout CACHE HASH
  python scripts/checkpoint_scores.py table CACHE EVIDENCE CHECKPOINTS HASH...

TARGETS is `7day`, `daily` or `7day,daily`, the label's own
(`targets_for`) by default. A unit adds only the targets a date's cache
file lacks; a file with no `target` column holds the 7-day target.

`table` rebuilds the rule's rows for the hashes given from the cache and
merges them into EVIDENCE (rows of other rules are kept) and the
hashes' outbreaks and grids into CHECKPOINTS.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TAG_PREFIX = "chronos-ft-"
SLOTS_TAG_PREFIX = "chronos-slots-"
"""The final run's releases (series slots), named `slots-<hash>` as the
paper pins them (`foundation.release_tag`)."""
PUBLISHED_FROM = "2026-11-16T09:00:00Z"
"""The first publish time scored: the paper uses only the checkpoints of
the continuous run the laptop trainer publishes from about 09:30 that
day; r3's and earlier runs' checkpoints are never scored."""
REFERENCE_PIN = "e8a1f5ae338b"
"""The checkpoint the reference runs pin, which they never load."""
RULE = "own-quantile-wis_crps-1000-common"
"""Sam's rule of 2026-11-16."""
OLD_RULE = "samples-100"
"""Rows written before the rule: WIS of 100 draws."""
BASELINE = "randomwalknowcast"
REFERENCE = "reference"
ZERO_SHOT = ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged")
FINETUNED = tuple(f"{m}_ft" for m in ZERO_SHOT)
TARGETS = ("7day", "daily")
"""The hub's 7-day target and each day's count (Sam, 2026-11-17 00:10)."""
DAILY_ZERO_SHOT = ("fm_chronos_daily_ragged",)
"""The zero-shot models with a daily target; the weekly ones have none."""
FINETUNED_DAILY = ("fm_chronos_daily_ragged_ft",)
"""The fine-tuned models scored on the daily target. A model plugs in
here if it has `quantiles(task, levels, target="daily")`, `daily(task)` or
`forecast_both(task)` (`bdbvhubrenewal.evaluation.marginals.daily_quantiles`).
"""
DAILY_RUNS = ("one-model",)
"""Runs (`run_of`) whose checkpoints take the daily target by default."""
DAILY_DONE = "daily.done"
"""In a unit's folder: written once every date holds the daily target."""
DAILY_DAYS = 28
"""The daily target's days run from this many days before the nowcast
date (the reference date for BDBV) to it, as the 7-day target's sums."""
COMPLETE = ("fm_chronos_daily_complete", "fm_chronos_weekly_complete")
"""Chronos-2 given only complete data points (no lags), zero-shot: a
reference beside the ragged zero-shot models (Sam, 2026-11-16 16:55)."""
COMPLETE_LABEL = "reference-complete"
"""The complete-data models' own run, merged into the reference files
once every date and origin is in (`merge_complete`)."""
COMPLETE_STAMP = "complete.merged"
"""In the cache: written once the complete-data forecasts are merged."""
N_DRAWS = 1000
GERMANY_CELL = ["origin", "location", "target_end_date"]
BDBV_CELL = ["origin", "truth_vintage", "location", "target_end_date"]
COLUMNS = [
    "study",
    "level",
    "model",
    "checkpoint",
    "outbreaks",
    "n",
    "n_origins",
    "first_origin",
    "last_origin",
    "rel_wis",
    "rel_log_wis",
    "coverage_50",
    "coverage_90",
    "coverage_95",
    "grid",
    "rule",
    "rel_crps",
    "rel_log_crps",
    "loss",
    "loss_zero_shot",
    "run",
    "target",
]
HELD_OUT_LEVELS = ("leaf", "national", "intermediate")
SORT = ["rule", "study", "level", "model", "outbreaks", "checkpoint"]
"""The evidence's row order, before the target."""


# Which checkpoints to score -------------------------------------------


def checkpoint_queue(
    releases: Iterable[Mapping[str, str]],
    scored: set[str],
    published_from: str = PUBLISHED_FROM,
) -> list[str]:
    """Return the checkpoints to score: slots ones first, oldest first.

    Only releases published from `published_from` are taken. The final
    run's `chronos-slots-<hash>` releases come first, as `slots-<hash>`,
    then the no-slots run's `chronos-ft-<hash>` ones, as `<hash>`. Scored
    checkpoints and releases of other prefixes are left out.
    """
    published = {}
    for r in releases:
        tag = r["tagName"]
        if r["publishedAt"] < published_from:
            continue
        if tag.startswith(SLOTS_TAG_PREFIX):
            label = "slots-" + tag.removeprefix(SLOTS_TAG_PREFIX)
            published[label] = (0, r["publishedAt"])
        elif tag.startswith(TAG_PREFIX):
            published[tag.removeprefix(TAG_PREFIX)] = (1, r["publishedAt"])
    todo = {h: t for h, t in published.items() if h not in scored}
    return sorted(todo, key=todo.__getitem__)


def release_of(label: str) -> str:
    """Return a checkpoint label's release tag."""
    from bdbvhubrenewal.forecasters.foundation import release_tag

    return release_tag(label)


def scored_checkpoints(evidence: pd.DataFrame) -> set[str]:
    """Return the checkpoints with rule rows for every study."""
    if "rule" not in evidence:
        return set()
    rows = evidence.loc[evidence["rule"] == RULE]
    studies = rows.groupby("checkpoint")["study"].agg(set)
    need = {"germany", "ebola", "held_out"}
    return {str(c) for c, s in studies.items() if need <= s}


# Rows ------------------------------------------------------------------


CURRENT_RUNS = {"one-model"}
"""Runs whose adapters this code loads (series bias, mixed layouts)."""


def run_of(settings: Mapping[str, Any]) -> str:
    """Return a checkpoint's run from its settings.

    `one-model`: the series bias with mixed-frequency layouts (a source
    such as `daily+weekly`); `attn-bias`: the earlier series-bias trials;
    `slots-zeroinit` and `slots`: the superseded slots runs; `no-slots`.
    """
    if settings.get("series_bias"):
        mixed = any("+" in str(s) for s in settings.get("sources", []))
        return "one-model" if mixed else "attn-bias"
    slots = settings.get("series_slots")
    if not slots:
        return "no-slots"
    return "slots-zeroinit" if slots.get("init") == "zero" else "slots"


def checkpoints_with_rows(evidence: pd.DataFrame) -> set[str]:
    """Return the checkpoints with any rule rows, scored or not."""
    if "rule" not in evidence:
        return set()
    rule = evidence.loc[evidence["rule"] == RULE, "checkpoint"].astype(str)
    return set(rule) - {"zero-shot"}


def outbreaks_and_grid(
    settings: Mapping[str, Any], release: Mapping[str, str]
) -> tuple[int, str]:
    """Return a checkpoint's training outbreaks and grid.

    From settings.json, else from the release's title and body
    ("2,000 simulated ... outbreaks", "round 1"), else 0 and "unknown".
    """
    text = f"{release.get('name', '')} {release.get('body', '')}"
    outbreaks = settings.get("n_outbreaks")
    if outbreaks is None:
        found = re.search(r"([\d,]+) (?:simulated )?(?:\w+ )?outbreaks", text)
        outbreaks = int(found.group(1).replace(",", "")) if found else 0
    grid = settings.get("grid")
    if grid is None:
        found = re.search(r"round (\d+)", text)
        grid = f"r{found.group(1)}" if found else "unknown"
    return int(outbreaks), str(grid)


def held_out_rows(checkpoint: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Return the trainer's held-out losses as rows, missing ones as NaN."""
    by = settings.get("held_out_loss_by_level") or {}
    rows = [
        {
            "level": "all",
            "loss": settings.get("held_out_loss", np.nan),
            "loss_zero_shot": settings.get("held_out_loss_start", np.nan),
        }
    ]
    rows += [
        {"level": lv, "loss": by[lv]} for lv in HELD_OUT_LEVELS if lv in by
    ]
    if "real_validation_loss" in settings:
        rows.append(
            {
                "level": "real_validation",
                "loss": settings["real_validation_loss"],
            }
        )
    out = pd.DataFrame(rows).assign(
        study="held_out",
        model="fm_chronos",
        checkpoint=checkpoint,
        n=settings.get("n_validation", 0),
        n_origins=0,
        rule=RULE,
    )
    out["loss"] = out["loss"].astype(float)
    return out.reindex(columns=COLUMNS)


def with_target(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a `target`, `7day` where they have none."""
    if "target" not in rows:
        return rows.assign(target="7day")
    return rows.assign(target=rows["target"].fillna("7day"))


def matched(rows: pd.DataFrame, cell: list[str]) -> pd.DataFrame:
    """Keep, in each level and target, the cells every model made."""
    rows = with_target(rows)
    key = rows["model"] + "@" + rows["checkpoint"]
    out = []
    for _, part in rows.assign(_k=key).groupby(["target", "level"]):
        n = part.groupby(cell)["_k"].nunique()
        full = n.index[n == part["_k"].nunique()]
        out.append(part.set_index(cell).loc[full].reset_index())
    return pd.concat(out, ignore_index=True).drop(columns="_k")


def relative(
    rows: pd.DataFrame, study: str, metrics: tuple[str, str]
) -> pd.DataFrame:
    """Return mean scores relative to the baseline, by level and model.

    `metrics` is the score and its log-scale score, such as
    `("wis", "log_wis")` or `("crps", "log_crps")`. Each target divides
    by the baseline's scores of that target.
    """
    rows = with_target(rows)
    covers = [
        c for c in ("coverage_50", "coverage_90", "coverage_95") if c in rows
    ]
    keys = ["target", "level", "model", "checkpoint"]
    mean = rows.groupby(keys)[[*metrics, *covers]].mean()
    mean["n"] = rows.groupby(keys).size()
    origins = rows.groupby(keys)["origin"]
    mean["n_origins"] = origins.nunique()
    mean["first_origin"] = origins.min()
    mean["last_origin"] = origins.max()
    base = mean.xs((BASELINE, "zero-shot"), level=("model", "checkpoint"))
    at = pd.MultiIndex.from_arrays(
        [
            mean.index.get_level_values("target"),
            mean.index.get_level_values("level"),
        ]
    )
    for m in metrics:
        mean[f"rel_{m}"] = mean[m] / base[m].reindex(at).to_numpy()
    out = mean.reset_index().assign(study=study, rule=RULE)
    return out.reindex(columns=COLUMNS)


def merge_evidence(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add rule rows to the evidence, replacing the rule's rows it had.

    Rows from before the rule (no `rule`) are marked `OLD_RULE` and kept.
    """
    old = old.reindex(columns=COLUMNS)
    old["rule"] = old["rule"].astype(object).fillna(OLD_RULE)
    old = old.loc[old["rule"] != RULE]
    out = pd.concat([old, new.reindex(columns=COLUMNS)], ignore_index=True)
    out["target"] = out["target"].astype(object).fillna("7day")
    return out.sort_values(
        [*SORT, "target"],
        kind="stable",
    ).reset_index(drop=True)


# Forecasting -----------------------------------------------------------


def common(model: Any) -> Any:
    """Return the model drawing from the stream every Chronos-2 shares."""
    from bdbvhubrenewal.evaluation.marginals import COMMON_STREAM

    if hasattr(model, "stream"):
        return dataclasses.replace(model, stream=COMMON_STREAM)
    return model


def label_models(label: str, target: str = "7day") -> tuple[str, ...]:
    """Return the models a label runs for a target."""
    if target == "daily":
        if label == REFERENCE:
            return (BASELINE, *DAILY_ZERO_SHOT)
        return () if label == COMPLETE_LABEL else FINETUNED_DAILY
    if label == REFERENCE:
        return (BASELINE, *ZERO_SHOT, *COMPLETE)
    if label == COMPLETE_LABEL:
        return COMPLETE
    return FINETUNED


def _pin(label: str) -> None:
    """Pin the checkpoint, so no process resolves "latest" mid-run."""
    if label not in (REFERENCE, COMPLETE_LABEL):
        os.environ["BDBV_CHRONOS_FT"] = label
    elif not os.environ.get("BDBV_CHRONOS_FT"):
        os.environ["BDBV_CHRONOS_FT"] = REFERENCE_PIN


def free_caches() -> None:
    """Free JAX's compiled functions and collect garbage.

    Each date or origin compiles JAX functions of new shapes (the
    random walk's sampling among them). Kept, they grew a unit past its
    6 GB scope, where LLVM could not allocate memory to compile the next
    (the zero-shot reference, 2026-11-16, after 29 of 80 German dates).
    """
    import gc

    import jax

    jax.clear_caches()
    gc.collect()


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a gzipped CSV atomically, so a stopped run leaves no part."""
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)


def germany_dates() -> list[str]:
    """Return the development half's nowcast dates."""
    from bdbvhubrenewal.evaluation import germany as de

    config = de.load_germany_config("config/germany.toml")
    return de.nowcast_dates(config, held_out=False)


def unit_root(cache: Path, study: str, label: str) -> Path:
    """Return a unit's cache folder, `<study>/<label>`, made if needed.

    One folder per checkpoint: a slots checkpoint's is `slots-<hash>`.
    Each date or origin is written there as it finishes.
    """
    root = cache / study / label
    root.mkdir(parents=True, exist_ok=True)
    return root


def merge_complete(cache: Path) -> None:
    """Add the complete-data forecasts to the reference files, once.

    Each reference file gains the `COMPLETE_LABEL` file's rows of models
    it does not hold yet, so a second merge adds nothing. Then
    `COMPLETE_STAMP` is written.
    """
    read = {"dtype": {"origin": str, "target_end_date": str}}
    for study in ("germany", "bdbv"):
        for f in sorted((cache / study / COMPLETE_LABEL).glob("*.csv.gz")):
            ref = cache / study / REFERENCE / f.name
            new = pd.read_csv(f, **read)
            if ref.exists():
                old = pd.read_csv(ref, **read)
                new = new.loc[~new["model"].isin(set(old["model"]))]
                if new.empty:
                    continue
                new = pd.concat([old, new], ignore_index=True)
            _write(new, ref)
    (cache / COMPLETE_STAMP).write_text("merged\n")


def ready_labels(
    cache: Path, study: str, labels: Sequence[str], target: str = "7day"
) -> list[str]:
    """Return the labels whose `study` unit is complete for `target`."""
    return [lb for lb in labels if unit_done(cache, study, lb, target)]


def daily_ready(cache: Path, study: str, labels: Sequence[str]) -> list[str]:
    """Return the labels with the daily target, if the reference has it.

    The daily target's baseline is the reference's `randomwalknowcast`.
    """
    ready = ready_labels(cache, study, labels, "daily")
    return ready if REFERENCE in ready else []


def targets_for(cache: Path, label: str) -> tuple[str, ...]:
    """Return the targets a label takes by default.

    The reference and the checkpoints of `DAILY_RUNS` take both; others,
    and checkpoints whose settings are not fetched yet, the 7-day only.
    """
    if label == REFERENCE:
        return TARGETS
    settings = cache / "heldout" / f"{label}.json"
    if label == COMPLETE_LABEL or not settings.exists():
        return ("7day",)
    run = run_of(json.loads(settings.read_text()))
    return TARGETS if run in DAILY_RUNS else ("7day",)


_READ = {"dtype": {"origin": str, "target_end_date": str}}


def read_cache(path: Path) -> pd.DataFrame:
    """Read a cache file, with `target` `7day` where it has none."""
    return with_target(pd.read_csv(path, **_READ))


def cached_targets(path: Path) -> set[str]:
    """Return the targets a cache file holds, none if it is missing."""
    if not path.exists():
        return set()
    head = pd.read_csv(path, nrows=0)
    if "target" not in head:
        return {"7day"}
    return set(pd.read_csv(path, usecols=["target"])["target"].dropna())


def _add(path: Path, parts: list[pd.DataFrame]) -> None:
    """Write a date's new targets beside the ones its file holds."""
    if path.exists():
        parts = [read_cache(path), *parts]
    _write(pd.concat(parts, ignore_index=True), path)


def germany_unit(
    cache: Path, label: str, targets: Sequence[str] = ("7day",)
) -> None:
    """Nowcast each development date, as own quantiles with truth.

    Only the `targets` a date's file lacks are run and added to it.
    """
    import time

    from bdbvhubrenewal.data.germany import hub_triangle
    from bdbvhubrenewal.evaluation import germany as de

    _pin(label)
    config = de.load_germany_config("config/germany.toml")
    triangle = hub_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    root = unit_root(cache, "germany", label)
    dates = de.nowcast_dates(config, held_out=False)
    for i, date in enumerate(dates, 1):
        path = root / f"{date}.csv.gz"
        todo = [t for t in targets if t not in cached_targets(path)]
        if not todo:
            continue
        parts, took = [], []
        for target in todo:
            start = time.monotonic()
            models = list(label_models(label, target))
            if target == "7day":
                fc = de.germany_forecasts(
                    models, triangle, [date], config.n_samples, config.seed
                )
                fc = de.with_truth(fc, triangle, config.truth_delay)
            else:
                fc = de.germany_daily_forecasts(
                    models, triangle, [date], config.n_samples, config.seed
                )
                fc = de.with_truth(fc, triangle, config.truth_delay, window=1)
            parts.append(fc.assign(target=target))
            took.append(f"{target} {time.monotonic() - start:.0f}s")
        _add(path, parts)
        free_caches()
        print(
            f"germany {label} {date} ({i}/{len(dates)}): {', '.join(took)}",
            flush=True,
        )
    if "daily" in targets:
        (root / DAILY_DONE).write_text("done\n")
    print(f"germany {label} done", flush=True)


def bdbv_origins() -> list[str]:
    """Return the release backtest's origins: releases with a later one."""
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.backtest import real_vintages

    config = bt.load_config()
    target = pd.read_csv("data/target.csv", dtype={"kind": str})
    real = real_vintages(target)
    return [v for v in real[:-1] if v not in config.skip_origins]


def origin_draws(frames: list[pd.DataFrame], origin: str) -> pd.DataFrame:
    """Return one origin's draws of every model, empty if none forecast."""
    cols = ["location", "target_end_date", "sample", "value", "model"]
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=cols)
    ).assign(origin=origin)
    out["value"] = out["value"].astype("float32")
    return out


def daily_window(drawn: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Keep the daily target's days: `DAILY_DAYS` back to `reference`."""
    day = pd.to_datetime(drawn["target_end_date"])
    end = pd.Timestamp(reference)
    first = end - pd.Timedelta(days=DAILY_DAYS)
    return drawn.loc[(day >= first) & (day <= end)]


def bdbv_unit(
    cache: Path, label: str, targets: Sequence[str] = ("7day",)
) -> None:
    """Draw 1,000 final-count samples per origin for each model.

    The daily target's draws are each model's daily draws (`daily`), of
    the days `DAILY_DAYS` back to the task's reference date. Only the
    `targets` an origin's file lacks are run and added to it.
    """
    from bdbvhubrenewal.data import load_daily
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation import store
    from bdbvhubrenewal.evaluation.marginals import daily_draws
    from bdbvhubrenewal.forecasters import get_model
    from bdbvhubrenewal.forecasters.base import (
        ModelUnavailableError,
        add_national,
    )

    _pin(label)
    config = bt.load_config()
    data = Path("data")
    target = pd.read_csv(data / "target.csv", dtype={"kind": str})
    tasks = store.origin_tasks(
        target, pd.read_csv(data / "delays.csv.gz"), config, load_daily(data)
    )
    origins = bdbv_origins()
    built = {
        t: [common(get_model(m)) for m in label_models(label, t)]
        for t in targets
    }
    root = unit_root(cache, "bdbv", label)
    for origin in origins:
        path = root / f"{origin}.csv.gz"
        todo = [t for t in targets if t not in cached_targets(path)]
        if not todo:
            continue
        task = dataclasses.replace(
            tasks[origin], n_samples=N_DRAWS, seed=config.seed
        )
        parts = []
        for target in todo:
            frames = []
            for model in built[target]:
                try:
                    if target == "7day":
                        drawn = model.forecast(task)
                    else:
                        drawn = daily_draws(model, task)
                except ModelUnavailableError as err:
                    print(f"{model.name} at {origin}: {err}", flush=True)
                    continue
                if drawn is None:
                    continue
                if "as_of" in drawn:
                    drawn = drawn.loc[drawn["as_of"].isna()].drop(
                        columns="as_of"
                    )
                if target == "daily":
                    drawn = daily_window(drawn, str(task.reference_date))
                frames.append(add_national(drawn).assign(model=model.name))
            parts.append(origin_draws(frames, origin).assign(target=target))
        _add(path, parts)
        free_caches()
        print(f"bdbv {label} {origin}: {', '.join(todo)}", flush=True)
    if "daily" in targets:
        (root / DAILY_DONE).write_text("done\n")
    print(f"bdbv {label} done", flush=True)


def held_out_unit(cache: Path, checkpoint: str) -> None:
    """Fetch a release's settings.json and its title and body."""
    root = cache / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    tag = release_of(checkpoint)
    repo = ["-R", "seabbs/BDBVHubRenewal"]
    out = str(root / f"{checkpoint}.json")
    get = ["gh", "release", "download", tag, *repo, "-p", "settings.json"]
    subprocess.run([*get, "-O", out, "--clobber"], check=True)
    view = subprocess.run(
        ["gh", "release", "view", tag, *repo, "--json", "name,body"],
        check=True,
        capture_output=True,
        text=True,
    )
    (root / f"{checkpoint}-release.json").write_text(view.stdout)
    print(f"heldout {checkpoint} done", flush=True)


def unit_done(
    cache: Path, study: str, label: str, target: str = "7day"
) -> bool:
    """Whether a unit's cache is complete for a target.

    Every date's file holds the 7-day target; the daily one is done once
    the unit has written `DAILY_DONE`.
    """
    if study == "heldout":
        return (cache / "heldout" / f"{label}-release.json").exists()
    want = germany_dates() if study == "germany" else bdbv_origins()
    files = (cache / study / label).glob("*.csv.gz")
    have = {p.name.removesuffix(".csv.gz") for p in files}
    if target == "daily" and not (cache / study / label / DAILY_DONE).exists():
        return False
    return set(want) <= have


# Scoring ---------------------------------------------------------------


def cell_scores(samples: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Score each forecast cell's draws by CRPS, log CRPS and coverage."""
    from bdbvhubrenewal.evaluation.scoring import coverage_sample, crps_sample

    keys = ["model", "origin", "location", "target_end_date"]
    rows = samples.merge(truth, on=["location", "target_end_date"])
    out = []
    for key, grp in rows.groupby(keys, sort=True):
        x = grp["value"].to_numpy(dtype=float)
        y = float(grp["truth"].iloc[0])
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "truth_vintage": grp["truth_vintage"].iloc[0],
                "truth": y,
                "crps": crps_sample(y, x),
                "log_crps": crps_sample(float(np.log1p(y)), np.log1p(x)),
                **{
                    f"coverage_{p}": float(coverage_sample(y, x, p / 100))
                    for p in (50, 90, 95)
                },
            }
        )
    return pd.DataFrame(
        out,
        columns=[
            *keys,
            "truth_vintage",
            "truth",
            "crps",
            "log_crps",
            "coverage_50",
            "coverage_90",
            "coverage_95",
        ],
    )


def _tag(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """Name each run's rows by checkpoint, the `_ft` suffix dropped."""
    return rows.assign(
        checkpoint=(
            "zero-shot" if label in (REFERENCE, COMPLETE_LABEL) else label
        ),
        model=rows["model"].str.removesuffix("_ft"),
    )


def _by_target(
    labels: Sequence[str], daily_labels: Sequence[str]
) -> dict[str, list[str]]:
    """Return each label's targets to score."""
    out: dict[str, list[str]] = {}
    for label in labels:
        out.setdefault(label, []).append("7day")
    for label in daily_labels:
        out.setdefault(label, []).append("daily")
    return out


def germany_rows(
    cache: Path, labels: Sequence[str], daily_labels: Sequence[str] = ()
) -> pd.DataFrame:
    """Return the German development half's rule rows.

    The 7-day target's from `labels` and the daily target's from
    `daily_labels`, where the hub teams take no part.
    """
    from bdbvhubrenewal.evaluation import germany as de

    sys.path.insert(0, str(Path(__file__).parent))
    from germany_chronos_dev import scores as germany_scores

    config = de.load_germany_config("config/germany.toml")
    teams = set(de.TEAM_NAMES.values())
    parts = []
    for label, targets in _by_target(labels, daily_labels).items():
        files = sorted((cache / "germany" / label).glob("*.csv.gz"))
        fc = pd.concat([read_cache(f) for f in files])
        fc = fc.loc[fc["origin"] < config.split_date]
        for target in targets:
            part = fc.loc[fc["target"] == target].drop(columns="target")
            if target == "daily":
                part = part.loc[~part["model"].isin(teams)]
            if part.empty:
                continue
            sc = germany_scores(part, config.truth_delay)
            parts.append(_tag(sc, label).assign(target=target))
    sc = pd.concat(parts, ignore_index=True).rename(columns={"pool": "level"})
    return relative(matched(sc, GERMANY_CELL), "germany", ("wis", "log_wis"))


def bdbv_daily_truth(daily: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Return each day's count in the first release `delay` days after it.

    A release lists only days with a case (`load_daily`), so each covers
    every day from the first onset to its cut-off, 0 where it has no row:
    the release reporting no case that day. The count is observed plus
    imputed onsets, as the weekly target's. The provinces are the places,
    and the country is their sum.

    Parameters
    ----------
    daily : pandas.DataFrame
        `load_daily` rows: `vintage`, `location`, `onset_date`,
        `observed`, `imputed` and `cutoff`.
    delay : int
        $D$, the days from a day to the cut-off of the release that gives
        its truth.

    Returns
    -------
    pandas.DataFrame
        As `rolling_truth`, one row per place and day with a release $D$
        days on, the day in `target_end_date`.
    """
    from bdbvhubrenewal.evaluation.nowcast import rolling_truth

    rows = daily.loc[daily["location"].str.count("-") == 1]
    places = sorted(set(rows["location"]))
    first = pd.Timestamp(rows["onset_date"].min())
    frames = []
    for (vintage, cutoff), own in rows.groupby(["vintage", "cutoff"]):
        days = pd.date_range(first, cutoff, freq="D").strftime("%Y-%m-%d")
        wide = (
            own.assign(total=own["observed"] + own["imputed"])
            .pivot_table(
                index="onset_date",
                columns="location",
                values="total",
                aggfunc="sum",
            )
            .reindex(index=days, columns=places)
            .fillna(0.0)
        )
        wide["CD"] = wide[places].sum(axis=1)
        long = wide.rename_axis(index="target_end_date").reset_index()
        long = long.melt(
            id_vars="target_end_date", var_name="location", value_name="total"
        )
        frames.append(long.assign(vintage=vintage, cutoff=cutoff))
    return rolling_truth(pd.concat(frames, ignore_index=True), delay)


def bdbv_rows(
    cache: Path, labels: Sequence[str], daily_labels: Sequence[str] = ()
) -> pd.DataFrame:
    """Return the BDBV release backtest's rule rows.

    The 7-day target's from `labels`, on the weekly truth, and the daily
    target's from `daily_labels`, on the daily truth
    (`bdbv_daily_truth`).
    """
    from bdbvhubrenewal.data import load_daily
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.nowcast import (
        resolve_truth_delay,
        rolling_truth,
    )

    config = bt.load_config()
    target = pd.read_csv("data/target.csv", dtype={"kind": str})
    real = target.loc[target["kind"] == "real"]
    delay = resolve_truth_delay(config, target)
    truth = {"7day": rolling_truth(real, delay)}
    if daily_labels:
        truth["daily"] = bdbv_daily_truth(load_daily(Path("data")), delay)
    truth = {
        t: v.astype({"target_end_date": str, "truth_vintage": str})
        for t, v in truth.items()
    }
    parts = []
    for label, targets in _by_target(labels, daily_labels).items():
        for f in sorted((cache / "bdbv" / label).glob("*.csv.gz")):
            draws = read_cache(f)
            for t in targets:
                part = draws.loc[draws["target"] == t].drop(columns="target")
                sc = _tag(cell_scores(part, truth[t]), label)
                parts.append(sc.assign(target=t))
    sc = pd.concat(parts, ignore_index=True)
    sc["level"] = (sc["location"].str.count("-") == 0).map(
        {True: "national", False: "provinces"}
    )
    sc = sc.loc[sc["location"].str.count("-") <= 1]
    return relative(matched(sc, BDBV_CELL), "ebola", ("crps", "log_crps"))


def table(
    cache: Path, evidence: Path, checkpoints: Path, hashes: Sequence[str]
) -> pd.DataFrame:
    """Rebuild the rule's rows for `hashes` and merge them into the files."""
    meta = {}
    for h in hashes:
        settings = json.loads((cache / "heldout" / f"{h}.json").read_text())
        release = json.loads(
            (cache / "heldout" / f"{h}-release.json").read_text()
        )
        meta[h] = (settings, *outbreaks_and_grid(settings, release))
    # Each study's rows come from the labels whose unit is finished, so a
    # slots checkpoint's BDBV rows land before its German run ends.
    labels = [REFERENCE, *hashes]
    new = pd.concat(
        [
            germany_rows(
                cache,
                ready_labels(cache, "germany", labels),
                daily_ready(cache, "germany", labels),
            ),
            bdbv_rows(
                cache,
                ready_labels(cache, "bdbv", labels),
                daily_ready(cache, "bdbv", labels),
            ),
            *(held_out_rows(h, m[0]) for h, m in meta.items()),
        ],
        ignore_index=True,
    )
    new["outbreaks"] = new["checkpoint"].map({h: m[1] for h, m in meta.items()})
    new["outbreaks"] = new["outbreaks"].fillna(0).astype(int)
    new["grid"] = new["checkpoint"].map({h: m[2] for h, m in meta.items()})
    runs = {h: run_of(m[0]) for h, m in meta.items()}
    new["run"] = new["checkpoint"].map(runs).fillna("zero-shot")
    old = pd.read_csv(evidence, dtype={"checkpoint": str})
    merge_evidence(old, new).round(4).to_csv(evidence, index=False)
    ck = pd.read_csv(checkpoints, dtype={"checkpoint": str})
    add = pd.DataFrame(
        [(h, m[1], m[2]) for h, m in meta.items()],
        columns=["checkpoint", "outbreaks", "grid"],
    )
    ck = pd.concat(
        [ck.loc[~ck["checkpoint"].isin(add["checkpoint"])], add],
        ignore_index=True,
    )
    ck.to_csv(checkpoints, index=False)
    return new


def main(argv: Sequence[str]) -> None:
    """Run one unit, or rebuild the table."""
    study, cache = argv[0], Path(argv[1])
    if study in ("germany", "bdbv"):
        label = argv[2]
        targets = (
            tuple(argv[3].split(","))
            if len(argv) > 3
            else targets_for(cache, label)
        )
        unknown = set(targets) - set(TARGETS)
        if unknown:
            raise SystemExit(f"unknown targets {sorted(unknown)}: {TARGETS}")
        unit = germany_unit if study == "germany" else bdbv_unit
        unit(cache, label, targets)
    elif study == "heldout":
        held_out_unit(cache, argv[2])
    elif study == "table":
        new = table(cache, Path(argv[2]), Path(argv[3]), argv[4:])
        print(new.round(3).to_string(index=False), flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
