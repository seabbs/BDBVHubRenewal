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

Relative scores divide by `randomwalknowcast` on the cells every model
and checkpoint of a level made. Each run of the models is a "label":
`reference` (randomwalknowcast and the zero-shot models) or a checkpoint
hash (the `_ft` models with `BDBV_CHRONOS_FT` set to it). Forecasts are
cached per date (Germany) and origin (BDBV), so a rerun resumes.

Usage:
  python scripts/checkpoint_scores.py germany CACHE LABEL
  python scripts/checkpoint_scores.py bdbv CACHE LABEL
  python scripts/checkpoint_scores.py heldout CACHE HASH
  python scripts/checkpoint_scores.py table CACHE EVIDENCE CHECKPOINTS HASH...

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
BACKLOG = (
    "6b7aaa8d4f55",
    "3d3bc724bbfa",
    "e8a1f5ae338b",
    "0cea0acef1ce",
    "2c2b22a03107",
)
"""Checkpoints published before this job, scored after any new one: the
r3 run's first, then the earlier runs'."""
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
]
HELD_OUT_LEVELS = ("leaf", "national", "intermediate")


# Which checkpoints to score -------------------------------------------


def checkpoint_queue(
    releases: Iterable[Mapping[str, str]],
    scored: set[str],
    backlog: Sequence[str] = BACKLOG,
) -> list[str]:
    """Return the checkpoints to score, in the order to score them.

    Checkpoints not in `backlog` come first, oldest published first,
    then the backlog in its own order. Scored checkpoints and releases
    of other prefixes are left out.
    """
    published = {
        r["tagName"].removeprefix(TAG_PREFIX): r["publishedAt"]
        for r in releases
        if r["tagName"].startswith(TAG_PREFIX)
    }
    todo = {h: t for h, t in published.items() if h not in scored}
    new = sorted((h for h in todo if h not in backlog), key=todo.__getitem__)
    return new + [h for h in backlog if h in todo]


def scored_checkpoints(evidence: pd.DataFrame) -> set[str]:
    """Return the checkpoints with rule rows for every study."""
    if "rule" not in evidence:
        return set()
    rows = evidence.loc[evidence["rule"] == RULE]
    studies = rows.groupby("checkpoint")["study"].agg(set)
    need = {"germany", "ebola", "held_out"}
    return {str(c) for c, s in studies.items() if need <= s}


# Rows ------------------------------------------------------------------


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


def matched(rows: pd.DataFrame, cell: list[str]) -> pd.DataFrame:
    """Keep, in each level, the cells every model and checkpoint made."""
    key = rows["model"] + "@" + rows["checkpoint"]
    out = []
    for _, part in rows.assign(_k=key).groupby("level"):
        n = part.groupby(cell)["_k"].nunique()
        full = n.index[n == part["_k"].nunique()]
        out.append(part.set_index(cell).loc[full].reset_index())
    return pd.concat(out, ignore_index=True).drop(columns="_k")


def relative(
    rows: pd.DataFrame, study: str, metrics: tuple[str, str]
) -> pd.DataFrame:
    """Return mean scores relative to the baseline, by level and model.

    `metrics` is the score and its log-scale score, such as
    `("wis", "log_wis")` or `("crps", "log_crps")`.
    """
    covers = [
        c for c in ("coverage_50", "coverage_90", "coverage_95") if c in rows
    ]
    keys = ["level", "model", "checkpoint"]
    mean = rows.groupby(keys)[[*metrics, *covers]].mean()
    mean["n"] = rows.groupby(keys).size()
    origins = rows.groupby(keys)["origin"]
    mean["n_origins"] = origins.nunique()
    mean["first_origin"] = origins.min()
    mean["last_origin"] = origins.max()
    base = mean.xs((BASELINE, "zero-shot"), level=("model", "checkpoint"))
    for m in metrics:
        mean[f"rel_{m}"] = (
            mean[m]
            / base[m].reindex(mean.index.get_level_values("level")).to_numpy()
        )
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
    return out.sort_values(
        ["rule", "study", "level", "model", "outbreaks", "checkpoint"],
        kind="stable",
    ).reset_index(drop=True)


# Forecasting -----------------------------------------------------------


def common(model: Any) -> Any:
    """Return the model drawing from the stream every Chronos-2 shares."""
    from bdbvhubrenewal.evaluation.marginals import COMMON_STREAM

    if hasattr(model, "stream"):
        return dataclasses.replace(model, stream=COMMON_STREAM)
    return model


def label_models(label: str) -> tuple[str, ...]:
    """Return the models a label runs."""
    return (BASELINE, *ZERO_SHOT) if label == REFERENCE else FINETUNED


def _pin(label: str) -> None:
    """Pin the checkpoint, so no process resolves "latest" mid-run."""
    if label != REFERENCE:
        os.environ["BDBV_CHRONOS_FT"] = label
    elif not os.environ.get("BDBV_CHRONOS_FT"):
        os.environ["BDBV_CHRONOS_FT"] = REFERENCE_PIN


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


def germany_unit(cache: Path, label: str) -> None:
    """Nowcast each development date, as own quantiles with truth."""
    from bdbvhubrenewal.data.germany import hub_triangle
    from bdbvhubrenewal.evaluation import germany as de

    _pin(label)
    config = de.load_germany_config("config/germany.toml")
    triangle = hub_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    root = cache / "germany" / label
    root.mkdir(parents=True, exist_ok=True)
    dates = de.nowcast_dates(config, held_out=False)
    for i, date in enumerate(dates, 1):
        path = root / f"{date}.csv.gz"
        if path.exists():
            continue
        fc = de.germany_forecasts(
            list(label_models(label)),
            triangle,
            [date],
            config.n_samples,
            config.seed,
        )
        _write(de.with_truth(fc, triangle, config.truth_delay), path)
        print(f"germany {label} {date} ({i}/{len(dates)})", flush=True)
    print(f"germany {label} done", flush=True)


def bdbv_origins() -> list[str]:
    """Return the release backtest's origins: releases with a later one."""
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.backtest import real_vintages

    config = bt.load_config()
    target = pd.read_csv("data/target.csv", dtype={"kind": str})
    real = real_vintages(target)
    return [v for v in real[:-1] if v not in config.skip_origins]


def bdbv_unit(cache: Path, label: str) -> None:
    """Draw 1,000 final-count samples per origin for each model."""
    from bdbvhubrenewal.data import load_daily
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation import store
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
    built = [common(get_model(m)) for m in label_models(label)]
    root = cache / "bdbv" / label
    root.mkdir(parents=True, exist_ok=True)
    for origin in origins:
        path = root / f"{origin}.csv.gz"
        if path.exists():
            continue
        task = dataclasses.replace(
            tasks[origin], n_samples=N_DRAWS, seed=config.seed
        )
        frames = []
        for model in built:
            try:
                drawn = model.forecast(task)
            except ModelUnavailableError as err:
                print(f"{model.name} at {origin}: {err}", flush=True)
                continue
            if "as_of" in drawn:
                drawn = drawn.loc[drawn["as_of"].isna()].drop(columns="as_of")
            frames.append(add_national(drawn).assign(model=model.name))
        out = pd.concat(frames, ignore_index=True).assign(origin=origin)
        out["value"] = out["value"].astype("float32")
        _write(out, path)
        print(f"bdbv {label} {origin}", flush=True)
    print(f"bdbv {label} done", flush=True)


def held_out_unit(cache: Path, checkpoint: str) -> None:
    """Fetch a release's settings.json and its title and body."""
    root = cache / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{TAG_PREFIX}{checkpoint}"
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


def unit_done(cache: Path, study: str, label: str) -> bool:
    """Whether a unit's cache is complete."""
    if study == "heldout":
        return (cache / "heldout" / f"{label}-release.json").exists()
    want = germany_dates() if study == "germany" else bdbv_origins()
    files = (cache / study / label).glob("*.csv.gz")
    have = {p.name.removesuffix(".csv.gz") for p in files}
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
        checkpoint="zero-shot" if label == REFERENCE else label,
        model=rows["model"].str.removesuffix("_ft"),
    )


def germany_rows(cache: Path, labels: Sequence[str]) -> pd.DataFrame:
    """Return the German development half's rule rows."""
    from bdbvhubrenewal.evaluation import germany as de

    sys.path.insert(0, str(Path(__file__).parent))
    from germany_chronos_dev import scores as germany_scores

    config = de.load_germany_config("config/germany.toml")
    read = {"dtype": {"origin": str, "target_end_date": str}}
    parts = []
    for label in labels:
        files = sorted((cache / "germany" / label).glob("*.csv.gz"))
        fc = pd.concat([pd.read_csv(f, **read) for f in files])
        fc = fc.loc[fc["origin"] < config.split_date]
        sc = germany_scores(fc, config.truth_delay)
        parts.append(_tag(sc, label))
    sc = pd.concat(parts, ignore_index=True).rename(columns={"pool": "level"})
    return relative(matched(sc, GERMANY_CELL), "germany", ("wis", "log_wis"))


def bdbv_rows(cache: Path, labels: Sequence[str]) -> pd.DataFrame:
    """Return the BDBV release backtest's rule rows."""
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.nowcast import (
        resolve_truth_delay,
        rolling_truth,
    )

    config = bt.load_config()
    target = pd.read_csv("data/target.csv", dtype={"kind": str})
    real = target.loc[target["kind"] == "real"]
    truth = rolling_truth(real, resolve_truth_delay(config, target))
    truth = truth.astype({"target_end_date": str, "truth_vintage": str})
    parts = []
    for label in labels:
        for f in sorted((cache / "bdbv" / label).glob("*.csv.gz")):
            draws = pd.read_csv(
                f, dtype={"origin": str, "target_end_date": str}
            )
            parts.append(_tag(cell_scores(draws, truth), label))
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
    labels = [REFERENCE, *hashes]
    new = pd.concat(
        [
            germany_rows(cache, labels),
            bdbv_rows(cache, labels),
            *(held_out_rows(h, m[0]) for h, m in meta.items()),
        ],
        ignore_index=True,
    )
    new["outbreaks"] = new["checkpoint"].map({h: m[1] for h, m in meta.items()})
    new["outbreaks"] = new["outbreaks"].fillna(0).astype(int)
    new["grid"] = new["checkpoint"].map({h: m[2] for h, m in meta.items()})
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
    if study == "germany":
        germany_unit(cache, argv[2])
    elif study == "bdbv":
        bdbv_unit(cache, argv[2])
    elif study == "heldout":
        held_out_unit(cache, argv[2])
    elif study == "table":
        new = table(cache, Path(argv[2]), Path(argv[3]), argv[4:])
        print(new.round(3).to_string(index=False), flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
