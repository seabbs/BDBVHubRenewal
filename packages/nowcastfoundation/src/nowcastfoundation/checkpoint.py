# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Score each published Chronos-2 checkpoint.

Each checkpoint (release `chronos-ft-<hash>` or `chronos-slots-<hash>`)
is a label, run beside `reference` (randomwalknowcast and zero-shot
Chronos-2, ragged and complete). A label's units forecast and cache;
`table` scores the caches into rows of `evidence/checkpoint-scores.csv`
with `rule` = `RULE`:

- `germany`: the development half of the German benchmark
  (config/germany.toml), quantile WIS and log WIS on each model's own
  quantiles (`evaluation.germany.germany_forecasts`), by level.
- `held_out`: the trainer's held-out loss from the release's
  settings.json (unit `release`).

BDBV is held out for the paper, so no BDBV rows are scored here.

Relative scores divide by randomwalknowcast's on the cells every model
and checkpoint of a level made. Every score comes from
`evaluation.scoring`, as the paper's and the dashboard's do. Rows are
keyed by checkpoint hash and run, with the checkpoint's outbreaks and
grid.

Usage:
  python -m nowcastfoundation.checkpoint germany CACHE LABEL
  python -m nowcastfoundation.checkpoint release CACHE HASH
  python -m nowcastfoundation.checkpoint table CACHE EVIDENCE \\
    CHECKPOINTS HASH...
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nowcastfoundation.sidecar import SIDECAR, with_sidecar

RULE: Final = "own-quantile-wis_crps-1000-common"
"""The scoring rule of 2026-11-16."""
OLD_RULE: Final = "samples-100"
"""Rows written before the rule: WIS of 100 draws."""
BASELINE: Final = "randomwalknowcast"
REFERENCE: Final = "reference"
REFERENCE_PIN: Final = "e8a1f5ae338b"
"""The checkpoint the reference pins, which its models never load."""
ZERO_SHOT: Final = ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged")
FINETUNED: Final = tuple(f"{m}_ft" for m in ZERO_SHOT)
COMPLETE: Final = ("fm_chronos_daily_complete", "fm_chronos_weekly_complete")
"""Zero-shot Chronos-2 given complete data points only, a reference."""
GERMANY_UNIT: Final = "germany-days"
"""The cache folder of the German units, which score every one of the
hub's 29 target days."""
GERMANY_CHUNK: Final = 5
"""German dates forecast together (`germany_unit`) unless `SCORE_DATES`
in the environment (local config) says otherwise. Each date has the
tasks of 7 weekday offsets, so a chunk holds 7 times its dates' tasks:
small enough for `score.mem_mb`."""
FINAL_OUTBREAKS: Final = 8400
"""The reported model is the live run's final checkpoint."""
FINAL_GRID: Final = "r5"
CURRENT_RUNS: Final = frozenset({"one-model"})
"""Runs whose adapters this code loads (series bias, mixed layouts)."""
HELD_OUT_LEVELS: Final = ("leaf", "national", "intermediate")
GERMANY_CELL: Final = ["origin", "location", "target_end_date"]
COVERAGE: Final = (50, 90)
"""Central intervals scored for coverage: 50% and 90%."""
GERMANY_COVERAGE: Final = (50, 95)
"""The German rows' intervals: the hub's quantiles have no 0.05 or 0.95,
so there is no 90% interval and its rows keep the 95% one the hub
scored (`evaluation.germany`)."""
WIS_PARTS: Final = ("dispersion", "overprediction", "underprediction")
"""WIS's components (Bracher et al. 2021), which sum to it."""
REPO: Final = "seabbs/BDBVHubRenewal"
COLUMNS: Final = [
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
    "chosen",
    "target",
    "daily_rule",
    "layout",
    "window",
    "rel_to",
    "wis",
    "crps",
    "n_blocks",
    "horizon_weeks",
    "zone_method",
    "factor",
    "factor_level",
    "dispersion",
    "overprediction",
    "underprediction",
    "precision",
]
"""The evidence table's columns, those of rows of other studies too."""
SORT: Final = ["rule", "study", "level", "model", "outbreaks", "checkpoint"]


# Labels and checkpoints ------------------------------------------------


def label_models(label: str) -> tuple[str, ...]:
    """Return the models a label runs."""
    if label == REFERENCE:
        return (BASELINE, *ZERO_SHOT, *COMPLETE)
    return FINETUNED


def run_of(settings: Mapping[str, Any]) -> str:
    """Return a checkpoint's run from its settings.

    `one-model`: the series bias with mixed-frequency layouts (a source
    such as `daily+weekly`); `attn-bias`: earlier series-bias trials;
    `slots-zeroinit` and `slots`: the slots runs; `no-slots`.
    """
    if settings.get("series_bias"):
        mixed = any("+" in str(s) for s in settings.get("sources", []))
        return "one-model" if mixed else "attn-bias"
    slots = settings.get("series_slots")
    if not slots:
        return "no-slots"
    return "slots-zeroinit" if slots.get("init") == "zero" else "slots"


def outbreaks_and_grid(
    settings: Mapping[str, Any], release: Mapping[str, str]
) -> tuple[int, str]:
    """Return a checkpoint's training outbreaks and grid.

    From settings.json, else the release's title and body ("2,000
    simulated outbreaks", "round 4"), else 0 and `unknown`.
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


def chosen(rows: pd.DataFrame) -> pd.Series:
    """Mark the reported model's rows: the live run's final checkpoint."""
    return (
        rows["run"].astype(str).isin(CURRENT_RUNS)
        & (rows["grid"].astype(str) == FINAL_GRID)
        & (rows["outbreaks"] == FINAL_OUTBREAKS)
    )


def held_out_rows(checkpoint: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Return the trainer's held-out losses as rows."""
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
        target="7day",
    )
    out["loss"] = out["loss"].astype(float)
    return out.reindex(columns=COLUMNS)


def unit_done(root: Path, wanted: Sequence[str]) -> bool:
    """Whether a unit's folder holds a file for every date or origin."""
    have = {p.name.removesuffix(".csv.gz") for p in root.glob("*.csv.gz")}
    return set(wanted) <= have


# Scores ----------------------------------------------------------------


def tag(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """Name a label's rows by checkpoint, the `_ft` suffix dropped."""
    return rows.assign(
        checkpoint="zero-shot" if label == REFERENCE else label,
        model=rows["model"].str.removesuffix("_ft"),
    )


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

    With the mean WIS and its components, and coverage, where the rows
    have them.
    """
    widths = (*COVERAGE, *GERMANY_COVERAGE)
    extra = [
        "wis",
        *WIS_PARTS,
        *(f"coverage_{p}" for p in dict.fromkeys(widths)),
    ]
    extra = [c for c in extra if c in rows and c not in metrics]
    keys = ["level", "model", "checkpoint"]
    mean = rows.groupby(keys)[[*metrics, *extra]].mean()
    mean["n"] = rows.groupby(keys).size()
    origins = rows.groupby(keys)["origin"]
    mean["n_origins"] = origins.nunique()
    mean["first_origin"] = origins.min()
    mean["last_origin"] = origins.max()
    base = mean.xs((BASELINE, "zero-shot"), level=("model", "checkpoint"))
    at = mean.index.get_level_values("level")
    for m in metrics:
        mean[f"rel_{m}"] = mean[m] / base[m].reindex(at).to_numpy()
    out = mean.reset_index().assign(study=study, rule=RULE, target="7day")
    return out.reindex(columns=COLUMNS)


def merge_evidence(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add rule rows to the evidence, replacing the checkpoints' old ones.

    The rule's rows of checkpoints not in `new` are kept, as are rows of
    other rules.
    """
    old = old.reindex(columns=COLUMNS)
    old["rule"] = old["rule"].astype(object).fillna(OLD_RULE)
    mine = set(new["checkpoint"].astype(str))
    old = old.loc[
        (old["rule"] != RULE) | ~old["checkpoint"].astype(str).isin(mine)
    ]
    out = pd.concat([old, new.reindex(columns=COLUMNS)], ignore_index=True)
    out["target"] = out["target"].astype(object).fillna("7day")
    return out.sort_values([*SORT, "target"], kind="stable").reset_index(
        drop=True
    )


# Units -----------------------------------------------------------------


def score_dates() -> int:
    """Return the German dates a unit forecasts together.

    `SCORE_DATES` from the local config (`flowlib.machine.torch_env`),
    else `GERMANY_CHUNK`: more is faster and holds more in memory.
    """
    return int(os.environ.get("SCORE_DATES") or GERMANY_CHUNK)


PRECISION_FILE: Final = "precision.txt"


def record_precision(root: Path) -> None:
    """Write the precision a unit forecasts in to its folder."""
    from nowcastfoundation.backends.chronos2 import (
        inference_precision,
        torch_device,
    )

    Path(root).mkdir(parents=True, exist_ok=True)
    precision = inference_precision(torch_device())
    (Path(root) / PRECISION_FILE).write_text(precision + "\n")


def unit_precision(root: Path) -> str:
    """Return the precision a unit forecast in: fp32 before it was kept."""
    path = Path(root) / PRECISION_FILE
    return path.read_text().strip() if path.exists() else "fp32"


def _pin(label: str) -> None:
    """Pin the checkpoint, so no process resolves `latest` mid-run."""
    os.environ["BDBV_CHRONOS_FT"] = (
        REFERENCE_PIN if label == REFERENCE else label
    )


def _free() -> None:
    """Free JAX's compiled functions, which grow with each origin."""
    import jax

    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a gzipped CSV atomically, so a stopped run leaves no part."""
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)


def german_tasks_dir(
    cache: Path, triangle: Path, n_samples: int, seed: int
) -> Path:
    """Return the store of the German tasks every checkpoint scores.

    Keyed on what the tasks are built from: the triangle file's bytes,
    the task-building code (`data.germany` and the `data.dates` it uses),
    the fields of the `ForecastTask` they are pickled as, and the draws
    and seed.
    """
    import hashlib
    import inspect

    from bdbvhubrenewal.data import germany as dg
    from nowcastutils import base, dates

    h = hashlib.sha256()
    with Path(triangle).open("rb") as f:
        for part in iter(lambda: f.read(1 << 24), b""):
            h.update(part)
    for module in (dg, dates):
        h.update(inspect.getsource(module).encode())
    fields = [f.name for f in dataclasses.fields(base.ForecastTask)]
    h.update(repr(fields).encode())
    h.update(f"{n_samples}-{seed}".encode())
    return Path(cache) / GERMANY_UNIT / f"tasks-{h.hexdigest()[:12]}"


def germany_unit(cache: Path, label: str) -> None:
    """Nowcast each German development date, own quantiles with truth."""
    from bdbvhubrenewal.data.germany import hub_triangle
    from bdbvhubrenewal.evaluation import germany as de

    _pin(label)
    config = de.load_germany_config("config/germany.toml")
    path = Path("data/raw/germany") / Path(config.triangle).name
    triangle = hub_triangle(path)
    # Built once for every checkpoint, then read back.
    tasks_dir = german_tasks_dir(cache, path, config.n_samples, config.seed)
    root = cache / GERMANY_UNIT / label
    root.mkdir(parents=True, exist_ok=True)
    record_precision(root)
    dates = de.nowcast_dates(config, held_out=False)
    todo = [d for d in dates if not (root / f"{d}.csv.gz").exists()]
    models = list(label_models(label))
    # Dates are forecast `score_dates()` at a time, Chronos-2 predicted for
    # all of them in a few batched passes, and written one file a date so
    # a stopped run resumes.
    step = score_dates()
    for i in range(0, len(todo), step):
        chunk = todo[i : i + step]
        start = time.monotonic()
        fc = de.germany_forecasts(
            models, triangle, chunk, config.n_samples, config.seed, tasks_dir
        )
        fc = de.with_truth(fc, triangle, config.truth_delay)
        for date in chunk:
            mine = fc.loc[fc["origin"] == date].reset_index(drop=True)
            _write(mine, root / f"{date}.csv.gz")
        _free()
        took = time.monotonic() - start
        done = len(dates) - len(todo) + i + len(chunk)
        print(
            f"germany {label} {len(chunk)} dates to {chunk[-1]} "
            f"({done}/{len(dates)}): {took:.0f} s",
            flush=True,
        )
    print(f"germany {label} done", flush=True)


def release_unit(cache: Path, label: str) -> None:
    """Fetch a release's settings.json and its title and body."""
    from nowcastfoundation.forecaster import release_tag

    root = cache / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    tag_ = release_tag(label)
    out = str(root / f"{label}.json")
    subprocess.run(
        ["gh", "release", "download", tag_, "-R", REPO, "-p", "settings.json",
         "-O", out, "--clobber"],
        check=True,
    )  # fmt: skip
    view = subprocess.run(
        ["gh", "release", "view", tag_, "-R", REPO, "--json",
         "name,body,assets"],
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip
    release = json.loads(view.stdout)
    # Offloaded checkpoints keep their held-out losses in a sidecar.
    if SIDECAR in {a["name"] for a in release.pop("assets", [])}:
        sidecar = root / f"{label}-{SIDECAR}"
        subprocess.run(
            ["gh", "release", "download", tag_, "-R", REPO, "-p", SIDECAR,
             "-O", str(sidecar), "--clobber"],
            check=True,
        )  # fmt: skip
        settings = json.loads(Path(out).read_text())
        merged = with_sidecar(settings, json.loads(sidecar.read_text()))
        Path(out).write_text(json.dumps(merged, indent=1))
    (root / f"{label}-release.json").write_text(json.dumps(release))


# Table -----------------------------------------------------------------


def _cached(root: Path) -> pd.DataFrame | None:
    files = sorted(root.glob("*.csv.gz"))
    if not files:
        return None
    dtype = {"origin": str, "target_end_date": str}
    rows = pd.concat([pd.read_csv(f, dtype=dtype) for f in files])
    if "target" not in rows:
        return rows
    # Files of the earlier script also hold the daily target.
    keep = rows["target"].fillna("7day") == "7day"
    return rows.loc[keep].drop(columns="target")


def _with_precision(
    rows: pd.DataFrame, root: Path, labels: Sequence[str]
) -> pd.DataFrame:
    """Give each checkpoint's rows the precision its unit forecast in."""
    at = {
        ("zero-shot" if lb == REFERENCE else lb): unit_precision(root / lb)
        for lb in labels
    }
    return rows.assign(precision=rows["checkpoint"].map(at))


def germany_rows(cache: Path, labels: Sequence[str]) -> pd.DataFrame:
    """Return the German development half's rule rows."""
    from bdbvhubrenewal.evaluation import germany as de

    config = de.load_germany_config("config/germany.toml")
    parts = []
    for label in labels:
        fc = _cached(cache / GERMANY_UNIT / label)
        if fc is None:
            continue
        fc = fc.loc[fc["origin"] < config.split_date]
        nat = de.score_germany(fc, config.truth_delay)
        logged = fc.assign(
            value=np.log1p(fc["value"].clip(lower=0)),
            truth=np.log1p(fc["truth"]),
        )
        log = de.score_germany(logged, config.truth_delay)
        log = log[["model", *GERMANY_CELL, "wis"]].rename(
            columns={"wis": "log_wis"}
        )
        sc = nat.merge(log, on=["model", *GERMANY_CELL])
        parts.append(tag(sc, label))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    sc = pd.concat(parts, ignore_index=True).rename(columns={"pool": "level"})
    out = relative(matched(sc, GERMANY_CELL), "germany", ("wis", "log_wis"))
    return _with_precision(out, cache / GERMANY_UNIT, labels)


def _existing(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=list(columns))
    return pd.read_csv(path)


def table(
    cache: Path, evidence: Path, checkpoints: Path, hashes: Sequence[str]
) -> pd.DataFrame:
    """Score the caches of `hashes` and the reference into the tables.

    Rows of other rules, and of this rule for other checkpoints, are kept.
    """
    meta = {}
    for h in hashes:
        settings = json.loads((cache / "heldout" / f"{h}.json").read_text())
        release = json.loads(
            (cache / "heldout" / f"{h}-release.json").read_text()
        )
        meta[h] = (settings, *outbreaks_and_grid(settings, release))
    labels = [REFERENCE, *hashes]
    # BDBV is held out for the paper: no BDBV rows are computed, and
    # those already in `evidence` are kept.
    new = pd.concat(
        [
            germany_rows(cache, labels),
            *(held_out_rows(h, m[0]) for h, m in meta.items()),
        ],
        ignore_index=True,
    )
    new["outbreaks"] = new["checkpoint"].map({h: m[1] for h, m in meta.items()})
    new["outbreaks"] = new["outbreaks"].fillna(0).astype(int)
    new["grid"] = new["checkpoint"].map({h: m[2] for h, m in meta.items()})
    runs = {h: run_of(m[0]) for h, m in meta.items()}
    new["run"] = new["checkpoint"].map(runs).fillna("zero-shot")
    new["chosen"] = chosen(new)
    merged = merge_evidence(_existing(evidence, COLUMNS), new)
    merged.round(4).to_csv(evidence, index=False)
    ck = _existing(checkpoints, ["checkpoint", "outbreaks", "grid"])
    add = pd.DataFrame(
        [(h, m[1], m[2]) for h, m in meta.items()],
        columns=["checkpoint", "outbreaks", "grid"],
    )
    ck = pd.concat(
        [ck.loc[~ck["checkpoint"].astype(str).isin(add["checkpoint"])], add],
        ignore_index=True,
    )
    ck.to_csv(checkpoints, index=False)
    return new


def main(argv: Sequence[str]) -> None:
    """Run one unit, or rebuild the tables."""
    if len(argv) < 3:
        raise SystemExit(__doc__)
    study, cache = argv[0], Path(argv[1])
    if study == "germany":
        germany_unit(cache, argv[2])
    elif study == "release":
        release_unit(cache, argv[2])
    elif study == "table" and len(argv) >= 4:
        new = table(cache, Path(argv[2]), Path(argv[3]), argv[4:])
        print(new.round(3).to_string(index=False), flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
