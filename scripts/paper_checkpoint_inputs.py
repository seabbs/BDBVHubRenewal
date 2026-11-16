"""Lay out native-quantile checkpoint scores for `checkpoint_curve.py`.

Subcommands:

- `releases GRID OUT`: write OUT, a `checkpoint,outbreaks` CSV of every
  published `chronos-ft-*` release whose settings.json has `grid` GRID,
  in order of outbreaks.
- `germany NATIVE BASELINE OUT`: OUT/zero-shot.csv.gz holds the baseline
  forecasts (BASELINE/<date>.csv.gz) and the zero-shot models' native
  quantiles (NATIVE/zero-shot/<date>.csv.gz); OUT/<checkpoint> links to
  NATIVE/<checkpoint>.
- `bdbv NATIVE CELLS OUT`: OUT/<label>/nowcast-scores.csv from
  NATIVE/<label>/quantiles.csv.gz, scored on the natural and log(x + 1)
  scales against the complete-target truth of CELLS (the release
  backtest's `nowcast-scores.csv`), whose `randomwalknowcast` rows join
  the zero-shot label.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.scoring import QUANTILE_LEVELS, score_quantiles

REPO = "seabbs/BDBVHubRenewal"
PREFIX = "chronos-ft-"
CELL = ["origin", "truth_vintage", "location", "target_end_date"]
BASELINE = "randomwalknowcast"


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def releases(grid: str, out: Path) -> None:
    """List the published checkpoints of one fine-tuning grid."""
    tags = json.loads(
        _gh(
            "release",
            "list",
            "-R",
            REPO,
            "--limit",
            "200",
            "--json",
            "tagName,isDraft",
        )
    )
    rows = []
    for t in tags:
        tag = t["tagName"]
        if t["isDraft"] or not tag.startswith(PREFIX):
            continue
        settings = json.loads(
            _gh(
                "release",
                "download",
                tag,
                "-R",
                REPO,
                "-p",
                "settings.json",
                "-O",
                "-",
            )
        )
        if settings.get("grid") == grid:
            rows.append((tag.removeprefix(PREFIX), settings["n_outbreaks"]))
    table = pd.DataFrame(rows, columns=["checkpoint", "outbreaks"])
    # One checkpoint per size: the latest published wins on a rerun.
    table = table.drop_duplicates("outbreaks").sort_values("outbreaks")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))


def germany(native: Path, baseline: Path, out: Path) -> None:
    """Lay out the German forecasts."""
    out.mkdir(parents=True, exist_ok=True)
    files = [
        *sorted(baseline.glob("*.csv.gz")),
        *sorted((native / "zero-shot").glob("*.csv.gz")),
    ]
    zero = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    zero.to_csv(out / "zero-shot.csv.gz", index=False, compression="gzip")
    for d in native.iterdir():
        link = out / d.name
        if d.name != "zero-shot" and not link.exists():
            link.symlink_to(d.resolve())


def _scores(q: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    keys = ("model", *CELL)
    cols = [*keys, "horizon", "quantile", "value"]
    nat = score_quantiles(q[cols], truth, keys, QUANTILE_LEVELS)
    logq = q.assign(value=np.log1p(q["value"].clip(lower=0)))
    logt = truth.assign(truth=np.log1p(truth["truth"]))
    log = score_quantiles(logq[cols], logt, keys, QUANTILE_LEVELS)
    log = log[[*keys, "wis"]].rename(columns={"wis": "log_wis"})
    return nat.merge(log, on=list(keys)).assign(target="complete")


def bdbv(native: Path, cells: Path, out: Path) -> None:
    """Score each label's native quantiles on both scales."""
    sc = pd.read_csv(cells)
    sc = sc.loc[sc["target"] == "complete"]
    truth = sc[[*CELL, "truth"]].drop_duplicates()
    for d in sorted(native.iterdir()):
        q = pd.read_csv(d / "quantiles.csv.gz")
        rows = _scores(q, truth)
        if d.name == "zero-shot":
            rows = pd.concat(
                [rows, sc.loc[sc["model"] == BASELINE]], ignore_index=True
            )
        (out / d.name).mkdir(parents=True, exist_ok=True)
        rows.to_csv(out / d.name / "nowcast-scores.csv", index=False)
        print(d.name, len(rows))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"releases": releases, "germany": germany, "bdbv": bdbv}[cmd](
        *(
            a if i == 0 and cmd == "releases" else Path(a)
            for i, a in enumerate(args)
        )
    )
