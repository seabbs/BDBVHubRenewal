"""Load and score the prospective Ebola hub forecasts.

The BDBV Modeling Hub collects 100 sample trajectories of weekly confirmed
cases by week of symptom onset (`wk inc confirmed cases`) for each round,
named by its `reference_date` (a Saturday). Rounds close at 23:59
Kinshasa time (22:59 UTC) on the Tuesday four days before it.

Three sources give samples.

- `hub_samples`: every team's files in a clone of the hub, our
  submissions included. A file may hold several forecasts; the one with
  the latest `forecast_date` is kept.
- `candidate_samples`: files of our models' forecasts in the layout of
  `data/candidate-forecasts.csv.gz`.
- `store_samples`: the fit store branch. Each model and origin keeps the
  first fit stored, and only if it was made before its round closed.

`to_quantiles` turns samples into the seven hubverse quantile levels the
hub used for its quantile round and its dashboard. `load` joins the
latest release of the hub's target data as `truth`. `score` gives the
weighted interval score (WIS) of each forecast on the natural scale.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "COLUMNS",
    "QUANTILE_LEVELS",
    "TARGET",
    "candidate_samples",
    "hub_samples",
    "hub_truth",
    "load",
    "round_close",
    "score",
    "store_samples",
    "to_quantiles",
]

TARGET: Final = "wk inc confirmed cases"
"""The target of the hub's weekly round."""

QUANTILE_LEVELS: Final = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
"""The hubverse quantile levels of the hub's quantile round."""

COLUMNS: Final = (
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "value",
    "truth",
)
"""The columns of the long table every study shares."""

KEYS: Final = ["model", "location", "origin_date", "target_date"]
SAMPLE_COLUMNS: Final = [*KEYS, "sample", "value"]


def round_close(origin_date: pd.Timestamp) -> pd.Timestamp:
    """Return when a round closes: 22:59 UTC on the Tuesday before."""
    day = pd.Timestamp(origin_date).normalize() - pd.Timedelta(days=4)
    return (day + pd.Timedelta(hours=22, minutes=59)).tz_localize("UTC")


def _samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename a hub-layout frame to the sample columns."""
    out = frame.rename(
        columns={
            "reference_date": "origin_date",
            "target_end_date": "target_date",
            "output_type_id": "sample",
        }
    )
    for col in ("origin_date", "target_date"):
        out[col] = pd.to_datetime(out[col])
    out["value"] = out["value"].astype(float)
    return out[SAMPLE_COLUMNS].reset_index(drop=True)


def hub_samples(hub: str | Path) -> pd.DataFrame:
    """Read every team's weekly-round samples from a clone of the hub."""
    frames = []
    for path in sorted(Path(hub, "model-output").glob("*/*.csv")):
        frame = pd.read_csv(path)
        if "target" not in frame or "output_type" not in frame:
            continue
        frame = frame[
            (frame["target"] == TARGET) & (frame["output_type"] == "sample")
        ]
        if frame.empty:
            continue
        if "forecast_date" in frame:
            latest = frame["forecast_date"].max()
            frame = frame[frame["forecast_date"] == latest]
        frames.append(frame.assign(model=path.parent.name))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return _samples(pd.concat(frames, ignore_index=True))


def candidate_samples(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read our models' samples from files of stored forecasts."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    if "kind" in frame:
        frame = frame[frame["kind"] == "real"]
    return _samples(frame)


def _git(repo: str | Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def store_samples(
    repo: str | Path, ref: str = "origin/fit-store"
) -> pd.DataFrame:
    """Read the prospective fits from the fit store's history.

    Every fit file ever committed to `ref` is read. For each model and
    origin the earliest good fit is kept, if it was made before the
    round it forecasts closed.
    """
    log = _git(
        repo,
        "log",
        "--reverse",
        "--format=commit %H",
        "--name-only",
        "--diff-filter=A",
        ref,
    ).decode()
    added: dict[str, str] = {}
    commit = ""
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.removeprefix("commit ")
        elif line.endswith(".json.gz"):
            added.setdefault(line, commit)
    first: dict[tuple[str, str], tuple[str, pd.DataFrame]] = {}
    for path, commit in added.items():
        fit = json.loads(
            gzip.decompress(_git(repo, "show", f"{commit}:{path}"))
        )
        meta = fit["meta"]
        if meta["status"] != "ok":
            continue
        key = (meta["model"], meta["origin"])
        if key in first and first[key][0] <= meta["fitted_at"]:
            continue
        table = fit["forecasts"]
        frame = pd.DataFrame(table["data"], columns=table["columns"])
        frame = frame[frame["kind"] == "real"]
        if frame.empty:
            continue
        first[key] = (meta["fitted_at"], _samples(frame))
    frames = []
    for fitted_at, out in first.values():
        close = round_close(out["origin_date"].iloc[0])
        if pd.Timestamp(fitted_at) <= close:
            frames.append(out)
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def hub_truth(hub: str | Path) -> pd.DataFrame:
    """Return the latest release of the hub's target data."""
    data = pd.read_csv(Path(hub, "target-data", "time-series.csv"))
    data = data[data["target"] == TARGET]
    data = data[data["as_of"] == data["as_of"].max()]
    return pd.DataFrame(
        {
            "location": data["location"],
            "target_date": pd.to_datetime(data["target_end_date"]),
            "truth": data["observation"].astype(float),
            "as_of": pd.to_datetime(data["as_of"]),
        }
    ).reset_index(drop=True)


def to_quantiles(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the quantiles of each forecast's samples."""
    levels = np.asarray(QUANTILE_LEVELS)
    rows = []
    for key, group in samples.groupby(KEYS, sort=True):
        values = np.quantile(group["value"].to_numpy(float), levels)
        rows.append(
            pd.DataFrame(
                dict(zip(KEYS, key, strict=True)), index=levels
            ).assign(quantile_level=levels, value=values)
        )
    if not rows:
        return pd.DataFrame(columns=[*KEYS, "quantile_level", "value"])
    return pd.concat(rows, ignore_index=True)


def load(
    hub: str | Path,
    candidates: Iterable[str | Path] = (),
    store: str | Path | None = None,
    store_ref: str = "origin/fit-store",
) -> pd.DataFrame:
    """Return every forecast as quantiles with the latest truth.

    `hub` is a clone of the hub. `candidates` are files of our models'
    stored forecasts and `store` a clone holding the fit store branch
    `store_ref`. A model and round in more than one source is taken from
    the first of hub, candidates and store.
    """
    parts = [hub_samples(hub), candidate_samples(candidates)]
    if store is not None:
        parts.append(store_samples(store, store_ref))
    seen: set[tuple[str, pd.Timestamp]] = set()
    kept = []
    for part in parts:
        pairs = set(zip(part["model"], part["origin_date"], strict=True))
        mask = [
            (m, o) not in seen
            for m, o in zip(part["model"], part["origin_date"], strict=True)
        ]
        kept.append(part[mask])
        seen |= pairs
    samples = pd.concat([k for k in kept if not k.empty], ignore_index=True)
    table = to_quantiles(samples)
    truth = hub_truth(hub).drop(columns="as_of")
    table = table.merge(truth, on=["location", "target_date"], how="left")
    table["horizon"] = (
        (table["target_date"] - table["origin_date"]).dt.days // 7
    ).astype(int)
    table["study"] = "ebola"
    return table[list(COLUMNS)].reset_index(drop=True)


def score(table: pd.DataFrame) -> pd.DataFrame:
    """Score each forecast with truth by WIS, its median and coverage.

    WIS follows the hubverse (scoringutils) definition: the absolute
    error of the median weighted by a half plus each central interval's
    score weighted by alpha over two, divided by the number of intervals
    plus a half. Coverage is of the 50%, 80% and 95% intervals.
    """
    keys = [c for c in (*KEYS, "horizon") if c in table]
    table = table.dropna(subset=["truth"])
    rows = []
    for key, group in table.groupby(keys, sort=True):
        q = dict(zip(group["quantile_level"], group["value"], strict=True))
        y = group["truth"].iloc[0]
        median = q[0.5]
        total = 0.5 * abs(y - median)
        cover = {}
        lower_levels = [lv for lv in QUANTILE_LEVELS if lv < 0.5]
        for lower in lower_levels:
            alpha = 2 * lower
            lo, hi = q[lower], q[round(1 - lower, 3)]
            interval = (hi - lo) + 2 / alpha * (max(lo - y, 0) + max(y - hi, 0))
            total += alpha / 2 * interval
            cover[f"cover_{round(100 * (1 - alpha))}"] = lo <= y <= hi
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "truth": y,
                "wis": total / (len(lower_levels) + 0.5),
                "ae_median": abs(y - median),
                **cover,
            }
        )
    columns = [
        *keys,
        "truth",
        "wis",
        "ae_median",
        "cover_95",
        "cover_80",
        "cover_50",
    ]
    return pd.DataFrame(rows, columns=columns)
