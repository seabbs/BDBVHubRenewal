"""Load and score the prospective Ebola hub forecasts.

The BDBV Modeling Hub collects 100 sample trajectories of weekly confirmed
cases by week of symptom onset (`wk inc confirmed cases`) for each round,
named by its `reference_date` (a Saturday). Rounds close at 23:59
Kinshasa time (22:59 UTC) on the Tuesday four days before it.

Three sources give samples.

- `hub_samples`: every team's files in a clone of the hub, our
  submissions and the hub's baseline (`HUB_BASELINE`) included. A file
  may hold several forecasts; the one with the latest `forecast_date` is
  kept.
- `candidate_samples`: files of our models' forecasts in the layout of
  `data/candidate-forecasts.csv.gz`.
- `store_samples`: the fit store branch. Each model and origin keeps the
  first fit stored, and only if it was made before its round closed.

`observations` gives the hub's counts as of each release.
`reporting_delay` estimates D, the age by which 98% of the national
count is reported, from the hub's release history. `load` gives every
forecast's samples, one row per sample, in the paper's schema with
`truth` left for the workflow to attach.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "COLUMNS",
    "DELAY_LEVEL",
    "HUB_BASELINE",
    "TARGET",
    "candidate_samples",
    "hub_samples",
    "load",
    "observations",
    "reporting_delay",
    "round_close",
    "store_samples",
]

TARGET: Final = "wk inc confirmed cases"
"""The target of the hub's weekly round."""

DELAY_LEVEL: Final = 0.98
"""Share of the national count reported by the truth delay, D."""

SKIP_RELEASES: Final = ("2026-08-18",)
"""Releases left out of the delay estimate.

The 2026-08-18 release has no delay file, is 50 days before the next and
its old weeks are short by up to 12% of their latest counts, which is a
revision rather than a reporting delay (as `config/backtest.toml`).
"""

COLUMNS: Final = (
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "sample_id",
    "value",
    "truth",
)
"""The columns of the long table every study shares."""

HUB_BASELINE: Final = "BDBVHub-baseline"
"""The hub's own baseline: an additive random walk from horizon -3."""

KEYS: Final = ["model", "location", "origin_date", "target_date"]
SAMPLE_COLUMNS: Final = [*KEYS, "sample_id", "value"]


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
            "output_type_id": "sample_id",
            "sample": "sample_id",
        }
    )
    for col in ("origin_date", "target_date"):
        out[col] = pd.to_datetime(out[col])
    out["value"] = out["value"].astype(float)
    out["sample_id"] = out["sample_id"].astype(int)
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


def _target(hub: str | Path) -> pd.DataFrame:
    data = pd.read_csv(Path(hub, "target-data", "time-series.csv"))
    data = data[data["target"] == TARGET]
    return data.assign(
        as_of=pd.to_datetime(data["as_of"]),
        target_date=pd.to_datetime(data["target_end_date"]),
    )


def observations(hub: str | Path) -> pd.DataFrame:
    """Return the hub's counts as of each release, in the paper's schema."""
    data = _target(hub)
    return pd.DataFrame(
        {
            "study": "ebola",
            "location": data["location"],
            "target_date": data["target_date"],
            "as_of_date": data["as_of"],
            "value": data["observation"].astype(float),
        }
    ).reset_index(drop=True)


def reporting_delay(
    hub: str | Path,
    location: str = "CD",
    level: float = DELAY_LEVEL,
    skip: Sequence[str] = SKIP_RELEASES,
) -> int | None:
    """Return the age in days by which `level` of the count is reported.

    Each earlier release's count of a week is set against the latest
    release's count of that week, at the week's age in the release
    (days from the week's end to the release, rounded up to whole weeks).
    The counts are pooled over releases at each age. The delay is the
    first age whose pooled share reaches `level`, or None if none does.
    """
    data = _target(hub)
    data = data[data["location"] == location]
    latest = data["as_of"].max()
    final = data.loc[data["as_of"] == latest].set_index("target_date")[
        "observation"
    ]
    early = data[
        (data["as_of"] < latest)
        & ~data["as_of"].isin(pd.to_datetime(list(skip)))
    ]
    early = early.assign(final=early["target_date"].map(final))
    # Weeks the latest release no longer has cannot be compared.
    early = early.dropna(subset=["final"])
    days = (early["as_of"] - early["target_date"]).dt.days
    early = early.assign(age=np.ceil(days / 7).astype(int))
    pooled = early.groupby("age")[["observation", "final"]].sum()
    share = pooled["observation"] / pooled["final"]
    reached = share.index[share >= level]
    return int(reached.min()) * 7 if len(reached) else None


def load(
    hub: str | Path,
    candidates: Iterable[str | Path] = (),
    store: str | Path | None = None,
    store_ref: str = "origin/fit-store",
) -> pd.DataFrame:
    """Return every forecast's samples, with `truth` left empty.

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
        pairs = list(zip(part["model"], part["origin_date"], strict=True))
        kept.append(part[[p not in seen for p in pairs]])
        seen |= set(pairs)
    table = pd.concat([k for k in kept if not k.empty], ignore_index=True)
    table["truth"] = np.nan
    table["horizon"] = (
        (table["target_date"] - table["origin_date"]).dt.days // 7
    ).astype(int)
    table["study"] = "ebola"
    table["quantile_level"] = np.nan
    return table[list(COLUMNS)].reset_index(drop=True)
