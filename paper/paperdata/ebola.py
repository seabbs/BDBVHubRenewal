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

`rolling_truth` gives each target week's count as reported `delay_days`
after the week ended, as the German nowcast hub did (Wolffram et al.
2023). Weeks whose release that far after them is not out yet have no
truth. `load` joins it to the samples as `truth`, one row per sample.
`score` scores the samples with the repo's sample scores
(`bdbvhubrenewal.evaluation.scoring`): CRPS on the natural and `log1p`
scales, bias, and 50% and 90% coverage. `relative_scores` divides each
model's mean CRPS by the hub's baseline's on the forecasts every compared
model made.
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
from bdbvhubrenewal.evaluation.scoring import (
    bias_sample,
    coverage_sample,
    crps_sample,
    relative_skill,
)

__all__ = [
    "COLUMNS",
    "DELAY_DAYS",
    "HUB_BASELINE",
    "SCORES",
    "TARGET",
    "candidate_samples",
    "hub_samples",
    "load",
    "relative_scores",
    "rolling_truth",
    "round_close",
    "score",
    "store_samples",
]

TARGET: Final = "wk inc confirmed cases"
"""The target of the hub's weekly round."""

DELAY_DAYS: Final = 28
"""Days after a target week ends at which its count is taken as truth."""

COLUMNS: Final = (
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "sample_id",
    "value",
    "truth",
)
"""The columns of the long table every study shares."""

HUB_BASELINE: Final = "BDBVHub-baseline"
"""The hub's own baseline: an additive random walk from horizon -3."""

KEYS: Final = ["model", "location", "origin_date", "target_date"]
SAMPLE_COLUMNS: Final = [*KEYS, "sample_id", "value"]
SCORES: Final = ("crps", "log_crps", "bias", "coverage_50", "coverage_90")
"""The scores `score` gives for each forecast."""


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


def rolling_truth(
    hub: str | Path, delay_days: int = DELAY_DAYS
) -> pd.DataFrame:
    """Return each target week's count as reported `delay_days` after it.

    The count is from the latest release on or before the week's end plus
    `delay_days`. A week is left out until a release on or after that
    day exists, so its truth no longer changes.
    """
    data = pd.read_csv(Path(hub, "target-data", "time-series.csv"))
    data = data[data["target"] == TARGET]
    as_of = pd.to_datetime(data["as_of"])
    week = pd.to_datetime(data["target_end_date"])
    due = week + pd.Timedelta(days=delay_days)
    data = data.assign(as_of=as_of, target_date=week, due=due)
    data = data[(data["as_of"] <= data["due"]) & (data["due"] <= as_of.max())]
    data = (
        data.sort_values("as_of")
        .groupby(["location", "target_date"], as_index=False)
        .last()
    )
    return pd.DataFrame(
        {
            "location": data["location"],
            "target_date": data["target_date"],
            "truth": data["observation"].astype(float),
        }
    ).reset_index(drop=True)


def load(
    hub: str | Path,
    candidates: Iterable[str | Path] = (),
    store: str | Path | None = None,
    store_ref: str = "origin/fit-store",
    delay_days: int = DELAY_DAYS,
) -> pd.DataFrame:
    """Return every forecast's samples with the rolling truth.

    `hub` is a clone of the hub. `candidates` are files of our models'
    stored forecasts and `store` a clone holding the fit store branch
    `store_ref`. A model and round in more than one source is taken from
    the first of hub, candidates and store. `truth` is missing where the
    release `delay_days` after the week is not out yet.
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
    truth = rolling_truth(hub, delay_days)
    table = table.merge(truth, on=["location", "target_date"], how="left")
    table["horizon"] = (
        (table["target_date"] - table["origin_date"]).dt.days // 7
    ).astype(int)
    table["study"] = "ebola"
    return table[list(COLUMNS)].reset_index(drop=True)


def score(table: pd.DataFrame) -> pd.DataFrame:
    """Score each forecast with truth from its samples.

    Returns one row per forecast: its keys, `horizon`, `truth`,
    `n_samples` and `SCORES`. `log_crps` is the CRPS of `log1p` of the
    samples against `log1p` of the truth. Bias is positive for
    overprediction.
    """
    keys = [*KEYS, "horizon"]
    table = table.dropna(subset=["truth"])
    rows = []
    for key, group in table.groupby(keys, sort=True):
        y = float(group["truth"].iloc[0])
        x = group["value"].to_numpy(float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "truth": y,
                "n_samples": x.size,
                "crps": crps_sample(y, x),
                "log_crps": crps_sample(float(np.log1p(y)), np.log1p(x)),
                "bias": bias_sample(y, x),
                "coverage_50": coverage_sample(y, x, 0.5),
                "coverage_90": coverage_sample(y, x, 0.9),
            }
        )
    return pd.DataFrame(rows, columns=[*keys, "truth", "n_samples", *SCORES])


def relative_scores(
    scores: pd.DataFrame,
    baseline: str = HUB_BASELINE,
    by: Sequence[str] = (),
) -> pd.DataFrame:
    """Return each model's mean CRPS over the baseline's.

    Only forecasts (location, round and target week) that every model in
    `scores` made are compared, so pass the models to compare. The
    baseline has 1. Also gives `log_crps`, bias and coverage means.
    """
    match = ["location", "origin_date", "target_date"]
    n_models = scores["model"].nunique()
    shared = (
        scores.groupby(match)["model"].nunique().loc[lambda n: n == n_models]
    )
    scores = scores.merge(shared.index.to_frame(index=False), on=match)
    return relative_skill(
        scores,
        baseline=baseline,
        by=by,
        metrics=("crps", "log_crps"),
        match_on=match,
        means=("crps", "log_crps", "bias", "coverage_50", "coverage_90"),
    )
