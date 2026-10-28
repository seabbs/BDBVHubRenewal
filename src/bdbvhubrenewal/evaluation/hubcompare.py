# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Compare our forecasts with the forecasts submitted to the hub.

`fetch_hub_submissions` downloads every team's metadata and its sample
forecasts for the weekly round (`wk inc confirmed cases`).
Files whose licence allows redistribution (`OPEN_LICENCES`) are kept
gzipped in a committed directory; the rest go to an uncommitted cache.
Files from the archived cumulative outbreak size round are skipped.

For each hub round, `our_forecasts` runs our unfitted models from the
matching vintage: the latest real vintage that serves the round if there
is one, otherwise a vintage reconstructed for the Saturday before it.
Fitted models are slow, so `stored_forecasts` takes theirs from backtest
and held-out forecast files for the same round.
The `source` column says which (`real`, `reconstructed`, or `hub`).

`score_comparison` scores every model on every week it forecast, against
the latest real vintage as reported then, and flags complete weeks: those
ending at least $K$ weeks before the truth vintage.
A round is pending until a real vintage later than its matching vintage
exists, as before then the truth is the data the forecasts were made from.
`summarise_comparison` reports complete weeks only, with WIS relative to
a baseline on the weeks both forecast.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml

from bdbvhubrenewal.data.dates import as_date, reference_date_for_vintage
from bdbvhubrenewal.data.reconstruct import (
    reconstruct_delays,
    reconstruct_vintages,
)
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.scoring import (
    relative_skill,
    score_forecasts,
    summarise_scores,
)
from bdbvhubrenewal.forecasters import get_model, list_models

__all__ = [
    "ATTRIBUTION",
    "OPEN_LICENCES",
    "QUANTILES",
    "WEEKLY_TARGET",
    "ComparisonConfig",
    "attribution",
    "fetch_hub_submissions",
    "forecast_quantiles",
    "load_comparison_config",
    "load_hub_submissions",
    "load_team_metadata",
    "match_vintage",
    "our_forecasts",
    "own_hub_ids",
    "resolve_baseline",
    "score_comparison",
    "stored_forecasts",
    "submission_dir",
    "summarise_comparison",
]

WEEKLY_TARGET: Final = "wk inc confirmed cases"
"""The target of the hub's current round."""

OPEN_LICENCES: Final = ("CC0-1.0", "CC-BY-4.0")
"""Licences under which a team's files may be committed here."""

QUANTILES: Final = {
    "q05": 0.05,
    "q25": 0.25,
    "q50": 0.5,
    "q75": 0.75,
    "q95": 0.95,
}
"""Quantiles kept for the plots."""

SAMPLE_COLUMNS: Final = (
    "model",
    "source",
    "reference_date",
    "location",
    "target_end_date",
    "sample",
    "value",
)

ATTRIBUTION: Final = (
    "Surveillance data: Institut National de Santé Publique (INSP) and the "
    "Incident Management Support Team (IMST), Democratic Republic of the "
    "Congo. Processed by the Atlantic Coast Center for Infectious Disease "
    "Dynamics and Analytics (ACCIDDA) and the Kinshasa School of Public "
    "Health, University of Kinshasa, and distributed through the BDBV "
    "Modeling Hub, https://github.com/InsightNet-US/BDBV-Modeling-Hub, "
    "accessed {accessed}."
)
"""The hub's required data credit ("Citing these data" in its README)."""


@dataclass(frozen=True)
class ComparisonConfig:
    """Settings for the hub comparison, from `config/hub-comparison.toml`.

    Attributes
    ----------
    baseline : str
        Model that WIS is relative to, if registered
        (`resolve_baseline`).
    stored : tuple of str
        Forecast files to take fitted models' forecasts from.
    locations : tuple of str
        Locations shown in the plots.
    ref : str
        Git ref of the hub to fetch from.
    """

    baseline: str = "persistencenow"
    stored: tuple[str, ...] = ()
    locations: tuple[str, ...] = ("CD",)
    ref: str = "main"


def load_comparison_config(
    path: Path | str = "config/hub-comparison.toml",
) -> ComparisonConfig:
    """Read the hub comparison settings.

    Parameters
    ----------
    path : pathlib.Path or str
        The TOML file.

    Returns
    -------
    ComparisonConfig
        The settings.
    """
    raw = tomllib.loads(Path(path).read_text()).get("comparison", {})
    default = ComparisonConfig()
    return ComparisonConfig(
        baseline=str(raw.get("baseline", default.baseline)),
        stored=tuple(raw.get("stored", default.stored)),
        locations=tuple(raw.get("locations", default.locations)),
        ref=str(raw.get("ref", default.ref)),
    )


def resolve_baseline(preferred: str) -> str:
    """Return the baseline for relative scores.

    Parameters
    ----------
    preferred : str
        The configured baseline.

    Returns
    -------
    str
        `preferred` if it is registered, else the registry baseline.
    """
    specs = list_models()
    if any(s.name == preferred for s in specs):
        return preferred
    return next(s.name for s in specs if s.baseline)


def own_hub_ids(path: Path | str = "config/hub.toml") -> dict[str, str]:
    """Return the hub ids of our models that submit under their own id.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub submission settings, with a `[models.<name>]` table per
        model that has its own id.

    Returns
    -------
    dict
        Model name to hub model id, such as
        `{"persistencenow": "seabbs-baseline"}`.
    """
    from bdbvhubrenewal.submission.hub import model_abbr_for, model_id

    raw = tomllib.loads(Path(path).read_text())
    return {
        name: model_id(model_abbr_for(name, path))
        for name in raw.get("models", {})
    }


def attribution(accessed: str) -> str:
    """Return the hub's data credit with the access date filled in.

    Parameters
    ----------
    accessed : str
        ISO date the data were accessed.

    Returns
    -------
    str
        The credit text.
    """
    return ATTRIBUTION.format(accessed=accessed)


def submission_dir(licence: str, keep: Path, cache: Path) -> Path:
    """Return where a team's files go, by their licence.

    Parameters
    ----------
    licence : str
        The licence in the team's metadata.
    keep : pathlib.Path
        The committed directory.
    cache : pathlib.Path
        The uncommitted cache.

    Returns
    -------
    pathlib.Path
        `keep` for `OPEN_LICENCES`, else `cache`.
    """
    return keep if licence in OPEN_LICENCES else cache


def fetch_hub_submissions(  # pragma: no cover - network
    dest: Path | str,
    cache: Path | str,
    ref: str = "main",
    token: str | None = None,
) -> dict[str, object]:
    """Download every team's metadata and weekly-round forecasts.

    Parameters
    ----------
    dest : pathlib.Path or str
        Committed directory, for metadata and openly licensed files.
    cache : pathlib.Path or str
        Uncommitted directory, for other files.
    ref : str
        Git ref of the hub.
    token : str, optional
        GitHub token. Defaults to `GITHUB_TOKEN` if set.

    Returns
    -------
    dict
        A manifest with the commit, the access date and each kept file,
        also written to `dest/manifest.json`.
    """
    from bdbvhubrenewal.data.tidy import HUB_REPO, _api, _raw

    token = token or os.environ.get("GITHUB_TOKEN")
    keep, other = Path(dest), Path(cache)
    head = _api(f"repos/{HUB_REPO}/commits/{ref}", token)["sha"]
    tree = _api(f"repos/{HUB_REPO}/git/trees/{head}?recursive=1", token)
    paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"]
    licences = {}
    for path in paths:
        parts = path.split("/")
        if parts[0] == "model-metadata" and path.endswith(".yml"):
            text = _raw(head, path, token)
            out = keep / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(text)
            meta = yaml.safe_load(text) or {}
            licences[parts[1].removesuffix(".yml")] = str(
                meta.get("license", "")
            )
    files = []
    for path in paths:
        parts = path.split("/")
        if parts[0] != "model-output" or len(parts) != 3:
            continue
        if not parts[2].endswith(".csv"):
            continue
        content = _raw(head, path, token)
        head_rows = pd.read_csv(io.BytesIO(content), nrows=5)
        if (
            "target" not in head_rows
            or not (head_rows["target"] == WEEKLY_TARGET).all()
        ):
            continue
        root = submission_dir(licences.get(parts[1], ""), keep, other)
        out = root / f"{path}.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.compress(content, mtime=0))
        files.append({"path": path, "committed": root == keep})
    manifest: dict[str, object] = {
        "repo": HUB_REPO,
        "ref": ref,
        "commit": head,
        "accessed": dt.date.today().isoformat(),
        "files": files,
    }
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_hub_submissions(*roots: Path | str) -> pd.DataFrame:
    """Read the weekly-round sample forecasts in some directories.

    Parameters
    ----------
    *roots : pathlib.Path or str
        Directories holding `model-output/<model>/*.csv` or `*.csv.gz`.
        Missing ones are skipped.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `source` (`hub`), `reference_date`, `location`,
        `target_end_date`, `sample` and `value`.
    """
    frames = []
    for root in roots:
        for path in sorted(Path(root).glob("model-output/*/*.csv*")):
            table = pd.read_csv(path, dtype={"output_type_id": str})
            rows = table.loc[
                (table["target"] == WEEKLY_TARGET)
                & (table["output_type"] == "sample")
            ]
            if rows.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "model": path.parent.name,
                        "source": "hub",
                        "reference_date": rows["reference_date"].astype(str),
                        "location": rows["location"].astype(str),
                        "target_end_date": rows["target_end_date"].astype(str),
                        "sample": pd.to_numeric(rows["output_type_id"]).astype(
                            "int64"
                        ),
                        "value": rows["value"].astype(float),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def load_team_metadata(root: Path | str) -> pd.DataFrame:
    """Read the team, model name and licence of each downloaded model.

    Parameters
    ----------
    root : pathlib.Path or str
        Directory holding `model-metadata/*.yml`.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `team_name`, `model_name`, `license` and
        `website_url`, sorted by model.
    """
    rows = []
    for path in sorted(Path(root).glob("model-metadata/*.yml")):
        meta = yaml.safe_load(path.read_text()) or {}
        rows.append(
            {
                "model": path.stem,
                "team_name": meta.get("team_name", ""),
                "model_name": meta.get("model_name", ""),
                "license": meta.get("license", ""),
                "website_url": meta.get("website_url", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["model", "team_name", "model_name", "license", "website_url"],
    )


def match_vintage(
    reference_date: dt.date | str, real_vintages: Sequence[str]
) -> tuple[str, str] | None:
    """Return the vintage our forecast for a hub round is made from.

    Parameters
    ----------
    reference_date : datetime.date or str
        The hub round.
    real_vintages : sequence of str
        Real vintages available.

    Returns
    -------
    tuple of str or None
        The vintage and its kind: the latest real vintage serving the
        round (`real`), else the Saturday a week before the reference date
        (`reconstructed`). None if that Saturday is after the latest real
        vintage, so it cannot be reconstructed.
    """
    ref = as_date(reference_date)
    serving = [v for v in real_vintages if reference_date_for_vintage(v) == ref]
    if serving:
        return max(serving), "real"
    day = ref - dt.timedelta(days=7)
    if not real_vintages or day > as_date(max(real_vintages)):
        return None
    return day.isoformat(), "reconstructed"


def our_forecasts(
    reference_dates: Sequence[str],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    models: Sequence[str],
    min_horizon: int | None = None,
) -> pd.DataFrame:
    """Run our models for hub rounds from the matching vintages.

    Parameters
    ----------
    reference_dates : sequence of str
        Hub reference dates.
    target, delays : pandas.DataFrame
        Tidy tables.
    config : BacktestConfig
        Horizons, samples and seed.
    models : sequence of str
        Registered model names.
    min_horizon : int, optional
        First week, in weeks from the reference date, to cover the weeks
        the hub forecasts did. Defaults to `config.min_horizon`.

    Returns
    -------
    pandas.DataFrame
        Output of `run_forecasts` with a `source` column (`real` or
        `reconstructed`). Rounds with no matching vintage are skipped.
    """
    if min_horizon is not None:
        config = replace(config, min_horizon=min_horizon)
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    built = [get_model(n) for n in models]
    frames = []
    for ref in sorted(set(reference_dates)):
        match = match_vintage(ref, vintages)
        if match is None or not built:
            continue
        vintage, source = match
        if source == "real":
            rows = real.loc[real["vintage"] == vintage]
            dly = delays
        else:
            rows = reconstruct_vintages(real, delays, [vintage])
            src = delays.loc[delays["vintage"] == delays["vintage"].max()]
            dly = reconstruct_delays(src, vintage).assign(kind="reconstructed")
        fc = run_forecasts(built, make_tasks(rows, dly, config))
        fc["source"] = source
        frames.append(fc)
    if not frames:
        return pd.DataFrame(columns=[*SAMPLE_COLUMNS, "horizon"])
    return pd.concat(frames, ignore_index=True)


def stored_forecasts(
    paths: Sequence[Path | str],
    reference_dates: Sequence[str],
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Take forecasts for hub rounds from stored forecast files.

    Parameters
    ----------
    paths : sequence of pathlib.Path or str
        Outputs of `run_forecasts`, such as the backtest and held-out
        forecasts. Missing files are skipped.
    reference_dates : sequence of str
        Hub reference dates.
    exclude : sequence of str
        Models not to take, such as those run live.

    Returns
    -------
    pandas.DataFrame
        Columns `SAMPLE_COLUMNS`, with `source` the vintage kind. For each
        model and round, real vintages win over reconstructed ones.
    """
    frames = []
    for path in paths:
        try:
            fc = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        fc = fc.loc[
            fc["reference_date"].astype(str).isin(list(reference_dates))
            & ~fc["model"].isin(list(exclude))
            & fc["kind"].isin(["real", "reconstructed"])
        ]
        frames.append(fc)
    if not frames:
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    fc = pd.concat(frames, ignore_index=True).drop_duplicates()
    best = fc.groupby(["model", "reference_date"])["kind"].transform(
        lambda k: "real" if (k == "real").any() else "reconstructed"
    )
    fc = fc.loc[fc["kind"] == best]
    return fc.rename(columns={"kind": "source"})[list(SAMPLE_COLUMNS)]


def score_comparison(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    complete_after_weeks: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model on the weeks it forecast, against the latest data.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Hub and our samples with `SAMPLE_COLUMNS`.
    target : pandas.DataFrame
        Target rows with every real vintage.
    complete_after_weeks : int
        $K$: a week is complete if it ended at least this many weeks
        before the truth vintage.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores (`score_forecasts` output with `source`, `vintage`,
        `truth_vintage` and `complete`), and the pending rounds with
        columns `reference_date`, `vintage` and `models`.
    """
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    latest = vintages[-1]
    truth = real.loc[
        real["vintage"] == latest, ["location", "target_end_date", "total"]
    ].rename(columns={"total": "truth"})
    cutoff = as_date(latest) - dt.timedelta(weeks=complete_after_weeks)
    scored, pending = [], []
    for ref, grp in forecasts[list(SAMPLE_COLUMNS)].groupby("reference_date"):
        match = match_vintage(str(ref), vintages)
        if match is None or match[0] >= latest:
            pending.append(
                {
                    "reference_date": str(ref),
                    "vintage": match[0] if match else None,
                    "models": ", ".join(sorted(set(grp["model"]))),
                }
            )
            continue
        grp = grp.assign(
            horizon=(
                pd.to_datetime(grp["target_end_date"]) - pd.Timestamp(str(ref))
            ).dt.days
            // 7
        )
        s = score_forecasts(
            grp,
            truth,
            keys=(
                "model",
                "source",
                "reference_date",
                "location",
                "target_end_date",
            ),
        )
        s["vintage"] = match[0]
        s["truth_vintage"] = latest
        s["complete"] = pd.to_datetime(s["target_end_date"]) <= pd.Timestamp(
            cutoff
        )
        scored.append(s)
    scores = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    return scores, pd.DataFrame(
        pending, columns=["reference_date", "vintage", "models"]
    )


def summarise_comparison(scores: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Summarise complete-week scores by round and model.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_comparison`.
    baseline : str
        Model that WIS is relative to.

    Returns
    -------
    pandas.DataFrame
        Columns `reference_date`, `model`, `source`, `n`, `wis`,
        `log_wis`, `relative_wis` and `relative_log_wis` (on the weeks
        shared with the baseline; missing if none), `coverage_50`,
        `coverage_90` and `bias`.
    """
    done = scores.loc[scores["complete"]]
    cols = [
        "reference_date",
        "model",
        "source",
        "n",
        "wis",
        "log_wis",
        "relative_wis",
        "relative_log_wis",
        "coverage_50",
        "coverage_90",
        "bias",
    ]
    if done.empty:
        return pd.DataFrame(columns=cols)
    out = summarise_scores(done, by=("reference_date", "model", "source"))
    rel = relative_skill(
        done,
        baseline=baseline,
        by=("reference_date",),
        metrics=("wis", "log_wis"),
        match_on=("reference_date", "location", "target_end_date"),
    )
    out = out.merge(
        rel[["model", "reference_date", "relative_wis", "relative_log_wis"]],
        on=["model", "reference_date"],
        how="left",
    )
    return (
        out[cols].sort_values(["reference_date", "wis"]).reset_index(drop=True)
    )


def forecast_quantiles(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return the plotted quantiles of each forecast.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Samples with `SAMPLE_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        One row per model, source, round, location and week, with the
        `QUANTILES` columns.
    """
    keys = ["model", "source", "reference_date", "location", "target_end_date"]
    grp = forecasts.groupby(keys)["value"]
    out = grp.quantile(np.array(list(QUANTILES.values()))).unstack()
    out.columns = list(QUANTILES)
    return out.reset_index()
