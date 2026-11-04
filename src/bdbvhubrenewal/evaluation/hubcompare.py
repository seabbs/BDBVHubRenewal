# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Compare our forecasts with the forecasts submitted to the hub.

`fetch_hub_submissions` downloads every team's metadata and its sample
forecasts for the weekly round (`wk inc confirmed cases`).
Files whose licence allows redistribution (`OPEN_LICENCES`) are kept
gzipped in a committed directory; the rest go to an uncommitted cache.
Files from the archived cumulative outbreak size round are skipped.

For each hub round, `our_forecasts` runs our unfitted models from the
latest real vintage that serves the round.
Rounds with no such vintage are skipped.
Fitted models are slow, so `stored_forecasts` takes theirs from backtest
and held-out forecast files for the same round.
The `source` column says which (`real` or `hub`).

`score_comparison` scores every model on the count each later real
vintage reports, from the vintage serving its round
(`bdbvhubrenewal.evaluation.nowcast.release_backtest`), with the complete
weeks of the latest vintage as a secondary check.
A round is pending until a real vintage later than its matching vintage
exists.
`summarise_comparison` gives WIS relative to a baseline on the weeks both
forecast, by round and target.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml

from bdbvhubrenewal.data.dates import as_date, reference_date_for_vintage
from bdbvhubrenewal.data.zones import weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.nowcast import (
    BASELINE,
    relative_energy,
    release_backtest,
)
from bdbvhubrenewal.evaluation.scoring import (
    relative_skill,
    scale_of,
    summarise_scores,
)
from bdbvhubrenewal.forecasters import get_model, list_models
from bdbvhubrenewal.forecasters.zones import zone_task

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
    "overall_comparison",
    "own_hub_ids",
    "resolve_baseline",
    "score_comparison",
    "stored_forecasts",
    "submission_dir",
    "summarise_comparison",
    "with_submissions",
    "zone_truth",
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
    submitted : str
        Directory holding the forecasts we submitted to the hub, as
        `model-output/<hub id>/*.csv.gz`.
    """

    baseline: str = "randomwalk"
    stored: tuple[str, ...] = ()
    locations: tuple[str, ...] = ("CD",)
    ref: str = "main"
    submitted: str = "data/our-submissions"


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
        submitted=str(raw.get("submitted", default.submitted)),
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
    """Return the hub ids of our submitted models.

    Parameters
    ----------
    path : pathlib.Path or str
        The hub submission settings: the model the `[submission]` id
        carries first, and a `[models.<name>]` table per model that has
        its own id.

    Returns
    -------
    dict
        Model name to hub model id, for the models set to submit, such as
        `{"renewalbvd": "seabbs-renewal", "randomwalknowcast":
        "seabbs-rwnowcast"}`.
    """
    from bdbvhubrenewal.submission.hub import model_id

    raw = tomllib.loads(Path(path).read_text())
    tables = {raw["submission"]["model"]: raw["submission"]}
    tables |= raw.get("models", {})
    return {
        name: model_id(table["model_abbr"])
        for name, table in tables.items()
        if table["submit"]
    }


def with_submissions(
    forecasts: pd.DataFrame,
    submitted: pd.DataFrame,
    hub_ids: Mapping[str, str],
) -> pd.DataFrame:
    """Put what we submitted in place of our forecasts for those rounds.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Samples with `SAMPLE_COLUMNS`.
    submitted : pandas.DataFrame
        Our submissions, from `load_hub_submissions`, by hub id.
    hub_ids : mapping
        Model name to hub id (`own_hub_ids`).

    Returns
    -------
    pandas.DataFrame
        `forecasts` without our model's rows for each round it submitted
        or the hub's copy of them, and the submitted rows under the model
        name with `source` `submitted`.
    """
    names = {hub_id: name for name, hub_id in hub_ids.items()}
    sub = submitted.assign(
        model=submitted["model"].map(names), source="submitted"
    )
    done = sub[["model", "reference_date"]].drop_duplicates()
    keep = forecasts.loc[~forecasts["model"].isin(list(names))].merge(
        done, on=["model", "reference_date"], how="left", indicator=True
    )
    keep = keep.loc[keep["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([keep, sub[list(SAMPLE_COLUMNS)]], ignore_index=True)


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
) -> str | None:
    """Return the real vintage our forecast for a hub round is made from.

    Parameters
    ----------
    reference_date : datetime.date or str
        The hub round.
    real_vintages : sequence of str
        Real vintages available.

    Returns
    -------
    str or None
        The latest real vintage serving the round, or None if there is
        none.
    """
    ref = as_date(reference_date)
    serving = [v for v in real_vintages if reference_date_for_vintage(v) == ref]
    return max(serving) if serving else None


def our_forecasts(
    reference_dates: Sequence[str],
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    models: Sequence[str],
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: Sequence[str],
    min_horizon: int | None = None,
) -> pd.DataFrame:
    """Run our models for hub rounds from the matching vintages.

    Each model runs on the country and provinces, and again on the health
    zones with cases at the vintage, as the zone backtest runs them.

    Parameters
    ----------
    reference_dates : sequence of str
        Hub reference dates.
    target, delays : pandas.DataFrame
        Tidy tables.
    config : BacktestConfig
        Horizons, samples and seed.
    models : sequence of str
        Registered model names that run at zone level.
    daily : pandas.DataFrame
        Daily onsets by release of the country and provinces.
    daily_zones : pandas.DataFrame
        Daily onsets by release of the health zones (`load_daily_zones`).
    zones : sequence of str
        The hub's zone codes (`hub_zones`).
    min_horizon : int, optional
        First week, in weeks from the reference date, to cover the weeks
        the hub forecasts did. Defaults to `config.min_horizon`.

    Returns
    -------
    pandas.DataFrame
        Output of `run_forecasts` with a `source` column (`real`). Rounds
        with no matching vintage are skipped.
    """
    if min_horizon is not None:
        config = replace(config, min_horizon=min_horizon)
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    built = [get_model(n) for n in models]
    frames = []
    for ref in sorted(set(reference_dates)):
        vintage = match_vintage(ref, vintages)
        if vintage is None or not built:
            continue
        rows = real.loc[real["vintage"] == vintage]
        tasks = make_tasks(rows, delays, config, daily=daily)
        counts = weekly_zone_counts(daily_zones, vintage)
        here = tuple(z for z in zones if z in set(counts["location"]))
        history = pd.concat([daily, daily_zones])
        tasks += [zone_task(t, counts, here, history) for t in tasks]
        fc = run_forecasts(built, tasks)
        fc["source"] = "real"
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
        forecasts. Missing files are skipped, and so are the rows of counts
        as reported by a later vintage (`as_of`).
    reference_dates : sequence of str
        Hub reference dates.
    exclude : sequence of str
        Models not to take, such as those run live.

    Returns
    -------
    pandas.DataFrame
        Columns `SAMPLE_COLUMNS`, with `source` the vintage kind.
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
            & (fc["kind"] == "real")
            & fc["as_of"].isna()
        ]
        frames.append(fc)
    if not frames:
        return pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    fc = pd.concat(frames, ignore_index=True).drop_duplicates()
    return fc.rename(columns={"kind": "source"})[list(SAMPLE_COLUMNS)]


def zone_truth(
    zones: pd.DataFrame, codes: Sequence[str], target: pd.DataFrame
) -> pd.DataFrame:
    """Return health zone counts by release in the target's layout.

    Parameters
    ----------
    zones : pandas.DataFrame
        Zone counts with `vintage`, `location`, `target_end_date` and
        `total` (`load_health_zones`). A zone and week missing from a
        release has no cases in it.
    codes : sequence of str
        The hub's zone codes (`hub_zones`).
    target : pandas.DataFrame
        The tidy target table. Each real release's weeks set its grid.

    Returns
    -------
    pandas.DataFrame
        Columns `kind` (`real`), `vintage`, `location`, `target_end_date`
        and `total`, for every zone and week of each release.
    """
    keys = ["location", "target_end_date"]
    frames = []
    real = target.loc[target["kind"] == "real"]
    for vintage, rows in real.groupby("vintage"):
        grid = pd.MultiIndex.from_product(
            [list(codes), sorted(set(rows["target_end_date"]))], names=keys
        )
        have = zones.loc[zones["vintage"] == vintage].set_index(keys)["total"]
        frames.append(
            have.reindex(grid, fill_value=0)
            .rename("total")
            .reset_index()
            .assign(kind="real", vintage=vintage)
        )
    return pd.concat(frames, ignore_index=True)[
        ["kind", "vintage", *keys, "total"]
    ]


def score_comparison(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    daily: pd.DataFrame,
    zones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every model on the count each later real vintage reports.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Hub and our samples with `SAMPLE_COLUMNS`.
    target, delays : pandas.DataFrame
        Tidy tables with every real vintage.
    config : BacktestConfig
        Settings of the tasks the rounds are scored from.
    daily : pandas.DataFrame
        Daily onsets by release, as for `make_tasks`.
    zones : pandas.DataFrame
        Health zone counts by release in the target's layout
        (`zone_truth`), so zone forecasts are scored too.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores (`ReleaseBacktest.scores` with `reference_date`,
        `source` and `horizon`), the energy scores
        (`ReleaseBacktest.energy` over every model of a round, with
        `reference_date` and `source`), and the pending rounds with
        columns `reference_date`, `vintage` and `models`.
    """
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    rounds, tasks, stored, pending = {}, [], [], []
    for ref, grp in forecasts[list(SAMPLE_COLUMNS)].groupby("reference_date"):
        match = match_vintage(str(ref), vintages)
        if match is None or match >= vintages[-1]:
            pending.append(
                {
                    "reference_date": str(ref),
                    "vintage": match,
                    "models": ", ".join(sorted(set(grp["model"]))),
                }
            )
            continue
        rounds[match] = str(ref)
        rows = real.loc[real["vintage"] == match]
        tasks += make_tasks(rows, delays, config, daily=daily)
        stored.append(grp.assign(vintage=match))
    pending_rounds = pd.DataFrame(
        pending, columns=["reference_date", "vintage", "models"]
    )
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(), pending_rounds
    samples = pd.concat(stored, ignore_index=True)
    result = release_backtest(
        tasks,
        daily,
        pd.concat([target, zones]),
        models=(),
        stored=samples,
        compare=set(samples["model"]),
    )
    sources = samples[["model", "vintage", "source"]].drop_duplicates()
    sources = sources.rename(columns={"vintage": "origin"})
    scores = result.scores.loc[result.scores["model"] != BASELINE]
    scores = scores.merge(sources, on=["model", "origin"])
    scores["reference_date"] = scores["origin"].map(rounds)
    scores["horizon"] = (
        pd.to_datetime(scores["target_end_date"])
        - pd.to_datetime(scores["reference_date"])
    ).dt.days // 7
    energy = result.energy.merge(sources, on=["model", "origin"])
    energy["reference_date"] = energy["origin"].map(rounds)
    return scores.reset_index(drop=True), energy, pending_rounds


def summarise_comparison(scores: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Summarise scores by round, target and model.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_comparison`.
    baseline : str
        Model that WIS is relative to.

    Returns
    -------
    pandas.DataFrame
        Columns `reference_date`, `target`, `model`, `source`, `n`, `wis`,
        `log_wis`, `relative_wis` and `relative_log_wis` (on the weeks
        shared with the baseline; missing if none), `coverage_50`,
        `coverage_90` and `bias`.
    """
    by = ("reference_date", "target")
    out = summarise_scores(scores, by=(*by, "model", "source"))
    rel = relative_skill(
        scores,
        baseline=baseline,
        by=by,
        metrics=("wis", "log_wis"),
        match_on=(*by, "truth_vintage", "location", "target_end_date"),
    )
    out = out.merge(
        rel[[*by, "model", "relative_wis", "relative_log_wis"]],
        on=[*by, "model"],
        how="left",
    )
    cols = [
        *by,
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
    return out[cols].sort_values([*by, "wis"]).reset_index(drop=True)


def overall_comparison(
    scores: pd.DataFrame, energy: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    """Summarise every model over all rounds, nationally and below.

    Parameters
    ----------
    scores, energy : pandas.DataFrame
        The scores and energy scores of `score_comparison`.
    baseline : str
        Model that every relative score divides by.

    Returns
    -------
    pandas.DataFrame
        On the reported target and the forecasts the baseline also made,
        one row per `scale` (`national`, `provinces` or `zones`) and
        `model`, with `hub` (a hub team's), `n`, `rel_wis`, `rel_log_wis`,
        `coverage_50`, `coverage_90`, `bias`, and `n_energy` (rounds and
        releases), `rel_energy` and `rel_log_energy`, as
        `OVERALL_METRICS`.
    """
    match = ["reference_date", "truth_vintage", "location", "target_end_date"]
    rep = scores.loc[scores["target"] == "reported"].assign(
        scale=scale_of(scores["location"])
    )
    rep = rep.merge(rep.loc[rep["model"] == baseline, match], on=match)
    rel = relative_skill(
        rep,
        baseline,
        by=("scale",),
        metrics=("wis", "log_wis"),
        match_on=match,
    )
    joint = relative_energy(
        energy.loc[energy["target"] == "reported"], ("scale",), baseline
    )
    keys = ["scale", "model"]
    cover = (
        rep.assign(hub=rep["source"] == "hub")
        .groupby(keys)
        .agg(
            hub=("hub", "all"),
            coverage_50=("coverage_50", "mean"),
            coverage_90=("coverage_90", "mean"),
            bias=("bias", "mean"),
        )
        .reset_index()
    )
    out = rel.merge(cover, on=keys).merge(joint, on=keys, how="left")
    out = out.rename(
        columns={"relative_wis": "rel_wis", "relative_log_wis": "rel_log_wis"}
    )
    return out[
        [
            *keys,
            "hub",
            "n",
            "rel_wis",
            "rel_log_wis",
            "coverage_50",
            "coverage_90",
            "bias",
            "n_energy",
            "rel_energy",
            "rel_log_energy",
        ]
    ]


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
