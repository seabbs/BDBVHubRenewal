# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Compare our forecasts with the forecasts submitted to the hub.

`fetch_hub_submissions` downloads the hub's `model-output/` files and
`model-metadata/` records for a list of models.
`load_hub_submissions` keeps the sample forecasts of the current round
(`wk inc confirmed cases`) and lists files from other rounds, such as the
archived cumulative outbreak size round, which target a different
quantity and are not scored.

For each hub reference date, `our_forecasts` runs our registered models
from the matching vintage: the latest real vintage that serves that round
if there is one, otherwise a vintage reconstructed for the Saturday before
it.
The `source` column says which (`real`, `reconstructed`, or `hub` for a
submitted file).

`score_comparison` scores every forecast for a reference date on the same
truth (the latest real vintage, as reported then) and the same locations
and weeks: those every model covered and the truth has.
A reference date is pending until a real vintage later than its matching
vintage exists.

`display_names` maps model ids to the labels shown: `named` keeps them,
`anonymised` replaces other teams with Team A, Team B and so on, and
`epiforecasts` keeps epiforecasts models only.
`bdbvhubrenewal.reporting.plots.plot_comparison` draws them side by side.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import string
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
from bdbvhubrenewal.evaluation.scoring import score_forecasts
from bdbvhubrenewal.forecasters import get_model

__all__ = [
    "ATTRIBUTION",
    "DISPLAY_MODES",
    "WEEKLY_TARGET",
    "ComparisonConfig",
    "attribution",
    "display_names",
    "fetch_hub_submissions",
    "load_comparison_config",
    "load_hub_submissions",
    "load_team_metadata",
    "match_vintage",
    "our_forecasts",
    "score_comparison",
]

WEEKLY_TARGET: Final = "wk inc confirmed cases"
"""The target of the hub's current round."""

DISPLAY_MODES: Final = ("named", "anonymised", "epiforecasts")
"""How teams are shown in the comparison."""

OWN_TEAM: Final = "epiforecasts"
"""Our team abbreviation on the hub."""

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
    models : tuple of str
        Hub model ids to fetch.
    display : str
        One of `DISPLAY_MODES`.
    ref : str
        Git ref of the hub to fetch from.
    """

    models: tuple[str, ...]
    display: str = "anonymised"
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

    Raises
    ------
    ValueError
        If the display mode is unknown.
    """
    raw = tomllib.loads(Path(path).read_text())["comparison"]
    cfg = ComparisonConfig(
        models=tuple(raw["models"]),
        display=str(raw.get("display", "anonymised")),
        ref=str(raw.get("ref", "main")),
    )
    display_names([], cfg.display)
    return cfg


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


def fetch_hub_submissions(  # pragma: no cover - network
    dest: Path | str,
    models: Sequence[str],
    ref: str = "main",
    token: str | None = None,
) -> dict[str, object]:
    """Download the hub's model output and metadata for some models.

    Parameters
    ----------
    dest : pathlib.Path or str
        Directory to write `model-output/` and `model-metadata/` into.
    models : sequence of str
        Model ids.
    ref : str
        Git ref of the hub.
    token : str, optional
        GitHub token. Defaults to `GITHUB_TOKEN` if set.

    Returns
    -------
    dict
        A manifest with the commit and access date, also written to
        `dest/manifest.json`.
    """
    from bdbvhubrenewal.data.tidy import HUB_REPO, _api, _raw

    token = token or os.environ.get("GITHUB_TOKEN")
    root = Path(dest)
    head = _api(f"repos/{HUB_REPO}/commits/{ref}", token)["sha"]
    tree = _api(f"repos/{HUB_REPO}/git/trees/{head}?recursive=1", token)
    wanted = set(models)
    paths = []
    for item in tree["tree"]:
        parts = item["path"].split("/")
        output = (
            len(parts) == 3
            and parts[0] == "model-output"
            and parts[1] in wanted
            and parts[2].endswith(".csv")
        )
        meta = (
            len(parts) == 2
            and parts[0] == "model-metadata"
            and parts[1].removesuffix(".yml") in wanted
        )
        if item["type"] == "blob" and (output or meta):
            paths.append(item["path"])
    for path in paths:
        out = root / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_raw(head, path, token))
    manifest: dict[str, object] = {
        "repo": HUB_REPO,
        "ref": ref,
        "commit": head,
        "accessed": dt.date.today().isoformat(),
        "files": sorted(paths),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_hub_submissions(
    root: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read downloaded hub files.

    Parameters
    ----------
    root : pathlib.Path or str
        Directory holding `model-output/<model>/*.csv`.

    Returns
    -------
    tuple of pandas.DataFrame
        The sample forecasts of the current round, with columns `model`,
        `reference_date`, `location`, `target_end_date`, `output_type`,
        `sample` and `value`; and one row per other file, with `model`,
        `reference_date`, `target`, `output_types` and `rows`.
    """
    weekly, other = [], []
    for path in sorted(Path(root).glob("model-output/*/*.csv")):
        model = path.parent.name
        table = pd.read_csv(path, dtype={"output_type_id": str})
        keep = (table["target"] == WEEKLY_TARGET) & (
            table["output_type"] == "sample"
        )
        if keep.any():
            rows = table.loc[keep]
            weekly.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "reference_date": rows["reference_date"].astype(str),
                        "location": rows["location"].astype(str),
                        "target_end_date": rows["target_end_date"].astype(str),
                        "output_type": "sample",
                        "sample": pd.to_numeric(rows["output_type_id"]).astype(
                            "int64"
                        ),
                        "value": rows["value"].astype(float),
                    }
                )
            )
        rest = table.loc[~keep]
        for (ref, target), grp in rest.groupby(["reference_date", "target"]):
            other.append(
                {
                    "model": model,
                    "reference_date": str(ref),
                    "target": str(target),
                    "output_types": ", ".join(
                        sorted(set(grp["output_type"].astype(str)))
                    ),
                    "rows": len(grp),
                }
            )
    weekly_df = (
        pd.concat(weekly, ignore_index=True)
        if weekly
        else pd.DataFrame(
            columns=[
                "model",
                "reference_date",
                "location",
                "target_end_date",
                "output_type",
                "sample",
                "value",
            ]
        )
    )
    other_df = pd.DataFrame(
        other,
        columns=["model", "reference_date", "target", "output_types", "rows"],
    )
    return weekly_df, other_df


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

    Returns
    -------
    pandas.DataFrame
        Output of `run_forecasts`, with `model` as the hub id
        `epiforecasts-<name>` and a `source` column (`real` or
        `reconstructed`). Rounds with no matching vintage are skipped.
    """
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    built = [get_model(n) for n in models]
    frames = []
    for ref in sorted(set(reference_dates)):
        match = match_vintage(ref, vintages)
        if match is None:
            continue
        vintage, source = match
        if source == "real":
            rows = real.loc[real["vintage"] == vintage]
            dly = delays
        else:
            rows = reconstruct_vintages(real, delays, [vintage])
            src = delays.loc[delays["vintage"] == delays["vintage"].max()]
            dly = reconstruct_delays(src, vintage).assign(kind="reconstructed")
        tasks = make_tasks(rows, dly, config)
        fc = run_forecasts(built, tasks)
        fc["source"] = source
        frames.append(fc)
    out = pd.concat(frames, ignore_index=True)
    out["model"] = f"{OWN_TEAM}-" + out["model"]
    return out


def score_comparison(
    hub: pd.DataFrame, ours: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score hub and our forecasts on the same truth, places and weeks.

    Parameters
    ----------
    hub : pandas.DataFrame
        Weekly samples from `load_hub_submissions`.
    ours : pandas.DataFrame
        Output of `our_forecasts`.
    target : pandas.DataFrame
        Target rows with every real vintage.

    Returns
    -------
    tuple of pandas.DataFrame
        The scores (`score_forecasts` output with `source` and
        `truth_vintage`), and the pending rounds with columns
        `reference_date`, `vintage` and `models`.
    """
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    latest = vintages[-1]
    truth = real.loc[
        real["vintage"] == latest, ["location", "target_end_date", "total"]
    ].rename(columns={"total": "truth"})
    cols = [
        "model",
        "source",
        "reference_date",
        "location",
        "target_end_date",
        "sample",
        "value",
    ]
    both = pd.concat(
        [hub.assign(source="hub")[cols], ours[cols]], ignore_index=True
    )
    scored, pending = [], []
    for ref, grp in both.groupby("reference_date"):
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
        cells = grp[["model", "source", "location", "target_end_date"]]
        cover = cells.drop_duplicates().groupby(["location", "target_end_date"])
        n_models = grp[["model", "source"]].drop_duplicates().shape[0]
        n_cover = cover.size()
        common = n_cover[n_cover == n_models].reset_index()[
            ["location", "target_end_date"]
        ]
        sel = grp.merge(common, on=["location", "target_end_date"])
        sel = sel.assign(
            horizon=(
                pd.to_datetime(sel["target_end_date"]) - pd.Timestamp(str(ref))
            ).dt.days
            // 7
        )
        s = score_forecasts(
            sel,
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
        scored.append(s)
    scores = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    return scores, pd.DataFrame(
        pending, columns=["reference_date", "vintage", "models"]
    )


def display_names(models: Sequence[str], mode: str) -> dict[str, str]:
    """Map model ids to the labels shown under a display mode.

    Parameters
    ----------
    models : sequence of str
        Hub model ids, including ours (`epiforecasts-*`).
    mode : str
        One of `DISPLAY_MODES`.

    Returns
    -------
    dict
        Model id to label. Models left out are not shown.

    Raises
    ------
    ValueError
        If `mode` is unknown.
    """
    if mode not in DISPLAY_MODES:
        raise ValueError(
            f"Unknown display mode {mode!r}; use one of {DISPLAY_MODES}."
        )
    own = [m for m in models if m.startswith(f"{OWN_TEAM}-")]
    others = sorted(m for m in models if m not in own)
    names = {m: m for m in own}
    if mode == "named":
        names |= {m: m for m in others}
    elif mode == "anonymised":
        letters = string.ascii_uppercase
        names |= {
            m: f"Team {letters[i % 26]}{'' if i < 26 else i // 26}"
            for i, m in enumerate(others)
        }
    return {m: names[m] for m in models if m in names}
