# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Score our models on the German hospitalisation nowcast benchmark.

The benchmark is that of Wolffram et al. (2023): nowcasts of the 7-day
hospitalisation incidence by Meldedatum in Germany, made each day from
2021-11-22 to 2022-04-29, of the sums ending 0 to 28 days before the
nowcast date, for Germany, its states and its age groups.
They are scored against the data as of 2022-08-08, by WIS and the coverage
of the 50% and 95% intervals, from the quantiles at `QUANTILE_LEVELS`.

`forecast_germany` runs every model on the tasks of each nowcast date
(`bdbvhubrenewal.data.germany.nowcast_tasks`) and keeps those quantiles of
its samples. Our models forecast the sums ending 0, 7, 14, 21 and 28 days
back (`HORIZONS`), so the teams are scored on the same.
`score_quantiles` scores them.
`team_scores` takes the hub teams' WIS from the paper's own score of each
quantile, and their coverage from their published quantiles. Forecasts
the paper scored with quantiles that are not published (filled from
later model versions or corrected) have no coverage.
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.germany import (
    ALL_AGES,
    GERMANY,
    HORIZONS,
    hub_forecasts,
    nowcast_tasks,
)
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.evaluation.scoring import wis_quantile
from bdbvhubrenewal.forecasters import VARIANTS, get_model, list_models

__all__ = [
    "COVERAGE",
    "QUANTILE_LEVELS",
    "SCORE_COLUMNS",
    "TEAM_NAMES",
    "GermanyConfig",
    "fetch_germany",
    "forecast_germany",
    "level_of",
    "load_germany_config",
    "nowcast_dates",
    "portable_models",
    "score_quantiles",
    "summarise_germany",
    "team_scores",
]

QUANTILE_LEVELS: Final = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
"""The quantile levels the hub asked for."""

COVERAGE: Final = {"coverage_50": (0.25, 0.75), "coverage_95": (0.025, 0.975)}
"""Each coverage score's interval, as quantile levels."""

SCORE_COLUMNS: Final = ("wis", *COVERAGE)
"""Scores of each forecast."""

TEAM_NAMES: Final = {
    "Epiforecasts-independent": "Epiforecasts",
    "ILM-prop": "ILM",
    "KIT-frozen_baseline": "KIT-frozen_baseline",
    "KIT-simple_nowcast": "KIT",
    "LMU_StaBLab-GAM_nowcast": "LMU",
    "NowcastHub-MeanEnsemble": "MeanEnsemble",
    "NowcastHub-MedianEnsemble": "MedianEnsemble",
    "RIVM-KEW": "RIVM",
    "RKI-weekly_report": "RKI",
    "SU-hier_bayes": "SU",
    "SZ-hosp_nowcast": "SZ",
}
"""The paper's name of each hub model."""

KEYS: Final = ["location", "age_group", "forecast_date", "target_end_date"]
"""Columns that identify a forecast of a model."""

_RKI_ERROR = 1e6
"""Quantiles above this are errors in RKI's file, which the paper replaced."""


@dataclass(frozen=True)
class GermanyConfig:
    """The benchmark's sources and settings (`config/germany.toml`).

    Attributes
    ----------
    hub_repo, hub_commit, triangle : str
        The hub repository, its commit and the reporting triangle's path.
    paper_repo, paper_commit : str
        The paper's evaluation repository and its commit.
    truth, scores : str
        Paths there of the truth and of the score of each quantile.
    submissions : tuple of str
        Paths there of the teams' submissions.
    first_date, last_date : str
        The first and last nowcast date.
    n_samples, seed : int
        Samples per forecast, and the seed.
    """

    hub_repo: str
    hub_commit: str
    triangle: str
    paper_repo: str
    paper_commit: str
    truth: str
    scores: str
    submissions: tuple[str, ...]
    first_date: str
    last_date: str
    n_samples: int
    seed: int


def load_germany_config(
    path: Path | str = "config/germany.toml",
) -> GermanyConfig:
    """Read the benchmark settings.

    Parameters
    ----------
    path : pathlib.Path or str
        The TOML file.

    Returns
    -------
    GermanyConfig
        The settings.
    """
    raw = tomllib.loads(Path(path).read_text())
    hub, paper, now = raw["hub"], raw["paper"], raw["nowcast"]
    return GermanyConfig(
        hub_repo=hub["repo"],
        hub_commit=hub["commit"],
        triangle=hub["triangle"],
        paper_repo=paper["repo"],
        paper_commit=paper["commit"],
        truth=paper["truth"],
        scores=paper["scores"],
        submissions=tuple(paper["submissions"]),
        first_date=now["first_date"],
        last_date=now["last_date"],
        n_samples=int(now["n_samples"]),
        seed=int(now["seed"]),
    )


def nowcast_dates(config: GermanyConfig) -> list[str]:
    """Return every nowcast date, oldest first.

    Parameters
    ----------
    config : GermanyConfig
        Gives the first and last date.

    Returns
    -------
    list of str
        ISO dates.
    """
    days = pd.date_range(config.first_date, config.last_date, freq="D")
    return [d.date().isoformat() for d in days]


def portable_models() -> list[str]:
    """Return the registered models that are not renewal variants.

    The renewal models use the BDBV hub's delay line data and outbreak
    estimates, so they do not run on other data.

    Returns
    -------
    list of str
        Model names, baseline first.
    """
    return [s.name for s in list_models() if s.name not in VARIANTS]


def fetch_germany(  # pragma: no cover - network
    config: GermanyConfig, dest: Path | str
) -> dict[str, object]:
    """Download the triangle and the paper's files at their commits.

    Parameters
    ----------
    config : GermanyConfig
        The sources.
    dest : pathlib.Path or str
        Directory the files go to, by base name, with `sources.json`.

    Returns
    -------
    dict
        The sources, also written to `dest/sources.json`.
    """
    from bdbvhubrenewal.data.tidy import _request

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    files = [(config.hub_repo, config.hub_commit, config.triangle)] + [
        (config.paper_repo, config.paper_commit, path)
        for path in (config.truth, config.scores, *config.submissions)
    ]
    for repo, commit, path in files:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        (out / Path(path).name).write_bytes(_request(url, None))
    sources: dict[str, object] = {
        "files": [
            {"repo": repo, "commit": commit, "path": path}
            for repo, commit, path in files
        ],
        "accessed": dt.date.today().isoformat(),
    }
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    return sources


def forecast_germany(
    models: Sequence[str],
    triangle: pd.DataFrame,
    dates: Sequence[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Run models on each nowcast date and keep the hub's quantiles.

    Parameters
    ----------
    models : sequence of str
        Registered model names.
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date
        (`bdbvhubrenewal.data.germany.read_triangle`).
    dates : sequence of str
        Nowcast dates.
    n_samples, seed : int
        Samples per forecast, and the seed.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `location`, `age_group`, `forecast_date`,
        `target_end_date`, `horizon` (days), `quantile` and `value`, the
        sample quantiles at `QUANTILE_LEVELS`.
    """
    built = [get_model(m) for m in models]
    frames = []
    for date in dates:
        for task in nowcast_tasks(triangle, date, n_samples, seed):
            samples = run_forecasts(built, [task])
            if samples.empty:
                continue
            frames.append(hub_forecasts(samples, task, date))
    keys = ["model", *KEYS, "horizon"]
    if not frames:
        return pd.DataFrame(columns=[*keys, "quantile", "value"])
    grp = pd.concat(frames).groupby(keys)["value"]
    out = grp.quantile(np.array(QUANTILE_LEVELS)).rename("value").reset_index()
    return out.rename(columns={f"level_{len(keys)}": "quantile"})


def level_of(location: pd.Series, age_group: pd.Series) -> pd.Series:
    """Return the paper's level of each place: national, states or age.

    Parameters
    ----------
    location, age_group : pandas.Series
        The hub's location and age group.

    Returns
    -------
    pandas.Series
        `national` for Germany at all ages, `age` for its age groups and
        `states` for the states.
    """
    level = np.where(
        location == GERMANY,
        np.where(age_group == ALL_AGES, "national", "age"),
        "states",
    )
    return pd.Series(level, index=location.index)


def _coverage(quantiles: pd.DataFrame) -> pd.DataFrame:
    """Return the coverage of each forecast's intervals from its quantiles."""
    wide = quantiles.pivot_table(
        index=["model", *KEYS, "truth"],
        columns="quantile",
        values="value",
        aggfunc="first",
    )
    y = wide.index.get_level_values("truth").to_numpy()
    out = pd.DataFrame(index=wide.index)
    for name, (lo, hi) in COVERAGE.items():
        inside = (wide[lo].to_numpy() <= y) & (y <= wide[hi].to_numpy())
        out[name] = np.where(wide[[lo, hi]].notna().all(axis=1), inside, np.nan)
    return out.dropna().reset_index()


def _with_truth(rows: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth = truth.rename(columns={"date": "target_end_date"})
    return rows.merge(truth, on=["location", "age_group", "target_end_date"])


def _finish(scores: pd.DataFrame) -> pd.DataFrame:
    scores["horizon"] = (
        pd.to_datetime(scores["target_end_date"])
        - pd.to_datetime(scores["forecast_date"])
    ).dt.days
    scores["level"] = level_of(scores["location"], scores["age_group"])
    cols = ["model", "level", *KEYS, "horizon", "truth", *SCORE_COLUMNS]
    return scores[cols].sort_values(cols[:5]).reset_index(drop=True)


def score_quantiles(
    quantiles: pd.DataFrame, truth: pd.DataFrame
) -> pd.DataFrame:
    """Score quantile forecasts against the truth.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `forecast_germany` output.
    truth : pandas.DataFrame
        The paper's truth: `date`, `location`, `age_group` and `truth`.

    Returns
    -------
    pandas.DataFrame
        One row per model and forecast: `model`, `level`, the forecast's
        keys, `horizon`, `truth` and `SCORE_COLUMNS`.
    """
    rows = _with_truth(quantiles, truth)
    cover = _coverage(rows)
    wide = rows.pivot_table(
        index=["model", *KEYS, "truth"],
        columns="quantile",
        values="value",
        aggfunc="first",
    )[list(QUANTILE_LEVELS)]
    wis = wis_quantile(
        wide.index.get_level_values("truth"), wide, QUANTILE_LEVELS
    )["wis"]
    scores = cover.merge(
        pd.DataFrame({"wis": wis}, index=wide.index).reset_index()
    )
    return _finish(scores)


def team_scores(
    paper: pd.DataFrame, submissions: pd.DataFrame, truth: pd.DataFrame
) -> pd.DataFrame:
    """Score the hub teams at our horizons, as the paper did.

    Parameters
    ----------
    paper : pandas.DataFrame
        The paper's score of each quantile (`data/scores/scores.csv.gz`):
        `model`, the forecast's keys, `target`, `type`, `quantile` and
        `score`. A forecast's WIS is the mean of its quantile scores.
    submissions : pandas.DataFrame
        The teams' submissions as the paper gathered them, with
        `model` (the hub's name), the forecast's keys, `type`, `quantile`,
        `value` and `retrospective`. A retrospective file replaces the
        prospective one of the same date.
    truth : pandas.DataFrame
        The paper's truth.

    Returns
    -------
    pandas.DataFrame
        As `score_quantiles`, for every forecast the paper scored. The
        coverage is missing where the quantiles are not published, or are
        above `_RKI_ERROR`.
    """
    targets = {f"{h} day ahead inc hosp" for h in HORIZONS}
    rows = paper.loc[
        (paper["type"] == "quantile") & paper["target"].isin(targets)
    ]
    wis = rows.groupby(["model", *KEYS])["score"].mean().rename("wis")
    sub = submissions.loc[
        (submissions["type"] == "quantile")
        & submissions["target"].isin(targets)
        & submissions["quantile"].isin(
            {q for c in COVERAGE.values() for q in c}
        )
    ]
    sub = sub.assign(model=sub["model"].map(TEAM_NAMES))
    retro = sub["retrospective"].fillna(False).astype(bool)
    has_retro = retro.groupby([sub["model"], sub["forecast_date"]]).transform(
        "any"
    )
    sub = sub.loc[retro | ~has_retro]
    sub = sub.loc[sub["value"] < _RKI_ERROR]
    cover = _coverage(_with_truth(sub, truth))
    scores = _with_truth(wis.reset_index(), truth).merge(cover, how="left")
    return _finish(scores)


def summarise_germany(
    scores: pd.DataFrame, by: Sequence[str] = ("level", "model")
) -> pd.DataFrame:
    """Average the scores over forecasts.

    Parameters
    ----------
    scores : pandas.DataFrame
        `score_quantiles` or `team_scores` output.
    by : sequence of str
        Columns to group by.

    Returns
    -------
    pandas.DataFrame
        The group columns, `n` and the mean of each of `SCORE_COLUMNS`
        over the forecasts that have it, with `n_coverage` the number
        that have coverage.
    """
    grp = scores.groupby(list(by))
    out = grp[list(SCORE_COLUMNS)].mean()
    out.insert(0, "n", grp.size())
    out["n_coverage"] = grp["coverage_50"].count()
    return out.reset_index()
