# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Score our models and the hub teams on the German nowcast benchmark.

The benchmark is that of Wolffram et al. (2023): nowcasts of the 7-day
hospitalisation incidence by Meldedatum in Germany, made each day from
2021-11-22 to 2022-04-29, for Germany, its states and its age groups.
The hub collected them as quantiles at `QUANTILE_LEVELS` and scored them
by WIS and the coverage of the 50% and 95% intervals. They are scored the
same way here.

`germany_backtest` runs our models on the tasks of each nowcast date
(`bdbvhubrenewal.data.germany.nowcast_tasks`), for the sums ending 0, 7,
14, 21 and 28 days back (`HORIZONS`), and keeps the quantiles of their
samples at `QUANTILE_LEVELS`.
`germany_teams` takes the hub teams' submitted quantiles at the same
horizons (`team_forecasts`).
Both are scored by `score_quantiles`.
Each sum's truth is its count in the data $D$ days after it ends, the
paper's 40-day truth (`rolling_truth`).
`relative_scores` divides each model's mean scores by `BASELINE`'s on the
forecasts both made.
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
from scipy.stats import norm

from bdbvhubrenewal.data.germany import (
    ALL_AGES,
    HORIZONS,
    PLACE_NAMES,
    hub_forecasts,
    hub_target,
    level,
    nowcast_tasks,
)
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.evaluation.scoring import relative_skill, score_quantiles
from bdbvhubrenewal.forecasters import get_model

__all__ = [
    "BASELINE",
    "QUANTILE_LEVELS",
    "SCORE_COLUMNS",
    "TEAM_NAMES",
    "GermanyConfig",
    "fetch_germany",
    "germany_backtest",
    "germany_teams",
    "load_germany_config",
    "nowcast_dates",
    "read_submissions",
    "relative_scores",
    "team_forecasts",
]

QUANTILE_LEVELS: Final = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
"""The quantile levels the hub collected."""

SCORE_COLUMNS: Final = (
    "wis",
    "dispersion",
    "overprediction",
    "underprediction",
    "ae_median",
    "coverage_50",
    "coverage_80",
    "coverage_95",
)
"""Scores of each forecast, from its quantiles at `QUANTILE_LEVELS`."""

BASELINE: Final = "randomwalknowcast"
"""The model relative scores divide by."""

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

_SUBMISSION: Final = (
    "model",
    "location",
    "age_group",
    "forecast_date",
    "target_end_date",
    "target",
    "quantile",
    "value",
)
"""Columns of a submission used, besides `type`."""

_RKI_ERROR: Final = 1e6
"""RKI quantiles above this are errors, which the paper replaced."""

_KEYS = ("model", "target", "origin", "truth_vintage", "location")
"""Columns that, with the week, identify a scored forecast."""


@dataclass(frozen=True)
class GermanyConfig:
    """The benchmark's source and settings (`config/germany.toml`).

    Attributes
    ----------
    repo, commit, triangle : str
        The hub repository, its commit and the reporting triangle's path.
    teams_repo, teams_commit : str
        The paper's evaluation repository and its commit.
    submissions : tuple of str
        Paths there of the hub teams' submissions.
    first_date, last_date : str
        The first and last nowcast date.
    models : tuple of str
        The models run.
    n_samples, seed : int
        Samples per forecast, and the seed.
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.
    """

    repo: str
    commit: str
    triangle: str
    teams_repo: str
    teams_commit: str
    submissions: tuple[str, ...]
    first_date: str
    last_date: str
    models: tuple[str, ...]
    n_samples: int
    seed: int
    truth_delay: int


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
    hub, teams, now = raw["hub"], raw["teams"], raw["nowcast"]
    return GermanyConfig(
        repo=hub["repo"],
        commit=hub["commit"],
        triangle=hub["triangle"],
        teams_repo=teams["repo"],
        teams_commit=teams["commit"],
        submissions=tuple(teams["submissions"]),
        first_date=now["first_date"],
        last_date=now["last_date"],
        models=tuple(now["models"]),
        n_samples=int(now["n_samples"]),
        seed=int(now["seed"]),
        truth_delay=int(raw["truth"]["delay"]),
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


def fetch_germany(  # pragma: no cover - network
    config: GermanyConfig, dest: Path | str
) -> dict[str, object]:
    """Download the triangle and the teams' submissions at their commits.

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
    files = [(config.repo, config.commit, config.triangle)] + [
        (config.teams_repo, config.teams_commit, path)
        for path in config.submissions
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


def _truth(triangle: pd.DataFrame, ends: set[str], delay: int) -> pd.DataFrame:
    """Each sum's count in the data `delay` days after it ends."""
    return rolling_truth(hub_target(triangle, sorted(ends)), delay)


def _score(
    quantiles: pd.DataFrame, truth: pd.DataFrame, delay: int
) -> pd.DataFrame:
    """Score quantile forecasts on the D-day truth, with their level."""
    rows = quantiles.merge(
        truth[["location", "target_end_date", "truth_vintage"]],
        on=["location", "target_end_date"],
    ).assign(target=f"{delay}-day")
    scores = score_quantiles(
        rows, truth, (*_KEYS, "target_end_date"), QUANTILE_LEVELS
    )
    scores["pool"] = scores["location"].map(level)
    return scores[
        [*_KEYS, "target_end_date", "horizon", "truth", *SCORE_COLUMNS, "pool"]
    ]


def germany_backtest(
    models: Sequence[str],
    triangle: pd.DataFrame,
    dates: Sequence[str],
    n_samples: int,
    seed: int,
    truth_delay: int,
) -> pd.DataFrame:
    """Nowcast each date with each model and score the nowcasts.

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
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.

    Returns
    -------
    pandas.DataFrame
        One row per forecast: `model`, `target` (`"<D>-day"`), `origin`
        (the nowcast date), `truth_vintage` (the report date of the
        truth), `location` (the place), `target_end_date`, `horizon`
        (days), `truth`, `SCORE_COLUMNS` and `pool` (the paper's level,
        `national`, `states` or `age`). Sums with no data $D$ days after
        them are not scored.
    """
    ends = {
        (dt.date.fromisoformat(d) + dt.timedelta(days=h)).isoformat()
        for d in dates
        for h in HORIZONS
    }
    truth = _truth(triangle, ends, truth_delay)
    built = [get_model(m) for m in models]
    keys = ["model", "origin", "location", "target_end_date", "horizon"]
    frames = []
    for date in dates:
        for task in nowcast_tasks(triangle, date, n_samples, seed):
            samples = run_forecasts(built, [task])
            if samples.empty:
                continue
            drawn = hub_forecasts(samples, task, date).rename(
                columns={"forecast_date": "origin"}
            )
            q = drawn.groupby(keys)["value"].quantile(np.array(QUANTILE_LEVELS))
            frames.append(
                q.rename_axis([*keys, "quantile"]).reset_index(name="value")
            )
    if not frames:
        return pd.DataFrame(
            columns=[*_KEYS, "target_end_date", *SCORE_COLUMNS, "pool"]
        )
    return _score(pd.concat(frames, ignore_index=True), truth, truth_delay)


def read_submissions(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Read the hub teams' submissions as the paper gathered them.

    Parameters
    ----------
    paths : sequence of pathlib.Path or str
        CSV files with the hub's columns, such as `submissions.csv.gz`
        of the paper's evaluation repository.

    Returns
    -------
    pandas.DataFrame
        The files stacked: `model`, `location`, `age_group`,
        `forecast_date`, `target_end_date`, `target`, `type`, `quantile`
        and `value`.
    """
    cols = [*_SUBMISSION[:6], "type", *_SUBMISSION[6:]]
    return pd.concat(
        [pd.read_csv(p, usecols=cols) for p in paths], ignore_index=True
    )[cols]


def _fix_rki(rows: pd.DataFrame) -> pd.DataFrame:
    """Replace RKI's errors by its -1 day nowcast, as the paper does."""
    bad = (rows["model"] == "RKI") & ~(rows["value"] <= _RKI_ERROR)
    ids = ["model", "location", "age_group", "forecast_date", "quantile"]
    prior = rows.loc[
        (rows["model"] == "RKI") & (rows["target"] == "-1 day ahead inc hosp"),
        [*ids, "value"],
    ]
    fixed = rows.loc[bad].merge(
        prior, on=ids, how="left", suffixes=("", "_prior")
    )
    fixed["value"] = np.fmin(fixed["value"], fixed["value_prior"])
    return pd.concat(
        [rows.loc[~bad], fixed.drop(columns="value_prior")], ignore_index=True
    )


def _fill_tenth(rows: pd.DataFrame) -> pd.DataFrame:
    """Add Epiforecasts' missing 10% quantiles, as the paper does.

    Epiforecasts left out the 10% quantile at times. It is filled from the
    normal through the 2.5% and 25% quantiles, sorted.
    """
    ids = ["model", "location", "age_group", "forecast_date", "target"]
    wide = rows.pivot_table(
        index=[*ids, "target_end_date"],
        columns="quantile",
        values="value",
        aggfunc="first",
    )
    epi = wide.index.get_level_values("model") == "Epiforecasts"
    gap = wide.loc[
        epi & wide[0.1].isna() & wide[[0.025, 0.25]].notna().all(axis=1)
    ]
    lo = gap[[0.025, 0.25]].min(axis=1)
    hi = gap[[0.025, 0.25]].max(axis=1)
    z_lo, z_hi = norm.ppf(0.025), norm.ppf(0.25)
    sd = (hi - lo) / (z_hi - z_lo)
    tenth = lo + sd * (norm.ppf(0.1) - z_lo)
    filled = pd.DataFrame(tenth.rename("value")).reset_index()
    filled["quantile"] = 0.1
    return pd.concat([rows, filled[rows.columns]], ignore_index=True)


def team_forecasts(submissions: pd.DataFrame) -> pd.DataFrame:
    """Return the hub teams' quantile nowcasts at our horizons.

    RKI's quantiles above `_RKI_ERROR` are replaced by its -1 day nowcast
    and Epiforecasts' missing 10% quantiles are filled (`_fill_tenth`), as
    the paper does. Nothing else is filled.

    Parameters
    ----------
    submissions : pandas.DataFrame
        `read_submissions` output.

    Returns
    -------
    pandas.DataFrame
        Columns `model` (the paper's name, `TEAM_NAMES`), `origin` (the
        forecast date), `location` (the place, as in `PLACE_NAMES`),
        `target_end_date`, `horizon` (days, as `HORIZONS`), `quantile` and
        `value`, for the places in `PLACE_NAMES`.

    Raises
    ------
    ValueError
        If a team is not in `TEAM_NAMES`.
    """
    rows = submissions.loc[submissions["type"] == "quantile", list(_SUBMISSION)]
    unknown = set(rows["model"]) - set(TEAM_NAMES)
    if unknown:
        raise ValueError(f"Teams with no name in TEAM_NAMES: {sorted(unknown)}")
    rows = rows.assign(model=rows["model"].map(TEAM_NAMES))
    targets = {f"{h} day ahead inc hosp": h for h in HORIZONS}
    rows = _fix_rki(rows)
    rows = _fill_tenth(rows.loc[rows["target"].isin(set(targets))])
    place = rows["location"].where(
        rows["age_group"] == ALL_AGES, rows["age_group"]
    )
    out = pd.DataFrame(
        {
            "model": rows["model"],
            "origin": rows["forecast_date"],
            "location": place,
            "target_end_date": rows["target_end_date"],
            "horizon": rows["target"].map(targets),
            "quantile": rows["quantile"],
            "value": rows["value"],
        }
    )
    keep = out["location"].isin(set(PLACE_NAMES))
    return out.loc[keep].reset_index(drop=True)


def germany_teams(
    submissions: pd.DataFrame, triangle: pd.DataFrame, truth_delay: int
) -> pd.DataFrame:
    """Score the hub teams' nowcasts as `germany_backtest` scores ours.

    Parameters
    ----------
    submissions : pandas.DataFrame
        `read_submissions` output.
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date, which give the truth.
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.

    Returns
    -------
    pandas.DataFrame
        As `germany_backtest`.
    """
    quantiles = team_forecasts(submissions)
    truth = _truth(triangle, set(quantiles["target_end_date"]), truth_delay)
    return _score(quantiles, truth, truth_delay)


def relative_scores(
    scores: pd.DataFrame, by: Sequence[str] = ("pool",)
) -> pd.DataFrame:
    """Divide each model's mean scores by `BASELINE`'s on the same forecasts.

    Parameters
    ----------
    scores : pandas.DataFrame
        `germany_backtest` and `germany_teams` output.
    by : sequence of str
        Columns of a cell other than `model`.

    Returns
    -------
    pandas.DataFrame
        One row per cell and model, over the forecasts `BASELINE` also
        made: `n`, `wis`, `rel_wis`, `ae_median`, `rel_ae_median`, and
        the coverage of the 50% and 95% intervals.
    """
    match = ("target", "origin", "truth_vintage", "location", "target_end_date")
    rel = relative_skill(
        scores,
        BASELINE,
        by=by,
        metrics=("wis", "ae_median"),
        match_on=match,
        means=("wis", "ae_median", "coverage_50", "coverage_95"),
    )
    rel = rel.rename(
        columns={
            "relative_wis": "rel_wis",
            "relative_ae_median": "rel_ae_median",
        }
    )
    cols = ["n", "wis", "rel_wis", "ae_median", "rel_ae_median"]
    return rel[[*by, "model", *cols, "coverage_50", "coverage_95"]]
