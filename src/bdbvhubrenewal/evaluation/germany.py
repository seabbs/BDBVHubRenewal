# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Score our models and the hub teams on the German nowcast benchmark.

The benchmark is that of Wolffram et al. (2023): nowcasts of the 7-day
hospitalisation incidence by Meldedatum in Germany, made each day from
2021-11-22 to 2022-04-29, for Germany, its states and its age groups.
The hub collected them as quantiles at `QUANTILE_LEVELS` and scored them
by WIS and the coverage of the 50% and 95% intervals. They are scored the
same way here.

The nowcast dates are split in two halves at `GermanyConfig.split_date`.
The first half is for developing and checking our setup; the second is
held out for the paper (`nowcast_dates`).

`germany_forecasts` runs our models on the tasks of each nowcast date
(`bdbvhubrenewal.data.germany.nowcast_tasks`), for the sums ending 0, 7,
14, 21 and 28 days back (`HORIZONS`), and keeps the quantiles of their
samples at `QUANTILE_LEVELS` (`hub_quantiles`).
`team_forecasts` takes the hub teams' submitted quantiles at the same
horizons.
`with_truth` adds each sum's truth, its count in the data $D$ days after
it ends, the paper's 40-day truth (`rolling_truth`).
`score_germany` scores both by `score_quantiles`, and `relative_scores`
divides each model's mean scores by a baseline's on the forecasts both
made: `randomwalknowcast` on the page, the hub's own baseline in the paper.
`paper_forecasts` writes the forecasts as the paper's long table.
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
    GERMANY,
    HORIZONS,
    PLACE_NAMES,
    STATES,
    hub_forecasts,
    hub_target,
    level,
    nowcast_tasks,
)
from bdbvhubrenewal.evaluation.backtest import run_forecasts
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.evaluation.scoring import relative_skill, score_quantiles
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.reporting.hubtables import hub_quantiles

__all__ = [
    "PAPER_COLUMNS",
    "QUANTILE_LEVELS",
    "SCORE_COLUMNS",
    "TEAM_NAMES",
    "GermanyConfig",
    "asof_sums",
    "fetch_germany",
    "germany_forecasts",
    "load_germany_config",
    "nowcast_dates",
    "paper_forecasts",
    "read_submissions",
    "relative_scores",
    "score_germany",
    "team_forecasts",
    "with_truth",
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

PAPER_COLUMNS: Final = (
    "study",
    "model",
    "location",
    "stratum",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "sample_id",
    "value",
    "truth",
)
"""Columns of the paper's long forecast table (`paper/schema.md`)."""

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
    split_date : str
        The first nowcast date of the held-out half. The dates before it
        are for development.
    models : tuple of str
        The models run.
    n_samples, seed : int
        Samples per forecast, and the seed.
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.
    baseline : str
        The model the page's relative scores divide by.
    paper_baseline : str
        The model the paper's relative scores divide by, the hub's own.
    hidden : tuple of str
        Models switched off on the page until the reader switches them on.
    plot_every : int
        The nowcast plots show every this many development dates.
    """

    repo: str
    commit: str
    triangle: str
    teams_repo: str
    teams_commit: str
    submissions: tuple[str, ...]
    first_date: str
    last_date: str
    split_date: str
    models: tuple[str, ...]
    n_samples: int
    seed: int
    truth_delay: int
    baseline: str
    paper_baseline: str
    hidden: tuple[str, ...]
    plot_every: int


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
        split_date=now["split_date"],
        models=tuple(now["models"]),
        n_samples=int(now["n_samples"]),
        seed=int(now["seed"]),
        truth_delay=int(raw["truth"]["delay"]),
        baseline=raw["scoring"]["baseline"],
        paper_baseline=raw["scoring"]["paper_baseline"],
        hidden=tuple(raw["page"]["hidden"]),
        plot_every=int(raw["page"]["plot_every"]),
    )


def nowcast_dates(config: GermanyConfig, held_out: bool = False) -> list[str]:
    """Return the nowcast dates of one half, oldest first.

    Parameters
    ----------
    config : GermanyConfig
        Gives the first, last and split date.
    held_out : bool
        The held-out half, from the split date on, rather than the
        development half before it.

    Returns
    -------
    list of str
        ISO dates.
    """
    days = pd.date_range(config.first_date, config.last_date, freq="D")
    dates = [d.date().isoformat() for d in days]
    return [d for d in dates if (d >= config.split_date) == held_out]


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


def asof_sums(
    triangle: pd.DataFrame, dates: Sequence[str], weeks: int = 8
) -> pd.DataFrame:
    """Return the 7-day sums as each nowcast date's data gave them.

    Parameters
    ----------
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date.
    dates : sequence of str
        Nowcast dates.
    weeks : int
        How far back the sums go, from each nowcast date.

    Returns
    -------
    pandas.DataFrame
        Columns `origin` (the nowcast date), `location` (the place),
        `target_end_date`, every day from `weeks` weeks before the
        nowcast date to it, and `total`, the sum reported by the nowcast
        date.
    """
    cols = ["origin", "location", "target_end_date", "total"]
    if not dates:
        return pd.DataFrame(columns=cols)
    first = dt.date.fromisoformat(min(dates)) - dt.timedelta(weeks=weeks)
    ends = pd.date_range(first, max(dates), freq="D").date.astype(str)
    sums = hub_target(triangle, list(ends))
    sums = sums.loc[sums["vintage"].isin(set(dates))].rename(
        columns={"vintage": "origin"}
    )
    back = pd.to_datetime(sums["origin"]) - pd.to_datetime(
        sums["target_end_date"]
    )
    keep = back.dt.days <= 7 * weeks
    return sums.loc[keep, cols].reset_index(drop=True)


def germany_forecasts(
    models: Sequence[str],
    triangle: pd.DataFrame,
    dates: Sequence[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Nowcast each date with each model, as quantiles at the hub's levels.

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
        One row per quantile: `model`, `origin` (the nowcast date),
        `location` (the place), `target_end_date`, `horizon` (days),
        `quantile` and `value`, the samples' quantiles at
        `QUANTILE_LEVELS` (`hub_quantiles`).
    """
    built = [get_model(m) for m in models]
    keys = ["model", "origin", "location", "target_end_date", "horizon"]
    frames = []
    for date in dates:
        for task in nowcast_tasks(triangle, date, n_samples, seed):
            samples = run_forecasts(built, [task])
            if samples.empty:
                continue
            drawn = hub_forecasts(samples, task, date).rename(
                columns={
                    "forecast_date": "origin",
                    "sample": "output_type_id",
                }
            )
            table = drawn.assign(output_type="sample")[
                [*keys, "output_type", "output_type_id", "value"]
            ]
            frames.append(hub_quantiles(table, QUANTILE_LEVELS))
    if not frames:
        return pd.DataFrame(columns=[*keys, "quantile", "value"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns="output_type").rename(
        columns={"output_type_id": "quantile"}
    )[[*keys, "quantile", "value"]]


def with_truth(
    forecasts: pd.DataFrame, triangle: pd.DataFrame, truth_delay: int
) -> pd.DataFrame:
    """Add each forecast's truth: the sum as reported $D$ days after it ends.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `germany_forecasts` or `team_forecasts` output.
    triangle : pandas.DataFrame
        Counts by Meldedatum and report date.
    truth_delay : int
        $D$, the days after a sum ends that its truth is reported by.

    Returns
    -------
    pandas.DataFrame
        `forecasts` with `truth` and `truth_vintage` (the report date of
        the truth), missing for sums with no data $D$ days after them.
    """
    ends = sorted(set(forecasts["target_end_date"]))
    if not ends:
        return forecasts.assign(truth=np.nan, truth_vintage=None)
    truth = rolling_truth(hub_target(triangle, ends), truth_delay)
    return forecasts.merge(
        truth, on=["location", "target_end_date"], how="left"
    )


def score_germany(forecasts: pd.DataFrame, truth_delay: int) -> pd.DataFrame:
    """Score quantile forecasts on their truth, as the hub scored them.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `with_truth` output.
    truth_delay : int
        $D$, which names the target.

    Returns
    -------
    pandas.DataFrame
        One row per forecast with a truth: `model`, `target`
        (`"<D>-day"`), `origin`, `truth_vintage`, `location`,
        `target_end_date`, `horizon` (days), `truth`, `SCORE_COLUMNS` and
        `pool` (the paper's level, `national`, `states` or `age`).
    """
    cols = [*_KEYS, "target_end_date", "horizon", "truth", *SCORE_COLUMNS]
    rows = forecasts.dropna(subset=["truth"]).assign(
        target=f"{truth_delay}-day"
    )
    if rows.empty:
        return pd.DataFrame(columns=[*cols, "pool"])
    truth = rows[
        ["location", "target_end_date", "truth_vintage", "truth"]
    ].drop_duplicates()
    scores = score_quantiles(
        rows.drop(columns="truth"),
        truth,
        (*_KEYS, "target_end_date"),
        QUANTILE_LEVELS,
    )
    scores["pool"] = scores["location"].map(level)
    return scores[[*cols, "pool"]]


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
    ).reindex(columns=[0.025, 0.1, 0.25])
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


def team_forecasts(
    submissions: pd.DataFrame, dates: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return the hub teams' quantile nowcasts at our horizons.

    RKI's quantiles above `_RKI_ERROR` are replaced by its -1 day nowcast
    and Epiforecasts' missing 10% quantiles are filled (`_fill_tenth`), as
    the paper does. Nothing else is filled. Quantiles that cross are
    sorted within their forecast.

    Parameters
    ----------
    submissions : pandas.DataFrame
        `read_submissions` output.
    dates : sequence of str, optional
        The nowcast dates kept. All if None.

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
    if dates is not None:
        rows = rows.loc[rows["forecast_date"].isin(set(dates))]
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
    out = out.loc[out["location"].isin(set(PLACE_NAMES))]
    # Under 1% of the submitted forecasts have quantiles that cross. Their
    # values are sorted so each forecast is a distribution.
    ids = ["model", "origin", "location", "target_end_date"]
    out = out.sort_values([*ids, "quantile"])
    out["value"] = out.groupby(ids)["value"].transform(np.sort)
    return out.reset_index(drop=True)


def relative_scores(
    scores: pd.DataFrame, by: Sequence[str], baseline: str
) -> pd.DataFrame:
    """Divide each model's mean scores by the baseline's on the same forecasts.

    Parameters
    ----------
    scores : pandas.DataFrame
        `score_germany` output, for our models and the teams.
    by : sequence of str
        Columns of a cell other than `model`.
    baseline : str
        The model to divide by, such as `GermanyConfig.baseline`.

    Returns
    -------
    pandas.DataFrame
        One row per cell and model, over the forecasts the baseline also
        made: `n`, `wis`, `rel_wis`, `ae_median`, `rel_ae_median`, and
        the coverage of the 50% and 95% intervals.
    """
    match = ("target", "origin", "truth_vintage", "location", "target_end_date")
    rel = relative_skill(
        scores,
        baseline,
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


def paper_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return forecasts as the paper's long table (`paper/schema.md`).

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `with_truth` output, for our models and the teams.

    Returns
    -------
    pandas.DataFrame
        `PAPER_COLUMNS`: `study` (`germany`), `model`, `location` (`DE` or
        a state), `stratum` (an age group of Germany, or `all`),
        `origin_date`, `target_date` (the last day of the 7-day sum),
        `horizon` (days), `quantile_level`, `sample_id` (empty), `value`
        and `truth`.
    """
    ages = ~forecasts["location"].isin({GERMANY, *STATES})
    return pd.DataFrame(
        {
            "study": "germany",
            "model": forecasts["model"],
            "location": forecasts["location"].where(~ages, GERMANY),
            "stratum": forecasts["location"].where(ages, "all"),
            "origin_date": forecasts["origin"],
            "target_date": forecasts["target_end_date"],
            "horizon": forecasts["horizon"],
            "quantile_level": forecasts["quantile"],
            "sample_id": pd.array([pd.NA] * len(forecasts), dtype="Int64"),
            "value": forecasts["value"],
            "truth": forecasts["truth"],
        },
        index=forecasts.index,
    )[list(PAPER_COLUMNS)].reset_index(drop=True)
