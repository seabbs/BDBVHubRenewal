# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""How much pairing the walk models' draws matters (the sensitivity page).

`randomwalk` and `randomwalknowcast` pair their province draws within
each week inside their forecast (`forecasters.copula.pair_walks`). Here
each forecasts a task once and gives its draws under both pairings
(`VARIANTS`): none (`off`) and the rolling $\rho_p$ it uses (`rolling`),
from its unpaired draws (`RandomWalk.unpaired`). Pairing only reorders
draws, so per-cell scores are equal across variants and only sums and
joint scores move.

`variant_scores` scores each variant of one forecast by level (national,
provinces or states or age groups, zones): per cell and on summed targets
(the country, which is the sum of the provinces; each place's total over
its weeks; the zones of a province each week), by sample CRPS and the
coverage of the 50% and 90% central intervals, and by the energy score of
each level's places and weeks as one vector. `summarise_pairing` gives the
mean of each score by study, level, model, variant and target.
"""

from __future__ import annotations

import dataclasses
from typing import Final, cast

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.scoring import (
    coverage_sample,
    crps_sample,
    energy_sample,
)
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastTask,
    ModelUnavailableError,
    add_national,
    get_model,
)
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk

__all__ = [
    "MODELS",
    "SCORE_COLUMNS",
    "VARIANTS",
    "summarise_pairing",
    "variant_forecasts",
    "variant_scores",
]

MODELS: Final = ("randomwalk", "randomwalknowcast")
"""The models that pair their draws."""

VARIANTS: Final = ("off", "rolling")
"""The pairings compared: none, and the rolling $\rho_p$ the models use."""

SCORE_COLUMNS: Final = (
    "study",
    "level",
    "model",
    "variant",
    "origin",
    "target",
    "metric",
    "value",
    "n",
)
"""Columns of `variant_scores`."""

_CELL = ["location", "target_end_date"]


def variant_forecasts(name: str, task: ForecastTask) -> dict[str, pd.DataFrame]:
    """Forecast a task once and return it under both pairings.

    Parameters
    ----------
    name : str
        A walk model (`MODELS`).
    task : ForecastTask
        The task.

    Returns
    -------
    dict
        Each variant's samples, by label (`VARIANTS`), with the country
        added as the sum of the provinces where the task has provinces.
        Empty where the model cannot forecast the task.
    """
    model: RandomWalk = dataclasses.replace(
        cast("RandomWalk", get_model(name)), keep_variants=True
    )
    try:
        paired = model.forecast(task)
    except ModelUnavailableError:
        return {}
    out = {}
    for label, drawn in zip(VARIANTS, (model.unpaired(), paired), strict=True):
        zones = drawn["location"].str.count("-") >= 2
        out[label] = drawn if zones.all() else add_national(drawn)
    return out


def _level(
    location: pd.Series, levels: dict[str, str], national: str
) -> pd.Series:
    """Each location's level: national, a named level, or zones."""
    zone = location.str.count("-") >= 2
    out = location.map(levels).fillna("provinces")
    out[location == national] = "national"
    out[zone] = "zones"
    return out


def _sums(
    rows: pd.DataFrame, by: list[str], target: str
) -> list[dict[str, object]]:
    """CRPS and coverage of per-sample sums of `rows` over `by` groups."""
    found: list[dict[str, object]] = []
    for _, grp in rows.groupby(by):
        draws = grp.groupby("sample")["value"].sum().to_numpy()
        obs = float(grp.drop_duplicates(_CELL)["truth"].sum())
        for metric, value in (
            ("crps", crps_sample(obs, draws)),
            ("cover50", float(coverage_sample(obs, draws, 0.5))),
            ("cover90", float(coverage_sample(obs, draws, 0.9))),
        ):
            found.append({"target": target, "metric": metric, "value": value})
    return found


def variant_scores(
    samples: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    study: str,
    model: str,
    variant: str,
    origin: str,
    levels: dict[str, str] | None = None,
    national: str = NATIONAL,
) -> pd.DataFrame:
    """Score one variant of one forecast by level.

    Parameters
    ----------
    samples : pandas.DataFrame
        `location`, `target_end_date`, `sample` and `value`.
    truth : pandas.DataFrame
        `location`, `target_end_date` and `truth`. Only cells with a truth
        are scored.
    study, model, variant, origin : str
        Labels.
    levels : dict, optional
        A level for places that are neither the country nor zones (such as
        age groups); `provinces` by default.
    national : str
        The country's code.

    Returns
    -------
    pandas.DataFrame
        `SCORE_COLUMNS`: the mean of each score over the forecast's cells
        or sums, and their number `n`.
    """
    rows = samples.merge(truth[[*_CELL, "truth"]], on=_CELL)
    if rows.empty:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    rows = rows.assign(level=_level(rows["location"], levels or {}, national))
    found: list[pd.DataFrame] = []
    for level, grp in rows.groupby("level"):
        scored: list[dict[str, object]] = [
            *_sums(grp, _CELL, "cell"),
            *_sums(grp, ["location"], "sum_weeks"),
        ]
        if level == "zones":
            province = grp["location"].str.rsplit("-", n=1).str[0]
            scored += _sums(
                grp.assign(province=province),
                ["province", "target_end_date"],
                "sum_zones",
            )
            groups = list(grp.assign(province=province).groupby("province"))
        else:
            groups = [(level, grp)]
        for _, own in groups:
            wide = own.pivot(index="sample", columns=_CELL, values="value")
            obs = (
                own.drop_duplicates(_CELL)
                .set_index(_CELL)["truth"]
                .reindex(wide.columns)
                .to_numpy()
            )
            x = wide.to_numpy()
            scored += [
                {"target": "joint", "metric": "energy",
                 "value": energy_sample(obs, x)},
                {"target": "joint", "metric": "log_energy",
                 "value": energy_sample(np.log1p(obs), np.log1p(x))},
            ]  # fmt: skip
        frame = pd.DataFrame(scored)
        summary = frame.groupby(["target", "metric"], as_index=False).agg(
            value=("value", "mean"), n=("value", "size")
        )
        found.append(summary.assign(level=level))
    out = pd.concat(found, ignore_index=True).assign(
        study=study, model=model, variant=variant, origin=origin
    )
    return out[list(SCORE_COLUMNS)]


def summarise_pairing(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the mean of each score by study, level, model and variant.

    Parameters
    ----------
    scores : pandas.DataFrame
        `variant_scores` rows of many origins.

    Returns
    -------
    pandas.DataFrame
        `study`, `level`, `model`, `variant`, `target`, `metric`, `value`
        (the mean over origins, weighted by each origin's `n`), `n` and
        `relative` (the value over the `off` variant's).
    """
    keys = ["study", "level", "model", "variant", "target", "metric"]
    weighted = scores.assign(total=scores["value"] * scores["n"])
    out = weighted.groupby(keys, as_index=False).agg(
        total=("total", "sum"), n=("n", "sum")
    )
    out["value"] = out["total"] / out["n"]
    base = out.loc[out["variant"] == "off"].set_index(
        [k for k in keys if k != "variant"]
    )["value"]
    idx = pd.MultiIndex.from_frame(out[[k for k in keys if k != "variant"]])
    out["relative"] = out["value"].to_numpy() / base.reindex(idx).to_numpy()
    return out[[*keys, "value", "n", "relative"]]


def bdbv_sensitivity(
    data: str,
    config_path: str = "config/backtest.toml",
    origins: list[str] | None = None,
) -> pd.DataFrame:
    """Score both pairings on the BDBV release backtest's origins.

    Each real origin (less `skip_origins` and the latest) is forecast for
    the provinces, with the country their sum. Zones are their own
    forecasts and are not paired, so they are not scored here. Truth is
    each week's count in the first release the backtest's truth delay
    after it.

    Parameters
    ----------
    data : str
        The data directory.
    config_path : str
        The backtest config.
    origins : list of str, optional
        Only these origins (a block of a chunked run).

    Returns
    -------
    pandas.DataFrame
        `variant_scores` rows of every model, variant, origin and level.
    """
    from bdbvhubrenewal.data import load_daily, load_delays, load_target
    from bdbvhubrenewal.evaluation.backtest import load_config, real_vintages
    from bdbvhubrenewal.evaluation.nowcast import (
        resolve_truth_delay,
        rolling_truth,
    )
    from bdbvhubrenewal.evaluation.store import origin_tasks

    config = load_config(config_path)
    target, delays = load_target(data), load_delays(data)
    daily = load_daily(data)
    real = target.loc[target["kind"] == "real"]
    vintages = real_vintages(real)
    truth = rolling_truth(real, resolve_truth_delay(config, target))
    tasks = origin_tasks(target, delays, config, daily)
    frames = []
    for origin in vintages[:-1]:
        if origin in config.skip_origins or origin not in tasks:
            continue
        if origins is not None and origin not in origins:
            continue
        for name in MODELS:
            for label, s in variant_forecasts(name, tasks[origin]).items():
                frames.append(
                    variant_scores(
                        s,
                        truth,
                        study="bdbv",
                        model=name,
                        variant=label,
                        origin=origin,
                    )
                )
    if not frames:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def germany_sensitivity(
    data: str,
    config_path: str = "config/germany.toml",
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """Score both pairings on the German development half.

    Each nowcast date's states tasks (with Germany their sum) and age
    groups tasks, one pair per weekday offset, are forecast as the German
    benchmark does (`data.germany.nowcast_tasks`, `hub_forecasts`). They
    give the 7-day sums ending every day from 0 to 28 days back. The truth
    is the sum reported the config's truth delay after it ends.

    Parameters
    ----------
    data : str
        The German data directory.
    config_path : str
        The German config.
    dates : list of str, optional
        Only these nowcast dates (a block of a chunked run).

    Returns
    -------
    pandas.DataFrame
        `variant_scores` rows of every model, variant, date and level.
    """
    from pathlib import Path

    from bdbvhubrenewal.data.germany import (
        AGE_GROUPS,
        hub_forecasts,
        hub_target,
        hub_triangle,
        nowcast_tasks,
    )
    from bdbvhubrenewal.evaluation import germany as de
    from bdbvhubrenewal.evaluation.nowcast import rolling_truth

    cfg = de.load_germany_config(config_path)
    triangle = hub_triangle(Path(data) / Path(cfg.triangle).name)
    run = de.nowcast_dates(cfg) if dates is None else list(dates)
    levels = dict.fromkeys(AGE_GROUPS, "ages")
    frames = []
    for date in run:
        for task in nowcast_tasks(triangle, date, cfg.n_samples, cfg.seed):
            for name in MODELS:
                drawn = variant_forecasts(name, task)
                if not drawn:
                    continue
                moved = {
                    label: hub_forecasts(
                        s.assign(model=name, as_of=np.nan), task, date
                    )
                    for label, s in drawn.items()
                }
                ends = sorted(
                    set(next(iter(moved.values()))["target_end_date"])
                )
                truth = rolling_truth(
                    hub_target(triangle, ends), cfg.truth_delay
                )
                for label, s in moved.items():
                    frames.append(
                        variant_scores(
                            s,
                            truth,
                            study="germany",
                            model=name,
                            variant=label,
                            origin=date,
                            levels={
                                **levels,
                                **{
                                    p: "states"
                                    for p in set(s["location"])
                                    if p not in levels and p != "DE"
                                },
                            },
                        )
                    )
    if not frames:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    # Germany's provinces are its states.
    out["level"] = out["level"].replace({"provinces": "states"})
    return out
