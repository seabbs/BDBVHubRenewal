# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""How much pairing the walk models' draws matters (the sensitivity page).

`randomwalk` and `randomwalknowcast` pair their draws inside their
forecast (`forecasters.copula.pair_walks`). Here each forecasts a task once
and gives its draws under every pairing (`VARIANTS`): none (`off`), the
prior (`prior`), and the rolling estimate at each shrinkage strength
(`rolling_w<w>`), by the step the model itself calls
(`RandomWalk.variant`). Pairing only reorders draws, so per-cell scores
are equal across variants and only sums and joint scores move.

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
    "WEIGHTS",
    "summarise_pairing",
    "variant_forecasts",
    "variant_scores",
]

MODELS: Final = ("randomwalk", "randomwalknowcast")
"""The models that pair their draws."""

WEIGHTS: Final = (0.0, 10.0, 30.0, 100.0)
"""Shrinkage strengths compared: the prior's weight in pairs."""

VARIANTS: Final = (
    ("off", None),
    ("prior", None),
    *((f"rolling_w{int(w)}", w) for w in WEIGHTS),
)
"""Each variant's label and shrinkage strength (None: not rolling)."""

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
    """Forecast a task once and return it under every pairing.

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
        model.forecast(task)
    except ModelUnavailableError:
        return {}
    out = {}
    for label, weight in VARIANTS:
        pairing = "rolling" if label.startswith("rolling") else label
        drawn = model.variant(pairing, weight)
        zones = drawn["location"].str.count("-") >= 2
        out[label] = drawn if zones.all() else add_national(drawn)
    return out


def _level(location: pd.Series, levels: dict[str, str]) -> pd.Series:
    """Each location's level: national, a named level, or zones."""
    zone = location.str.count("-") >= 2
    out = location.map(levels).fillna("provinces")
    out[location == NATIONAL] = "national"
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

    Returns
    -------
    pandas.DataFrame
        `SCORE_COLUMNS`: the mean of each score over the forecast's cells
        or sums, and their number `n`.
    """
    rows = samples.merge(truth[[*_CELL, "truth"]], on=_CELL)
    if rows.empty:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    rows = rows.assign(level=_level(rows["location"], levels or {}))
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
