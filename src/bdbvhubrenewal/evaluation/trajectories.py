# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""How pairing draws into trajectories changes a forecast's scores.

A model that draws each place and week on its own is paired by the
rolling copula (`bdbvhubrenewal.forecasters.copula`).
`copula_arms` makes the forecasts of each arm of the comparison from the
same draws: `unpaired`, `fixed` (the copula's prior), `rolling` (the
rolling copula) and `rolling_w25` (the rolling copula with the prior
worth 25 pairs, not 100).
`copula_scores` scores each arm's forecast at each origin: the energy
and variogram scores of the vector of every place and week, on
$\log(x + 1)$, the coverage of each place's sum over the weeks and of
each week's sum over the places, and the mean WIS of the places and of
the country.
Pairing leaves each place's WIS as it is, and changes the country's
only where the country is the sum of the places.
`copula_rho` lists each arm's correlations by origin.
"""

from __future__ import annotations

import zlib
from typing import Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.scoring import (
    coverage_sample,
    energy_sample,
    variogram_sample,
    wis_sample,
)
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.copula import (
    PRIOR,
    RollingCopula,
    pair_samples,
)

__all__ = [
    "ARMS",
    "RHO_COLUMNS",
    "SCORE_COLUMNS",
    "SENSITIVITY_WEIGHT",
    "copula_arms",
    "copula_rho",
    "copula_scores",
]

ARMS: Final = ("unpaired", "fixed", "rolling", "rolling_w25")
"""The arms of the comparison, in order."""

SENSITIVITY_WEIGHT: Final = 25.0
"""The prior's weight in pairs in the `rolling_w25` arm."""

SCORE_COLUMNS: Final = (
    "study",
    "strata",
    "model",
    "arm",
    "origin_date",
    "energy_log",
    "variogram_log",
    "cover50_weeks",
    "cover90_weeks",
    "cover50_places",
    "cover90_places",
    "wis_strata",
    "wis_national",
)
"""Columns of `copula_scores`."""

RHO_COLUMNS: Final = (
    "study",
    "strata",
    "model",
    "arm",
    "origin_date",
    "n_scores",
    "rho_t",
    "rho_p",
    "rho_n",
    "rho_z",
)
"""Columns of `copula_rho`."""

_CELL = ["location", "target_end_date"]


def copula_arms(
    unpaired: pd.DataFrame,
    truth: pd.DataFrame,
    seed: int,
    national: str = NATIONAL,
    fixed: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make every arm's forecasts from the same draws.

    Parameters
    ----------
    unpaired : pandas.DataFrame
        Draws of each place and week on their own: `model`, `origin`,
        `location`, `target_end_date`, `sample` and `value`.
    truth : pandas.DataFrame
        `rolling_truth`: `location`, `target_end_date`, `truth` and
        `truth_vintage`, from which the rolling arms estimate.
    seed : int
        Seed.
    national : str
        The country's location code.
    fixed : pandas.DataFrame, optional
        The `fixed` arm, as `unpaired`, for a model that pairs by the
        prior inside itself. By default `unpaired` paired by the prior.
        The rolling arms pair it again.

    Returns
    -------
    tuple of pandas.DataFrame
        The forecasts of every arm, with `arm`. Then each arm's copula by
        model and origin: `arm`, `model`, `origin`, `n_scores` and the
        three correlations.
    """
    if fixed is None:
        frames = []
        for (model, origin), rows in unpaired.groupby(["model", "origin"]):
            key = [seed, zlib.crc32(f"{model}|{origin}".encode())]
            rng = np.random.default_rng(key)
            frames.append(pair_samples(rows, PRIOR, rng, national))
        fixed = pd.concat(frames, ignore_index=True)
    prior = (
        fixed[["model", "origin"]]
        .drop_duplicates()
        .assign(
            n_scores=0,
            rho_time=PRIOR.rho_time,
            rho_provinces=PRIOR.rho_provinces,
            rho_national=PRIOR.rho_national,
            rho_zones=PRIOR.rho_zones,
        )
    )
    rolling, fits = RollingCopula(national=national).pair(fixed, truth, seed)
    w25, fits_w25 = RollingCopula(
        weight=SENSITIVITY_WEIGHT, national=national
    ).pair(fixed, truth, seed)
    samples = pd.concat(
        [
            unpaired.assign(arm="unpaired"),
            fixed.assign(arm="fixed"),
            rolling.assign(arm="rolling"),
            w25.assign(arm="rolling_w25"),
        ],
        ignore_index=True,
    )
    copulas = pd.concat(
        [
            prior.assign(arm="fixed"),
            fits.assign(arm="rolling"),
            fits_w25.assign(arm="rolling_w25"),
        ],
        ignore_index=True,
    )
    return samples, copulas


def _covered(
    rows: pd.DataFrame, by: str, levels: tuple[float, ...] = (0.5, 0.9)
) -> list[float]:
    """Share of `by` groups whose per-sample sum covers the truth's sum."""
    hits: dict[float, list[bool]] = {level: [] for level in levels}
    for _, grp in rows.groupby(by):
        draws = grp.groupby("sample")["value"].sum().to_numpy()
        obs = grp.drop_duplicates(_CELL)["truth"].sum()
        for level in levels:
            hits[level].append(coverage_sample(obs, draws, level))
    return [float(np.mean(hits[level])) if hits[level] else np.nan
            for level in levels]  # fmt: skip


def _wis(rows: pd.DataFrame) -> float:
    """Mean WIS of the place-week forecasts in `rows`."""
    if rows.empty:
        return np.nan
    scores = [
        wis_sample(float(grp["truth"].iloc[0]), grp["value"])["wis"]
        for _, grp in rows.groupby(_CELL)
    ]
    return float(np.mean(scores))


def copula_scores(
    samples: pd.DataFrame,
    truth: pd.DataFrame,
    study: str,
    strata: str,
    national: str = NATIONAL,
) -> pd.DataFrame:
    """Score each arm's forecast at each origin.

    Parameters
    ----------
    samples : pandas.DataFrame
        `copula_arms` forecasts of one study and strata.
    truth : pandas.DataFrame
        `location`, `target_end_date` and `truth`. Only place-weeks with a
        truth are scored.
    study, strata : str
        Labels of the forecasts, such as `ebola` and `provinces`.
    national : str
        The country's location code.

    Returns
    -------
    pandas.DataFrame
        `SCORE_COLUMNS`, one row per model, arm and origin.
    """
    rows = samples.merge(truth[[*_CELL, "truth"]], on=_CELL)
    out = []
    for (model, arm, origin), grp in rows.groupby(["model", "arm", "origin"]):
        wide = grp.pivot_table(
            index="sample", columns=_CELL, values="value"
        ).dropna(axis=1)
        obs = (
            grp.drop_duplicates(_CELL)
            .set_index(_CELL)["truth"]
            .reindex(wide.columns)
            .to_numpy()
        )
        draws = wide.to_numpy()
        places = grp.loc[grp["location"] != national]
        weeks = _covered(places, "location")
        across = _covered(places, "target_end_date")
        out.append(
            {
                "study": study,
                "strata": strata,
                "model": model,
                "arm": arm,
                "origin_date": origin,
                "energy_log": energy_sample(np.log1p(obs), np.log1p(draws)),
                "variogram_log": variogram_sample(
                    np.log1p(obs), np.log1p(draws)
                ),
                "cover50_weeks": weeks[0],
                "cover90_weeks": weeks[1],
                "cover50_places": across[0],
                "cover90_places": across[1],
                "wis_strata": _wis(places),
                "wis_national": _wis(grp.loc[grp["location"] == national]),
            }
        )
    return pd.DataFrame(out, columns=list(SCORE_COLUMNS))


def copula_rho(fits: pd.DataFrame, study: str, strata: str) -> pd.DataFrame:
    """List each arm's correlations by model and origin.

    Parameters
    ----------
    fits : pandas.DataFrame
        The copulas of `copula_arms`.
    study, strata : str
        Labels, as for `copula_scores`.

    Returns
    -------
    pandas.DataFrame
        `RHO_COLUMNS`.
    """
    out = fits.rename(
        columns={
            "origin": "origin_date",
            "rho_time": "rho_t",
            "rho_provinces": "rho_p",
            "rho_national": "rho_n",
            "rho_zones": "rho_z",
        }
    ).assign(study=study, strata=strata)
    return out[list(RHO_COLUMNS)].reset_index(drop=True)
