# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A Gaussian copula that pairs a model's marginal draws into trajectories.

A model that gives each place and step only its marginal, such as
Chronos-2's quantiles, draws each cell independently.
`GaussianCopula.couple` pairs those draws into trajectories: each
trajectory draws one correlated normal vector $z$ over places and steps,
and every cell's draws are reordered across trajectories so they rank as
$z$ does.
This is the Gaussian copula with $u = \Phi(z)$ read off each cell's own
draws, so each cell keeps exactly the draws it had.

The correlation of $z$ is a place correlation times a step correlation.
Between steps it is $\rho_t^{|h - h'|}$ for weeks, and $\rho_t^{|h - h'|/7}$
for days.
Between places it is $\rho_p$ for two provinces and $\rho_n$ for the
country and a province, with the country first.
`estimate_copula` estimates the three from the normal scores of matured
weeks at past origins (`normal_scores`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import ndtri

from bdbvhubrenewal.forecasters.base import NATIONAL

__all__ = [
    "MAX_RHO",
    "GaussianCopula",
    "estimate_copula",
    "normal_scores",
]

MAX_RHO: Final = 0.95
"""Largest correlation `estimate_copula` gives."""

_CELL = ["model", "origin", "location", "target_end_date"]


def _factor(corr: np.ndarray) -> np.ndarray:
    """Return $A$ with $A A^T$ a correlation matrix near `corr`.

    Negative eigenvalues are set to zero and the rows rescaled to unit
    length. A row left with no length keeps its own unit normal.
    """
    values, vectors = np.linalg.eigh(corr)
    a = vectors * np.sqrt(np.clip(values, 0.0, None))
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    empty = norm[:, 0] == 0.0
    a[empty] = np.eye(len(a))[empty]
    norm[empty] = 1.0
    return np.asarray(a / norm)


@dataclass(frozen=True)
class GaussianCopula:
    r"""Correlations of a Gaussian copula over places and weeks.

    Attributes
    ----------
    rho_time : float
        $\rho_t$, between consecutive weeks of a place.
    rho_provinces : float
        $\rho_p$, between two provinces in a week.
    rho_national : float
        $\rho_n$, between the country and a province in a week.
    """

    rho_time: float = 0.0
    rho_provinces: float = 0.0
    rho_national: float = 0.0

    def __post_init__(self) -> None:
        """Check each correlation is in [0, 1)."""
        for name in ("rho_time", "rho_provinces", "rho_national"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1 (not 1).")

    def normals(
        self,
        rng: np.random.Generator,
        n: int,
        n_places: int,
        n_steps: int,
        step_days: int = 7,
    ) -> np.ndarray:
        r"""Draw correlated standard normals over places and steps.

        Where $\rho_p$ and $\rho_n$ give no correlation matrix, the
        nearby one is used (`_factor`).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator.
        n : int
            Trajectories.
        n_places : int
            Places, the country first.
        n_steps : int
            Steps.
        step_days : int
            Days in a step, 7 for weeks and 1 for days.

        Returns
        -------
        numpy.ndarray
            `[n, n_places, n_steps]`.
        """
        place = np.full((n_places, n_places), self.rho_provinces)
        place[0, :] = place[:, 0] = self.rho_national
        np.fill_diagonal(place, 1.0)
        lag = np.abs(np.subtract.outer(np.arange(n_steps), np.arange(n_steps)))
        step = self.rho_time ** (lag * step_days / 7)
        e = rng.standard_normal((n, n_places, n_steps))
        return np.asarray(
            np.einsum(
                "pi,nij,tj->npt",
                _factor(place),
                e,
                _factor(step),
                optimize=True,
            )
        )

    def couple(
        self, draws: np.ndarray, rng: np.random.Generator, step_days: int = 7
    ) -> np.ndarray:
        """Pair independent draws into trajectories.

        Parameters
        ----------
        draws : numpy.ndarray
            Draws of each place and step, `[n, n_places, n_steps]`, the
            country first.
        rng : numpy.random.Generator
            Random generator.
        step_days : int
            Days in a step, 7 for weeks and 1 for days.

        Returns
        -------
        numpy.ndarray
            The same draws of each cell, reordered across trajectories so
            they rank as the copula's normals do.
        """
        n, n_places, n_steps = draws.shape
        z = self.normals(rng, n, n_places, n_steps, step_days)
        rank = np.argsort(np.argsort(z, axis=0), axis=0)
        return np.take_along_axis(np.sort(draws, axis=0), rank, axis=0)


def normal_scores(
    forecasts: pd.DataFrame, truth: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    r"""Return the normal score of each truth in its forecast's draws.

    The score is $\Phi^{-1}$ of the randomised PIT: with $m$ draws, $b$
    below the truth and $e$ equal to it, $(b + U (e + 1)) / (m + 1)$ for
    $U$ uniform.
    A forecast whose draws are all equal gives no score, as its PIT is
    $U$ alone.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Columns `model`, `origin`, `location`, `target_end_date`, `sample`
        and `value`.
    truth : pandas.DataFrame
        Columns `location`, `target_end_date` and `truth`, plus any
        columns it should be matched on.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    pandas.DataFrame
        One row per forecast with a truth and spread: `model`, `origin`,
        `location`, `target_end_date` and `z`.
    """
    on = [c for c in truth.columns if c != "truth"]
    rows = forecasts.merge(truth.dropna(subset=["truth"]), on=on)
    rows = rows.assign(
        below=rows["value"] < rows["truth"],
        equal=rows["value"] == rows["truth"],
    )
    out = rows.groupby(_CELL, as_index=False).agg(
        below=("below", "sum"),
        equal=("equal", "sum"),
        m=("value", "size"),
        spread=("value", "nunique"),
    )
    out = out.loc[out["spread"] > 1].reset_index(drop=True)
    u = 1.0 - rng.uniform(size=len(out))
    pit = (out["below"] + u * (out["equal"] + 1)) / (out["m"] + 1)
    return out[_CELL].assign(z=ndtri(pit.to_numpy()))


def _corr(pairs: pd.DataFrame) -> float:
    """Pearson correlation of `z_a` and `z_b`, in [0, `MAX_RHO`]."""
    if len(pairs) < 2:
        return 0.0
    r = float(np.corrcoef(pairs["z_a"], pairs["z_b"])[0, 1])
    return float(np.clip(np.nan_to_num(r), 0.0, MAX_RHO))


def estimate_copula(scores: pd.DataFrame) -> GaussianCopula:
    r"""Estimate the copula's correlations from normal scores.

    Each is the correlation of the scores of pairs from one forecast:
    $\rho_t$ of a place's consecutive weeks, $\rho_p$ of two provinces in a
    week, and $\rho_n$ of the country and a province in a week. Each is
    floored at zero and capped at `MAX_RHO`.

    Parameters
    ----------
    scores : pandas.DataFrame
        `normal_scores`: `model`, `origin`, `location`,
        `target_end_date` and `z`.

    Returns
    -------
    GaussianCopula
    """
    week = pd.to_datetime(scores["target_end_date"])
    scores = scores.drop(columns="target_end_date").assign(week=week)
    forecast = ["model", "origin"]
    later = scores.assign(week=scores["week"] - pd.Timedelta(days=7))
    time = scores.merge(
        later, on=[*forecast, "location", "week"], suffixes=("_a", "_b")
    )
    same = scores.merge(scores, on=[*forecast, "week"], suffixes=("_a", "_b"))
    a_national = same["location_a"] == NATIONAL
    b_national = same["location_b"] == NATIONAL
    provinces = same.loc[
        ~a_national & ~b_national & (same["location_a"] < same["location_b"])
    ]
    national = same.loc[a_national & ~b_national]
    return GaussianCopula(_corr(time), _corr(provinces), _corr(national))
