# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A Gaussian copula that pairs the walk models' province draws.

`randomwalk` and `randomwalknowcast` draw each province on its own, so
the country, the sum of the provinces, is too narrow.
`pair_walks` pairs the provinces' draws within each week: each week, each
trajectory draws one normal per province with correlation $\rho_p$ between
any two, and each cell's draws are reordered across trajectories so they
rank as the normals do (the Schaake shuffle, Clark et al. 2004, as
ensemble copula coupling uses it, Schefzik et al. 2013).
So each cell keeps exactly the draws it had, whether it is nowcast or
walking. Weeks are not paired with each other, so a province's own weeks
are not kept as one trajectory, and zones and the country are not paired
at all.
`estimate_rho` estimates $\rho_p$ from normal scores of a model's own
matured errors (`error_scores`), shrunk towards a prior as a guard for
thin windows.
"""

from __future__ import annotations

import zlib
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import ndtri

from bdbvhubrenewal.forecasters.base import NATIONAL

__all__ = [
    "MAX_RHO",
    "PRIOR",
    "PRIOR_WEIGHT",
    "WINDOW_WEEKS",
    "error_scores",
    "estimate_rho",
    "pair_walks",
]

MAX_RHO: Final = 0.95
"""Largest $\rho_p$ `estimate_rho` gives."""

PRIOR: Final = 0.7
"""The $\rho_p$ estimates are shrunk towards."""

PRIOR_WEIGHT: Final = 10.0
"""Pairs the prior counts as in `estimate_rho`'s shrinkage, a guard for
thin windows only. Chosen on the release backtest from 0, 10, 30 and 100,
which scored within 1% of one another; 10 is the weakest that still
guards a window with few pairs."""

WINDOW_WEEKS: Final = 8
"""The latest weeks of releases whose errors are used (`error_scores`)."""


def _provinces(
    locations: pd.Series | tuple[str, ...], national: str
) -> np.ndarray:
    """Return whether each location is a province, not the country or a zone."""
    codes = pd.Series(list(locations), dtype=object).astype(str)
    return ((codes != national) & (codes.str.count("-") < 2)).to_numpy()


def estimate_rho(
    scores: pd.DataFrame,
    prior: float = PRIOR,
    weight: float = PRIOR_WEIGHT,
    national: str = NATIONAL,
) -> dict[str, float | int | bool]:
    r"""Estimate $\rho_p$ from normal scores, with its record.

    $r$ is the Pearson correlation of the scores of every pair of two
    provinces in the same week of one forecast; the country and zones are
    left out. With $n$ pairs and the prior $\rho_0$ the estimate is
    $(n r + w \rho_0) / (n + w)$, the prior's where the pairs give no $r$,
    then floored at zero and capped at `MAX_RHO`.

    Parameters
    ----------
    scores : pandas.DataFrame
        `error_scores`: `model`, `origin`, `location`,
        `target_end_date` and `z`.
    prior : float
        $\rho_0$.
    weight : float
        $w$, the pairs the prior counts as.
    national : str
        The country's location code.

    Returns
    -------
    dict
        `rho_provinces`, `n_pairs` (the pairs behind it, 0 if it fell
        back) and `fallback` (whether it fell back to the prior for want of
        pairs that define it).
    """
    fallback: dict[str, float | int | bool] = {
        "rho_provinces": prior,
        "n_pairs": 0,
        "fallback": True,
    }
    if scores.empty:
        return fallback
    rows = scores.loc[_provinces(scores["location"], national)]
    key = ["model", "origin", "target_end_date"]
    pairs = rows.merge(rows, on=key, suffixes=("_a", "_b"))
    pairs = pairs.loc[pairs["location_a"] < pairs["location_b"]]
    r = None
    if len(pairs) >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(pairs["z_a"], pairs["z_b"])[0, 1]
        r = float(r) if np.isfinite(r) else None
    if r is None:
        return fallback
    n = len(pairs)
    rho = float(np.clip((n * r + weight * prior) / (n + weight), 0, MAX_RHO))
    return {"rho_provinces": rho, "n_pairs": int(n), "fallback": False}


def error_scores(
    added: np.ndarray,
    remaining: np.ndarray,
    valid: np.ndarray,
    size: np.ndarray,
    releases: list[str],
    ages: np.ndarray,
    locations: tuple[str, ...],
    model: str,
    seed: int,
    window_weeks: int = WINDOW_WEEKS,
) -> pd.DataFrame:
    r"""Return normal scores of a model's own nowcasts at earlier releases.

    At each earlier release $r$ the model nowcast the count still to come
    of each place and week back, as a negative binomial with mean
    `remaining` and size $\phi_k$; the origin release reports what came
    (`added`). The score is $\Phi^{-1}$ of the randomised PIT of `added`
    in that distribution, with the uniform drawn from a hash of the seed,
    model, release, place and week back, so a score does not depend on the
    others. Releases more than `window_weeks` weeks before the
    origin are left out, as are cells with no nowcast spread.

    Parameters
    ----------
    added, remaining, valid : numpy.ndarray
        `chain.Errors` fields, `[R, L, K]`.
    size : numpy.ndarray
        Negative binomial size of each place and week back, `[L, K]`.
    releases : list of str
        The releases, `[R]`.
    ages : numpy.ndarray
        Days from each release's cut-off to the origin's, `[R]`.
    locations : tuple of str
        The places, `[L]`.
    model : str
        The model, as the scores' `model`.
    seed : int
        Seed of the uniform draws.
    window_weeks : int
        The latest weeks of releases used.

    Returns
    -------
    pandas.DataFrame
        `model`, `origin` (the release), `location`, `target_end_date`
        (a date 7 days apart per week back, so consecutive weeks pair) and
        `z`, as `estimate_rho` takes them.
    """
    from scipy.stats import nbinom

    cols = ["model", "origin", "location", "target_end_date", "z"]
    keep = np.asarray(ages) <= 7 * window_weeks
    if not keep.any():
        return pd.DataFrame(columns=cols)
    mu = np.asarray(remaining, float)[keep]
    y = np.asarray(added, float)[keep]
    ok = np.asarray(valid, bool)[keep] & (mu > 0)
    phi = np.broadcast_to(np.asarray(size, float)[None], mu.shape)
    ok &= np.isfinite(phi) & (phi > 0)
    r, loc, k = np.nonzero(ok)
    if r.size == 0:
        return pd.DataFrame(columns=cols)
    n, m, yy = phi[r, loc, k], mu[r, loc, k], np.maximum(y[r, loc, k], 0)
    p = n / (n + m)
    below = nbinom.cdf(yy - 1, n, p)
    at = nbinom.pmf(yy, n, p)
    rel = np.asarray(releases, dtype=object)[keep][r]
    codes = np.asarray(locations, dtype=object)[loc]
    keys = [
        f"{seed}|{model}|{a}|{b}|{c}"
        for a, b, c in zip(rel, codes, k, strict=True)
    ]
    u = (np.array([zlib.crc32(x.encode()) for x in keys]) + 0.5) / 2.0**32
    pit = np.clip(below + u * at, 1e-12, 1 - 1e-12)
    week = pd.Timestamp("2000-01-01") - pd.to_timedelta(7 * k, unit="D")
    return pd.DataFrame(
        {
            "model": model,
            "origin": rel,
            "location": codes,
            "target_end_date": week.strftime("%Y-%m-%d"),
            "z": ndtri(pit),
        }
    )[cols]


def pair_walks(
    values: np.ndarray,
    locations: tuple[str, ...],
    rho: float,
    rng: np.random.Generator,
    national: str = NATIONAL,
) -> np.ndarray:
    r"""Pair a walk model's province draws within each week.

    Each week, each trajectory draws one normal per place, with
    correlation `rho` between any two, and each province's draws that week
    are reordered to rank as its normals do, whether the province is
    nowcast or walking that week. So every week's provinces are paired,
    but a province's own weeks are not kept as one trajectory, which
    nothing submitted or scored uses. The country and zones keep their
    draws as they are.

    Parameters
    ----------
    values : numpy.ndarray
        Draws `[S, L, H]`.
    locations : tuple of str
        The places' codes.
    rho : float
        $\rho_p$, in [0, 1).
    rng : numpy.random.Generator
        Random generator.
    national : str
        The country's code.

    Returns
    -------
    numpy.ndarray
        `values` with each province cell's draws reordered across
        trajectories.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be between 0 and 1 (not 1).")
    n, n_places, n_steps = values.shape
    paired = _provinces(locations, national)
    if not paired.any():
        return values
    z = np.sqrt(rho) * rng.standard_normal((n, 1, n_steps)) + np.sqrt(
        1 - rho
    ) * rng.standard_normal((n, n_places, n_steps))
    rank = np.argsort(np.argsort(z, axis=0), axis=0)
    index = np.take_along_axis(
        np.argsort(values, axis=0, kind="stable"), rank, axis=0
    )
    index[:, ~paired] = np.arange(n)[:, None, None]
    return np.take_along_axis(values, index, axis=0)
