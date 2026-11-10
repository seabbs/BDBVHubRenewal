# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Introductions into places with no recent cases.

A place with no case reported in the latest `k` weeks is silent
(`silent_places`).
It is exactly 0 until its first introduction.
An introduction comes each week with probability $q$ times the share of
the week not yet reported, with $q$ pooled over the place and its
siblings (`introduction_rates`).
It brings a count drawn from Poisson(1) given at least 1, and the place
then walks on $\log(1 + y)$ with its walk spread (`silent_trajectories`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from bdbvhubrenewal.forecasters.walk import random_walks

__all__ = [
    "introduction_rates",
    "silent_places",
    "silent_trajectories",
]


def silent_places(reported: np.ndarray, cut_week: int, k: int) -> np.ndarray:
    """Return the places with no case in the `k` weeks to `cut_week`.

    Parameters
    ----------
    reported : numpy.ndarray
        Weekly reported counts, `[L, W]`.
    cut_week : int
        Index of the week holding the cut-off.
    k : int
        Weeks without a case.

    Returns
    -------
    numpy.ndarray
        `[L]`, true where silent.
    """
    recent = reported[:, cut_week - k + 1 : cut_week + 1]
    return np.asarray(recent.sum(axis=1) == 0)


def introduction_rates(
    reported: np.ndarray, start: np.ndarray, parent: np.ndarray, k: int
) -> np.ndarray:
    """Return each place's weekly probability of a first case.

    A week is at risk when its place reported no case in the `k` weeks
    before it, and it is an introduction when the place reports a case
    in it.
    The probability is the introductions over the weeks at risk of the
    place and its siblings, the children of its parent, so zones pool
    within their province and provinces within the country.
    Weeks up to each place's start week count, so only complete weeks
    and data before the origin are used.

    Parameters
    ----------
    reported : numpy.ndarray
        Weekly reported counts, `[L, W]`.
    start : numpy.ndarray
        Start week of each place, `[L]`.
    parent : numpy.ndarray
        Index of each place's parent, or -1, `[L]`.
    k : int
        Silent weeks before a week at risk.

    Returns
    -------
    numpy.ndarray
        `[L]`, 0 where no week is at risk.
    """
    pos = reported > 0
    before = np.cumsum(np.pad(pos, ((0, 0), (1, 0))), axis=1)
    risk = np.zeros_like(pos)
    risk[:, k:] = before[:, k:-1] == before[:, : -k - 1]
    risk &= np.arange(pos.shape[1])[None, :] <= start[:, None]
    first = (risk & pos).sum(axis=1)
    risk_n = risk.sum(axis=1)
    # The country has no siblings, so it is a group of its own.
    n = parent.size
    group = np.where(parent >= 0, parent, n + np.arange(n))
    first_g = np.bincount(group, first, 2 * n)[group]
    risk_g = np.bincount(group, risk_n, 2 * n)[group]
    return first_g / np.maximum(risk_g, 1)


def silent_trajectories(
    key: jax.Array,
    silent: np.ndarray,
    q: np.ndarray,
    unreported: np.ndarray,
    start: np.ndarray,
    sigma: np.ndarray,
    h: int,
    n: int,
) -> np.ndarray:
    r"""Draw the counts of silent places after their start week.

    A silent place is exactly 0 until its first introduction.
    Week `start + 1 + i` brings one with probability $q$ times the share
    of the week not yet reported, so an introduction is likelier in
    recent weeks, whose cases may still be reported, and has probability
    $q$ in weeks without data.
    The introduction brings a count drawn from Poisson(1) given at least
    1, and the place then walks on $\log(1 + y)$ with spread `sigma`.

    Parameters
    ----------
    key : jax.Array
        Random key.
    silent : numpy.ndarray
        True where a place is silent, `[L]`.
    q : numpy.ndarray
        Weekly probability of a first case (`introduction_rates`), `[L]`.
    unreported : numpy.ndarray
        Share of each week not yet reported, `[L, W]`.
        Weeks after `W` are wholly unreported.
    start : numpy.ndarray
        Week after which the counts are drawn, `[L]`.
    sigma : numpy.ndarray
        Standard deviation of a weekly walk step, `[L]`.
    h : int
        Weeks to draw.
    n : int
        Samples to draw.

    Returns
    -------
    numpy.ndarray
        Counts for weeks `start + 1` to `start + h`, `[L, n, h]`, and 0
        for places that are not silent.
    """
    out = np.zeros((silent.size, n, h))
    idx = np.flatnonzero(silent)
    if idx.size == 0:
        return out
    u = start[idx, None] + 1 + np.arange(h)[None, :]
    share = np.take_along_axis(
        np.pad(unreported[idx], ((0, 0), (0, h + 1)), constant_values=1.0),
        u,
        axis=1,
    )
    k_walk, k_intro, k_count = jax.random.split(key, 3)
    v = np.asarray(jax.random.uniform(k_count, (idx.size, n)), np.float64)
    x0 = stats.poisson.ppf(np.exp(-1.0) + v * -np.expm1(-1.0), 1.0)
    keys = jax.random.split(k_walk, idx.size)
    _, drawn = random_walks(
        keys,
        jnp.asarray(x0, jnp.float32),
        jnp.asarray(sigma[idx], jnp.float32),
        h,
    )
    y = np.asarray(drawn, dtype=np.float64)
    hit = np.asarray(jax.random.uniform(k_intro, (idx.size, n, h))) < (
        q[idx, None, None] * share[:, None, :]
    )
    first = np.where(hit.any(axis=2), np.argmax(hit, axis=2), h)
    lag = np.arange(h)[None, None, :] - first[:, :, None]
    # The introduced count is the first week's, and the walk from it
    # gives the weeks after.
    walked = np.take_along_axis(y, np.clip(lag - 1, 0, h - 1), axis=2)
    paths = np.where(lag >= 1, walked, 0.0)
    out[idx] = np.where(lag == 0, x0[:, :, None], paths)
    return out
