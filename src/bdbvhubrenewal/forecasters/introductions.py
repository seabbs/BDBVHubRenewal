# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Introductions into places with no recent cases.

A place with no case reported in the latest `k` weeks is silent
(`silent_places`).
It is exactly 0 until its first introduction.
An introduction comes each week with probability $q$ times the share of
the week not yet reported (`introduction_probability`), with $q$ pooled
over the place and its siblings (`introduction_rates`).
It brings a count drawn from Poisson(1) given at least 1, and the place
then walks on $\log(1 + y)$ with its walk spread
(`walk_with_introductions`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from bdbvhubrenewal.forecasters.walk import random_walk

__all__ = [
    "introduction_probability",
    "introduction_rates",
    "silent_places",
    "walk_with_introductions",
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


def introduction_probability(
    q: np.ndarray, complete: np.ndarray, start: np.ndarray, h: int
) -> np.ndarray:
    """Return each place's weekly introduction probability after `start`.

    Week `start + 1 + i` has probability $q$ times the share of the week
    not yet reported, so an introduction is likelier in recent weeks,
    whose cases may still be reported, and has probability $q$ in weeks
    without data.

    Parameters
    ----------
    q : numpy.ndarray
        Weekly probability of a first case, `[L]`.
    complete : numpy.ndarray
        Share of each week reported, `[L, W]`.
    start : numpy.ndarray
        Week after which introductions are drawn, `[L]`.
    h : int
        Weeks to cover.

    Returns
    -------
    numpy.ndarray
        `[L, H]`.
    """
    n_data = complete.shape[1]
    u = start[:, None] + 1 + np.arange(h)[None, :]
    seen = np.take_along_axis(np.pad(complete, ((0, 0), (0, h + 1))), u, axis=1)
    seen = np.where(u < n_data, np.clip(seen, 0.0, 1.0), 0.0)
    return q[:, None] * (1.0 - seen)


def walk_with_introductions(
    key: jax.Array,
    x0: np.ndarray,
    sigma: np.ndarray,
    silent: np.ndarray,
    intro: np.ndarray,
) -> np.ndarray:
    """Return walk counts for every place from `x0` of shape `[L, S]`.

    Each place walks with spread `sigma` `[L]`. A `silent` `[L]` place is
    0 until its first introduction, drawn week by week with probability
    `intro` `[L, H]`. The introduction brings a count drawn from
    Poisson(1) given at least 1, and the place walks from it.
    """
    h = intro.shape[1]
    k_walk, k_intro, k_count = jax.random.split(key, 3)
    u = np.asarray(jax.random.uniform(k_count, x0.shape), np.float64)
    first_count = stats.poisson.ppf(np.exp(-1.0) + u * -np.expm1(-1.0), 1.0)
    x0 = np.where(silent[:, None], first_count, x0)
    keys = jax.random.split(k_walk, x0.shape[0])
    _, drawn = jax.vmap(lambda k, x, s: random_walk(k, x, s, h))(
        keys,
        jnp.asarray(x0, jnp.float32),
        jnp.asarray(sigma, jnp.float32),
    )
    y: np.ndarray = np.asarray(drawn, dtype=np.float64)
    hit = (
        np.asarray(jax.random.uniform(k_intro, (*x0.shape, h)))
        < (intro[:, None, :])
    )
    first = np.where(hit.any(axis=2), np.argmax(hit, axis=2), h)
    first = np.where(silent[:, None], first, 0)
    lag = np.arange(h)[None, None, :] - first[:, :, None]
    # A silent place has its introduced count in its first week and walks
    # from it after.
    step = np.where(silent[:, None, None], lag - 1, lag)
    walked = np.take_along_axis(y, np.clip(step, 0, h - 1), axis=2)
    out = np.where(step >= 0, walked, 0.0)
    return np.where(silent[:, None, None] & (lag == 0), x0[:, :, None], out)
