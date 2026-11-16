# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Introductions into places with no recent cases.

A place with no case reported in the latest `k` weeks is silent
(`silent_places`).
It is exactly 0 until its first introduction.
An introduction comes each week with probability $q$ times the share of
the week not yet reported (`introduction_rates`).
$q$ is the rate of a first case for a place with none yet, and of a
re-introduction for a place seen before, over the `k` weeks to the
place's start week. It combines a local rate, pooled over the place and
its siblings, and a global rate, pooled over the other places of its
level, as independent sources: $q = 1 - (1 - q_l)(1 - q_g)$.
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
    r"""Return each place's weekly probability of an introduction.

    A place with no case yet takes the rate of a first case: the weeks
    with a case over the weeks with none before them.
    A place seen before takes the rate of a re-introduction: the weeks
    with a case over the weeks with an earlier case but none in the `k`
    weeks before them.
    Each rate counts the `k` weeks to each place's start week, the weeks
    that set the spread of the walk. Only complete weeks before the
    origin are used.
    The local rate $q_l$ pools the place and its siblings, the children
    of its parent, so zones pool within their province. The global rate
    $q_g$ pools the other places of the same level (zones of the other
    provinces); provinces and the country have none, so their $q_g = 0$.
    The two are independent sources, $q = 1 - (1 - q_l)(1 - q_g)$, so $q$
    is 0 only where the whole level has no introduction of that kind.

    Parameters
    ----------
    reported : numpy.ndarray
        Weekly reported counts, `[L, W]`.
    start : numpy.ndarray
        Start week of each place, `[L]`.
    parent : numpy.ndarray
        Index of each place's parent, or -1, `[L]`.
    k : int
        Weeks counted, and silent weeks before a re-introduction.

    Returns
    -------
    numpy.ndarray
        `[L]`, 0 where neither pool has a week at risk with a case.
    """
    pos = reported > 0
    before = np.cumsum(np.pad(pos, ((0, 0), (1, 0))), axis=1)[:, :-1]
    week = np.arange(pos.shape[1])[None, :]
    upto = week <= start[:, None]
    window = upto & (week > start[:, None] - k)
    quiet = np.zeros_like(pos)
    quiet[:, k:] = before[:, k:] == before[:, :-k]
    fresh, seen = upto & (before == 0), upto & quiet & (before > 0)
    first = _pooled(fresh & window, pos, parent)
    again = _pooled(seen & window, pos, parent)
    out: np.ndarray = np.where(pos.any(axis=1), again, first)
    return out


def _depth(parent: np.ndarray) -> np.ndarray:
    """Return each place's level, its steps from the root, `[L]`."""
    depth = np.zeros(parent.size, dtype=int)
    up = parent.copy()
    while (up >= 0).any():
        depth += up >= 0
        up = np.where(up >= 0, parent[np.maximum(up, 0)], -1)
    return depth


def _pooled(
    risk: np.ndarray, pos: np.ndarray, parent: np.ndarray
) -> np.ndarray:
    """Combine the sibling pool's rate with the rest of the level's.

    `risk` `[L, W]` marks each place's weeks at risk in the window.
    """
    n = parent.size
    group = np.where(parent >= 0, parent, n + np.arange(n))
    hits_p = (risk & pos).sum(axis=1).astype(float)
    weeks_p = risk.sum(axis=1).astype(float)
    hits = np.bincount(group, hits_p, 2 * n)[group]
    weeks = np.bincount(group, weeks_p, 2 * n)[group]
    local = hits / np.maximum(weeks, 1)
    level = _depth(parent)
    hits_l = np.bincount(level, hits_p)[level] - hits
    weeks_l = np.bincount(level, weeks_p)[level] - weeks
    rest = hits_l / np.maximum(weeks_l, 1)
    out: np.ndarray = 1 - (1 - local) * (1 - rest)
    return out


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
