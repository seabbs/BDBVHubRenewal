# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Introductions into places with no recent cases.

A place with no case reported in the latest `k` weeks is silent
(`silent_places`).
It is exactly 0 until its first introduction.
An introduction comes each week with probability $q$ times the share of
the week not yet reported (`introduction_rates`).
$q$ is the rate of a first case for a place with none yet, and of a
re-introduction for a place seen before, over the `k` weeks to the
place's start week, pooled over the place and its siblings.
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
    """Return each place's weekly probability of an introduction.

    Only the `k` weeks to each place's start week count, the weeks that
    set the spread of the walk, so only complete weeks before the origin
    are used.
    A place with no case yet takes the rate of a first case: the weeks
    with a case over the weeks with none before them.
    A place seen before takes the rate of a re-introduction: the weeks
    with a case over the weeks with an earlier case but none in the `k`
    weeks before them.
    Each rate pools the place and its siblings, the children of its
    parent, so zones pool within their province.
    Where the siblings have no introduction in the window, there is no
    estimate, and the place takes the pool of its parent's level: all
    zones of the country for a zone.
    The country has no siblings, so it is a group of its own.

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
        `[L]`, 0 where neither the place's pool nor any above it has an
        introduction.
    """
    pos = reported > 0
    before = np.cumsum(np.pad(pos, ((0, 0), (1, 0))), axis=1)[:, :-1]
    week = np.arange(pos.shape[1])[None, :]
    window = (week <= start[:, None]) & (week > start[:, None] - k)
    quiet = np.zeros_like(pos)
    quiet[:, k:] = before[:, k:] == before[:, :-k]
    first = _pooled(window & (before == 0), pos, parent)
    again = _pooled(window & quiet & (before > 0), pos, parent)
    return np.where(pos.any(axis=1), again, first)


def _pooled(
    risk: np.ndarray, pos: np.ndarray, parent: np.ndarray
) -> np.ndarray:
    """Pool the introductions over the weeks at risk, as `introduction_rates`.

    The pool of a place at level `j` above it is every place at its depth
    with the same ancestor `j` levels up, its siblings at `j = 1`. The
    nearest pool with an introduction gives the rate.
    """
    hits = (risk & pos).sum(axis=1).astype(float)
    weeks = risk.sum(axis=1).astype(float)
    n = parent.size
    depth = np.zeros(n, dtype=int)
    up = parent.copy()
    while (up >= 0).any():
        depth += up >= 0
        up = np.where(up >= 0, parent[np.maximum(up, 0)], -1)
    # A root is a pool of its own.
    root = parent < 0
    out = np.where(root, hits / np.maximum(weeks, 1), 0.0)
    todo = ~root
    anc = parent.copy()
    while todo.any():
        live = anc >= 0
        _, pool = np.unique(
            depth[live] * (n + 1) + anc[live], return_inverse=True
        )
        pool_hits = np.bincount(pool, hits[live])[pool]
        pool_weeks = np.bincount(pool, weeks[live])[pool]
        take = np.zeros(n, dtype=bool)
        take[live] = pool_hits > 0
        take &= todo
        rate = np.zeros(n)
        rate[live] = pool_hits / np.maximum(pool_weeks, 1)
        out[take] = rate[take]
        todo &= ~take & live
        anc = np.where(live, parent[np.maximum(anc, 0)], -1)
        todo &= anc >= 0
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
