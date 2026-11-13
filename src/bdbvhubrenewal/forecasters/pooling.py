# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""The one own-or-pooled rule.

Places form a hierarchy, zone then province then country.
Each node has an own estimate, from its own counts, errors or weekly
changes, and a pooled estimate, from its children with a level of at
least `min_level` cases a week.
A parent's counts are the sum of its children's, so the parent is not
in its own pool unless no child qualifies.
A node counts as qualifying when its level is at least `min_level`, and
for the walk spread also when its estimate exists (a net variance above
0).
A qualifying node uses its own estimate.
Otherwise it uses the pooled estimate of its nearest ancestor that
qualifies, else the country's pooled estimate.
The delay, the spread of the walk and the observation model all use this
rule with one threshold, except that a zone below `ZONE_DELAY_MIN_LEVEL`
takes its province's delay (`chain.fit_delay`).
"""

from __future__ import annotations

import numpy as np

__all__ = ["MIN_LEVEL", "ZONE_DELAY_MIN_LEVEL", "pool_matrix", "source_index"]

MIN_LEVEL = 10.0
"""Cases a week at which a node uses its own estimates."""

ZONE_DELAY_MIN_LEVEL = 50.0
"""Cases a week at which a zone uses its own delay, not its province's.

Chosen by the zone backtest of `randomwalknowcast` over 5, 10, 20, 30, 50
and 100 cases a week: the smallest whose WIS was within 2% of the best on
both the reported and the complete target (PR #178).
"""


def pool_matrix(
    level: np.ndarray, parent: np.ndarray, min_level: float = MIN_LEVEL
) -> np.ndarray:
    """Return the matrix that sums a node's qualifying children.

    Parameters
    ----------
    level : numpy.ndarray
        Cases a week of each node, `[L]`.
    parent : numpy.ndarray
        Index of each node's parent, or -1, `[L]`.
    min_level : float
        Level at which a child enters its parent's pool.

    Returns
    -------
    numpy.ndarray
        `[L, L]`, with entry `[n, m]` 1 if $m$ is a child of $n$ with a
        level of at least `min_level`, or if $m = n$ and no child of $n$
        qualifies.
    """
    n = len(level)
    out = np.zeros((n, n))
    for m in range(n):
        if parent[m] >= 0 and level[m] >= min_level:
            out[parent[m], m] = 1.0
    alone = out.sum(axis=1) == 0
    out[alone, alone] = 1.0
    return out


def source_index(
    level: np.ndarray,
    parent: np.ndarray,
    n_own: np.ndarray | None = None,
    n_pooled: np.ndarray | None = None,
    min_level: float = MIN_LEVEL,
    min_err: int = 1,
) -> np.ndarray:
    """Return the node whose estimate each node uses.

    Parameters
    ----------
    level : numpy.ndarray
        Cases a week of each node, `[L]`.
    parent : numpy.ndarray
        Index of each node's parent, or -1, `[L]`.
    n_own, n_pooled : numpy.ndarray, optional
        Errors behind each node's own and pooled estimates, `[L, K]`.
        Without them, only the level counts and the result is `[L]`.
    min_level : float
        Level at which a node qualifies.
    min_err : int
        Errors a node's estimate needs, where they are given.

    Returns
    -------
    numpy.ndarray
        Integer index `i`, `[L]` or `[L, K]`. It is an index into the
        own estimates for `i < L` and into the pooled estimates for
        `i >= L`, as `i - L`. Each estimate is gathered with it.
    """
    level = np.asarray(level, dtype=float)
    parent = np.asarray(parent, dtype=int)
    n = level.size
    big = (level >= min_level)[:, None]
    if n_own is None or n_pooled is None:
        ok_own, ok_pool = big, big
    else:
        ok_own = big & (np.asarray(n_own) >= min_err)
        ok_pool = big & (np.asarray(n_pooled) >= min_err)
    k_n = ok_own.shape[1]
    out = np.empty((n, k_n), dtype=int)
    for node in range(n):
        root = node
        while parent[root] >= 0:
            root = parent[root]
        for k in range(k_n):
            if ok_own[node, k]:
                out[node, k] = node
                continue
            a = parent[node]
            while a >= 0 and not ok_pool[a, k]:
                a = parent[a]
            out[node, k] = n + (a if a >= 0 else root)
    return out[:, 0] if n_own is None or n_pooled is None else out
