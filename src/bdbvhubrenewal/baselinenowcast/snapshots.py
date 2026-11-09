"""Snapshot data extension (E1) and the hub week aggregator.

R's ``estimate_delay`` needs a report in every delay cell of the
triangle or a right truncation.
Release snapshots are published on a few days only, so most cells of
the triangle are unobserved.
E1 is the one named extension of the port.
It takes the chain ladder ratio of sums over the cells observed in
both releases of a pair and skips the others, which are not filled.
Pairs of any gap share one daily log development factor per age
(``log_factors``), and ``growth`` sums it between two ages.

Axes.
``counts`` is ``(n_loc, n_release, n_day)``, the count of onset day t
in release r.
``observed`` is ``(n_release, n_day)``, True where day t has data in
release r, and ``cut[r]`` is the index of the cut-off of release r,
its latest onset day (``bdbvhubrenewal.data.releases.release_cutoff``).
The age of day t in release r is ``cut[r] - t``.
Pair p is releases ``(pairs[p], pairs[p] + 1)`` with step
``cut[r + 1] - cut[r]``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import nnls

Aggregator = Callable[[Array, np.ndarray], tuple[Array, np.ndarray]]
"""Aggregates reference times: `(values, mask)` to `(values, mask)`."""


@partial(jax.jit, static_argnames="max_delay")
def _pair_sums(
    counts: Array,
    observed: Array,
    cut: Array,
    pairs: Array,
    max_delay: int,
) -> tuple[Array, Array]:
    n_day = counts.shape[-1]
    t = cut[pairs][:, None] - jnp.arange(max_delay)[None, :]
    inside = (t >= 0) & (t < n_day)
    tc = jnp.clip(t, 0, n_day - 1)
    r = pairs[:, None]
    both = inside & observed[r, tc] & observed[r + 1, tc]
    den = jnp.where(both, counts[:, r, tc], 0.0)
    num = jnp.where(both, counts[:, r + 1, tc], 0.0)
    return num, den


def pair_sums(
    counts: Any,
    observed: Any,
    cut: Any,
    pairs: Any,
    max_delay: int,
) -> tuple[Array, Array]:
    """Numerator and denominator per pair and age (extension E1).

    For pair ``p = (r, r + 1)`` and age ``x = 0 .. max_delay - 1`` at
    release r, the cell is day ``t = cut[r] - x``.
    ``den`` is its count in release r and ``num`` its count in
    release r + 1.
    Cells not observed in both releases, or outside the days, are
    skipped and add 0.

    Parameters
    ----------
    counts : array_like
        Counts by location, release and day.
    observed : array_like of bool
        True where a count is observed, the same shape.
    cut : array_like of int
        Each release's cut-off day.
    pairs : array_like of int
        First release index of each pair.
    max_delay : int
        Ages to use.

    Returns
    -------
    tuple of jax.Array
        ``(num, den)``, each ``(n_loc, n_pair, max_delay)``.
    """
    c = jnp.asarray(counts)
    return _pair_sums(
        c,
        jnp.asarray(observed, bool),
        jnp.asarray(cut, int),
        jnp.asarray(pairs, int),
        int(max_delay),
    )


def log_factors(num: Any, den: Any, gaps: Any) -> np.ndarray:
    r"""Daily log development factors shared by all pairs (extension E1).

    A day of age $a$ at release $r$ is age $a + s$ at release $r + 1$,
    with $s$ the pair's gap in days.
    Releases are batched reports, so its growth over the gap is
    $\exp \sum_{j=a}^{a+s-1} g(j)$, with $g(j) = 0$ from ``max_delay``.
    The pairs of each gap are summed by age before the ratio, as
    baselinenowcast sums over reference times.
    Every gap and age with a denominator gives one equation
    $\log(\sum \mathrm{num} / \sum \mathrm{den}) = \sum g$,
    weighted by $\sum \mathrm{den}$, and $g \ge 0$ is the
    non-negative least squares fit.
    With a gap of 1 only, $g(a)$ is the log of the chain ladder factor.

    Parameters
    ----------
    num, den : array_like
        ``(n_loc, n_pair, max_delay)``, from `pair_sums`.
    gaps : array_like of int
        Each pair's gap in days.

    Returns
    -------
    numpy.ndarray
        $g$, ``(n_loc, max_delay)``.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    gaps = np.asarray(gaps, int)
    n_loc, _, d = num.shape
    uniq = np.unique(gaps)
    member = (gaps[:, None] == uniq[None, :]).astype(float)
    n_sum = np.einsum("lpa,ps->lsa", num, member)
    d_sum = np.einsum("lpa,ps->lsa", den, member)
    age = np.arange(d)
    j = np.arange(d)[None, None, :]
    lo = age[None, :, None]
    rows = ((j >= lo) & (j < lo + uniq[:, None, None])).reshape(-1, d)
    g = np.zeros((n_loc, d))
    for n in range(n_loc):
        w = d_sum[n].ravel()
        # A zero denominator gives a row of weight 0, so no equation.
        seen = w > 0
        y = np.log(np.where(seen, n_sum[n].ravel() / np.where(seen, w, 1), 1))
        root = np.sqrt(w)
        g[n] = nnls(rows * root[:, None], y * root)[0]
    return g


def growth(g: Any, start: Any, stop: Any) -> np.ndarray:
    r"""Growth $\exp \sum_{j=\mathrm{start}}^{\mathrm{stop}-1} g(j)$.

    Parameters
    ----------
    g : array_like
        Daily log factors, ``(n_loc, max_delay)``, from `log_factors`.
    start, stop : array_like of int
        Ages that broadcast together, clipped to ``0 .. max_delay``, so
        growth from ``max_delay`` on is 1.

    Returns
    -------
    numpy.ndarray
        ``(n_loc, *shape)``.
    """
    g = np.asarray(g, float)
    d = g.shape[1]
    total = np.concatenate([np.zeros((g.shape[0], 1)), np.cumsum(g, 1)], 1)
    a, b = np.broadcast_arrays(
        np.clip(np.asarray(start, int), 0, d),
        np.clip(np.asarray(stop, int), 0, d),
    )
    return np.exp(total[:, b] - total[:, a])


def fill_row(x: Any, cdf: Any) -> np.ndarray:
    r"""Return a row's count by its last column, as R's ``apply_delay``.

    Each later column $j$ adds $(x_{j-1} + 1 - F_{j-1}) / F_{j-1}\,
    (F_j - F_{j-1})$, with $x_{j-1}$ the count up to column $j - 1$
    (R ``.calc_modified_expectation``).
    The $+1$ gives a row with nothing reported yet a positive count.

    Parameters
    ----------
    x : array_like
        Count reported by the first column.
    cdf : array_like
        Share reported by each column, the first that of ``x``, on the
        last axis. It broadcasts with ``x`` on the others.

    Returns
    -------
    numpy.ndarray
        The count by the last column.
    """
    cdf = np.asarray(cdf, float)
    total = np.asarray(x, float)
    for j in range(1, cdf.shape[-1]):
        prev = cdf[..., j - 1]
        total = total + (total + 1 - prev) / prev * (cdf[..., j] - prev)
    return total


def window_pairs(cut: Any, max_delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs whose step is at most ``max_delay``, and their steps.

    A longer step has no age inside the window to attach a factor to.
    Its releases stay in the data and in the counts.

    Parameters
    ----------
    cut : array_like of int
        Each release's cut-off day.
    max_delay : int
        Longest step kept, in days.

    Returns
    -------
    tuple of numpy.ndarray
        The first release index and the step of each pair.
    """
    steps = np.diff(np.asarray(cut, dtype=int))
    keep = np.flatnonzero((steps >= 1) & (steps <= max_delay))
    return keep, steps[keep]


def hub_week_matrix(
    first_onset: Any, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrix of hub weeks (weeks by days) and the Saturday labels.

    Hub weeks run Sunday to Saturday and carry the Saturday's date.
    A first week that starts after its Sunday keeps the days present.
    No onset day is dropped.
    The last week keeps only the days present, so the cut-off week is
    partial unless the last day is a Saturday.

    Parameters
    ----------
    first_onset : datetime.date or str
        Date of the first day.
    n_days : int
        Number of days.

    Returns
    -------
    tuple of numpy.ndarray
        The matrix, weeks by days, and the weeks' Saturdays.
    """
    start = np.datetime64(first_onset, "D")
    days = start + np.arange(n_days)
    # numpy weekday: 1970-01-01 was a Thursday; Sunday = 0 below.
    dow = (days.astype(int) + 4) % 7
    sat = days + (6 - dow)
    saturdays = np.unique(sat)
    w = (sat[None, :] == saturdays[:, None]).astype(float)
    return w, saturdays


def hub_weeks(first_onset: Any) -> Aggregator:
    """Build an aggregator summing onset days into hub weeks.

    The returned callable has the ``ref_time_aggregator`` signature.
    It builds the week matrix from the number of rows of the matrix it
    is given, whose last row is that matrix's cut-off day.
    An aggregated cell is unobserved if any of its inputs is.

    Parameters
    ----------
    first_onset : datetime.date or str
        Date of the first onset day.

    Returns
    -------
    Aggregator
        The aggregator.
    """

    def aggregate(values: Array, mask: np.ndarray) -> tuple[Array, np.ndarray]:
        v = jnp.asarray(values)
        m = np.asarray(mask, dtype=bool)
        w, _ = hub_week_matrix(first_onset, v.shape[0])
        wj = jnp.asarray(w, dtype=v.dtype)
        out = wj @ jnp.where(jnp.asarray(m), v, 0.0)
        ok = (w @ (~m).astype(float)) == 0
        return jnp.where(jnp.asarray(ok), out, 0.0), ok

    return aggregate
