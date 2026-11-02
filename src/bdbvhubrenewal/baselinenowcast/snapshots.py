"""Snapshot data extension (E1) and the hub week aggregator.

R's ``estimate_delay`` needs a report in every delay cell of the
triangle or a right truncation.
Release snapshots are published on a few days only, so most cells of
the triangle are unobserved.
E1 is the one named extension of the port.
It takes the chain ladder ratio of sums over the cells observed in
both releases of a pair and skips the others, which are not filled.

Axes.
``counts`` is ``(n_loc, n_release, n_day)``, the count of onset day t
in release r.
``observed`` is ``(n_release, n_day)``, True where day t has data in
release r, and ``cut[r]`` is the index of the latest onset day of
release r.
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

Aggregator = Callable[[Array, np.ndarray], tuple[Array, np.ndarray]]


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
    Both are ``(n_loc, n_pair, max_delay)``.
    """
    c = jnp.asarray(counts)
    return _pair_sums(
        c,
        jnp.asarray(observed, bool),
        jnp.asarray(cut, int),
        jnp.asarray(pairs, int),
        int(max_delay),
    )


@jax.jit
def _pair_factors(num: Array, den: Array, member: Array) -> Array:
    n = jnp.einsum("lpx,ps->lsx", num, member)
    d = jnp.einsum("lpx,ps->lsx", den, member)
    return 1.0 + (n - d) / jnp.maximum(d, 1.0)


def pair_factors(
    num: Any, den: Any, step_index: Any, n_steps: int | None = None
) -> Array:
    """Development factors per step and age (extension E1).

    ``step_index`` maps pairs to steps.
    A vector gives the slot ``0 .. n_steps - 1`` of each pair's one step.
    A matrix ``(n_pair, n_steps)`` of 0 and 1 lets a pair enter several
    steps, as when every pair is pooled into every step.
    The numerators and denominators of the pairs of a step are summed
    before the ratio, as baselinenowcast sums over reference times.
    The factor is ``1 + (sum(num) - sum(den)) / max(sum(den), 1)``.
    This is ``1 + mult`` with the ``mult`` of ``.calculate_mult_factor``,
    and equals ``sum(num) / sum(den)`` where ``sum(den) >= 1``.
    Returns ``F`` of shape ``(n_loc, n_steps, max_delay)``.
    """
    idx = jnp.asarray(step_index)
    if idx.ndim == 1:
        if n_steps is None:
            raise ValueError("`n_steps` is needed with a vector step index.")
        idx = idx[:, None] == jnp.arange(int(n_steps))[None, :]
    return _pair_factors(
        jnp.asarray(num), jnp.asarray(den), idx.astype(jnp.asarray(num).dtype)
    )


def window_pairs(cut: Any, max_delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs whose step is at most ``max_delay``, and their steps.

    A longer step has no age inside the window to attach a factor to.
    Its releases stay in the data and in the counts.
    Returns the first release index and the step of each pair.
    """
    steps = np.diff(np.asarray(cut, dtype=int))
    keep = np.flatnonzero((steps >= 1) & (steps <= max_delay))
    return keep, steps[keep]


def exact_plan(
    ages: Any, step: int, slot: int, max_delay: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chain steps of the latest pair, from each age to ``max_delay``.

    Day of age ``a`` takes the fewest steps ``M`` with
    ``a + M * step >= max_delay``, at ages ``a, a + step, ...``.
    There is no tail factor past ``max_delay``, and ages from
    ``max_delay`` have no step.
    Returns ``(slot, age, mask)`` of shape ``(n, M_max)``, padded
    where ``mask`` is False.
    """
    a = np.asarray(ages, dtype=int)
    if np.any(a < 0):
        raise ValueError("`ages` must not be negative.")
    if step < 1:
        raise ValueError("`step` must be at least 1.")
    n_steps = np.maximum(-(-(max_delay - a) // step), 0)
    m_max = max(int(n_steps.max(initial=0)), 1)
    m = np.arange(m_max)[None, :]
    mask = m < n_steps[:, None]
    ages_m = a[:, None] + m * step
    return (
        np.where(mask, slot, 0),
        np.where(mask, ages_m, 0),
        mask,
    )


@jax.jit
def _chain_growth(
    factors: Array, slot: Array, age: Array, mask: Array
) -> Array:
    f = factors[:, slot, age]
    return jnp.prod(jnp.where(mask[None], f, 1.0), axis=-1)


def chain_growth(factors: Any, plan: tuple[Any, Any, Any]) -> Array:
    """Growth to ``max_delay`` per location and day (extension E1).

    ``plan`` is ``(slot, age, mask)`` as from ``exact_plan``.
    The growth is the product of the factors at the planned steps.
    Returns ``(n_loc, n)``.
    """
    slot, age, mask = plan
    return _chain_growth(
        jnp.asarray(factors),
        jnp.asarray(slot, int),
        jnp.asarray(age, int),
        jnp.asarray(mask, bool),
    )


def hub_week_matrix(
    first_onset: Any, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrix of hub weeks (weeks by days) and the Saturday labels.

    Hub weeks run Sunday to Saturday and carry the Saturday's date.
    A first week that starts after its Sunday keeps the days present.
    No onset day is dropped.
    The last week keeps only the days present, so the cut-off week is
    partial unless the last day is a Saturday.
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
