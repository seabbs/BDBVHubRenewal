"""Release data extension (E1) and the hub week aggregator.

R's ``estimate_delay`` needs every cell of the triangle to be observed
or right truncated.
Release data observe only the report days on which a release was
published, so each release increment is interval censored in the
delay.
The functions here turn releases into increments with their delay
intervals, expand increments into a triangle for a given delay pmf (the
E-step) and iterate the E-step with R's ``estimate_delay`` unchanged as
the M-step.

Axes.
Rows are onset days ``0 .. n_ref - 1`` shared by every release.
``cutoffs[k]`` is the row index of the latest onset day in release k.
Delay is ``cutoffs[k] - t``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ._checks import coerce
from .delay import _estimate_kernel, estimate_delay
from .triangle import preprocess_negative_values

Aggregator = Callable[[Array, np.ndarray], tuple[Array, np.ndarray]]


def release_increments(
    cum: Any,
    release_mask: Any,
    cutoffs: Any,
    max_delay: int,
) -> tuple[Array, np.ndarray, np.ndarray, np.ndarray]:
    """Increments between consecutive releases and their delay intervals.

    ``cum`` is ``(n_ref, n_release)`` cumulative counts per release and
    ``release_mask`` is True where onset day t is in release k
    (``t <= cutoffs[k]``).
    Returns ``(increments, lo, hi, mask)``.
    The increment ``I_k(t) = N_k(t) - N_{k-1}(t)`` covers delays
    ``lo .. hi`` inclusive with ``lo = max(c_{k-1} - t + 1, 0)`` and
    ``hi = min(c_k - t, max_delay)``.
    The first increment is clipped to ``max_delay``.
    Later increments with ``c_{k-1} - t >= max_delay`` are dropped
    (mask False), as R drops reports beyond ``max_delay``.
    """
    values, rm = coerce(cum, release_mask)
    c = np.asarray(cutoffs, dtype=int)
    n_ref, n_rel = rm.shape
    if c.shape != (n_rel,):
        raise ValueError("`cutoffs` needs one entry per release")
    if np.any(np.diff(c) <= 0):
        raise ValueError("`cutoffs` must be strictly increasing")
    t = np.arange(n_ref)[:, None]
    prev = np.concatenate([[-(10**9)], c[:-1]])[None, :]
    lo = np.maximum(prev - t + 1, 0)
    hi = np.minimum(c[None, :] - t, max_delay)
    mask = rm & (lo <= hi)
    before = jnp.concatenate(
        [jnp.zeros((n_ref, 1), values.dtype), values[:, :-1]], axis=1
    )
    inc = jnp.where(jnp.asarray(mask), values - before, 0.0)
    return inc, np.where(mask, lo, 0), np.where(mask, hi, 0), mask


def _weights(lo: Array, hi: Array, mask: Array, n_delay: int) -> Array:
    """Return the interval indicator, (n_ref, n_release, n_delay)."""
    d = jnp.arange(n_delay)
    inside = (d >= lo[..., None]) & (d <= hi[..., None]) & mask[..., None]
    return inside.astype(jnp.result_type(float))


@jax.jit
def _expand(increments: Array, inside: Array, pmf: Array) -> Array:
    """Run the E-step (see ``expand_release_triangle``)."""
    w = inside * pmf
    denom = jnp.sum(w, axis=-1, keepdims=True)
    count = jnp.sum(inside, axis=-1, keepdims=True)
    share = jnp.where(
        denom > 0,
        w / jnp.where(denom > 0, denom, 1.0),
        inside / jnp.where(count > 0, count, 1.0),
    )
    return jnp.einsum("tk,tkd->td", increments, share)


def _triangle_mask(n_ref: int, n_delay: int, cutoff: int) -> np.ndarray:
    t = np.arange(n_ref)[:, None]
    d = np.arange(n_delay)[None, :]
    return (t + d) <= cutoff


def expand_release_triangle(
    increments: Any,
    lo: Any,
    hi: Any,
    release_mask: Any,
    delay_pmf: Any,
    cutoff: int | None = None,
) -> tuple[Array, np.ndarray]:
    """Spread increments over their intervals by the delay pmf (E-step).

    ``X(t, d) = sum_k I_k(t) 1[d in J_k(t)] p(d) / sum_{J_k(t)} p``
    with an even split when the denominator is 0.
    Returns the triangle and the right truncation mask
    ``t + d <= cutoff``, where ``cutoff`` defaults to the last row.
    ``release_mask`` is the mask returned by ``release_increments``.
    """
    pmf = jnp.asarray(delay_pmf)
    inc = jnp.asarray(increments, dtype=pmf.dtype)
    inside = _weights(
        jnp.asarray(lo),
        jnp.asarray(hi),
        jnp.asarray(release_mask, bool),
        pmf.shape[0],
    ).astype(pmf.dtype)
    x = _expand(inc, inside, pmf)
    n_ref = inc.shape[0]
    mask = _triangle_mask(
        n_ref, pmf.shape[0], n_ref - 1 if cutoff is None else cutoff
    )
    return jnp.where(jnp.asarray(mask), x, 0.0), mask


def estimate_delay_releases(
    increments: Any,
    lo: Any,
    hi: Any,
    release_mask: Any,
    n: int,
    max_delay: int | None = None,
    cutoff: int | None = None,
    tol: float = 1e-6,
    max_iter: int = 5000,
    preprocess: bool = True,
) -> tuple[Array, Array, Array]:
    """Delay pmf from release increments by EM (extension E1).

    Starts from the uniform pmf on ``0 .. max_delay``.
    The E-step is ``expand_release_triangle`` and the M-step is
    ``estimate_delay`` on the right truncated triangle with ``n`` rows.
    Iterates in ``lax.while_loop`` until the largest change in the pmf
    is below ``tol`` or ``max_iter`` is reached.
    Negative increments are first moved to earlier releases with
    ``preprocess_negative_values`` (releases play the part of report
    columns).
    Returns ``(pmf, n_iter, max_change)`` as arrays, so the call can be
    ``vmap``ped or ``jit``ted over locations.
    """
    lo_a = np.asarray(lo, dtype=int)
    hi_a = np.asarray(hi, dtype=int)
    rm = np.asarray(release_mask, dtype=bool)
    if max_delay is None:
        max_delay = int(hi_a[rm].max())
    n_delay = max_delay + 1
    inc = jnp.asarray(increments)
    if preprocess:
        inc = preprocess_negative_values(inc, rm, validate=False)
    n_ref = inc.shape[0]
    mask = _triangle_mask(
        n_ref, n_delay, n_ref - 1 if cutoff is None else cutoff
    )
    inside = _weights(
        jnp.asarray(lo_a), jnp.asarray(hi_a), jnp.asarray(rm), n_delay
    ).astype(inc.dtype)
    mask_j = jnp.asarray(mask)
    p0 = jnp.full((n_delay,), 1.0 / n_delay, dtype=inc.dtype)
    # Validate the structure once with the first E-step.
    first = jnp.where(mask_j, _expand(inc, inside, p0), 0.0)
    estimate_delay(first, mask, n=n)
    n_use = int(n)

    def m_step(p: Array) -> Array:
        x = jnp.where(mask_j, _expand(inc, inside, p), 0.0)
        return _estimate_kernel(x[-n_use:], mask_j[-n_use:])

    @jax.jit
    def run(p_init: Array) -> tuple[Array, Array, Array]:
        def cond(s: tuple[Array, Array, Array]) -> Array:
            _, i, change = s
            return (change >= tol) & (i < max_iter)

        def body(s: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            p, i, _ = s
            p_new = m_step(p)
            return p_new, i + 1, jnp.max(jnp.abs(p_new - p))

        init = (
            p_init,
            jnp.zeros((), jnp.int32),
            jnp.asarray(jnp.inf, p_init.dtype),
        )
        return lax.while_loop(cond, body, init)

    p, i, change = run(p0)
    return p, i, change


def hub_week_matrix(
    first_onset: Any, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrix of hub weeks (weeks by days) and the Saturday labels.

    Hub weeks run Sunday to Saturday and carry the Saturday's date.
    A first week that starts before ``first_onset`` is dropped.
    The last week keeps only the days present, so the cut-off week is
    partial unless the last day is a Saturday.
    """
    start = np.datetime64(first_onset, "D")
    days = start + np.arange(n_days)
    # numpy weekday: 1970-01-01 was a Thursday; Sunday = 0 below.
    dow = (days.astype(int) + 4) % 7
    sat = days + (6 - dow)
    first_sunday = start + ((7 - ((start.astype(int) + 4) % 7)) % 7)
    keep_from = int((first_sunday - start).astype(int))
    saturdays = np.unique(sat[keep_from:])
    w = (sat[None, :] == saturdays[:, None]).astype(float)
    w[:, :keep_from] = 0.0
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
