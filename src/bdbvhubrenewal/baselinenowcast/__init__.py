# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Stand-ins for the JAX port of baselinenowcast 0.2.2 on branch `bncjax`.

Only the functions the random walk models call are here, with the
signatures of the port's snapshot extension (E1).
The port replaces this module.

E1 is the one named extension to 0.2.2.
Releases are daily snapshots, so a cell is observed when the release
covers its onset day.
Unobserved cells are skipped rather than filled.
Factors are ratios of sums over the cells observed in both releases of a
pair, by age on the onset clock and by the pair's step in days.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["ChainPlan", "chain_growth", "pair_factors", "pair_sums"]


class ChainPlan(NamedTuple):
    """Chain steps for each onset day, padded to a common length.

    Attributes
    ----------
    step : jax.Array
        Index of the step size, shape `[T, M]`.
    age : jax.Array
        Age in days at each step, shape `[T, M]`.
    valid : jax.Array
        Whether the step is used, shape `[T, M]`.
    """

    step: Array
    age: Array
    valid: Array


@partial(jax.jit, static_argnames="D")
def pair_sums(
    counts: Array,
    observed: Array,
    cut: Array,
    pairs: Array,
    D: int,  # noqa: N803
) -> tuple[Array, Array]:
    """Sum counts by age over the cells observed in both releases of a pair.

    Parameters
    ----------
    counts : jax.Array
        Counts by location, release and onset day, shape `[L, R, T]`.
    observed : jax.Array
        Whether a release covers an onset day, shape `[R, T]`.
    cut : jax.Array
        Index of each release's latest onset day, shape `[R]`.
    pairs : jax.Array
        Earlier and later release of each pair, shape `[P, 2]`.
    D : int
        Number of ages, 0 to `D - 1`.

    Returns
    -------
    tuple of jax.Array
        Numerator (later release) and denominator (earlier release),
        each of shape `[L, P, D]`.
    """
    t = jnp.arange(counts.shape[-1])
    early, late = pairs[:, 0], pairs[:, 1]
    age = cut[early][:, None] - t[None, :]
    use = observed[early] & observed[late] & (age >= 0) & (age < D)
    onehot = (age[:, None, :] == jnp.arange(D)[None, :, None]) & use[:, None, :]
    onehot = onehot.astype(counts.dtype)
    num = jnp.einsum("lpt,pdt->lpd", counts[:, late, :], onehot)
    den = jnp.einsum("lpt,pdt->lpd", counts[:, early, :], onehot)
    return num, den


@partial(jax.jit, static_argnames="n_steps")
def pair_factors(
    num: Array, den: Array, step_of_pair: Array, n_steps: int
) -> tuple[Array, Array]:
    """Sum the pairs that share a step and take the ratio.

    Parameters
    ----------
    num, den : jax.Array
        Output of `pair_sums`, shape `[L, P, D]`.
    step_of_pair : jax.Array
        Index of each pair's step, shape `[P]`.
    n_steps : int
        Number of distinct steps.

    Returns
    -------
    tuple of jax.Array
        Factors `[L, S, D]`, set to 1 where not identified, and whether
        each is identified (the denominator is above 0).
    """
    one = jax.nn.one_hot(step_of_pair, n_steps, dtype=num.dtype)
    top = jnp.einsum("lpd,ps->lsd", num, one)
    bottom = jnp.einsum("lpd,ps->lsd", den, one)
    ident = bottom > 0
    return jnp.where(ident, top / jnp.where(ident, bottom, 1.0), 1.0), ident


@jax.jit
def chain_growth(factors: Array, plan: ChainPlan) -> Array:
    """Chain the factors along each onset day's plan.

    Parameters
    ----------
    factors : jax.Array
        Factors, shape `[L, S, D]`.
    plan : ChainPlan
        Steps for each onset day.

    Returns
    -------
    jax.Array
        Growth from the day's age to the end of the chain, `[L, T]`.
    """
    age = jnp.clip(plan.age, 0, factors.shape[-1] - 1)
    f = factors[:, plan.step, age]
    return jnp.prod(jnp.where(plan.valid[None], f, 1.0), axis=-1)
