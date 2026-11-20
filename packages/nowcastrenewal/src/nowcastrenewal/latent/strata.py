# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""A renewal over strata with depletion, imports, shifts and variants.

`strata_renewal` is the simulator's renewal step: a pure function of the
starting infections, the reproduction numbers and the generation time, in
the form of PyRenew's renewal scans and `model.renewal.coupled_renewal`. It
adds what those lack:

- mixing between strata in counts (`mixing.count_mixing`), with a share
  that may change each day;
- a choice of susceptible depletion (`LinearDepletion`, the simulator's
  default, or `ExponentialDepletion`, the fitted models' form);
- imports added after depletion (`imports.Imports`);
- shifts in R that may react to the previous day's effective R
  (`Shifts`);
- variants that raise R and return immune people to the susceptibles
  (`Variants`).

The effective reproduction number is recorded after each day's
infections, $R_t S_{t+1} / N$, and shifts react to it on the next day.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .mixing import count_mixing

__all__ = [
    "Depletion",
    "ExponentialDepletion",
    "LinearDepletion",
    "Shifts",
    "StrataRun",
    "Variants",
    "strata_renewal",
]

# The logistic in shifts and variants runs over `days`, from about 10% to
# 90% of the way: 2 * log(9) is about 4.4.
_SPREAD = 4.4


class Depletion(Protocol):
    """New infections from R, the mixed force, susceptibles and imports."""

    def __call__(
        self, r: Array, force: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each stratum's new infections, at most its susceptibles."""
        ...


@dataclass(frozen=True)
class LinearDepletion:
    r"""$\min(R S F / N + \iota, S)$: linear in the susceptible share.

    The simulator's default (the R engine's form).
    """

    def __call__(
        self, r: Array, force: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each stratum's new infections, at most its susceptibles."""
        return jnp.minimum(r * s / pop * force + imports, s)


@dataclass(frozen=True)
class ExponentialDepletion:
    r"""$\min(S \{1 - e^{-R F / N}\} + \iota, S)$.

    PyRenew's `logistic_susceptibility_adjustment` (Bhatt et al. 2023,
    eq. 6) with the raw incidence $R F$, written with `expm1` as
    `model.renewal.coupled_renewal` does, so small forces keep their digits.
    To first order in $R F / N$ it is `LinearDepletion`.
    """

    def __call__(
        self, r: Array, force: Array, s: Array, pop: Array, imports: Array
    ) -> Array:
        """Return each stratum's new infections, at most its susceptibles."""
        return jnp.minimum(-s * jnp.expm1(-r * force / pop) + imports, s)


def _vec(x: Sequence[float] | ArrayLike) -> Array:
    return jnp.atleast_1d(jnp.asarray(x, dtype=float))


@dataclass(frozen=True)
class Shifts:
    r"""Shifts in R, each along a logistic over `days` centred on `at`.

    Shift $j$ multiplies R by $e^{\pm s_j}$ at full strength. Its direction
    is set on the first day at or after `at[j] - days[j] / 2`: a reactive
    shift lowers R in strata whose previous day's effective R is at least
    1 and raises it where it is below; the others go up where `up[j]`.

    Attributes
    ----------
    at, size, days : array_like
        Each shift's centre (day), size $s_j$ and length (days).
    reactive, up : array_like of bool
        Whether each shift reacts to the effective R, and if not whether
        it goes up.
    """

    at: Sequence[float] | ArrayLike = field(default=())
    size: Sequence[float] | ArrayLike = field(default=())
    days: Sequence[float] | ArrayLike = field(default=())
    reactive: Sequence[bool] | ArrayLike = field(default=())
    up: Sequence[bool] | ArrayLike = field(default=())

    def arrays(self) -> dict[str, Array]:
        """Return the shifts as float arrays, one entry per shift."""
        return {
            k: _vec(getattr(self, k))
            for k in ("at", "size", "days", "reactive", "up")
        }


@dataclass(frozen=True)
class Variants:
    r"""Variants, each raising R along a logistic and escaping immunity.

    Variant $j$ multiplies R by $1 + (m_j - 1)\,\sigma((t - a_j) /
    (d_j / 4.4))$, and on the first day at or after `at[j]` returns a share
    `escape[j]` of the immune, $N - S$, to the susceptibles.

    Attributes
    ----------
    at, escape, multiplier, days : array_like
        Each variant's centre (day), escape share, R multiplier at full
        strength and length (days).
    """

    at: Sequence[float] | ArrayLike = field(default=())
    escape: Sequence[float] | ArrayLike = field(default=())
    multiplier: Sequence[float] | ArrayLike = field(default=())
    days: Sequence[float] | ArrayLike = field(default=())

    def multiplier_on(self, t: ArrayLike) -> Array:
        """Return the variants' combined R multiplier on days `t`.

        Parameters
        ----------
        t : array_like
            Days, 1 the first.

        Returns
        -------
        Array
            The product over variants, the shape of `t`.
        """
        t = jnp.asarray(t, dtype=float)
        at, mult, days = _vec(self.at), _vec(self.multiplier), _vec(self.days)
        rise = jax.nn.sigmoid(
            (t[:, None] - at[None, :]) / (days[None, :] / _SPREAD)
        )
        return jnp.prod(1 + (mult[None, :] - 1) * rise, axis=1)


class StrataRun(NamedTuple):
    """A strata renewal's output.

    Attributes
    ----------
    infections : Array
        `(G + n, k)`, the starting infections then each day's.
    r : Array
        R after shifts and variants, `(n, k)`.
    reff : Array
        $R_t S_{t+1} / N$, `(n, k)`.
    susceptible : Array
        Susceptibles after each day, `(n, k)`.
    """

    infections: Array
    r: Array
    reff: Array
    susceptible: Array


def strata_renewal(
    start: ArrayLike,
    gt: ArrayLike,
    rt: ArrayLike,
    population: ArrayLike,
    mixing: tuple[ArrayLike, ArrayLike] | None = None,
    imports: ArrayLike | None = None,
    shifts: Shifts | None = None,
    variants: Variants | None = None,
    depletion: Depletion | None = None,
) -> StrataRun:
    r"""Run each stratum's renewal in counts.

    Each day, with $F$ the generation-time-weighted past infections and
    $W_t$ from `mixing.count_mixing`, the new infections are
    `depletion(R_t, W_t F, S_t, N, imports_t)`, after shifts and variants
    have set $R_t$ and escape has refilled $S_t$.

    Parameters
    ----------
    start : array_like
        Infections over the generation time's span before day 1, `(G, k)`.
        They count towards depletion.
    gt : array_like
        Generation time pmf on 1, ..., G days.
    rt : array_like
        Reproduction numbers before shifts and variants, `(n, k)`.
    population : array_like
        Each stratum's population, `(k,)`.
    mixing : tuple of array_like, optional
        The kernel $K$ `(k, k)`, rows receive, and its share
        $\varepsilon_t$ `(n,)` (`mixing.PlaceMixing`). `None` is none.
    imports : array_like, optional
        Imported infections, `(n, k)` (`imports.Imports`).
    shifts : Shifts, optional
        Shifts in R.
    variants : Variants, optional
        Variants.
    depletion : Depletion, optional
        `LinearDepletion` (the default) or `ExponentialDepletion`.

    Returns
    -------
    StrataRun
        The infections, R, effective R and susceptibles.
    """
    start_ = jnp.asarray(start, dtype=float)
    rt_ = jnp.asarray(rt, dtype=float)
    pop = jnp.asarray(population, dtype=float)
    n, k = rt_.shape
    deplete = LinearDepletion() if depletion is None else depletion
    rev_gt = jnp.flip(jnp.asarray(gt, dtype=float))
    if mixing is None:
        W = jnp.broadcast_to(jnp.eye(k), (n, k, k))
    else:
        W = count_mixing(mixing[0], jnp.broadcast_to(mixing[1], (n,)))
    imp = jnp.zeros((n, k)) if imports is None else jnp.asarray(imports)
    sh = (shifts or Shifts()).arrays()
    va = variants or Variants()
    va_at, va_escape = _vec(va.at), _vec(va.escape)
    n_sh = sh["at"].shape[0]
    days = jnp.arange(1, n + 1, dtype=float)
    va_mult = va.multiplier_on(days)
    s0 = jnp.maximum(pop - start_.sum(axis=0), 0)

    def step(carry: Any, inputs: Any) -> tuple[Any, tuple[Array, ...]]:
        hist, s, prev, mult, set_, escaped = carry
        t, r, w, im, vm = inputs
        # Shifts: the direction is fixed on the first day each is due.
        due = (~set_) & (t >= sh["at"] - sh["days"] / 2)
        up = jnp.where(
            sh["reactive"][:, None] > 0,
            prev[None, :] < 1,
            (sh["up"][:, None] > 0) & jnp.ones((n_sh, k), dtype=bool),
        )
        new_mult = jnp.exp(jnp.where(up, 1.0, -1.0) * sh["size"][:, None])
        mult = jnp.where(due[:, None], new_mult, mult)
        set_ = set_ | due
        weight = jax.nn.sigmoid((t - sh["at"]) / (sh["days"] / _SPREAD))
        factor = jnp.where(set_[:, None], 1 + (mult - 1) * weight[:, None], 1.0)
        r = r * jnp.prod(factor, axis=0) * vm
        # Variants: escape once, on the first day at or after `at`.
        go = (~escaped) & (t >= va_at)
        for j in range(va_at.shape[0]):
            s = jnp.where(go[j], s + va_escape[j] * (pop - s), s)
        escaped = escaped | go
        new = deplete(r, w @ (rev_gt @ hist), s, pop, im)
        s = s - new
        reff = r * s / pop
        hist = jnp.concatenate([hist[1:], new[None, :]])
        return (hist, s, reff, mult, set_, escaped), (new, r, reff, s)

    carry = (
        start_,
        s0,
        jnp.ones(k),
        jnp.ones((n_sh, k)),
        jnp.zeros(n_sh, dtype=bool),
        jnp.zeros(va_at.shape[0], dtype=bool),
    )
    _, (new, r, reff, s) = jax.lax.scan(
        step, carry, (days, rt_, W, imp, va_mult)
    )
    return StrataRun(jnp.concatenate([start_, new]), r, reff, s)
