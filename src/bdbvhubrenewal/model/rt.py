# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Pooled time-varying effects shared by the latent and observation parts.

Rt and the missing-onset fraction use the same structure as PyRenew's
`SubpopulationInfections`: a national temporal process plus province
deviations with a sum-to-zero constraint.
"""

from __future__ import annotations

from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.util
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


class KnotWalk(dist.Distribution):
    r"""Random walk over knots, with the knots themselves as the variable.

    $$k_0 \sim N(m, s), \qquad k_j - k_{j-1} \sim N(0, \sigma).$$

    Sampling the knots (total log Rt per week) rather than a level plus
    steps removes the stiff direction where the level and every step move
    together (BVDOutbreakSize #1124). The prior is unchanged.
    Batch shape is that of `loc`; the event is the `num_knots` knots.
    """

    arg_constraints: ClassVar[dict[str, Any]] = {
        "loc": dist.constraints.real,
        "scale": dist.constraints.positive,
        "step_sd": dist.constraints.positive,
    }
    support = dist.constraints.real_vector
    reparametrized_params: ClassVar[list[str]] = ["loc", "scale", "step_sd"]

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        step_sd: ArrayLike,
        num_knots: int,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale, self.step_sd = dist.util.promote_shapes(
            loc, scale, step_sd
        )
        self.num_knots = num_knots
        batch = jax.lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(step_sd)
        )
        super().__init__(batch, (num_knots,), validate_args=validate_args)

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draw knots: a start and cumulative steps."""
        shape = sample_shape + self.batch_shape + (self.num_knots,)
        eps = jax.random.normal(key, shape)
        start = self.loc[..., None] + self.scale[..., None] * eps[..., :1]
        steps = self.step_sd[..., None] * eps[..., 1:]
        return jnp.concatenate(
            [start, start + jnp.cumsum(steps, axis=-1)], axis=-1
        )

    def log_prob(self, value: Array) -> Array:
        """Return the log density: normal start plus normal increments."""
        start = dist.Normal(self.loc, self.scale).log_prob(value[..., 0])
        steps = dist.Normal(0.0, self.step_sd[..., None]).log_prob(
            jnp.diff(value, axis=-1)
        )
        return jnp.asarray(start + steps.sum(-1))


def _damped_sum(phi: ArrayLike, steps: Array) -> Array:
    r"""Return $y_j = \sum_{i \le j} \phi^{j - i} s_i$ by a matrix product.

    Lower-triangular with powers of $\phi$ (all ones for a random walk),
    which is faster than a scan or `cumsum` on CPU.
    """
    m = steps.shape[0]
    lag = jnp.arange(m)[:, None] - jnp.arange(m)[None, :]
    weights = jnp.where(lag >= 0, phi ** jnp.maximum(lag, 0), 0.0)
    return weights @ steps


class NonCentredRandomWalk:
    r"""Random walk or AR(1), frozen before the data, extendable past them.

    $$x(t) = x_0 \ (t \le k), \qquad x(t) = \phi\, x(t - 1) + \sigma z_t
    \ (t > k), \qquad z_t \sim N(0, 1),$$

    with $\phi = 1$ (random walk, the prior of PyRenew's `RandomWalk`)
    or $\phi = 2^{-1/h}$ for a half-life $h$ steps from `half_life_rv`
    (mean-reverting towards 0, for province deviations). PyRenew samples
    the steps one site per step in a scan; here they are one vectorised
    site. The first `frozen` steps are zero, so nothing moves before the
    data begin.

    Fit and forecast axes. With `n_fit` set, only the first `n_fit`
    points are sampled as the main site, which is all the fit sees. A
    longer axis (forecast mode) adds a separate `{prefix}_z_future` site
    of standardised steps that `Predictive` draws from the prior given
    the posterior, so steps with no data never enter NUTS (no funnel).

    `knots=(m, s)` samples the knots themselves (`KnotWalk`) on the fit
    axis, with the first knot $N(m, s)$ in place of `initial_value`
    (random walk only).

    Parameters
    ----------
    innovation_sd_rv : RandomVariable
        Step SD $\sigma$.
    frozen : int, optional
        Leading steps fixed at zero.
    knots : tuple of float, optional
        Sample the knots (`KnotWalk`).
    n_fit : int, optional
        Length of the fit axis.
    half_life_rv : RandomVariable, optional
        Half-life in steps; makes the process AR(1).
    """

    step_size: int = 1

    def __init__(
        self,
        innovation_sd_rv: RandomVariable,
        frozen: int = 0,
        knots: tuple[float, float] | None = None,
        n_fit: int | None = None,
        half_life_rv: RandomVariable | None = None,
    ) -> None:
        self.innovation_sd_rv = innovation_sd_rv
        self.frozen = frozen
        self.knots = knots
        self.n_fit = n_fit
        self.half_life_rv = half_life_rv

    def sample(
        self,
        n_timepoints: int,
        initial_value: float | ArrayLike | None = None,
        n_processes: int = 1,
        name_prefix: str = "rw",
        *,
        first_day_dow: int | None = None,
    ) -> Array:
        """Sample trajectories.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, n_processes)`.
        """
        n_past = (
            n_timepoints
            if self.n_fit is None
            else min(self.n_fit, n_timepoints)
        )
        x0 = jnp.broadcast_to(
            jnp.asarray(0.0 if initial_value is None else initial_value),
            (n_processes,),
        )
        frozen = min(self.frozen, n_past - 1)
        n_free = n_past - 1 - frozen
        sd = self.innovation_sd_rv()
        phi = (
            1.0
            if self.half_life_rv is None
            else 0.5 ** (1.0 / self.half_life_rv())
        )
        if self.knots is not None:
            loc, scale = self.knots
            k = numpyro.sample(
                f"{name_prefix}_knots",
                KnotWalk(
                    jnp.full((n_processes,), loc),
                    jnp.full((n_processes,), scale),
                    jnp.broadcast_to(sd, (n_processes,)),
                    n_free + 1,
                ),
            ).T
            past = jnp.vstack([jnp.repeat(k[:1], frozen, axis=0), k])
        else:
            z = numpyro.sample(
                f"{name_prefix}_z",
                dist.Normal(0.0, 1.0).expand([n_free, n_processes]),
            )
            free = _damped_sum(phi, sd * z)
            decay = phi ** jnp.arange(1, n_free + 1)[:, None]
            past = jnp.vstack(
                [jnp.repeat(x0[None, :], frozen + 1, axis=0), decay * x0 + free]
            )
        n_future = n_timepoints - n_past
        if n_future == 0:
            return past
        z_future = numpyro.sample(
            f"{name_prefix}_z_future",
            dist.Normal(0.0, 1.0).expand([n_future, n_processes]),
        )
        decay = phi ** jnp.arange(1, n_future + 1)[:, None]
        future = decay * past[-1] + _damped_sum(phi, sd * z_future)
        return jnp.vstack([past, future])


class PooledInitialValue(RandomVariable):
    r"""Starting values for each province drawn around a national value.

    $$v_p = \mu + \sigma \sum_j Q_{pj} z_j, \qquad z_j \sim N(0, 1),$$

    with $Q$ from `sum_to_zero_basis`, so the province offsets sum to
    zero and $\mu$ is their mean.

    with $\mu \sim N(m, s)$ and $\sigma \sim N^+(0, \tau)$.

    Parameters
    ----------
    name : str
        Prefix for the sample sites `{name}_mean`, `{name}_sd` and
        `{name}_z`.
    n_groups : int
        Number of provinces.
    loc, scale : float
        Mean $m$ and SD $s$ of the national value.
    pooling_sd : float
        Scale $\tau$ of the half-normal prior on the province spread.
    fixed_mean : bool, optional
        Fix $\mu = m$, when the national level is sampled elsewhere (as
        the first knot of a `KnotWalk`).
    centred : bool, optional
        Sample the $P - 1$ offsets on the basis directly (centred, the
        default, as BVDOutbreakSize #1083), rather than standardised.
    active : array_like, optional
        Provinces with their own offset; the others start at $\mu$.
    """

    def __init__(
        self,
        name: str,
        n_groups: int,
        loc: float,
        scale: float,
        pooling_sd: float,
        fixed_mean: bool = False,
        centred: bool = True,
        active: ArrayLike | None = None,
    ) -> None:
        super().__init__(name=name)
        self.active = None if active is None else np.asarray(active, bool)
        self.fixed_mean = fixed_mean
        self.centred = centred
        self.n_groups = n_groups
        self.loc = loc
        self.scale = scale
        self.pooling_sd = pooling_sd

    def sample(self, **kwargs: object) -> Array:
        """Sample one starting value per province.

        Returns
        -------
        ArrayLike
            Shape `(n_groups,)`.
        """
        mu = (
            self.loc
            if self.fixed_mean
            else numpyro.sample(
                f"{self.name}_mean", dist.Normal(self.loc, self.scale)
            )
        )
        n_active = (
            self.n_groups if self.active is None else int(self.active.sum())
        )
        if n_active <= 1:
            return jnp.full((self.n_groups,), mu)
        sd = numpyro.sample(f"{self.name}_sd", dist.HalfNormal(self.pooling_sd))
        basis = jnp.asarray(
            sum_to_zero_basis(self.n_groups, active=self.active)
        )
        if self.centred:
            offsets = numpyro.sample(
                f"{self.name}_offsets",
                dist.Normal(0.0, sd).expand([n_active - 1]),
            )
            return jnp.asarray(mu + basis @ offsets)
        z = numpyro.sample(
            f"{self.name}_z",
            dist.Normal(0.0, 1.0).expand([n_active - 1]),
        )
        return jnp.asarray(mu + sd * (basis @ z))


def split_initial_value(
    initial: ArrayLike,
    n_groups: int,
    weights: ArrayLike | None = None,
    active: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Split starting values into a national value and province deviations.

    Mirrors `SubpopulationInfections._split_initial_log_rt`: a scalar gives
    zero deviations, a vector gives its mean and the centred deviations.

    Parameters
    ----------
    initial : ArrayLike
        Scalar or shape `(n_groups,)`.
    n_groups : int
        Number of provinces.

    Returns
    -------
    tuple of ArrayLike
        National start (scalar) and deviation starts, shape `(n_groups,)`.
    """
    initial = jnp.asarray(initial)
    if initial.ndim == 0:
        return initial, jnp.zeros(n_groups)
    if weights is None and active is None:
        start = jnp.mean(initial)
    else:
        w = jnp.ones(n_groups) if weights is None else jnp.asarray(weights)
        if active is not None:
            w = w * jnp.asarray(active, dtype=float)
        start = (w @ initial) / w.sum()
    return start, initial - start


def sum_to_zero_basis(
    n: int,
    weights: ArrayLike | None = None,
    active: ArrayLike | None = None,
) -> np.ndarray:
    r"""Orthonormal basis of $\{\delta \in R^n : w^\top \delta = 0\}$.

    With no weights, $w = 1$ and the columns sum to zero. With weights
    (case counts by province), deviations from the national value
    average to zero weighted by cases, so the national value is the
    case-weighted one.

    With `active`, the basis spans only the active (outbreak) provinces:
    inactive rows are zero, so those provinces take the national value,
    and the constraint uses the active weights only.

    Returns
    -------
    numpy.ndarray
        Shape `(n, k - 1)`, with `k` active provinces (all by default),
        orthonormal columns orthogonal to `w`.
    """
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    on = np.ones(n, bool) if active is None else np.asarray(active, bool)
    _, _, vt = np.linalg.svd(w[on][None, :])
    out = np.zeros((n, int(on.sum()) - 1))
    out[on] = vt[1:].T
    return out


def pooled_walk(
    baseline_process: TemporalProcess,
    deviation_process: TemporalProcess,
    initial: ArrayLike,
    n_timepoints: int,
    n_groups: int,
    name_prefix: str,
    first_day_dow: int | None = None,
    weights: ArrayLike | None = None,
    active: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    r"""Sample a national process plus sum-to-zero province deviations.

    $$x_p(t) = b(t) + \sum_j Q_{pj} \eta_j(t),$$

    where $b$ follows `baseline_process`, $Q$ is an orthonormal basis of
    the sum-to-zero subspace (`sum_to_zero_basis`) and the $P - 1$
    coordinates $\eta_j$ follow `deviation_process`, which shares one
    innovation SD (partial pooling). The deviations sum to zero exactly
    with no redundant direction to sample (BVDOutbreakSize #855). Each
    deviation has the prior of $P$ independent walks centred on their
    mean.

    Parameters
    ----------
    baseline_process, deviation_process : TemporalProcess
        PyRenew temporal processes, for example a `WeeklyTemporalProcess`
        wrapping a `RandomWalk`.
    initial : ArrayLike
        Starting value(s), scalar or shape `(n_groups,)`.
    n_timepoints, n_groups : int
        Output shape.
    name_prefix : str
        Prefix for the sample sites.
    first_day_dow : int, optional
        Day of week of element 0, needed by calendar-aligned processes.
    weights : ArrayLike, optional
        Province weights; deviations then satisfy $w^\top \delta = 0$.
    active : ArrayLike, optional
        Provinces with their own deviation, `(n_groups,)`, or by time,
        `(n_timepoints, n_groups)`. The others follow the national process
        exactly. The basis spans the provinces ever active. With a mask by
        time, inactive deviations are set to zero and the rest re-centred
        so the weighted constraint holds over the active ones at each
        time; a province that goes quiet drops back to the national value
        and picks its deviation up again when it returns.

    Returns
    -------
    tuple of ArrayLike
        National trajectory `(n_timepoints, 1)` and province trajectories
        `(n_timepoints, n_groups)`.
    """
    mask = None if active is None else np.asarray(active, bool)
    ever = None if mask is None else (mask if mask.ndim == 1 else mask.any(0))
    base_start, dev_start = split_initial_value(
        initial, n_groups, weights, ever
    )
    baseline = baseline_process.sample(
        n_timepoints=n_timepoints,
        initial_value=base_start,
        name_prefix=f"{name_prefix}_baseline",
        first_day_dow=first_day_dow,
    )
    if n_groups == 1:
        return baseline, baseline
    n_active = n_groups if ever is None else int(ever.sum())
    if n_active <= 1:
        return baseline, jnp.repeat(baseline, n_groups, axis=1)
    basis = jnp.asarray(sum_to_zero_basis(n_groups, weights, ever))
    coords = deviation_process.sample(
        n_timepoints=n_timepoints,
        n_processes=n_active - 1,
        initial_value=basis.T @ dev_start,
        name_prefix=f"{name_prefix}_deviations",
        first_day_dow=first_day_dow,
    )
    dev = coords @ basis.T
    if mask is not None and mask.ndim == 2:
        if mask.shape != (n_timepoints, n_groups):
            raise ValueError(
                "The active mask must be (n_timepoints, n_groups)."
            )
        on = jnp.asarray(mask, dtype=dev.dtype)
        w = jnp.ones(n_groups) if weights is None else jnp.asarray(weights)
        dev = on * dev
        total = on @ w
        shift = (dev @ w) / jnp.where(total > 0, total, 1.0)
        dev = dev - shift[:, None] * on
    return baseline, baseline + dev


class MissingOnsetFraction(RandomVariable):
    r"""Time-varying probability that a case is reported without an onset.

    $$\operatorname{logit} \pi_p(t) = b(t) + \delta_p(t),$$

    from `pooled_walk`, indexed by report day $t$. With small innovation
    SDs it stays near its starting values, so it reduces to a constant
    $\pi_p$ when the data show no change.

    Parameters
    ----------
    name : str
        Prefix for the sample sites.
    initial_rv : RandomVariable
        Starting logit values, scalar or one per province.
    baseline_process, deviation_process : TemporalProcess
        Temporal processes for the national and province parts.
    """

    def __init__(
        self,
        name: str,
        initial_rv: RandomVariable,
        baseline_process: TemporalProcess,
        deviation_process: TemporalProcess,
    ) -> None:
        super().__init__(name=name)
        self.initial_rv = initial_rv
        self.baseline_process = baseline_process
        self.deviation_process = deviation_process

    def sample(
        self,
        n_timepoints: int,
        n_groups: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Array:
        r"""Sample $\pi_p(t)$.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, n_groups)`, values in (0, 1).
        """
        _, logit_pi = pooled_walk(
            self.baseline_process,
            self.deviation_process,
            self.initial_rv(),
            n_timepoints,
            n_groups,
            self.name,
            first_day_dow,
        )
        pi = jax.nn.sigmoid(logit_pi)
        numpyro.deterministic(self.name, pi)
        return pi
