# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""A non-centred random walk or AR(1) with separate fit and forecast steps."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable


def _damped_sum(phi: ArrayLike, steps: Array) -> Array:
    r"""Return $y_j = \sum_{i \le j} \phi^{j - i} s_i$ by a matrix product.

    Lower-triangular with powers of $\phi$ (all ones for a random walk),
    which is faster than a scan or `cumsum` on CPU.

    Returns
    -------
    Array
        The damped cumulative sums, the shape of `steps`.
    """
    m = steps.shape[0]
    lag = jnp.arange(m)[:, None] - jnp.arange(m)[None, :]
    weights = jnp.where(lag >= 0, phi ** jnp.maximum(lag, 0), 0.0)
    return weights @ steps


class NonCentredRandomWalk(TemporalProcess):
    r"""Random walk or AR(1), frozen before the data, extendable past them.

    $$x(t) = x_0 \ (t \le k), \qquad x(t) = \phi\, x(t - 1) + \sigma z_t
    \ (t > k), \qquad z_t \sim N(0, 1),$$

    with $\phi = 1$ (random walk, the prior of PyRenew's `RandomWalk`)
    or $\phi = 2^{-1/h}$ for a half-life $h$ steps from `half_life_rv`
    (mean-reverting towards 0). PyRenew samples the steps one site per
    step in a scan; here they are one vectorised site, `{prefix}_z`. The
    first `frozen` steps are zero, so nothing moves before the data
    begin.

    With `n_fit` set, only the first `n_fit` points use `{prefix}_z`,
    which is all a fit sees. A longer axis adds a separate
    `{prefix}_z_future` site that `Predictive` draws from the prior given
    the posterior, so steps with no data never enter NUTS.
    `forecast_half_life` $h_f$ makes the steps past the fit axis AR(1)
    with $\phi_f = 2^{-1/h_f}$, so a random walk reverts towards 0 there
    and the fit is unchanged.

    With a normal prior $x_0 \sim N(\mu_0, s_0)$ as `initial_value`, the
    start is sampled through its anchor $A = x_0 + m / c$, the mean of
    $x$ over the fit points from point $k$ divided by $c$, at
    `{prefix}_anchor`. Here $m$ is the mean of the steps' part and $c$
    the mean of $\phi^{t - k}$, so $A \sim N(\mu_0 + m / c, s_0)$ and
    $x_0 = A - m / c$. This is a change of variables with Jacobian 1, so
    the model is unchanged. The data fix the level of the walk over
    their period much more tightly than its start, and the anchor is
    close to that level.
    """

    step_size: int = 1

    def __init__(
        self,
        innovation_sd_rv: RandomVariable,
        frozen: int = 0,
        n_fit: int | None = None,
        half_life_rv: RandomVariable | None = None,
        forecast_half_life: float | None = None,
    ) -> None:
        r"""Initialize the process.

        Parameters
        ----------
        innovation_sd_rv
            Step SD $\sigma$.
        frozen
            Leading steps fixed at zero.
        n_fit
            Length of the fit axis. `None` samples every step as one site.
        half_life_rv
            Half-life in steps; makes the process AR(1).
        forecast_half_life
            Half-life in steps of the reversion past the fit axis.

        Raises
        ------
        TypeError
            If `innovation_sd_rv` or `half_life_rv` is not a
            `RandomVariable`.
        """
        if not isinstance(innovation_sd_rv, RandomVariable):
            raise TypeError("innovation_sd_rv must be a RandomVariable.")
        if half_life_rv is not None and not isinstance(
            half_life_rv, RandomVariable
        ):
            raise TypeError("half_life_rv must be a RandomVariable.")
        self.innovation_sd_rv = innovation_sd_rv
        self.frozen = frozen
        self.n_fit = n_fit
        self.half_life_rv = half_life_rv
        self.forecast_half_life = forecast_half_life

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NonCentredRandomWalk(innovation_sd_rv={self.innovation_sd_rv!r}"
            f", frozen={self.frozen}, n_fit={self.n_fit}, "
            f"half_life_rv={self.half_life_rv!r}, "
            f"forecast_half_life={self.forecast_half_life})"
        )

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

        Parameters
        ----------
        n_timepoints
            Number of time points.
        initial_value
            Starting value of each process, or a normal prior for it,
            sampled through the anchor. Defaults to 0.
        n_processes
            Number of independent processes.
        name_prefix
            Prefix of the sample site names.
        first_day_dow
            Unused. See `pyrenew.latent.TemporalProcess`.

        Returns
        -------
        Array
            Shape `(n_timepoints, n_processes)`.
        """
        n_past = (
            n_timepoints
            if self.n_fit is None
            else min(self.n_fit, n_timepoints)
        )
        frozen = min(self.frozen, n_past - 1)
        n_free = n_past - 1 - frozen
        sd = self.innovation_sd_rv()
        phi = (
            1.0
            if self.half_life_rv is None
            else 0.5 ** (1.0 / self.half_life_rv())
        )
        z = numpyro.sample(
            f"{name_prefix}_z",
            dist.Normal(0.0, 1.0).expand([n_free, n_processes]),
        )
        free = _damped_sum(phi, sd * z)
        decay = phi ** jnp.arange(1, n_free + 1)[:, None]
        if isinstance(initial_value, dist.Normal):
            # Points k onwards are x0 * (1, decay) + (0, free), so m / c
            # is the sum of free over the sum of (1, decay).
            shift = free.sum(axis=0) / (1 + decay.sum())
            anchor = numpyro.sample(
                f"{name_prefix}_anchor",
                dist.Normal(
                    initial_value.loc + shift, initial_value.scale
                ).expand([n_processes]),
            )
            x0 = anchor - shift
        else:
            x0 = jnp.broadcast_to(
                jnp.asarray(0.0 if initial_value is None else initial_value),
                (n_processes,),
            )
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
        if self.forecast_half_life is not None:
            phi = 0.5 ** (1.0 / self.forecast_half_life)
        decay = phi ** jnp.arange(1, n_future + 1)[:, None]
        future = decay * past[-1] + _damped_sum(phi, sd * z_future)
        return jnp.vstack([past, future])
