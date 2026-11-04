# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Delay pmfs: the censored parametric pmf and a fixed time-varying input.

`FixedTimeVarying` returns a matrix $f_s(d)$ of shape
`(n_timepoints, dmax + 1)`, the delay pmf for cases with onset on day $s$.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


def censored_pmf(
    family: str, mean: ArrayLike, cv: ArrayLike, dmax: int
) -> Array:
    r"""Daily double interval censored pmf of a delay, in closed form.

    With the primary event uniform over a day and the secondary censored
    to its day (primarycensored with `pwindow = swindow = 1`),

    $$f(d) = G(d + 1) - 2 G(d) + G(d - 1), \qquad
    G(x) = \int_0^x F(t)\, dt = x F_1(x) - m F_2(x),$$

    for $x > 0$ and $G = 0$ otherwise, with $m$ the mean. For a Gamma
    with shape $k$, $F_1$ and $F_2$ are Gamma cdfs with shapes $k$ and
    $k + 1$; for a log-normal, $F_1(x) = \Phi(z)$ and
    $F_2(x) = \Phi(z - \sigma)$ with $z = (\log x - \mu) / \sigma$.
    Since $G(x) = x - m + Q(x)$ with $Q(x) = m S_2(x) - x S_1(x)$ and
    $S = 1 - F$, and second differences of a linear term vanish, the right
    tail uses $Q$: each delay takes whichever of $G$ and $Q$ is smaller
    there. In float32 the cdfs are good to about $10^{-6}$ and the
    second difference scales that by the delay, so against a float64
    reference the error is at most $3 \times 10^{-5}$ for means of 2 to 15
    days and cv 0.3 to 2. Truncated at `dmax` and renormalised, as
    `primarycensored::dpcens(..., D = dmax + 1)`.
    The simulator (`bdbvhubrenewal.data.simulate`) draws its delays with
    it; `bdbvhubrenewal.model.model.generation_interval` uses the same
    form in float64, once.

    Parameters
    ----------
    family : {"gamma", "lognormal"}
        Delay family.
    mean, cv : ArrayLike
        Mean (days) and coefficient of variation.
    dmax : int
        Maximum delay.

    Returns
    -------
    ArrayLike
        Pmf on 0..dmax.
    """
    mean = jnp.asarray(mean)
    if family == "gamma":
        shape = 1 / jnp.square(cv)
        rate = shape / mean

        def cdfs(x: Array) -> tuple[Array, Array, Array, Array]:
            f1 = jax.scipy.special.gammainc(shape, rate * x)
            f2 = jax.scipy.special.gammainc(shape + 1, rate * x)
            s1 = jax.scipy.special.gammaincc(shape, rate * x)
            s2 = jax.scipy.special.gammaincc(shape + 1, rate * x)
            return f1, f2, s1, s2

    else:
        sdlog = jnp.sqrt(jnp.log1p(jnp.square(cv)))
        mu = jnp.log(mean) - jnp.square(sdlog) / 2
        norm = jax.scipy.stats.norm

        def cdfs(x: Array) -> tuple[Array, Array, Array, Array]:
            z = (jnp.log(x) - mu) / sdlog
            return (
                norm.cdf(z),
                norm.cdf(z - sdlog),
                norm.cdf(-z),
                norm.cdf(sdlog - z),
            )

    x = jnp.arange(-1.0, dmax + 2.0)
    pos = x > 0
    xs = jnp.where(pos, x, 1.0)
    f1, f2, s1, s2 = cdfs(xs)
    g = jnp.where(pos, xs * f1 - mean * f2, 0.0)
    # Q differs from G by x - m, which the second difference removes, but
    # Q is only valid for x > 0.
    q = jnp.where(pos, mean * s2 - xs * s1, mean - x)

    def second(h: Array) -> Array:
        return h[2:] - 2 * h[1:-1] + h[:-2]

    right = jnp.arange(dmax + 1) >= mean
    pmf = jnp.where(right, second(q), second(g))
    pmf = jnp.maximum(pmf, 1e-12)
    return pmf / pmf.sum()


class FixedTimeVarying(RandomVariable):
    """A fixed time-varying input: the hub's delay in the total-stream steps.

    Stands in for `HazardDelay` with a matrix fixed beforehand. It is
    data, so it adds no sites.
    Rows past the end are filled with the last row, so the same input
    covers the forecast horizon.

    Parameters
    ----------
    name : str
        Name of the input.
    value : array_like
        Shape `(n_rows, k)`.
    """

    def __init__(self, name: str, value: ArrayLike) -> None:
        super().__init__(name=name)
        self.value = jnp.asarray(value)

    def sample(self, n_timepoints: int, **kwargs: object) -> Array:
        """Return the input padded or cut to `n_timepoints` rows.

        Parameters
        ----------
        n_timepoints : int
            Rows to return.
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, k)`.
        """
        idx = jnp.minimum(jnp.arange(n_timepoints), self.value.shape[0] - 1)
        return self.value[idx]
