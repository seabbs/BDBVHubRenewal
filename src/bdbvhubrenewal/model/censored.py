# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Double interval censored daily pmfs of continuous delays.

A delay $X$ from a primary event, uniform over a day, to a secondary
event recorded to its day has the pmf of primarycensored
(`dpcens(..., pwindow = 1, swindow = 1)`), here in closed form for the
Gamma and log-normal. `CensoredDelayPMF` samples its parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
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


class CensoredDelayPMF(RandomVariable):
    r"""A daily double interval censored delay pmf with sampled parameters.

    `censored_pmf` of `family` with mean and coefficient of variation
    from `mean_rv` and `cv_rv`, truncated at `dmax`, recorded as
    `{name}`.
    """

    def __init__(
        self,
        name: str,
        family: str,
        dmax: int,
        mean_rv: RandomVariable,
        cv_rv: RandomVariable,
    ) -> None:
        """Initialize the pmf.

        Parameters
        ----------
        name
            Name of the recorded pmf.
        family
            `"gamma"` or `"lognormal"`.
        dmax
            Maximum delay.
        mean_rv, cv_rv
            Mean (days) and coefficient of variation.

        Raises
        ------
        TypeError
            If `mean_rv` or `cv_rv` is not a `RandomVariable`.
        """
        for label, rv in (("mean_rv", mean_rv), ("cv_rv", cv_rv)):
            if not isinstance(rv, RandomVariable):
                raise TypeError(f"{label} must be a RandomVariable.")
        super().__init__(name=name)
        self.family = family
        self.dmax = dmax
        self.mean_rv = mean_rv
        self.cv_rv = cv_rv

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CensoredDelayPMF(name={self.name!r}, family={self.family!r}, "
            f"dmax={self.dmax}, mean_rv={self.mean_rv!r}, "
            f"cv_rv={self.cv_rv!r})"
        )

    def sample(self, **kwargs: object) -> Array:
        """Sample the pmf.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Pmf on 0..dmax.
        """
        pmf = censored_pmf(self.family, self.mean_rv(), self.cv_rv(), self.dmax)
        numpyro.deterministic(self.name, pmf)
        return pmf
