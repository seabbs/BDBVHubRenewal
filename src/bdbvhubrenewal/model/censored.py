# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Double interval censored daily pmfs of continuous delays.

A delay $X$ from a primary event, uniform over a day, to a secondary
event recorded to its day has the pmf of primarycensored
(`dpcens(..., pwindow = 1, swindow = 1)`), here in closed form for the
Gamma, log-normal and generalised gamma. `CensoredPMF` gives that pmf as
a PyRenew random variable with a random mean and SD.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable

Cdfs = Callable[[Array], tuple[Array, Array, Array, Array]]
"""Returns $F_1$, $F_2$ and their upper tails $S_1$, $S_2$ at delays."""


def _from_cdfs(cdfs: Cdfs, mean: Array, dmax: int) -> Array:
    """Pmf on 0..dmax from $F_1$, $F_2$ and the mean, as `censored_pmf`."""
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

    return _from_cdfs(cdfs, mean, dmax)


def gengamma_censored_pmf(
    mu: ArrayLike, sigma: ArrayLike, q: ArrayLike, dmax: int
) -> Array:
    r"""Daily double interval censored pmf of a generalised gamma delay.

    The generalised gamma of flexsurv's `pgengamma` with $Q > 0$:
    $Y = Q^{-2} e^{Q w} \sim \text{Gamma}(Q^{-2}, 1)$ with
    $w = (\log x - \mu) / \sigma$. With $k = Q^{-2}$ and $s = \sigma / Q$,
    $X = e^{\mu} (Y / k)^s$, so the partial first moment is
    $E[X; X \le x] = m P(k + s, y)$ with mean
    $m = e^{\mu} k^{-s} \Gamma(k + s) / \Gamma(k)$ and $P$ the
    regularised lower incomplete gamma at $y = k e^{Q w}$. Hence
    $F_1 = P(k, y)$ and $F_2 = P(k + s, y)$ in `censored_pmf`'s closed
    form. Truncated at `dmax` and renormalised.

    Parameters
    ----------
    mu, sigma, q : ArrayLike
        Location (log days), scale and shape $Q > 0$.
    dmax : int
        Maximum delay.

    Returns
    -------
    ArrayLike
        Pmf on 0..dmax.

    """
    k = 1 / jnp.square(q)
    s = sigma / q
    mean = jnp.exp(
        mu
        - s * jnp.log(k)
        + jax.scipy.special.gammaln(k + s)
        - jax.scipy.special.gammaln(k)
    )

    def cdfs(x: Array) -> tuple[Array, Array, Array, Array]:
        y = k * jnp.exp(q * (jnp.log(x) - mu) / sigma)
        return (
            jax.scipy.special.gammainc(k, y),
            jax.scipy.special.gammainc(k + s, y),
            jax.scipy.special.gammaincc(k, y),
            jax.scipy.special.gammaincc(k + s, y),
        )

    return _from_cdfs(cdfs, mean, dmax)


class CensoredPMF(RandomVariable):
    """Double interval censored daily pmf with a random mean and SD.

    Returns `censored_pmf` on 0..dmax at the sampled mean and SD, so the
    delay's uncertainty enters the model through `mean_rv` and `sd_rv`.
    """

    def __init__(
        self,
        name: str,
        family: str,
        mean_rv: RandomVariable,
        sd_rv: RandomVariable,
        dmax: int,
    ) -> None:
        """Initialize the pmf.

        Parameters
        ----------
        name
            Name of the pmf.
        family
            Delay family, `"gamma"` or `"lognormal"`.
        mean_rv, sd_rv
            Mean and SD of the delay in days.
        dmax
            Maximum delay, so the pmf has `dmax + 1` entries.

        Raises
        ------
        TypeError
            If `mean_rv` or `sd_rv` is not a `RandomVariable`.
        ValueError
            If `family` is not `"gamma"` or `"lognormal"`.
        """
        if not isinstance(mean_rv, RandomVariable) or not isinstance(
            sd_rv, RandomVariable
        ):
            raise TypeError("mean_rv and sd_rv must be RandomVariables.")
        if family not in ("gamma", "lognormal"):
            raise ValueError("family must be 'gamma' or 'lognormal'.")
        super().__init__(name=name)
        self.family = family
        self.mean_rv = mean_rv
        self.sd_rv = sd_rv
        self.dmax = dmax

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CensoredPMF(name={self.name!r}, family={self.family!r}, "
            f"mean_rv={self.mean_rv!r}, sd_rv={self.sd_rv!r}, "
            f"dmax={self.dmax})"
        )

    def sample(self, **kwargs: object) -> Array:
        """Sample the mean and SD and return the pmf.

        Parameters
        ----------
        **kwargs
            Ignored.

        Returns
        -------
        Array
            Pmf on 0..dmax.
        """
        mean = self.mean_rv()
        return censored_pmf(self.family, mean, self.sd_rv() / mean, self.dmax)
