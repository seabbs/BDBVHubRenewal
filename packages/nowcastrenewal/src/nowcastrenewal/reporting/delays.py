# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Discretised reporting delays: the parametric families and their pmfs.

A delay $T$ from reference to report has a continuous distribution $F$ of
one of `FAMILIES`. Two discretisations are used:

- `double_censored_log_pmf` and `double_censored_logit_hazard`: both the
  reference and the report day are censored to whole days, with the
  uniform interval approximation $p_d = \{F(d + 1) - F(d - 1)\} / Z$ on
  $d = 0, \dots, D - 1$, where $Z = F(D) + F(D - 1)$ normalises over the
  window. This is the delay of the reporting triangle; its logit hazard
  is the baseline that the hazard effects shift.
- `interval_pmf`: $p_d \propto F(d + 1) - F(d)$, a delay from the start of
  the reference day. The simulator uses it only to size the NB1
  dispersion (its $\sum_d p_d^2$).

`ParametricDelay` is a PyRenew `RandomVariable` that returns the double
censored pmf for locations and scales from other components, one per
group and day.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import Array
from jax.scipy.special import gammainc, gammaincc, ndtr
from jax.typing import ArrayLike
from pyrenew.metaclass import RandomVariable
from scipy import stats

__all__ = [
    "FAMILIES",
    "DelayParameters",
    "ParametricDelay",
    "cdf_and_survival",
    "double_censored_log_pmf",
    "double_censored_logit_hazard",
    "interval_pmf",
    "log_cdf",
]

FAMILIES = ("exponential", "lognormal", "gamma")
r"""The parametric families, each by a location $\mu$ and scale $\sigma$:
exponential with rate $e^{-\mu}$; lognormal with meanlog $\mu$ and sdlog
$\sigma$; gamma with shape $e^{\mu}$ and rate $\sigma$."""


def _log1m_exp(x: Array) -> Array:
    return jnp.log(-jnp.expm1(x))


def cdf_and_survival(
    family: str, mu: ArrayLike, sigma: ArrayLike, x: ArrayLike
) -> tuple[Array, Array]:
    r"""Return $F(x)$ and $S(x) = 1 - F(x)$, each to full precision.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale, broadcast against `x`.
    x : array_like
        Positive delays.

    Returns
    -------
    tuple of Array
        $F(x)$ and $S(x)$.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`.
    """
    mu, sigma, x = jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(x)
    if family == "exponential":
        rate_x = jnp.exp(-mu) * x
        return -jnp.expm1(-rate_x), jnp.exp(-rate_x)
    if family == "lognormal":
        z = (jnp.log(x) - mu) / sigma
        return ndtr(z), ndtr(-z)
    if family == "gamma":
        shape = jnp.exp(mu)
        return gammainc(shape, sigma * x), gammaincc(shape, sigma * x)
    raise ValueError(f"family must be one of {FAMILIES}")


def log_cdf(
    family: str, mu: ArrayLike, sigma: ArrayLike, x: ArrayLike
) -> Array:
    r"""Return $\log F(x)$ for a family's location and scale.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale, broadcast against `x`.
    x : array_like
        Positive delays.

    Returns
    -------
    Array
        $\log F(x)$.
    """
    return jnp.log(cdf_and_survival(family, mu, sigma, x)[0])


class _Cdf:
    r"""$F$ at delays $-1, 0, 1, \dots, D$, with $F(-1) = F(0) = 0$.

    Differences $F(b) - F(a)$ are taken from $F$ below the median and
    from $S$ above it, so neither loses digits.
    """

    def __init__(
        self, family: str, mu: ArrayLike, sigma: ArrayLike, dmax: int
    ) -> None:
        if dmax < 3:
            raise ValueError("dmax must be at least 3")
        mu = jnp.asarray(mu)[..., None]
        sigma = jnp.asarray(sigma)[..., None]
        f, s = cdf_and_survival(
            family, mu, sigma, jnp.arange(1, dmax + 1, dtype=float)
        )
        lead = (*f.shape[:-1], 2)
        self.f = jnp.concatenate([jnp.zeros(lead, f.dtype), f], axis=-1)
        self.s = jnp.concatenate([jnp.ones(lead, s.dtype), s], axis=-1)
        self.dmax = dmax

    def __call__(self, a: Array, b: Array) -> Array:
        r"""Return $F(b) - F(a)$ for indices $a \le b$ (delays a - 1, b - 1)."""
        fa, fb = self.f[..., a], self.f[..., b]
        sa, sb = self.s[..., a], self.s[..., b]
        return jnp.where(fb < 0.5, fb - fa, sa - sb)


def double_censored_log_pmf(
    family: str, mu: ArrayLike, sigma: ArrayLike, dmax: int
) -> Array:
    r"""Return the double censored delay's log pmf on 0, ..., dmax - 1.

    $\log p_d$ with $p_d = \{F(d + 1) - F(d - 1)\} / \{F(D) + F(D - 1)\}$
    and $F(0) = F(-1) = 0$.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale; leading dimensions are kept.
    dmax : int
        Delays in the window, $D$ (at least 3).

    Returns
    -------
    Array
        Shape `(..., dmax)`.
    """
    diff = _Cdf(family, mu, sigma, dmax)
    d = jnp.arange(dmax)
    # Index i holds F(i - 1).
    z = diff.f[..., dmax + 1] + diff.f[..., dmax]
    return jnp.log(diff(d, d + 2)) - jnp.log(z)[..., None]


def double_censored_logit_hazard(
    family: str, mu: ArrayLike, sigma: ArrayLike, dmax: int
) -> Array:
    r"""Return the double censored delay's logit hazard on 0, ..., dmax - 1.

    $h_d = p_d / (1 - \sum_{i < d} p_i)$, so

    $$\operatorname{logit} h_d = \log\{F(d + 1) - F(d - 1)\}
    - \log[\{F(D) - F(d + 1)\} + \{F(D - 1) - F(d)\}],$$

    and $h_{D - 1} = 1$ (a logit of $+\infty$), so every report falls in
    the window.

    Parameters
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : array_like
        Location and scale; leading dimensions are kept.
    dmax : int
        Delays in the window, $D$ (at least 3).

    Returns
    -------
    Array
        Shape `(..., dmax)`.
    """
    diff = _Cdf(family, mu, sigma, dmax)
    d = jnp.arange(dmax)
    top = jnp.full_like(d, dmax + 1)
    rest = diff(d + 2, top) + diff(d + 1, top - 1)
    logit = jnp.log(diff(d, d + 2)) - jnp.log(rest)
    return logit.at[..., -1].set(jnp.inf)


@dataclass(frozen=True)
class DelayParameters:
    r"""A family's location $\mu$ and scale $\sigma$ from the grid's values.

    Attributes
    ----------
    family : str
        One of `FAMILIES`.
    mu, sigma : float
        Location and scale (`FAMILIES`); the exponential's scale is unused.
    """

    family: str
    mu: float
    sigma: float

    @classmethod
    def from_values(
        cls, family: str, values: Mapping[str, Any]
    ) -> DelayParameters:
        """Return the parameters from a simulation's values.

        Lognormal from `delay_median` and `delay_sdlog`; gamma from
        `delay_mean` and `delay_cv` (shape $1 / cv^2$); exponential from
        `delay_mean`.

        Parameters
        ----------
        family : str
            One of `FAMILIES`.
        values : mapping
            The values.

        Returns
        -------
        DelayParameters
            The parameters.

        Raises
        ------
        ValueError
            If `family` is not one of `FAMILIES`.
        """
        v = values
        if family == "lognormal":
            return cls(family, math.log(v["delay_median"]), v["delay_sdlog"])
        if family == "gamma":
            shape = 1 / v["delay_cv"] ** 2
            return cls(family, math.log(shape), shape / v["delay_mean"])
        if family == "exponential":
            return cls(family, math.log(v["delay_mean"]), 1.0)
        raise ValueError(f"family must be one of {FAMILIES}")


def interval_pmf(
    family: str, values: Mapping[str, Any], max_delay: int
) -> np.ndarray:
    r"""Return $p_d \propto F(d + 1) - F(d)$ on 0, ..., max_delay - 1.

    For `"none"` (a hazard with no parametric part) the delay is geometric
    with hazard $1 / (1 + \text{delay\_mean})$.

    Parameters
    ----------
    family : str
        One of `FAMILIES` or `"none"`.
    values : mapping
        The values (`DelayParameters.from_values`).
    max_delay : int
        Delays in the window.

    Returns
    -------
    numpy.ndarray
        The pmf, summing to 1.

    Raises
    ------
    ValueError
        If `family` is unknown.
    """
    d = np.arange(max_delay + 1, dtype=float)
    v = values
    if family == "lognormal":
        cdf = stats.lognorm.cdf(d, v["delay_sdlog"], scale=v["delay_median"])
    elif family == "gamma":
        shape = 1 / v["delay_cv"] ** 2
        cdf = stats.gamma.cdf(d, shape, scale=v["delay_mean"] / shape)
    elif family == "exponential":
        cdf = stats.expon.cdf(d, scale=v["delay_mean"])
    elif family == "none":
        hazard = 1 / (1 + v["delay_mean"])
        cdf = 1 - (1 - hazard) ** d
    else:
        raise ValueError(f"unknown family {family!r}")
    pmf = np.diff(cdf)
    return np.asarray(pmf / pmf.sum(), dtype=float)


class ParametricDelay(RandomVariable):
    """The double censored delay pmf for given locations and scales.

    Parameters
    ----------
    name : str
        Name of the component; the pmf is recorded as `"{name}::pmf"`.
    family : str
        One of `FAMILIES`.
    mu_rv, sigma_rv : RandomVariable
        Return the location and scale, of any matching shape (for example
        one per group and reference day).
    dmax : int
        Delays in the window.

    Raises
    ------
    ValueError
        If `family` is not one of `FAMILIES`.
    """

    def __init__(
        self,
        name: str,
        family: str,
        mu_rv: RandomVariable,
        sigma_rv: RandomVariable,
        dmax: int,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        super().__init__(name=name)
        self.family = family
        self.mu_rv = mu_rv
        self.sigma_rv = sigma_rv
        self.dmax = dmax

    def sample(self, **kwargs: object) -> Array:
        """Return the pmf.

        Parameters
        ----------
        **kwargs
            Ignored, for the PyRenew interface.

        Returns
        -------
        Array
            Shape `(*mu.shape, dmax)`.
        """
        pmf = jnp.exp(
            double_censored_log_pmf(
                self.family, self.mu_rv(), self.sigma_rv(), self.dmax
            )
        )
        numpyro.deterministic(f"{self.name}::pmf", pmf)
        return pmf
