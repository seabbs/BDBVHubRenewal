# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Time-varying onset-to-report delay pmfs, pooled across provinces.

Each class returns a matrix $f_s(d)$ of shape `(n_timepoints, dmax + 1)`,
the delay pmf for cases with onset on day $s$.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.latent.temporal_processes import (
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.randomvariable import DistributionalVariable

from .rt import NonCentredRandomWalk


def hazard_to_pmf(logit_hazard: ArrayLike) -> ArrayLike:
    r"""Convert discrete-time hazards to a pmf on the last axis.

    $$f(d) = h(d) \prod_{j < d} \{1 - h(j)\}, \qquad h(D) = 1,$$

    so each row sums to one. The cumulative sum of $\log(1 - h)$ is a
    product with a triangular matrix, which is faster than `cumsum` on
    CPU, forward and backward.

    Parameters
    ----------
    logit_hazard : ArrayLike
        Logit hazards for delays 0..D-1, shape `(..., D)`.

    Returns
    -------
    ArrayLike
        Pmf on 0..D, shape `(..., D + 1)`.
    """
    k = logit_hazard.shape[-1]
    log_h = jax.nn.log_sigmoid(logit_hazard)
    log_1mh = jax.nn.log_sigmoid(-logit_hazard)
    # Column d sums the first d terms, so the result is the log survival
    # before each delay, with zero at d = 0.
    upper = jnp.triu(jnp.ones((k, k + 1), logit_hazard.dtype), 1)
    log_surv_before = log_1mh @ upper
    zero = jnp.zeros((*log_h.shape[:-1], 1), log_h.dtype)
    log_h = jnp.concatenate([log_h, zero], axis=-1)
    return jnp.exp(log_h + log_surv_before)


class HazardDelay(RandomVariable):
    r"""Nonparametric delay with a smooth baseline and a time shift.

    $$\operatorname{logit} h_s(d) = \beta(d) + \gamma(s),$$

    where $\beta$ is a random walk over delay $d$ (a smoothness prior)
    and $\gamma$ is a temporal process over onset day $s$ that shifts all
    hazards together (proportional odds).
    The pmf follows from `hazard_to_pmf`. When $\gamma$ has a small
    innovation SD, the delay is near constant.

    Parameters
    ----------
    name : str
        Prefix for the sample sites.
    dmax : int
        Maximum delay $D$.
    time_process : TemporalProcess
        Process for $\gamma(s)$, starting at 0.
    hazard0_loc : float, optional
        Prior mean of $\beta(0)$.
    smoothness_sd : float, optional
        Scale of the half-normal prior on the step SD of $\beta$.
    """

    def __init__(
        self,
        name: str,
        dmax: int,
        time_process: TemporalProcess,
        hazard0_loc: float = -2.0,
        smoothness_sd: float = 0.1,
    ) -> None:
        super().__init__(name=name)
        self.dmax = dmax
        self.time_process = time_process
        self.shape_process = NonCentredRandomWalk(
            DistributionalVariable(
                f"{name}_smooth_sd", dist.HalfNormal(smoothness_sd)
            )
        )
        self.hazard0_loc = hazard0_loc

    def sample(
        self, n_timepoints: int, first_day_dow: int | None = None, **kwargs
    ) -> ArrayLike:
        """Sample the delay matrix.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, dmax + 1)`.
        """
        beta0 = numpyro.sample(
            f"{self.name}_hazard0", dist.Normal(self.hazard0_loc, 1.0)
        )
        beta = self.shape_process.sample(
            n_timepoints=self.dmax,
            initial_value=beta0,
            name_prefix=f"{self.name}_shape",
        )[:, 0]
        gamma = self.time_process.sample(
            n_timepoints=n_timepoints,
            initial_value=0.0,
            name_prefix=f"{self.name}_time",
            first_day_dow=first_day_dow,
        )
        weekly = isinstance(self.time_process, WeeklyTemporalProcess) and (
            first_day_dow == self.time_process.start_dow
        )
        if weekly:
            # The shift is constant within weeks, so compute one row per
            # week and repeat it.
            gamma = gamma[::7]
        pmf = hazard_to_pmf(beta[None, :] + gamma)
        mean = pmf @ jnp.arange(self.dmax + 1)
        if weekly:
            pmf = jnp.repeat(pmf, 7, axis=0)[:n_timepoints]
            mean = jnp.repeat(mean, 7, axis=0)[:n_timepoints]
        numpyro.deterministic(f"{self.name}_mean", mean)
        return pmf


def censored_pmf(
    family: str, mean: ArrayLike, cv: ArrayLike, dmax: int, n_nodes: int = 16
) -> ArrayLike:
    r"""Daily double interval censored pmf of a continuous delay.

    With the primary event uniform over a day and the secondary censored
    to its day (primarycensored with `pwindow = swindow = 1`),

    $$f(d) = F_{pc}(d + 1) - F_{pc}(d), \qquad
    F_{pc}(t) = \int_0^1 F(t - u)\, du,$$

    computed as $\int_0^1 \{F(d + 1 - u) - F(d - u)\} du$ by
    Gauss-Legendre quadrature, which keeps each term on the scale of the
    pmf (the closed form differences cumulative integrals and loses digits
    in float32). Truncated at `dmax` and renormalised, as
    `primarycensored::dpcens(..., D = dmax + 1)`; matches it to 2e-4.
    Gamma and log-normal families, parameterised by mean and coefficient
    of variation.

    Parameters
    ----------
    family : {"gamma", "lognormal"}
        Delay family.
    mean, cv : ArrayLike
        Mean (days) and coefficient of variation.
    dmax : int
        Maximum delay.
    n_nodes : int, optional
        Quadrature nodes.

    Returns
    -------
    ArrayLike
        Pmf on 0..dmax.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = jnp.asarray((nodes + 1) / 2)
    w = jnp.asarray(weights / 2)
    d = jnp.arange(dmax + 1)[:, None]

    def cdf(x):
        pos = x > 0
        x = jnp.where(pos, x, 1.0)
        if family == "gamma":
            shape = 1 / cv**2
            value = jax.scipy.special.gammainc(shape, x * shape / mean)
        else:
            sdlog = jnp.sqrt(jnp.log1p(cv**2))
            mu = jnp.log(mean) - sdlog**2 / 2
            value = jax.scipy.stats.norm.cdf((jnp.log(x) - mu) / sdlog)
        return jnp.where(pos, value, 0.0)

    pmf = (cdf(d + 1 - u) - cdf(d - u)) @ w
    pmf = jnp.maximum(pmf, 1e-12)
    return pmf / pmf.sum()


class ParametricDelay(RandomVariable):
    r"""Censored parametric delay with a time-varying mean.

    $$\log \mu(s) = \log \mu_0 + \gamma(s),$$

    with a fixed coefficient of variation $c$ and the pmf from
    `censored_pmf` for the chosen family.

    Parameters
    ----------
    name : str
        Prefix for the sample sites.
    dmax : int
        Maximum delay $D$.
    time_process : TemporalProcess
        Process for $\gamma(s)$, starting at 0.
    family : {"gamma", "lognormal"}, optional
        Delay family.
    mean_loc, mean_scale : float, optional
        Log-normal prior on $\mu_0$ (median and log SD).
    """

    def __init__(
        self,
        name: str,
        dmax: int,
        time_process: TemporalProcess,
        family: str = "gamma",
        mean_loc: float = 8.0,
        mean_scale: float = 0.5,
    ) -> None:
        super().__init__(name=name)
        self.dmax = dmax
        self.time_process = time_process
        self.family = family
        self.mean_loc = mean_loc
        self.mean_scale = mean_scale

    def sample(
        self, n_timepoints: int, first_day_dow: int | None = None, **kwargs
    ) -> ArrayLike:
        """Sample the delay matrix.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, dmax + 1)`.
        """
        mu0 = numpyro.sample(
            f"{self.name}_mean0",
            dist.LogNormal(jnp.log(self.mean_loc), self.mean_scale),
        )
        cv = numpyro.sample(f"{self.name}_cv", dist.LogNormal(0.0, 0.3))
        gamma = self.time_process.sample(
            n_timepoints=n_timepoints,
            initial_value=0.0,
            name_prefix=f"{self.name}_time",
            first_day_dow=first_day_dow,
        )[:, 0]
        mean = mu0 * jnp.exp(gamma)
        pmf = jax.vmap(lambda m: censored_pmf(self.family, m, cv, self.dmax))(
            mean
        )
        numpyro.deterministic(
            f"{self.name}_mean", pmf @ jnp.arange(self.dmax + 1)
        )
        return pmf


class FixedTimeVarying(RandomVariable):
    """A fixed time-varying input, for the fallback configuration.

    Stands in for `HazardDelay` or `MissingOnsetFraction` with a matrix
    estimated from data beforehand. It is data, so it adds no sites.
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

    def sample(self, n_timepoints: int, **kwargs) -> ArrayLike:
        """Return the input padded or cut to `n_timepoints` rows.

        Returns
        -------
        ArrayLike
            Shape `(n_timepoints, k)`.
        """
        idx = jnp.minimum(jnp.arange(n_timepoints), self.value.shape[0] - 1)
        return self.value[idx]


def empirical_delay_matrix(
    delay_counts: ArrayLike,
    cutoff: int,
    window: int = 2,
    prior_weight: float = 1.0,
) -> np.ndarray:
    r"""Smoothed time-varying delay pmf from the hub's delay file.

    `delay_counts[w, d]` counts known-onset cases with onset in week $w$
    (days $7w$ to $7w + 6$) and delay $d$, reported by day `cutoff`.
    Week $w$ is pooled with its neighbours within `window` weeks. Delays
    beyond the longest delay observable for that week,
    $L_w = T - 7w - 6$, are right truncated, so the pmf is completed with
    a pooled tail,

    $$\hat f_w(d) = (1 - q_w)\frac{c_w(d)}{\sum_{j \le L_w} c_w(j)}
    \ (d \le L_w), \qquad \hat f_w(d) = q_w \frac{\bar f(d)}
    {\sum_{j > L_w} \bar f(j)}\ (d > L_w),$$

    where $\bar f$ is the pooled pmf over weeks with full follow-up and
    $q_w = \sum_{j > L_w} \bar f(j)$. A pseudo-count of
    `prior_weight` times $\bar f$ smooths sparse weeks.

    Parameters
    ----------
    delay_counts : array_like
        Counts, shape `(n_weeks, dmax + 1)`.
    cutoff : int
        Data cut-off day on the same daily axis.
    window : int, optional
        Half-width of the pooling window in weeks.
    prior_weight : float, optional
        Pseudo-count weight on the pooled pmf.

    Returns
    -------
    numpy.ndarray
        Daily matrix, shape `(7 * n_weeks, dmax + 1)`.
    """
    counts = np.asarray(delay_counts, dtype=float)
    n_weeks, n_d = counts.shape
    longest = cutoff - 7 * np.arange(n_weeks) - 6
    full = longest >= n_d - 1
    base = counts[full].sum(axis=0) if full.any() else counts.sum(axis=0)
    base = (base + 0.5) / (base + 0.5).sum()
    out = np.empty_like(counts)
    for w in range(n_weeks):
        lo, hi = max(0, w - window), min(n_weeks, w + window + 1)
        c = counts[lo:hi].sum(axis=0) + prior_weight * base
        cap = int(np.clip(min(longest[lo:hi]), 0, n_d - 1))
        head = c[: cap + 1] / c[: cap + 1].sum()
        tail_mass = base[cap + 1 :].sum()
        out[w, : cap + 1] = (1 - tail_mass) * head
        out[w, cap + 1 :] = base[cap + 1 :]
    return np.repeat(out, 7, axis=0)


def empirical_missing_fraction(
    known: ArrayLike,
    imputed: ArrayLike,
    window: int = 2,
    shrink: float = 5.0,
    shift: int = 0,
) -> np.ndarray:
    r"""Smoothed missing-onset fraction by province and week, from data.

    The imputed share of onsets in each week, pooled over neighbouring
    weeks and shrunk towards the national share,

    $$\hat\pi_p(w) = \frac{\sum_{w'} m_p(w') + a \bar\pi}
    {\sum_{w'} \{k_p(w') + m_p(w')\} + a},$$

    with $k$ known and $m$ imputed onset counts and $a$ = `shrink`.
    Weeks with no data take the last available value.
    These shares are by onset week, but $\pi$ is by report day, so the
    daily values move `shift` days later (the mean delay), and the first
    days take the first value.

    Parameters
    ----------
    known, imputed : array_like
        Weekly counts, shape `(n_weeks, P)`, NaN where unobserved.
    window : int, optional
        Half-width of the pooling window in weeks.
    shrink : float, optional
        Pseudo-count $a$ towards the national share.
    shift : int, optional
        Days from onset to report day.

    Returns
    -------
    numpy.ndarray
        Daily matrix, shape `(7 * n_weeks, P)`.
    """
    k = np.nan_to_num(np.asarray(known, dtype=float))
    m = np.nan_to_num(np.asarray(imputed, dtype=float))
    national = (m.sum() + 1) / (k.sum() + m.sum() + 2)
    out = np.full(k.shape, national)
    for w in range(k.shape[0]):
        lo, hi = max(0, w - window), min(k.shape[0], w + window + 1)
        mw, tw = m[lo:hi].sum(axis=0), (k[lo:hi] + m[lo:hi]).sum(axis=0)
        out[w] = (mw + shrink * national) / (tw + shrink)
        if w > 0 and (k[w] + m[w]).sum() == 0:
            out[w] = out[w - 1]
    daily = np.repeat(out, 7, axis=0)
    if shift <= 0:
        return daily
    lead = np.repeat(daily[:1], shift, axis=0)
    return np.vstack([lead, daily[:-shift]])
