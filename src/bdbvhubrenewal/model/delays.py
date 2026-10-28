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
from jax import Array
from jax.typing import ArrayLike
from pyrenew.latent.temporal_processes import (
    TemporalProcess,
    WeeklyTemporalProcess,
)
from pyrenew.metaclass import RandomVariable
from pyrenew.randomvariable import DistributionalVariable

from .rt import NonCentredRandomWalk


def hazard_to_pmf(logit_hazard: Array) -> Array:
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

    $$\operatorname{logit} h_s(d) = \beta(d) + \gamma(s)
    \ [+ \delta_{(s + d) \bmod 7}],$$

    where $\beta$ is a random walk over delay $d$ (a smoothness prior)
    and $\gamma$ is a temporal process over onset day $s$ that shifts all
    hazards together (proportional odds). With `report_dow`, $\delta$ is
    a sum-to-zero effect of the day of the week of the report day,
    $\delta = \sigma_\delta (z - \bar z)$ with $z_k \sim N(0, 1)$ and
    $\sigma_\delta \sim N^+(0, 0.5^2)$.
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
        Scale of the half-normal prior on the step SD of $\beta$. Real
        fits put the step SD at about 0.2 (2026-10-24: 90% CrI 0.16 to
        0.25), in the tail of a scale of 0.1, so the default is 0.4.
    report_dow : bool, optional
        Add the report day-of-week effect $\delta$.
    """

    def __init__(
        self,
        name: str,
        dmax: int,
        time_process: TemporalProcess,
        hazard0_loc: float = -2.0,
        smoothness_sd: float = 0.4,
        report_dow: bool = False,
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
        self.report_dow = report_dow

    def sample(
        self,
        n_timepoints: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Array:
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
        weekly = (
            not self.report_dow
            and isinstance(self.time_process, WeeklyTemporalProcess)
            and first_day_dow == self.time_process.start_dow
        )
        if weekly:
            # The shift is constant within weeks, so compute one row per
            # week and repeat it.
            gamma = gamma[::7]
        logit = beta[None, :] + gamma
        if self.report_dow:
            logit = logit + self._report_dow(n_timepoints, first_day_dow or 0)
        pmf = hazard_to_pmf(logit)
        mean = pmf @ jnp.arange(self.dmax + 1)
        if weekly:
            pmf = jnp.repeat(pmf, 7, axis=0)[:n_timepoints]
            mean = jnp.repeat(mean, 7, axis=0)[:n_timepoints]
        numpyro.deterministic(f"{self.name}_mean", mean)
        return pmf

    def _report_dow(self, n: int, first_day_dow: int) -> Array:
        r"""Return $\delta$ of each report day, by onset day and delay."""
        sd = numpyro.sample(f"{self.name}_report_dow_sd", dist.HalfNormal(0.5))
        z = numpyro.sample(
            f"{self.name}_report_dow_z", dist.Normal(0.0, 1.0).expand([7])
        )
        effect = numpyro.deterministic(
            f"{self.name}_report_dow", sd * (z - z.mean())
        )
        # Rows repeat every 7 onset days, so index 7 rows and tile them.
        day = first_day_dow + np.arange(7)[:, None] + np.arange(self.dmax)
        rows = effect[day % 7]
        return jnp.tile(rows, (-(-n // 7), 1))[:n]


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
    days and cv 0.3 to 2 (16-node Gauss-Legendre quadrature reached
    $5.7 \times 10^{-5}$), and the Gamma value and gradient are about 20
    times faster. Truncated at `dmax` and renormalised, as
    `primarycensored::dpcens(..., D = dmax + 1)`.
    `bdbv_hub_renewal.model.model.generation_interval` uses the same form
    in float64, once.

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
        self,
        n_timepoints: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Array:
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

    def sample(self, n_timepoints: int, **kwargs: object) -> Array:
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
