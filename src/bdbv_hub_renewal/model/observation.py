# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Known and hub-imputed onsets by province, with right truncation.

Let $\lambda_p(s)$ be expected confirmed cases with onset on day $s$,
$f_s(d)$ the onset-to-report delay, $\pi_p(t)$ the probability that a case
reported on day $t$ has no onset, $T$ the vintage cut-off and $\hat f$
the hub's empirical delay for that vintage. Then

$$\kappa_p(s, d) = \lambda_p(s) f_s(d) \{1 - \pi_p(s + d)\}$$

is the expected number of known-onset cases with onset $s$ and delay
$d$, and known onsets by $T$ are $\sum_{d \le T - s} \kappa_p(s, d)$.
Reports by report day are $R_p(t) = \sum_d \lambda_p(t - d) f_{t-d}(d)$,
missing-onset cases are $M_p(t) = \pi_p(t) R_p(t)$, and the hub spreads
those reported by $T$ back with $\hat f$, giving expected imputed onsets

$$\iota_p(s) = \sum_{t = s}^{T} \hat f(t - s) M_p(t).$$
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.deterministic import DeterministicPMF
from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import CountNoise
from pyrenew.observation.types import ObservationSample
from pyrenew.time import daily_to_weekly

_FLOOR = 1e-6


def backsample(missing: ArrayLike, pmf: ArrayLike) -> ArrayLike:
    r"""Spread missing-onset cases back from report day with a delay pmf.

    $a(s) = \sum_d m(s + d) \hat f(d)$, the hub's back-sampling in
    expectation (`bdbv_hub_renewal.observation.expected_imputed_onsets`).
    Mass that would fall before day 0 is lost.

    Parameters
    ----------
    missing : ArrayLike
        Missing-onset cases by report day, shape `(n,)`.
    pmf : ArrayLike
        Delay pmf on 0..D.

    Returns
    -------
    ArrayLike
        Expected cases by onset day, shape `(n,)`.
    """
    n = missing.shape[0]
    return jnp.convolve(missing[::-1], pmf, mode="full")[:n][::-1]


def skew(x: ArrayLike, width: int) -> ArrayLike:
    r"""Shift row $s$ of `x` right by $s$: `out[s, s + d] = x[s, d]`.

    Pad, flatten and reshape, so the adjoint is also a reshape and needs
    no scatter. Requires `width >= n + D` for `x` of shape `(n, D + 1)`.

    Returns
    -------
    ArrayLike
        Shape `(n, width)`.
    """
    n, k = x.shape
    padded = jnp.pad(x, ((0, 0), (0, width + 1 - k)))
    return padded.reshape(-1)[: n * width].reshape(n, width)


def unskew(b: ArrayLike, k: int) -> ArrayLike:
    """Inverse of `skew`: `out[s, d] = b[s, s + d]` for `d < k`.

    Returns
    -------
    ArrayLike
        Shape `(n, k)`.
    """
    n, width = b.shape
    flat = jnp.pad(b.reshape(-1), (0, n))
    return flat.reshape(n, width + 1)[:, :k]


def onset_expectations(
    onsets: ArrayLike,
    delay: ArrayLike,
    missing: ArrayLike,
    hub_delay: ArrayLike,
    cutoff: int,
    final: bool = True,
) -> dict[str, ArrayLike]:
    r"""Compute expected known and imputed onsets by the cut-off and eventually.

    With $A_{st} = f_s(t - s)$ the onset-by-report-day delay matrix and
    $H_{st} = \hat f(t - s)$ the hub's back-sampling matrix, both banded
    and of shape `(n, n + D)`, and $\pi$ extended to report days past the
    end (last row repeated):

    - known by $T$: $\lambda \odot (A_{:, \le T} (1 - \pi)_{\le T})$;
    - missing by report day: $M = \pi \odot A^\top \lambda$;
    - imputed by $T$: $H_{:, \le T} M_{\le T}$, zero past $T$.

    Every term is a dense matrix product, so autodiff needs no gather or
    scatter adjoints. Equal to the direct version in
    `tests/reference_kernels.py`.

    Parameters
    ----------
    onsets : ArrayLike
        $\lambda_p(s)$, shape `(n, P)`.
    delay : ArrayLike
        $f_s(d)$, shape `(n, D + 1)`.
    missing : ArrayLike
        $\pi_p(t)$ by report day, shape `(n, P)`.
    hub_delay : ArrayLike
        The hub's empirical pmf $\hat f$.
    cutoff : int
        Cut-off day $T$ on the same axis. Static.
    final : bool, optional
        Also compute the final counts. Fitting does not need them, and
        skipping them saves about a fifth of the gradient cost.

    Returns
    -------
    dict
        `known` and `imputed` by $T$, `known_final` and `imputed_final`
        once all reports are in (all `(n, P)`; equal to the counts by $T$
        when `final` is false), and `triangle`, the
        expected known cases by onset day and delay reported by $T$,
        summed over provinces, `(n, D + 1)`.
    """
    n, n_d = delay.shape
    width = n + n_d - 1
    pi_ext = missing[jnp.minimum(jnp.arange(width), n - 1)]
    keep = 1 - pi_ext
    a = skew(delay, width)
    h = skew(jnp.broadcast_to(hub_delay, (n, n_d)), width)
    t1 = cutoff + 1
    a_t = a[:, :t1]
    known = onsets * (a_t @ keep[:t1])
    missing_by_t = pi_ext[:t1] * (a_t.T @ onsets)
    imputed = h[:, :t1] @ missing_by_t
    if final:
        known_final = onsets * (a @ keep)
        imputed_final = h @ (pi_ext * (a.T @ onsets))
    else:
        known_final, imputed_final = known, imputed
    # Expected known cells reported by T, summed over provinces, as a
    # band on (onset, report day), then back to (onset, delay).
    band = a_t * (onsets @ keep[:t1].T)
    band = jnp.pad(band, ((0, 0), (0, width - t1)))
    return {
        "known": known,
        "imputed": imputed,
        "known_final": known_final,
        "imputed_final": imputed_final,
        "triangle": unskew(band, n_d),
    }


class OnsetStreams(BaseObservationProcess):
    r"""Known and hub-imputed weekly onsets for each province.

    Expected confirmed onsets are $\lambda_p(s) = \alpha(s) I_p(s)$,
    with $I_p$ the latent infections dated by onset and $\alpha$ the
    time-varying ascertainment. `onset_expectations` gives the two streams.
    Each is summed to weeks and scored with its own negative binomial.
    The delays of known-onset cases, pooled over provinces and tabulated
    by onset week and delay, are scored with a multinomial given the
    weekly totals,

    $$P(d \mid w) = \frac{\sum_{s \in w} \sum_p \kappa_p(s, d)
    \mathbb{1}[s + d \le T]}{\sum_{s \in w} \sum_{d'} \sum_p
    \kappa_p(s, d') \mathbb{1}[s + d' \le T]},$$

    which is the conditional part of the Poisson reporting triangle, so
    it does not double count the known stream.
    Each stream's site, `{name}_known_obs` and `{name}_imputed_obs`, has
    two rows: the vintage counts and the final (fully reported) counts,
    which are never observed and so are drawn by `Predictive`. With
    `forecast=True` their sum is recorded as `{name}_final_total`.

    Parameters
    ----------
    name : str
        Prefix for all sites.
    ascertainment_rate_rv : RandomVariable
        $\alpha$, scalar or one value per day of the model axis.
    missing_fraction_rv : RandomVariable or None
        Returns $\pi_p(t)$, shape `(n, P)`. `None` fits one stream,
        `total`, of known plus imputed onsets truncated by `delay_rv`
        (the simpler steps of the model ladder).
    delay_rv : RandomVariable
        Returns $f_s(d)$, shape `(n, D + 1)`.
    known_noise, imputed_noise : CountNoise
        PyRenew noise models for the two streams (`known_noise` for the
        single stream).
    kernel_rv : LearntKernel, optional
        Learnt kernel whose `import_kernel` replaces the fixed one.
    import_scale_rv : RandomVariable, optional
        Scale $\kappa$ of importations into provinces with no cases,
        $\mu_q(s) = \kappa \sum_p K_{pq} \lambda_p(s)$, scored with a
        Poisson likelihood on their weekly counts (`inactive`).
    start_dow : int, optional
        First day of the reporting week (6 is Sunday).
    """

    def __init__(
        self,
        name: str,
        ascertainment_rate_rv: RandomVariable,
        missing_fraction_rv: RandomVariable | None,
        delay_rv: RandomVariable,
        known_noise: CountNoise,
        imputed_noise: CountNoise | None = None,
        import_scale_rv: RandomVariable | None = None,
        start_dow: int = 6,
        kernel_rv: RandomVariable | None = None,
    ) -> None:
        super().__init__(
            name, DeterministicPMF(f"{name}_identity", jnp.ones(1))
        )
        self.ascertainment_rate_rv = ascertainment_rate_rv
        self.missing_fraction_rv = missing_fraction_rv
        self.delay_rv = delay_rv
        self.known_noise = known_noise
        self.imputed_noise = imputed_noise
        self.import_scale_rv = import_scale_rv
        self.kernel_rv = kernel_rv
        self.start_dow = start_dow

    def lookback_days(self) -> int:
        """Return 0: the latent process is already dated by onset."""
        return 0

    def infection_resolution(self) -> str:
        """Return `"subpop"`."""
        return "subpop"

    def _predicted_obs(self, infections: ArrayLike) -> ArrayLike:
        """Return infections unchanged; see `onset_expectations`."""
        return infections

    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        cutoff: int | None = None,
        known: ArrayLike | None = None,
        imputed: ArrayLike | None = None,
        **kwargs: object,
    ) -> None:
        """Check the cut-off and the shapes of the weekly data.

        Raises
        ------
        ValueError
            If the cut-off is off the model axis or the weekly arrays do
            not have shape `(n_total // 7, n_subpops)`.
        """
        if cutoff is not None and not 0 <= cutoff < n_total:
            raise ValueError(f"cutoff {cutoff} is outside [0, {n_total}).")
        for label, obs in (("known", known), ("imputed", imputed)):
            if obs is not None and jnp.shape(obs) != (n_total // 7, n_subpops):
                raise ValueError(
                    f"{label} has shape {jnp.shape(obs)}, expected "
                    f"{(n_total // 7, n_subpops)}."
                )

    def _weekly(self, x: ArrayLike, first_day_dow: int) -> ArrayLike:
        return daily_to_weekly(x, first_day_dow, self.start_dow)

    def sample(
        self,
        infections: ArrayLike,
        first_day_dow: int,
        cutoff: int,
        hub_delay_pmf: ArrayLike,
        known: ArrayLike | None = None,
        imputed: ArrayLike | None = None,
        delay_counts: ArrayLike | None = None,
        inactive: ArrayLike | None = None,
        import_kernel: ArrayLike | None = None,
        forecast: bool = False,
    ) -> ObservationSample:
        """Score or sample the two onset streams.

        Parameters
        ----------
        infections : ArrayLike
            Infections by province, `(n_total, P)`.
        first_day_dow : int
            Day of week of element 0 of the model axis.
        cutoff : int
            Vintage cut-off $T$ as an index on the model axis.
        hub_delay_pmf : ArrayLike
            The hub's empirical delay pmf for this vintage.
        known, imputed : ArrayLike, optional
            Weekly counts on the model's week grid, `(n_weeks, P)`, NaN
            where unobserved.
        delay_counts : ArrayLike, optional
            Known-onset cases by onset week and delay, pooled over
            provinces, `(n_weeks, D + 1)`.
        inactive : ArrayLike, optional
            Weekly total counts in provinces with no cases, `(n_weeks, Q)`.
        import_kernel : ArrayLike, optional
            Kernel rows from the modelled provinces to those with no
            cases, `(P, Q)`. Needed with `import_scale_rv`.
        forecast : bool, optional
            Record the sum of the final draws of both streams.

        Returns
        -------
        ObservationSample
            Sampled known counts and the weekly expected known counts.
        """
        n, n_prov = infections.shape
        alpha = jnp.asarray(self.ascertainment_rate_rv())
        alpha = alpha[:, None] if alpha.ndim == 1 else alpha
        onsets = alpha * infections
        delay = self.delay_rv(n_timepoints=n, first_day_dow=first_day_dow)
        if self.missing_fraction_rv is None:
            pi = jnp.zeros_like(onsets)
        else:
            pi = self.missing_fraction_rv(
                n_timepoints=n, n_groups=n_prov, first_day_dow=first_day_dow
            )
        exp = onset_expectations(
            onsets, delay, pi, jnp.asarray(hub_delay_pmf), cutoff, forecast
        )
        weekly = {k: self._weekly(v, first_day_dow) for k, v in exp.items()}
        for key, value in weekly.items():
            self._deterministic(f"expected_{key}", value)
        self._deterministic("expected_daily", onsets)

        if self.missing_fraction_rv is None:
            total = None
            if known is not None:
                total = known + imputed
            draws = self._score(
                "total",
                self.known_noise,
                weekly["known"],
                weekly["known_final"],
                total,
            )
            final = draws[1]
        else:
            draws = self._score(
                "known",
                self.known_noise,
                weekly["known"],
                weekly["known_final"],
                known,
            )
            imputed_draws = self._score(
                "imputed",
                self.imputed_noise,
                weekly["imputed"],
                weekly["imputed_final"],
                imputed,
            )
            final = draws[1] + imputed_draws[1]
            if delay_counts is not None:
                self._score_delays(
                    weekly["triangle"], jnp.asarray(delay_counts)
                )
        if self.kernel_rv is not None and import_kernel is not None:
            # Gravity flows to idle provinces with the learnt decay.
            import_kernel = self.kernel_rv.import_kernel
        if self.import_scale_rv is not None and import_kernel is not None:
            self._score_imports(
                onsets, jnp.asarray(import_kernel), inactive, first_day_dow
            )
        if forecast:
            self._deterministic("final_total", final)
        return ObservationSample(observed=draws[0], predicted=weekly["known"])

    def _score_imports(
        self,
        onsets: ArrayLike,
        kernel: ArrayLike,
        counts: ArrayLike | None,
        first_day_dow: int,
    ) -> None:
        """Importation-only expectation and likelihood for idle provinces."""
        daily = self.import_scale_rv() * onsets @ kernel
        self._deterministic("expected_imports_daily", daily)
        weekly = jnp.maximum(self._weekly(daily, first_day_dow), _FLOOR)
        mask = None if counts is None else ~jnp.isnan(counts)
        with numpyro.handlers.mask(mask=True if mask is None else mask):
            numpyro.sample(
                self._sample_site_name("inactive_obs"),
                dist.Poisson(weekly),
                obs=None if counts is None else jnp.where(mask, counts, 0.0),
            )

    def _score(
        self,
        stream: str,
        noise: CountNoise,
        by_cutoff: ArrayLike,
        final: ArrayLike,
        obs: ArrayLike | None,
    ) -> ArrayLike:
        """Score the vintage counts and draw the final counts in one site.

        Row 0 of the site is the vintage (scored where `obs` is not NaN),
        row 1 the final count once all reports are in, which is never
        observed. One site keeps one concentration draw per stream.
        """
        predicted = jnp.maximum(jnp.stack([by_cutoff, final]), _FLOOR)
        if obs is None:
            return noise.sample(
                self._sample_site_name(f"{stream}_obs"), predicted
            )
        obs = jnp.stack([obs, jnp.full_like(obs, jnp.nan)])
        mask = ~jnp.isnan(obs)
        return noise.sample(
            self._sample_site_name(f"{stream}_obs"),
            predicted,
            obs=jnp.where(mask, obs, 0.0),
            mask=mask,
        )

    def _score_delays(self, triangle: ArrayLike, counts: ArrayLike) -> None:
        totals = counts.sum(axis=1)
        probs = triangle / jnp.maximum(
            triangle.sum(axis=1, keepdims=True), 1e-12
        )
        probs = jnp.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)
        with numpyro.handlers.mask(mask=totals > 0):
            numpyro.sample(
                self._sample_site_name("delays"),
                dist.Multinomial(total_count=totals, probs=probs),
                obs=counts,
            )


class RtPseudoObservation:
    r"""External Rt estimates as a tempered pseudo-observation.

    For groups $G$ of provinces (the nation, or BVDOutbreakSize patches)
    and the day $s_b$ of the external estimate, the model's group Rt is
    onset weighted over the week of $s_b$,

    $$R_G(s_b) = \frac{\sum_{p \in G} I_p R_p(s_b)}{\sum_{p \in G} I_p},$$

    and the summaries enter as

    $$\mu_N \sim N(\log R_N(s_b), \sigma_N / \sqrt{\alpha}), \qquad
    d_k \sim N(\log R_k(s_b) - \log R_N(s_b), \tau_k / \sqrt{\alpha}).$$

    This is a power prior on the external summary with tempering
    $\alpha$. With $\alpha = 0$ nothing is added, so the model is
    unchanged. PyRenew has no observation process on a latent Rt, hence
    this small class; it is called from the latent process, which holds
    both Rt and infections.

    Parameters
    ----------
    alpha : float
        Tempering, in [0, 1].
    day : int
        $s_b$ as a day index from the first data day.
    groups : array_like
        Boolean membership, `(G, P)`.
    values, scales : array_like
        Log-scale summaries and their SDs, `(G,)`.
    kinds : array_like
        0 for the national level, 1 for a deviation from it, `(G,)`.
    """

    def __init__(self, alpha, day, groups, values, scales, kinds) -> None:
        self.alpha = float(alpha)
        self.day = int(day)
        self.groups = jnp.asarray(groups, dtype=float)
        self.values = jnp.asarray(values, dtype=float)
        self.scales = jnp.asarray(scales, dtype=float)
        self.kinds = jnp.asarray(kinds, dtype=int)

    def log_group_rt(self, rt: ArrayLike, infections: ArrayLike, n_init: int):
        """Return log $R_G(s_b)$ for each group and for the nation.

        Returns
        -------
        tuple of ArrayLike
            `(G,)` group values and the national scalar.
        """
        week = n_init + 7 * (self.day // 7)
        w = infections[week : week + 7].sum(axis=0)
        r = rt[n_init + self.day]

        def log_rt(member):
            return jnp.log((member * w * r).sum(-1) / (member * w).sum(-1))

        return log_rt(self.groups), log_rt(jnp.ones_like(w))

    def sample(self, rt: ArrayLike, infections: ArrayLike, n_init: int):
        """Add the pseudo-likelihood and record its diagnostics."""
        group, national = self.log_group_rt(rt, infections, n_init)
        model = jnp.where(self.kinds == 0, group, group - national)
        numpyro.deterministic("bvd_log_rt_model", model)
        numpyro.deterministic("bvd_z", (self.values - model) / self.scales)
        if self.alpha > 0:
            numpyro.sample(
                "bvd_rt_obs",
                dist.Normal(model, self.scales / jnp.sqrt(self.alpha)),
                obs=self.values,
            )
