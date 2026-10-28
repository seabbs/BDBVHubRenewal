# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Known and hub-imputed onsets by province, with right truncation.

Let $\lambda_p(s)$ be expected confirmed cases with onset on day $s$,
$f_s(d)$ the onset-to-report delay, $\pi_p(t)$ the probability that a case
reported on day $t$ has no onset, $T$ the vintage cut-off and $\hat f$
the hub's empirical delay for that vintage. Then

$$\kappa_p(s, d) = \lambda_p(s) f_s(d) \{1 - \pi_p(s + d)\}$$

is the expected number of known-onset cases with onset $s$ and delay
$d$. A case reported on day $t$ appears in the vintage at $T$ with
probability $c(t) = F_\ell(T - t)$, with $F_\ell$ the cdf of the report
to publication lag $\ell$ on $0..L$. Known onsets by $T$ are
$\sum_d \kappa_p(s, d) c(s + d)$.
Reports by report day are $R_p(t) = \sum_d \lambda_p(t - d) f_{t-d}(d)$,
missing-onset cases are $M_p(t) = \pi_p(t) R_p(t)$, and the hub spreads
those published by $T$ back with $\hat f$, giving expected imputed onsets

$$\iota_p(s) = \sum_{t = s}^{T} \hat f(t - s) c(t) M_p(t).$$
"""

from __future__ import annotations

from functools import reduce
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from pyrenew.deterministic import DeterministicPMF
from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import CountNoise
from pyrenew.observation.types import ObservationSample
from pyrenew.time import daily_to_weekly

_FLOOR = 1e-6


def backsample(missing: Array, pmf: Array) -> Array:
    r"""Spread missing-onset cases back from report day with a delay pmf.

    $a(s) = \sum_d m(s + d) \hat f(d)$, the hub's back-sampling in
    expectation (`bdbv_hub_renewal.data.delays.expected_imputed_onsets`).
    Mass that would fall before day 0 is lost.

    Parameters
    ----------
    missing : Array
        Missing-onset cases by report day, shape `(n,)`.
    pmf : Array
        Delay pmf on 0..D.

    Returns
    -------
    Array
        Expected cases by onset day, shape `(n,)`.
    """
    n = missing.shape[0]
    return jnp.convolve(missing[::-1], pmf, mode="full")[:n][::-1]


def _nowcast(draws: Array, vintage: Array | None) -> Array:
    """Add the draw of later reports to the vintage counts, or their draw."""
    if vintage is None:
        return draws[0] + draws[1]
    return jnp.where(jnp.isnan(vintage), draws[0], vintage) + draws[1]


def _recentred(draws: Array, vintage: Array | None) -> Array:
    """Take the final draw where it was centred on the vintage count."""
    if vintage is None:
        return draws[0] + draws[1]
    return jnp.where(jnp.isnan(vintage), draws[0] + draws[1], draws[1])


def skew(x: Array, width: int) -> Array:
    r"""Shift row $s$ of `x` right by $s$: `out[s, s + d] = x[s, d]`.

    Pad, flatten and reshape, so the adjoint is also a reshape and needs
    no scatter. Requires `width >= n + D` for `x` of shape `(n, D + 1)`.

    Returns
    -------
    Array
        Shape `(n, width)`.
    """
    n, k = x.shape
    padded = jnp.pad(x, ((0, 0), (0, width + 1 - k)))
    return padded.reshape(-1)[: n * width].reshape(n, width)


def unskew(b: Array, k: int) -> Array:
    """Inverse of `skew`: `out[s, d] = b[s, s + d]` for `d < k`.

    Returns
    -------
    Array
        Shape `(n, k)`.
    """
    n, width = b.shape
    flat = jnp.pad(b.reshape(-1), (0, n))
    return flat.reshape(n, width + 1)[:, :k]


def published_by(n: int, lag_pmf: Array | None) -> Array:
    r"""Probability that a report on day $t$ is published by day $n - 1$.

    $c(t) = F_\ell(n - 1 - t)$, with $F_\ell$ the cdf of the report to
    publication lag pmf on $0..L$, so only the last $L$ days are partly
    published.

    Parameters
    ----------
    n : int
        Number of report days, the last being the cut-off. Static.
    lag_pmf : Array or None
        Lag pmf on $0..L$. `None` is no lag.

    Returns
    -------
    Array
        Shape `(n,)`.
    """
    if lag_pmf is None:
        return jnp.ones(n)
    cdf = jnp.cumsum(jnp.asarray(lag_pmf))[::-1]
    k = cdf.shape[0]
    if n >= k:
        return jnp.concatenate([jnp.ones(n - k), cdf])
    return cdf[k - n :]


def onset_expectations(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool = True,
    publication_lag: Array | None = None,
    weekly: bool = False,
) -> dict[str, Array]:
    r"""Compute expected known and imputed onsets by the cut-off and eventually.

    With $A_{st} = f_s(t - s)$ the onset-by-report-day delay matrix and
    $H_{st} = \hat f(t - s)$ the hub's back-sampling matrix, both banded
    and of shape `(n, n + D)`, and $\pi$ extended to report days past the
    end (last row repeated), and $c$ from `published_by`:

    - known by $T$: $\lambda \odot (A_{:, \le T} (c (1 - \pi))_{\le T})$;
    - missing by report day: $M = \pi \odot A^\top \lambda$;
    - imputed by $T$: $H_{:, \le T} (c M)_{\le T}$, zero past $T$.

    Every term is a dense matrix product, so autodiff needs no gather or
    scatter adjoints. Equal to the direct version in
    `tests/reference_kernels.py`.

    Parameters
    ----------
    onsets : Array
        $\lambda_p(s)$, shape `(n, P)`.
    delay : Array
        $f_s(d)$, shape `(n, D + 1)`.
    missing : Array
        $\pi_p(t)$ by report day, shape `(n, P)`.
    hub_delay : Array
        The hub's empirical pmf $\hat f$.
    cutoff : int
        Cut-off day $T$ on the same axis. Static.
    final : bool, optional
        Also compute the final counts. Fitting does not need them, and
        skipping them saves about a fifth of the gradient cost.
    publication_lag : Array, optional
        Pmf of the report to publication lag on $0..L$. `None` publishes
        every report on its report day.
    weekly : bool, optional
        The axis starts on a week boundary and has whole weeks. Then
        `_weekly_blocks` gives the same values at a cost linear, not
        quadratic, in the length of the outbreak.

    Returns
    -------
    dict
        `known` and `imputed` by $T$, `known_final` and `imputed_final`
        once all reports are in (all `(n, P)`; equal to the counts by $T$
        when `final` is false), and `triangle`, the
        expected known cases by onset day and delay published by $T$,
        summed over provinces, `(n, D + 1)`.
    """
    if weekly:
        return _weekly_blocks(
            onsets, delay, missing, hub_delay, cutoff, final, publication_lag
        )
    n, n_d = delay.shape
    width = n + n_d - 1
    pi_ext = missing[jnp.minimum(jnp.arange(width), n - 1)]
    keep = 1 - pi_ext
    a = skew(delay, width)
    h = skew(jnp.broadcast_to(hub_delay, (n, n_d)), width)
    t1 = cutoff + 1
    a_t = a[:, :t1]
    c = published_by(t1, publication_lag)[:, None]
    seen = c * keep[:t1]
    known = onsets * (a_t @ seen)
    missing_by_t = c * pi_ext[:t1] * (a_t.T @ onsets)
    imputed = h[:, :t1] @ missing_by_t
    if final:
        known_final = onsets * (a @ keep)
        imputed_final = h @ (pi_ext * (a.T @ onsets))
    else:
        known_final, imputed_final = known, imputed
    # Expected known cells published by T, summed over provinces, as a
    # band on (onset, report day), then back to (onset, delay).
    band = a_t * (onsets @ seen.T)
    band = jnp.pad(band, ((0, 0), (0, width - t1)))
    return {
        "known": known,
        "imputed": imputed,
        "known_final": known_final,
        "imputed_final": imputed_final,
        "triangle": unskew(band, n_d),
    }


def _windows(x: Array, n_w: int, k: int) -> Array:
    """`out[w, i] = x[7w + i]` for `i < 7k`, from stacked reshapes.

    Returns
    -------
    Array
        Shape `(n_w, 7 k, P)` for `x` of shape `(m, P)`.
    """
    pad = 7 * (n_w + k) - x.shape[0]
    blocks = jnp.pad(x, ((0, pad), (0, 0))).reshape(n_w + k, 7, -1)
    out = jnp.stack([blocks[j : j + n_w] for j in range(k)], axis=1)
    return out.reshape(n_w, 7 * k, -1)


def _overlap_add(r: Array, n_w: int, k: int, width: int) -> Array:
    """Sum windows back onto the day axis, the adjoint of `_windows`.

    Returns
    -------
    Array
        Shape `(width, P)`.
    """
    r = r.reshape(n_w, k, 7, -1)
    total = reduce(
        jnp.add,
        [jnp.pad(r[:, j], ((j, k - j), (0, 0), (0, 0))) for j in range(k)],
    )
    return total.reshape(7 * (n_w + k), -1)[:width]


def _weekly_blocks(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool,
    publication_lag: Array | None,
) -> dict[str, Array]:
    r"""`onset_expectations` on an axis of whole weeks.

    Onsets in week $w$ reach report days $7w$ to $7w + 6 + D$ only, so
    each term is a product of a $7 \times 7k$ banded block $G_w$, with
    $G_w[j, i] = f_{7w + j}(i - j)$ and $7k \ge D + 7$, and a window of the
    report-day axis. Windows are stacked reshapes and their adjoint is
    an overlap-add, so there is still no gather or scatter. The cost is
    linear in the number of weeks where the dense version is quadratic:
    about 2 times faster for the value and gradient at twice the length
    of the 2026-10-24 vintage, and the same at that length.

    Returns
    -------
    dict
        As `onset_expectations`.
    """
    n, n_d = delay.shape
    n_w, width = n // 7, n + n_d - 1
    k = -(-(n_d + 6) // 7)
    span = 7 * k
    g = jax.vmap(lambda f: skew(f, span))(delay.reshape(n_w, 7, n_d))
    gh = jnp.broadcast_to(
        skew(jnp.broadcast_to(hub_delay, (7, n_d)), span), (n_w, 7, span)
    )
    lam = onsets.reshape(n_w, 7, -1)
    pi_ext = missing[jnp.minimum(jnp.arange(width), n - 1)]
    keep = 1 - pi_ext
    t1 = cutoff + 1
    c = jnp.concatenate(
        [published_by(t1, publication_lag), jnp.zeros(width - t1)]
    )[:, None]

    def ahead(blocks: Array, x: Array) -> Array:
        # sum_d f_w(d) x(s + d) for every onset day s.
        out = jnp.einsum("wji,wip->wjp", blocks, _windows(x, n_w, k))
        return out.reshape(n, -1)

    by_report = jnp.einsum("wji,wjp->wip", g, lam)
    miss = pi_ext * _overlap_add(by_report, n_w, k, width)
    seen = c * keep
    known = onsets * ahead(g, seen)
    imputed = ahead(gh, c * miss)
    if final:
        known_final = onsets * ahead(g, keep)
        imputed_final = ahead(gh, miss)
    else:
        known_final, imputed_final = known, imputed
    cells = jnp.einsum("wjp,wip->wji", lam, _windows(seen, n_w, k)) * g
    triangle = jax.vmap(lambda b: unskew(b, n_d))(cells).reshape(n, n_d)
    return {
        "known": known,
        "imputed": imputed,
        "known_final": known_final,
        "imputed_final": imputed_final,
        "triangle": triangle,
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
    c(s + d)}{\sum_{s \in w} \sum_{d'} \sum_p \kappa_p(s, d') c(s + d')},$$

    with the same publication weights $c$ as the known stream, so the
    two truncate alike.

    which is the conditional part of the Poisson reporting triangle, so
    it does not double count the known stream.
    Each stream's site, `{name}_known_obs` and `{name}_imputed_obs`, has
    two rows: the vintage counts and the counts still to be reported
    after the cut-off, which are never observed and so are drawn by
    `Predictive`. With `forecast=True` the final counts are recorded as
    `{name}_final_total`: the vintage counts where known (`vintage`),
    else the row 0 draw, plus the row 1 draw of the rest, as in
    epinowcast, so draws are never below the known onsets reported.
    The hub re-imputes missing onsets at each vintage, so a week's imputed
    count can fall as well as rise. Its final count is drawn from the
    imputed stream's negative binomial centred on the current imputed
    count plus the count expected from cases reported later.

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
    publication_lag : RandomVariable or array_like, optional
        Pmf of the lag from report to publication in a vintage, on
        $0..L$: a random variable to estimate it, or a fixed pmf.
        `None` is no lag.
    delay_overdispersion_rv : RandomVariable, optional
        $1 / \sqrt{\psi}$, with $\psi$ capped at $10^4$. If given, the
        delay counts are Dirichlet-multinomial with concentration
        $\psi P(d \mid w)$ instead of multinomial, so that the pooled
        delays of thousands of cases do not make the delay overconfident.
    weekly_axis : bool, optional
        The model axis starts on `start_dow` with whole weeks, so the
        expectations can use per-week blocks (`onset_expectations` with
        `weekly`).
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
        publication_lag: RandomVariable | Array | None = None,
        delay_overdispersion_rv: RandomVariable | None = None,
        weekly_axis: bool = False,
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
        self.publication_lag = publication_lag
        self.delay_overdispersion_rv = delay_overdispersion_rv
        self.weekly_axis = weekly_axis

    def lookback_days(self) -> int:
        """Return 0: the latent process is already dated by onset."""
        return 0

    def infection_resolution(self) -> str:
        """Return `"subpop"`."""
        return "subpop"

    def _predicted_obs(self, infections: Array) -> Array:
        """Return infections unchanged; see `onset_expectations`."""
        return infections

    def validate_data(
        self,
        n_total: int,
        n_subpops: int,
        cutoff: int | None = None,
        known: Array | None = None,
        imputed: Array | None = None,
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

    def _weekly(self, x: Array, first_day_dow: int) -> Array:
        return jnp.asarray(daily_to_weekly(x, first_day_dow, self.start_dow))

    def sample(
        self,
        infections: Array,
        first_day_dow: int,
        cutoff: int,
        hub_delay_pmf: Array,
        known: Array | None = None,
        imputed: Array | None = None,
        delay_counts: Array | None = None,
        inactive: Array | None = None,
        import_kernel: Array | None = None,
        forecast: bool = False,
        vintage: Array | None = None,
        as_of: int | None = None,
    ) -> ObservationSample:
        """Score or sample the two onset streams.

        Parameters
        ----------
        infections : Array
            Infections by province, `(n_total, P)`.
        first_day_dow : int
            Day of week of element 0 of the model axis.
        cutoff : int
            Vintage cut-off $T$ as an index on the model axis.
        hub_delay_pmf : Array
            The hub's empirical delay pmf for this vintage.
        known, imputed : Array, optional
            Weekly counts on the model's week grid, `(n_weeks, P)`, NaN
            where unobserved.
        delay_counts : Array, optional
            Known-onset cases by onset week and delay, pooled over
            provinces, `(n_weeks, D + 1)`.
        inactive : Array, optional
            Weekly total counts in provinces with no cases, `(n_weeks, Q)`.
        import_kernel : Array, optional
            Kernel rows from the modelled provinces to those with no
            cases, `(P, Q)`. Needed with `import_scale_rv`.
        forecast : bool, optional
            Record the sum of the final draws of both streams.
        vintage : Array, optional
            Known and imputed vintage counts, `(2, n_weeks, P)`, NaN
            where unobserved or held out. With `forecast`, the final
            draws add the cases still to be reported to these.
        as_of : int, optional
            With `forecast`, a later cut-off on the model axis: the
            "final" draws are then the counts as reported by it (the
            vintage counts plus the cases reported between the two
            cut-offs), for scoring against a later vintage as reported.

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
        if self.weekly_axis and (n % 7 or first_day_dow != self.start_dow):
            raise ValueError("Week blocks need an axis of whole weeks.")
        if self.missing_fraction_rv is None:
            pi = jnp.zeros_like(onsets)
        else:
            pi = self.missing_fraction_rv(
                n_timepoints=n, n_groups=n_prov, first_day_dow=first_day_dow
            )
        lag = self.publication_lag
        if isinstance(lag, RandomVariable):
            lag = lag()
        exp = onset_expectations(
            onsets,
            delay,
            pi,
            jnp.asarray(hub_delay_pmf),
            cutoff,
            forecast,
            publication_lag=lag,
            weekly=self.weekly_axis,
        )
        weekly = {k: self._weekly(v, first_day_dow) for k, v in exp.items()}
        if as_of is not None:
            later = onset_expectations(
                onsets,
                delay,
                pi,
                jnp.asarray(hub_delay_pmf),
                as_of,
                False,
                publication_lag=lag,
                weekly=self.weekly_axis,
            )
            weekly["known_final"] = self._weekly(later["known"], first_day_dow)
            weekly["imputed_final"] = self._weekly(
                later["imputed"], first_day_dow
            )
        for key, value in weekly.items():
            self._deterministic(f"expected_{key}", value)
        self._deterministic("expected_daily", onsets)

        if self.missing_fraction_rv is None:
            total = None
            if known is not None and imputed is not None:
                total = known + imputed
            draws = self._score(
                "total",
                self.known_noise,
                weekly["known"],
                weekly["known_final"],
                total,
            )
            final = _nowcast(draws, None if vintage is None else vintage.sum(0))
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
                centre=None if vintage is None else vintage[1],
            )
            seen = (None, None) if vintage is None else vintage
            final = _nowcast(draws, seen[0]) + _recentred(
                imputed_draws, seen[1]
            )
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
        onsets: Array,
        kernel: Array,
        counts: Array | None,
        first_day_dow: int,
    ) -> None:
        """Importation-only expectation and likelihood for idle provinces."""
        if self.import_scale_rv is None:
            raise ValueError("Importations need `import_scale_rv`.")
        daily = self.import_scale_rv() * onsets @ kernel
        self._deterministic("expected_imports_daily", daily)
        weekly = jnp.maximum(self._weekly(daily, first_day_dow), _FLOOR)
        if counts is None:
            numpyro.sample(
                self._sample_site_name("inactive_obs"), dist.Poisson(weekly)
            )
            return
        mask = ~jnp.isnan(counts)
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample(
                self._sample_site_name("inactive_obs"),
                dist.Poisson(weekly),
                obs=jnp.where(mask, counts, 0.0),
            )

    def _score(
        self,
        stream: str,
        noise: CountNoise,
        by_cutoff: Array,
        final: Array,
        obs: Array | None,
        centre: Array | None = None,
    ) -> Array:
        """Score the vintage counts and draw the final counts in one site.

        Row 0 of the site is the vintage (scored where `obs` is not NaN),
        row 1 the count reported after the cut-off, which is never
        observed. One site keeps one concentration draw per stream.
        Where `centre` is not NaN, row 1 is instead the final count,
        centred on `centre` plus the count expected after the cut-off.
        """
        rest = final - by_cutoff
        if centre is not None:
            rest = jnp.where(jnp.isnan(centre), rest, centre + rest)
        predicted = jnp.maximum(jnp.stack([by_cutoff, rest]), _FLOOR)
        if obs is None:
            return cast(
                Array,
                noise.sample(
                    self._sample_site_name(f"{stream}_obs"), predicted
                ),
            )
        obs = jnp.stack([obs, jnp.full_like(obs, jnp.nan)])
        mask = ~jnp.isnan(obs)
        return cast(
            Array,
            noise.sample(
                self._sample_site_name(f"{stream}_obs"),
                predicted,
                obs=jnp.where(mask, obs, 0.0),
                mask=mask,
            ),
        )

    def _score_delays(self, triangle: Array, counts: Array) -> None:
        totals = counts.sum(axis=1)
        probs = triangle / jnp.maximum(
            triangle.sum(axis=1, keepdims=True), 1e-12
        )
        probs = jnp.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)
        if self.delay_overdispersion_rv is None:
            counts_dist = dist.Multinomial(total_count=totals, probs=probs)
        else:
            # Capped at 1e4: near the multinomial the float32 log density
            # is noisy enough to stall NUTS step-size adaptation.
            psi = 1 / (jnp.square(self.delay_overdispersion_rv()) + 1e-4)
            counts_dist = dist.DirichletMultinomial(
                concentration=psi * probs, total_count=totals
            )
        with numpyro.handlers.mask(mask=totals > 0):
            numpyro.sample(
                self._sample_site_name("delays"), counts_dist, obs=counts
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

    def __init__(
        self,
        alpha: float,
        day: int,
        groups: ArrayLike,
        values: ArrayLike,
        scales: ArrayLike,
        kinds: ArrayLike,
    ) -> None:
        self.alpha = float(alpha)
        self.day = int(day)
        self.groups = jnp.asarray(groups, dtype=float)
        self.values = jnp.asarray(values, dtype=float)
        self.scales = jnp.asarray(scales, dtype=float)
        self.kinds = jnp.asarray(kinds, dtype=int)

    def log_group_rt(
        self,
        rt: Array,
        infections: Array,
        n_init: int,
        fractions: Array | None = None,
    ) -> tuple[Array, Array]:
        """Return log $R_G(s_b)$ for each group and for the nation.

        Infections are per resident, so the weights are scaled by the
        population `fractions` to be proportional to counts.

        Returns
        -------
        tuple of Array
            `(G,)` group values and the national scalar.
        """
        week = n_init + 7 * (self.day // 7)
        w = infections[week : week + 7].sum(axis=0)
        if fractions is not None:
            w = w * fractions
        r = rt[n_init + self.day]

        def log_rt(member: Array) -> Array:
            return jnp.log((member * w * r).sum(-1) / (member * w).sum(-1))

        return log_rt(self.groups), log_rt(jnp.ones_like(w))

    def sample(
        self,
        rt: Array,
        infections: Array,
        n_init: int,
        fractions: Array | None = None,
    ) -> None:
        """Add the pseudo-likelihood and record its diagnostics."""
        group, national = self.log_group_rt(rt, infections, n_init, fractions)
        model = jnp.where(self.kinds == 0, group, group - national)
        numpyro.deterministic("bvd_log_rt_model", model)
        numpyro.deterministic("bvd_z", (self.values - model) / self.scales)
        if self.alpha > 0:
            numpyro.sample(
                "bvd_rt_obs",
                dist.Normal(model, self.scales / jnp.sqrt(self.alpha)),
                obs=self.values,
            )
