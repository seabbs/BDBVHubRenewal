# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Known and hub-imputed onsets by province, with right truncation.

Let $\lambda_p(s)$ be expected confirmed cases with onset on day $s$,
$f_s(d)$ the onset-to-report delay, $\pi_p(t)$ the probability that a case
reported on day $t$ has no onset, $T$ the vintage cut-off and $\hat f$
the hub's imputation delay for that release. Then

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
import numpy as np
import numpyro
from jax import Array
from jax.typing import ArrayLike
from pyrenew.convolve import compute_prop_already_reported
from pyrenew.latent.temporal_processes import TemporalProcess
from pyrenew.metaclass import RandomVariable
from pyrenew.observation.base import BaseObservationProcess
from pyrenew.observation.noise import CountNoise
from pyrenew.observation.types import ObservationSample
from pyrenew.time import daily_to_weekly

from .composition_noise import CompositionNoise, MultinomialNoise
from .pooled import pooled_groups, pooled_walk
from .publication_lag import ReleasePairLikelihood

_FLOOR = 1e-6


NO_LAG = np.ones(1)
"""Publication lag pmf with all mass at lag 0.

Every report is published on its report day. A NumPy array, so that
importing the package does not start JAX before the host device count
is set (`cli.host_devices`).
"""


def backsample(missing: Array, pmf: Array) -> Array:
    r"""Spread missing-onset cases back from report day with a delay pmf.

    $a(s) = \sum_d m(s + d) \hat f(d)$, the hub's back-sampling in
    expectation (`bdbvhubrenewal.data.delays.expected_imputed_onsets`).
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

    Parameters
    ----------
    x : jax.Array
        Shape `(n, D + 1)`.
    width : int
        Columns of the output, at least `n + D`.

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

    Parameters
    ----------
    b : jax.Array
        A skewed matrix, as from `skew`.
    k : int
        Columns to return.

    Returns
    -------
    Array
        Shape `(n, k)`.
    """
    n, width = b.shape
    flat = jnp.pad(b.reshape(-1), (0, n))
    return flat.reshape(n, width + 1)[:, :k]


def onset_expectations(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool = True,
    publication_lag: ArrayLike = NO_LAG,
    weekly: bool = False,
) -> dict[str, Array]:
    r"""Compute expected known and imputed onsets by the cut-off and eventually.

    With $A_{st} = f_s(t - s)$ the onset-by-report-day delay matrix and
    $H_{st} = \hat f(t - s)$ the hub's back-sampling matrix, both banded
    and of shape `(n, n + D)`, and $\pi$ extended to report days past the
    end (last row repeated), and $c$ the share of reports published by
    $T$ (PyRenew's `compute_prop_already_reported`):

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
        The hub's imputation pmf $\hat f$.
    cutoff : int
        Cut-off day $T$ on the same axis. Static.
    final : bool, optional
        Also compute the final counts. Fitting does not need them, and
        skipping them saves about a fifth of the gradient cost.
    publication_lag : Array, optional
        Pmf of the report to publication lag on $0..L$. The default, all
        mass at lag 0, publishes every report on its report day.
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
    c = compute_prop_already_reported(jnp.asarray(publication_lag), t1, 0)
    c = c[:, None]
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


def onset_expectations_by_place(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool = True,
    publication_lag: ArrayLike = NO_LAG,
    weekly: bool = False,
) -> dict[str, Array]:
    r"""`onset_expectations` with a delay for each province.

    Each province is computed with its own delay $f_{p,s}$, so the
    triangle stays by province.

    Parameters
    ----------
    onsets : Array
        $\lambda_p(s)$, shape `(n, P)`.
    delay : Array
        $f_{p,s}(d)$, shape `(P, n, D + 1)`.
    missing : Array
        $\pi_p(t)$ by report day, shape `(n, P)`.
    hub_delay, cutoff, final, publication_lag, weekly
        As `onset_expectations`.

    Returns
    -------
    dict
        As `onset_expectations`, with `triangle` of shape `(n, P, D + 1)`.
    """

    def one(lam: Array, f: Array, pi: Array) -> dict[str, Array]:
        return onset_expectations(
            lam[:, None], f, pi[:, None], hub_delay, cutoff, final,
            publication_lag, weekly,
        )  # fmt: skip

    out = jax.vmap(one, in_axes=(1, 0, 1))(onsets, delay, missing)
    streams = ("known", "imputed", "known_final", "imputed_final")
    return {k: out[k][..., 0].T for k in streams} | {
        "triangle": jnp.moveaxis(out["triangle"], 0, 1)
    }


def _week_windows(x: Array, n_w: int, span: int) -> Array:
    """`out[p, w, m] = x[p, 7w + m]` for `m < span`, from stacked reshapes.

    Returns
    -------
    Array
        Shape `(P, n_w, span)` for `x` of shape `(P, m)`.
    """
    k = span // 7
    pad = 7 * (n_w + k) - x.shape[1]
    blocks = jnp.pad(x, ((0, 0), (0, pad))).reshape(x.shape[0], n_w + k, 7)
    return jnp.concatenate([blocks[:, j : j + n_w] for j in range(k)], axis=2)


def _week_overlap_add(r: Array, width: int) -> Array:
    """Sum week windows back onto the day axis, the adjoint of `_week_windows`.

    Returns
    -------
    Array
        Shape `(P, width)` for `r` of shape `(P, n_w, span)`.
    """
    n_p, n_w, span = r.shape
    k = span // 7
    r = r.reshape(n_p, n_w, k, 7)
    total = reduce(
        jnp.add,
        [jnp.pad(r[:, :, j], ((0, 0), (j, k - j), (0, 0))) for j in range(k)],
    )
    return total.reshape(n_p, 7 * (n_w + k))[:, :width]


def weekly_onset_expectations(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool = True,
    publication_lag: ArrayLike = NO_LAG,
    shared: np.ndarray | None = None,
) -> dict[str, Array]:
    r"""Weekly sums of `onset_expectations_by_place` for a weekly delay.

    When each province's delay is the same on every day of a week,
    $f_{p,s} = f_{p,w}$ for $s$ in week $w$, the expected known cells
    published by $T$, summed over the onset days of week $w$, are

    $$f_{p,w}(d) \sum_{j=0}^{6} \lambda_p(7w + j) \{1 - \pi_p(7w + j + d)\}
    c(7w + j + d),$$

    and the known onsets of the week are their sum over $d$. Reports by
    report day are each week's onsets convolved with its delay, and the
    hub's back-sampling summed to weeks is one weekly kernel. Every term
    works on arrays of `(P, n_w, D + 7)` at most, where the daily blocks
    work on one row per day, and nothing is computed by day only to be
    summed to weeks.

    Parameters
    ----------
    onsets : Array
        $\lambda_p(s)$, shape `(n, P)`, with `n` whole weeks.
    delay : Array
        $f_{p,w}(d)$, shape `(P, n // 7, D + 1)`.
    missing : Array
        $\pi_p(t)$ by report day, shape `(n, P)`.
    hub_delay, cutoff, final, publication_lag
        As `onset_expectations`.
    shared : numpy.ndarray, optional
        Boolean `(P,)`, static: provinces whose delay and missing
        fraction are the same, such as those that follow the pooled
        values (`HazardDelay.shared_groups`). Their terms are one product
        with the shared delay for all of them, rather than one per
        province. Only the first shared province's delay and missing
        fraction are read.

    Returns
    -------
    dict
        The weekly sums of `onset_expectations_by_place`: `known`,
        `imputed`, `known_final` and `imputed_final` of shape
        `(n // 7, P)`, and `triangle` of shape `(n // 7, P, D + 1)`.
    """
    if shared is None or not np.any(shared):
        return _weekly_by_place(
            onsets, delay, missing, hub_delay, cutoff, final, publication_lag
        )
    shared = np.asarray(shared, dtype=bool)
    own, same = np.flatnonzero(~shared), np.flatnonzero(shared)
    parts = []
    if own.size:
        parts.append(
            _weekly_by_place(
                onsets[:, own],
                delay[own],
                missing[:, own],
                hub_delay,
                cutoff,
                final,
                publication_lag,
            )
        )
    parts.append(
        _weekly_shared(
            onsets[:, same],
            delay[same[0]],
            missing[:, same[0]],
            hub_delay,
            cutoff,
            final,
            publication_lag,
        )
    )
    order = np.argsort(np.concatenate([own, same]))
    return {
        k: jnp.concatenate([part[k] for part in parts], axis=1)[:, order]
        for k in parts[0]
    }


def _publication_share(
    publication_lag: ArrayLike, cutoff: int, width: int
) -> Array:
    """Share published by the cut-off, by report day, zero past it."""
    t1 = cutoff + 1
    return jnp.concatenate(
        [
            compute_prop_already_reported(jnp.asarray(publication_lag), t1, 0),
            jnp.zeros(width - t1),
        ]
    )


def _hub_by_week(hub_delay: Array, span: int) -> Array:
    r"""Hub back-sampling of a week, $\sum_j \hat f(m - j)$, $m <$ `span`."""
    n_d = hub_delay.shape[0]
    return reduce(
        jnp.add,
        [jnp.pad(hub_delay, (j, span - n_d - j)) for j in range(7)],
    )


def _extend(pi: Array, width: int) -> Array:
    """Repeat the last report day of `pi` (`(..., n)`) up to `width` days."""
    n = pi.shape[-1]
    tail = jnp.broadcast_to(pi[..., -1:], (*pi.shape[:-1], width - n))
    return jnp.concatenate([pi, tail], axis=-1)


def _weekly_by_place(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool,
    publication_lag: ArrayLike,
) -> dict[str, Array]:
    """`weekly_onset_expectations` with a delay for each province."""
    n, n_p = onsets.shape
    n_w, n_d = delay.shape[1], delay.shape[2]
    width = n + n_d - 1
    span = 7 * -(-(n_d + 6) // 7)
    lam = onsets.T.reshape(n_p, n_w, 7)
    pi_ext = _extend(missing.T, width)
    keep = 1 - pi_ext
    c = _publication_share(publication_lag, cutoff, width)
    hub = _hub_by_week(hub_delay, span)

    def by_onset_week(x: Array) -> Array:
        # sum_j lambda(7w + j) x(7w + j + d), shape (P, n_w, D + 1).
        win = _week_windows(x, n_w, span)
        return reduce(
            jnp.add,
            [lam[:, :, j : j + 1] * win[:, :, j : j + n_d] for j in range(7)],
        )

    def imputed_by_week(x: Array) -> Array:
        return jnp.sum(_week_windows(x, n_w, span) * hub, axis=-1).T

    by_report = reduce(
        jnp.add,
        [
            jnp.pad(
                lam[:, :, j : j + 1] * delay,
                ((0, 0), (0, 0), (j, span - n_d - j)),
            )
            for j in range(7)
        ],
    )
    miss = pi_ext * _week_overlap_add(by_report, width)
    triangle = delay * by_onset_week(c * keep)
    known = triangle.sum(-1).T
    imputed = imputed_by_week(c * miss)
    if final:
        known_final = (delay * by_onset_week(keep)).sum(-1).T
        imputed_final = imputed_by_week(miss)
    else:
        known_final, imputed_final = known, imputed
    return {
        "known": known,
        "imputed": imputed,
        "known_final": known_final,
        "imputed_final": imputed_final,
        "triangle": jnp.moveaxis(triangle, 0, 1),
    }


def _weekly_shared(
    onsets: Array,
    delay: Array,
    missing: Array,
    hub_delay: Array,
    cutoff: int,
    final: bool,
    publication_lag: ArrayLike,
) -> dict[str, Array]:
    """`weekly_onset_expectations` for provinces with one delay.

    `delay` is `(n_w, D + 1)` and `missing` `(n,)`, the same for every
    province, so each term is one product over the provinces.
    """
    n, n_p = onsets.shape
    n_w, n_d = delay.shape
    width = n + n_d - 1
    span = 7 * -(-(n_d + 6) // 7)
    lam = onsets.T.reshape(n_p, n_w, 7)
    pi_ext = _extend(missing, width)
    keep = 1 - pi_ext
    c = _publication_share(publication_lag, cutoff, width)
    hub = _hub_by_week(hub_delay, span)

    def hankel(x: Array) -> Array:
        # x(7w + j + d), shape (n_w, 7, D + 1).
        win = _week_windows(x[None], n_w, span)[0]
        return jnp.stack([win[:, j : j + n_d] for j in range(7)], axis=1)

    def imputed_by_week(x: Array) -> Array:
        return jnp.sum(_week_windows(x, n_w, span) * hub, axis=-1).T

    # f_w(m - j), shape (n_w, 7, span).
    shifted = jnp.stack(
        [jnp.pad(delay, ((0, 0), (j, span - n_d - j))) for j in range(7)],
        axis=1,
    )
    by_report = jnp.einsum("pwj,wjm->pwm", lam, shifted)
    miss = pi_ext * _week_overlap_add(by_report, width)
    triangle = delay * jnp.einsum("pwj,wjd->pwd", lam, hankel(c * keep))
    known = triangle.sum(-1).T
    imputed = imputed_by_week(c * miss)
    if final:
        per_day = jnp.sum(delay[:, None, :] * hankel(keep), axis=-1)
        known_final = jnp.einsum("pwj,wj->wp", lam, per_day)
        imputed_final = imputed_by_week(miss)
    else:
        known_final, imputed_final = known, imputed
    return {
        "known": known,
        "imputed": imputed,
        "known_final": known_final,
        "imputed_final": imputed_final,
        "triangle": jnp.moveaxis(triangle, 0, 1),
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
    publication_lag: ArrayLike,
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
        [
            compute_prop_already_reported(jnp.asarray(publication_lag), t1, 0),
            jnp.zeros(width - t1),
        ]
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

    Expected confirmed onsets are
    $\lambda_p(s) = \alpha(s) \sum_\tau h(\tau) I_p(s - \tau)$,
    with $I_p$ the latent infections, $h$ the infection-to-onset pmf
    (`temporal_pmf_rv`, the incubation period) and $\alpha$ the
    time-varying ascertainment.
    `onset_expectations_by_place` gives the two streams, with each
    province's own delay $f_{p,s}$.
    Each is summed to weeks and scored with its own negative binomial.
    The delays of known-onset cases, tabulated by province, onset week
    and delay, are scored given each province's weekly totals
    (`delay_noise`) with probabilities

    $$P(d \mid p, w) = \frac{\sum_{s \in w} \kappa_p(s, d)
    c(s + d)}{\sum_{s \in w} \sum_{d'} \kappa_p(s, d') c(s + d')},$$

    with the same publication weights $c$ as the known stream, so the
    two truncate alike. This is the conditional part of the Poisson
    reporting triangle, so it does not double count the known stream.
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
    temporal_pmf_rv : RandomVariable
        Infection-to-onset pmf $h$ on lags $0..H$.
    ascertainment_rate_rv : RandomVariable
        $\alpha$, scalar or one value per day of the model axis.
    missing_fraction_rv : RandomVariable or None
        Returns $\pi_p(t)$, shape `(n, P)`. `None` fits one stream,
        `total`, of known plus imputed onsets truncated by `delay_rv`
        (the simpler steps of the model ladder).
    delay_rv : RandomVariable
        Returns $f_{p,s}(d)$, shape `(P, n, D + 1)`, given `n_timepoints`
        and `n_groups`, or one delay shared by every province,
        `(n, D + 1)`.
    known_noise, imputed_noise : CountNoise
        PyRenew noise models for the two streams (`known_noise` for the
        single stream).
    start_dow : int, optional
        First day of the reporting week (6 is Sunday).
    publication_lag : RandomVariable or array_like, optional
        Pmf of the lag from report to publication in a vintage, on
        $0..L$: a random variable to estimate it, or a fixed pmf.
        The default, all mass at lag 0, is no lag.
    delay_noise : CompositionNoise, optional
        Noise model of the delay counts given the weekly totals.
        Defaults to `MultinomialNoise`.
    weekly_axis : bool, optional
        The model axis starts on `start_dow` with whole weeks, so the
        expectations can use per-week blocks (`onset_expectations` with
        `weekly`).
    weekly_delay : bool, optional
        On a weekly axis, ask `delay_rv` for one row per week
        (`by_week`). When it gives one, as `HazardDelay` does for a delay
        constant within weeks, the weekly expectations come straight
        from `weekly_onset_expectations`.
    pair_likelihood : ReleasePairLikelihood, optional
        Fit `publication_lag` to which rows each delay file kept of the
        last (`release_pairs`).
    """

    def __init__(
        self,
        name: str,
        temporal_pmf_rv: RandomVariable,
        ascertainment_rate_rv: RandomVariable,
        missing_fraction_rv: RandomVariable | None,
        delay_rv: RandomVariable,
        known_noise: CountNoise,
        imputed_noise: CountNoise | None = None,
        start_dow: int = 6,
        publication_lag: RandomVariable | ArrayLike = NO_LAG,
        delay_noise: CompositionNoise | None = None,
        weekly_axis: bool = False,
        pair_likelihood: ReleasePairLikelihood | None = None,
        weekly_delay: bool = True,
    ) -> None:
        super().__init__(name, temporal_pmf_rv)
        self.ascertainment_rate_rv = ascertainment_rate_rv
        self.missing_fraction_rv = missing_fraction_rv
        self.delay_rv = delay_rv
        self.known_noise = known_noise
        self.imputed_noise = imputed_noise
        self.start_dow = start_dow
        self.publication_lag = publication_lag
        self.delay_noise = (
            MultinomialNoise() if delay_noise is None else delay_noise
        )
        self.weekly_axis = weekly_axis
        self.pair_likelihood = pair_likelihood
        self.weekly_delay = weekly_delay

    def lookback_days(self) -> int:
        """Return the days of infections before an onset that it needs.

        Returns
        -------
        int
            The length of the infection-to-onset pmf less one.
        """
        # Only the shape is needed, so a random pmf is traced, not sampled.
        pmf = jax.eval_shape(numpyro.handlers.seed(self.temporal_pmf_rv, 0))
        return int(pmf.shape[0]) - 1

    def infection_resolution(self) -> str:
        """Return the resolution of the infections the process reads.

        Returns
        -------
        str
            Always `"subpop"`, one series per province.
        """
        return "subpop"

    def _predicted_obs(self, infections: Array) -> Array:
        """Convolve infections with the infection-to-onset pmf.

        Column by column, aligned so day $t$ of the output is day $t$ of
        the input, with no infections before the axis. The seeding days
        feed the weekly sums, so they are partial sums rather than NaN.

        Parameters
        ----------
        infections : Array
            Infections by province, `(n_total, P)`.

        Returns
        -------
        Array
            Expected infections by onset day, the shape of `infections`.
        """
        pmf = jnp.asarray(self.temporal_pmf_rv())
        n = infections.shape[0]
        return jax.vmap(
            lambda col: jnp.convolve(col, pmf, mode="full")[:n],
            in_axes=1,
            out_axes=1,
        )(infections)

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

        Parameters
        ----------
        n_total : int
            Days on the model axis.
        n_subpops : int
            Number of provinces.
        cutoff : int, optional
            Day index of the data cut-off.
        known, imputed : jax.Array, optional
            Weekly known-onset and imputed-onset counts.
        **kwargs
            Ignored, for the PyRenew interface.

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

    def _shared_groups(self, n_prov: int) -> np.ndarray | None:
        """Provinces with the same delay and missing fraction, if known."""
        sources = [self.delay_rv]
        if self.missing_fraction_rv is not None:
            sources.append(self.missing_fraction_rv)
        if not all(hasattr(rv, "shared_groups") for rv in sources):
            return None
        shared = np.ones(n_prov, dtype=bool)
        for rv in sources:
            shared &= np.asarray(rv.shared_groups(n_prov), dtype=bool)
        return shared

    def _weekly(self, x: Array, first_day_dow: int) -> Array:
        flat = x.reshape(x.shape[0], -1)
        weekly = jnp.asarray(
            daily_to_weekly(flat, first_day_dow, self.start_dow)
        )
        return weekly.reshape(weekly.shape[0], *x.shape[1:])

    def sample(
        self,
        infections: Array,
        first_day_dow: int,
        cutoff: int,
        hub_delay_pmf: Array,
        known: Array | None = None,
        imputed: Array | None = None,
        delay_counts: Array | None = None,
        forecast: bool = False,
        vintage: Array | None = None,
        as_of: int | None = None,
        release_pairs: dict[str, Array] | None = None,
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
            Known-onset cases by onset week, province and delay,
            `(n_weeks, P, D + 1)`.
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
        release_pairs : dict, optional
            Rows of each delay file and how many the previous file had
            (`data.releases.release_pairs`), scored by `pair_likelihood`.

        Returns
        -------
        ObservationSample
            Sampled known counts and the weekly expected known counts.
        """
        n, n_prov = infections.shape
        alpha = jnp.asarray(self.ascertainment_rate_rv())
        alpha = alpha[:, None] if alpha.ndim == 1 else alpha
        onsets = alpha * self._predicted_obs(infections)
        by_week = self.weekly_axis and self.weekly_delay
        delay = self.delay_rv(
            n_timepoints=n,
            n_groups=n_prov,
            first_day_dow=first_day_dow,
            by_week=by_week,
        )
        # One row per week only when the delay rv gave one.
        by_week = by_week and delay.shape[-2] == n // 7 != n
        # A delay shared by every province (the ladder's fixed delay).
        delay = jnp.broadcast_to(delay, (n_prov, *delay.shape[-2:]))
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
        hub = jnp.asarray(hub_delay_pmf)

        shared = self._shared_groups(n_prov) if by_week else None

        def expect(cut: int, final: bool) -> dict[str, Array]:
            # Expectations summed to the weeks of the model axis.
            if by_week:
                return weekly_onset_expectations(
                    onsets, delay, pi, hub, cut, final, lag, shared
                )
            exp = onset_expectations_by_place(
                onsets,
                delay,
                pi,
                hub,
                cut,
                final,
                publication_lag=lag,
                weekly=self.weekly_axis,
            )
            return {k: self._weekly(v, first_day_dow) for k, v in exp.items()}

        weekly = expect(cutoff, forecast)
        if as_of is not None:
            later = expect(as_of, False)
            weekly["known_final"] = later["known"]
            weekly["imputed_final"] = later["imputed"]
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
        if self.pair_likelihood is not None and release_pairs is not None:
            self.pair_likelihood.score(lag, release_pairs)
        if forecast:
            self._deterministic("final_total", final)
        return ObservationSample(observed=draws[0], predicted=weekly["known"])

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
        When fitting, row 1 is unobserved and adds nothing, so only row 0
        is scored, and row 1 of the returned draws is zero.
        """
        site = self._sample_site_name(f"{stream}_obs")
        if obs is None:
            rest = final - by_cutoff
            if centre is not None:
                rest = jnp.where(jnp.isnan(centre), rest, centre + rest)
            predicted = jnp.maximum(jnp.stack([by_cutoff, rest]), _FLOOR)
            return cast(Array, noise.sample(site, predicted))
        mask = ~jnp.isnan(obs)
        seen = noise.sample(
            site,
            jnp.maximum(by_cutoff, _FLOOR),
            obs=jnp.where(mask, obs, 0.0),
            mask=mask,
        )
        return jnp.stack([seen, jnp.zeros_like(seen)])

    def _score_delays(self, triangle: Array, counts: Array) -> None:
        totals = counts.sum(axis=-1)
        probs = triangle / jnp.maximum(
            triangle.sum(axis=-1, keepdims=True), 1e-12
        )
        probs = jnp.maximum(probs, 1e-12)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        # A week with no counts adds exactly 0, so no mask is needed.
        self.delay_noise.sample(
            self._sample_site_name("delays"), probs, totals, obs=counts
        )


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
    active : numpy.ndarray, optional
        Boolean, one per province: provinces with a deviation, the rest
        following the pooled trend (`pooled_walk`). `None` is all.
    """

    def __init__(
        self,
        name: str,
        initial_rv: RandomVariable,
        baseline_process: TemporalProcess,
        deviation_process: TemporalProcess,
        active: np.ndarray | None = None,
    ) -> None:
        super().__init__(name=name)
        self.initial_rv = initial_rv
        self.baseline_process = baseline_process
        self.deviation_process = deviation_process
        self.active = active

    def shared_groups(self, n_groups: int) -> np.ndarray:
        """Return the provinces whose missing fraction is the pooled one.

        Parameters
        ----------
        n_groups : int
            Number of provinces.

        Returns
        -------
        numpy.ndarray
            Boolean `(n_groups,)`: the provinces outside `active`
            (`pooled_groups`).
        """
        return pooled_groups(self.active, n_groups)

    def sample(
        self,
        n_timepoints: int,
        n_groups: int,
        first_day_dow: int | None = None,
        **kwargs: object,
    ) -> Array:
        r"""Sample $\pi_p(t)$.

        Parameters
        ----------
        n_timepoints : int
            Days to sample.
        n_groups : int
            Number of provinces.
        first_day_dow : int, optional
            Weekday of the first day, passed by PyRenew.
        **kwargs
            Ignored, for the PyRenew interface.

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
            active=self.active,
        )
        pi = jax.nn.sigmoid(logit_pi)
        numpyro.deterministic(self.name, pi)
        return pi
