# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Expected onsets reported by any cut-off, in NumPy, from one draw.

The observation process (`observation.onset_expectations`) gives the
expected known and imputed onsets reported by one cut-off, fixed when
the model is traced. Scoring against the releases after a fit needs
them at cut-offs not known when it was fitted, so this module computes
them for a whole grid of cut-offs at once, from the forecast pass's
deterministic sites, without the model.

With $f_s(d)$ the delay of onset day $s$, $\pi(t)$ the missing-onset
fraction of report day $t$ (its last value past the axis), $\hat f$ the
hub's back-sampling pmf and $G(x)$ the share of a day's reports published
$x$ days later (the publication lag cdf, 0 for $x < 0$):

- known by $c$: $\lambda(s) \sum_d f_s(d) (1 - \pi(s + d)) G(c - s - d)$;
- missing by report day: $M(t) = \pi(t) \sum_d \lambda(t - d) f_{t-d}(d)$;
- imputed by $c$: $\sum_d \hat f(d) M(s + d) G(c - s - d)$.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["expected_reported_by"]


def _lag_cdf(
    lag: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the publication lag cdf at `x` days, 0 before 0 and 1 past it."""
    cdf = np.cumsum(lag)
    idx = np.clip(x, 0, len(cdf) - 1).astype(int)
    out = np.where(x < 0, 0.0, cdf[idx])
    return np.where(x >= len(cdf), 1.0, out)


def expected_reported_by(
    onsets: ArrayLike,
    delay: ArrayLike,
    missing: ArrayLike,
    lag: ArrayLike,
    hub: ArrayLike,
    cutoffs: ArrayLike,
    days: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    r"""Return the expected known and imputed onsets reported by cut-offs.

    Equal to `expected_known` and `expected_imputed` (by the vintage's
    cut-off), and `expected_*_final` (by `as_of`, or once all reports are
    in), of the model's forecast pass at the same draw.

    Parameters
    ----------
    onsets : array_like
        $\lambda_p(s)$, `(n, P)` (`onsets_expected_daily`).
    delay : array_like
        $f_{p,s}(d)$, `(P, n, D + 1)`, or by week `(P, n // 7, D + 1)`
        (`onsets_delay_pmf`).
    missing : array_like
        $\pi_p(t)$, `(n, P)` (`missing_fraction`).
    lag : array_like
        The publication lag pmf on $0..L$ (`onsets_publication_lag`).
    hub : array_like
        The hub's back-sampling pmf on $0..D$.
    cutoffs : array_like
        Cut-off days on the model axis. `numpy.inf` gives the final
        counts.
    days : array_like
        Onset days on the model axis to return.

    Returns
    -------
    dict
        `known` and `imputed`, each `(len(cutoffs), len(days), P)`.
    """
    lam = np.asarray(onsets, dtype=float)
    f = np.asarray(delay, dtype=float)
    if f.shape[1] != lam.shape[0]:
        # A weekly delay: every day of week w takes f_{p,w}.
        f = np.repeat(f, 7, axis=1)[:, : lam.shape[0]]
    f = np.moveaxis(f, 0, -1)  # (n, D + 1, P)
    pi = np.asarray(missing, dtype=float)
    hub_pmf = np.asarray(hub, dtype=float)
    n, n_d, n_p = f.shape
    width = n + n_d - 1
    pi_ext = pi[np.minimum(np.arange(width), n - 1)]
    s = np.asarray(days, dtype=int)
    d = np.arange(n_d)
    report = s[:, None] + d[None, :]  # (S, D + 1)
    # Expected reports by (onset day, delay), before publication.
    known = lam[s][:, None, :] * f[s] * (1 - pi_ext[report])
    sent = np.zeros((width + n_d, n_p))
    for j in range(n_d):
        sent[j : j + n] += lam * f[:, j, :]
    m = pi_ext * sent[:width]
    m = np.vstack([m, np.zeros((n_d, n_p))])
    imputed = hub_pmf[None, :, None] * m[report]
    cuts = np.asarray(cutoffs, dtype=float)
    share = _lag_cdf(
        np.asarray(lag, dtype=float),
        cuts[:, None, None] - report[None, :, :],
    )  # (C, S, D + 1)
    return {
        "known": np.einsum("csd,sdp->csp", share, known),
        "imputed": np.einsum("csd,sdp->csp", share, imputed),
    }
