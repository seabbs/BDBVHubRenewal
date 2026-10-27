# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Expected values of the two observed streams, including the hub's imputation.

A NumPy reference for the model's observation layer, for one province.
The model phase writes the same maths as PyRenew observation processes
in JAX and checks them against these functions.

Notation, all by day:

- $C(s)$: confirmed cases with onset on day $s$ (onsets times
  ascertainment $\alpha(s)$).
- $f_s(d)$: our delay pmf for onset day $s$ (time varying).
- $\pi(t)$: the share of cases reported on day $t$ without an onset.
- $T$: the vintage cut-off.

Reports by report day are $\rho(t) = \sum_s C(s) f_s(t - s)$.

**Known onsets.**
Cases with onset $s$, reported by $T$ with an onset recorded:
$$K_T(s) = \sum_{t \le T} C(s)\, f_s(t - s)\,(1 - \pi(t)).$$

**Missing onsets by report day** (latent, not published):
$$M(t) = \pi(t)\,\rho(t).$$

**The hub's imputation.**
The hub gives each missing-onset case an onset by subtracting a delay
drawn from $\hat g_T$, the empirical pmf of all published delays in the
vintage, pooled over provinces and onset weeks.
The expected imputed onsets on day $s$ are
$$I_T(s) = \sum_{t \le T} M(t)\, \hat g_T(t - s).$$
This is the expected back-sampling allocation (as in, for example,
`backsampling_allocate` from EpidemicDates).
Summing $K_T$ and $I_T$ over an onset week gives the expected hub counts,
so the forecast of the target carries the imputation's own bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdbv_hub_renewal.delays import DEFAULT_MAX_DELAY, empirical_pmf

__all__ = [
    "delay_matrix",
    "expected_imputed_onsets",
    "expected_known_onsets",
    "expected_missing_by_report",
    "hub_delay_pmf",
    "to_onset_weeks",
]


def delay_matrix(pmfs: NDArray[np.float64], n_days: int) -> NDArray[np.float64]:
    """Build the onset-day by report-day matrix of delay probabilities.

    Parameters
    ----------
    pmfs : numpy.ndarray
        One delay pmf per onset day, shape (n_days, max_delay + 1), or a
        single pmf used for every day.
    n_days : int
        Number of days.

    Returns
    -------
    numpy.ndarray
        `P[s, t] = f_s(t - s)`, shape (n_days, n_days). Mass past the last
        day is dropped.
    """
    p = np.atleast_2d(np.asarray(pmfs, dtype=np.float64))
    if p.shape[0] == 1:
        p = np.repeat(p, n_days, axis=0)
    out = np.zeros((n_days, n_days))
    width = p.shape[1]
    for s in range(n_days):
        end = min(n_days, s + width)
        out[s, s:end] = p[s, : end - s]
    return out


def expected_known_onsets(
    confirmed: NDArray[np.float64],
    delays: NDArray[np.float64],
    missing: NDArray[np.float64],
    cutoff: int,
) -> NDArray[np.float64]:
    r"""Return expected known-onset cases by onset day at a cut-off.

    Parameters
    ----------
    confirmed : numpy.ndarray
        Confirmed cases by onset day, $C(s)$.
    delays : numpy.ndarray
        Delay pmfs, as accepted by `delay_matrix`.
    missing : numpy.ndarray
        $\pi(t)$ by report day.
    cutoff : int
        Last report day included, $T$.

    Returns
    -------
    numpy.ndarray
        $K_T(s)$ by onset day.
    """
    n = confirmed.size
    p = delay_matrix(delays, n)[:, : cutoff + 1]
    keep = 1 - np.asarray(missing, dtype=np.float64)[: cutoff + 1]
    return np.asarray(confirmed * (p @ keep), dtype=np.float64)


def expected_missing_by_report(
    confirmed: NDArray[np.float64],
    delays: NDArray[np.float64],
    missing: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return expected missing-onset cases by report day, $M(t)$.

    Parameters
    ----------
    confirmed : numpy.ndarray
        Confirmed cases by onset day.
    delays : numpy.ndarray
        Delay pmfs, as accepted by `delay_matrix`.
    missing : numpy.ndarray
        $\pi(t)$ by report day.

    Returns
    -------
    numpy.ndarray
        $M(t)$ by report day.
    """
    p = delay_matrix(delays, confirmed.size)
    reports = confirmed @ p
    return np.asarray(missing * reports, dtype=np.float64)


def hub_delay_pmf(
    delays: pd.DataFrame,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> NDArray[np.float64]:
    r"""Return the hub's back-sampling delay pmf for one vintage.

    The hub draws from "the full delay distribution" of the published
    delay data of a defined vintage (hub issue 61), so this pools over
    provinces and onset weeks.
    How the hub treats negative delays, very long delays and which
    vintage it draws from are not published; here negative delays and
    delays above `max_delay` are dropped.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one vintage.
    max_delay : int
        Longest delay kept.

    Returns
    -------
    numpy.ndarray
        $\hat g_T$ on days 0 to `max_delay`.
    """
    return empirical_pmf(delays["delay"], max_delay)


def expected_imputed_onsets(
    missing_by_report: NDArray[np.float64],
    hub_pmf: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return the hub's expected imputed onsets by onset day, $I_T(s)$.

    Parameters
    ----------
    missing_by_report : numpy.ndarray
        $M(t)$ up to the cut-off.
    hub_pmf : numpy.ndarray
        $\hat g_T$ from `hub_delay_pmf`.

    Returns
    -------
    numpy.ndarray
        Expected imputed onsets by onset day. Mass before day 0 is lost.
    """
    m = np.asarray(missing_by_report, dtype=np.float64)
    g = np.asarray(hub_pmf, dtype=np.float64)
    out = np.zeros(m.size)
    for d in range(min(g.size, m.size)):
        out[: m.size - d] += m[d:] * g[d]
    return np.asarray(out, dtype=np.float64)


def to_onset_weeks(
    daily: NDArray[np.float64], start: str | pd.Timestamp
) -> pd.Series:
    """Sum daily values into epi weeks ending on Saturdays.

    Parameters
    ----------
    daily : numpy.ndarray
        Values by day from `start`.
    start : str or pandas.Timestamp
        Date of day 0.

    Returns
    -------
    pandas.Series
        Weekly sums indexed by the ISO date of each week's Saturday.
    """
    dates = pd.date_range(pd.Timestamp(start), periods=len(daily), freq="D")
    week = dates + pd.to_timedelta((5 - dates.dayofweek) % 7, unit="D")
    return (
        pd.Series(daily, index=week.strftime("%Y-%m-%d")).groupby(level=0).sum()
    )
