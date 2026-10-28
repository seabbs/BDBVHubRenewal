# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""How much of a week's final count is reported by a later date.

A case with onset on day $W - u$ of the week ending $W$ ($u = 0..6$,
equally likely), onset-to-report delay $d$ and publication lag $\ell$
is in a vintage dated $W + t$ if $d + \ell - u \le t$. So the share of
the week's final count in that vintage is

$$
s(t) = \frac{1}{7}\sum_{u=0}^{6} P(d + \ell - u \le t),
$$

with $P$ the empirical distribution of the delays.
Only onset weeks that ended at least `DEFAULT_MAX_DELAY` plus the lag
before the delay vintage are used, as their delays are fully observed.
$\ell$ is the longest publication lag, so $s$ errs low.

`complete_weeks` gives the fewest whole weeks $K$ with
$s(7K) \ge$ `threshold`. Scoring treats a truth week as complete if it
ended at least $K$ weeks before the truth vintage.

`truncated_delay_cdf` instead uses the recent onset weeks, whose delays
are right truncated, and corrects for it. A case with onset week $W$ is
in the vintage dated $T$ only if its delay $d \le \tau = T - (W - 3) -
\ell$, taking the onset mid-week. The maximum likelihood estimate of the
delay cdf under right truncation (Lynden-Bell, 1971) is

$$
F(x) = \prod_{y > x} \Big\{1 - \frac{n(y)}{R(y)}\Big\},
\qquad
R(y) = \#\{i : d_i \le y \le \tau_i\},
$$

with $n(y)$ the cases with delay $y$. It is the delay step of the EM
nowcast in EpidemicDates with no missing onsets. `share_from_cdf` turns
it into $s(t)$ as above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY
from bdbvhubrenewal.data.reconstruct import PUBLICATION_LAG_DAYS

__all__ = [
    "complete_weeks",
    "reported_share",
    "share_from_cdf",
    "truncated_delay_cdf",
]


def reported_share(
    delays: pd.DataFrame,
    vintage: str,
    days: ArrayLike,
    lag: int = PUBLICATION_LAG_DAYS,
) -> NDArray[np.float64]:
    """Return the share of a week's final count reported by each day.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data with `vintage`, `onset_week` and `delay` columns.
        Rows of `vintage` are used.
    vintage : str
        The delay vintage.
    days : array-like of int
        Days after the end of the onset week.
    lag : int
        Days from report to publication.

    Returns
    -------
    numpy.ndarray
        $s(t)$ for each day $t$, or ones if no onset week has settled.
    """
    rows = delays.loc[delays["vintage"] == vintage]
    cut = pd.Timestamp(as_date(vintage)) - pd.Timedelta(
        days=DEFAULT_MAX_DELAY + lag
    )
    settled = rows.loc[pd.to_datetime(rows["onset_week"]) <= cut, "delay"]
    t = np.asarray(days, dtype=np.int64)
    if settled.empty:
        return np.ones(t.shape)
    d = settled.to_numpy(dtype=np.int64).clip(min=0) + lag
    # Days from the week's end to publication, over the 7 onset days.
    after = (d[:, None] - np.arange(7)[None, :]).ravel()
    return np.asarray(
        (after[None, :] <= t[:, None]).mean(axis=1), dtype=np.float64
    )


def complete_weeks(
    delays: pd.DataFrame,
    vintage: str,
    threshold: float = 0.95,
    lag: int = PUBLICATION_LAG_DAYS,
    max_weeks: int = 12,
) -> int:
    r"""Return the fewest whole weeks after which a week counts as complete.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data, as for `reported_share`.
    vintage : str
        The delay vintage.
    threshold : float
        Share of the final count that must be reported.
    lag : int
        Days from report to publication.
    max_weeks : int
        Longest lag returned.

    Returns
    -------
    int
        $K$, the smallest number of weeks with $s(7K) \ge$ `threshold`,
        or `max_weeks` if none is.
    """
    weeks = np.arange(max_weeks + 1)
    share = reported_share(delays, vintage, 7 * weeks, lag)
    ok = np.nonzero(share >= threshold)[0]
    return int(weeks[ok[0]]) if ok.size else max_weeks


def truncated_delay_cdf(
    delays: pd.DataFrame,
    vintage: str,
    window: int = 84,
    lag: int = PUBLICATION_LAG_DAYS,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> NDArray[np.float64] | None:
    r"""Return the delay cdf of recent onset weeks, corrected for truncation.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data with `vintage`, `onset_week` and `delay` columns.
        Rows of `vintage` are used.
    vintage : str
        The delay vintage.
    window : int
        Use onset weeks ending fewer than this many days before
        `vintage`.
    lag : int
        Days from report to publication.
    max_delay : int
        Longest delay. Longer delays are counted at `max_delay`.

    Returns
    -------
    numpy.ndarray or None
        $F(0), \dots, F(D)$ with $D$ = `max_delay`, or None if no rows
        fall in the window.
    """
    rows = delays.loc[delays["vintage"] == vintage]
    end = pd.Timestamp(as_date(vintage))
    age = (end - pd.to_datetime(rows["onset_week"])).dt.days.to_numpy()
    keep = age < window
    if not keep.any():
        return None
    x = rows["delay"].to_numpy(dtype=np.int64)[keep].clip(0, max_delay)
    # A case in the vintage has a delay of at most tau.
    tau = np.clip(age[keep] + 3 - lag, x, max_delay)
    n = np.bincount(x, minlength=max_delay + 1)
    entered = n.cumsum()
    left = np.bincount(tau + 1, minlength=max_delay + 2)[: max_delay + 1]
    at_risk = entered - left.cumsum()
    hazard = n / np.maximum(at_risk, 1)
    # F(x) is the product of 1 - hazard over y > x.
    tail = np.cumprod((1 - hazard)[::-1])[::-1]
    return np.asarray(np.append(tail[1:], 1.0), dtype=np.float64)


def share_from_cdf(
    cdf: NDArray[np.float64],
    days: ArrayLike,
    lag: int = PUBLICATION_LAG_DAYS,
) -> NDArray[np.float64]:
    """Return the share of a week's final count reported by each day.

    Parameters
    ----------
    cdf : numpy.ndarray
        Delay cdf on `0..D`, as from `truncated_delay_cdf`.
    days : array-like of int
        Days after the end of the onset week.
    lag : int
        Days from report to publication.

    Returns
    -------
    numpy.ndarray
        $s(t)$ for each day $t$, averaged over the 7 onset days.
    """
    t = np.asarray(days, dtype=np.int64)
    out = np.zeros(t.shape)
    for u in range(7):
        k = t + u - lag
        out += np.where(k < 0, 0.0, cdf[np.clip(k, 0, cdf.size - 1)])
    return out / 7
