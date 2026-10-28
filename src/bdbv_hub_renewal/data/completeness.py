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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from bdbv_hub_renewal.data.dates import as_date
from bdbv_hub_renewal.data.delays import DEFAULT_MAX_DELAY
from bdbv_hub_renewal.data.reconstruct import PUBLICATION_LAG_DAYS

__all__ = ["complete_weeks", "reported_share"]


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
