# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Onset-to-report delay distributions.

The interface is small.
A `DelayModel` is fitted to the delay line data of one vintage and
returns a `DelayFit`.
A `DelayFit` gives a daily pmf for any province and onset week, so a
model can use delays that vary by place and time.

`EmpiricalDelayModel` is implemented now.
It takes the empirical pmf over a rolling window of onset weeks and
shrinks each province towards the pooled pmf, so small provinces and
recent weeks stay stable.
It ignores right truncation and the weekly censoring of onset, so recent
delays are biased short.

The model phase adds a parametric estimator that handles both.
Each delay is in whole days from the onset day to the report day, so it
is double interval censored at daily resolution, with pmf $p_\theta(d)$
from `bdbvhubrenewal.model.censored.censored_pmf`.
Only the onset week $W$ (days $W-6$ to $W$) is published, so right
truncation at vintage $T$ is averaged over the onset day:

$$
P_\theta(\text{reported by } T \mid W) = \frac{1}{7}\sum_{u=0}^{6}
\sum_{d=0}^{T - W + u} p_\theta(d).
$$

Each case contributes $p_\theta(d) / P_\theta(\text{reported by } T
\mid W)$ to the likelihood.
$\theta$ varies smoothly over onset weeks and is partially pooled over
provinces (see the Methods page).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import integrate, special

from bdbvhubrenewal.data.dates import as_date

__all__ = [
    "DEFAULT_MAX_DELAY",
    "HUB_DOUBLE_CENSORED_FROM",
    "HUB_GENGAMMA",
    "DelayFit",
    "DelayModel",
    "EmpiricalDelayFit",
    "EmpiricalDelayModel",
    "delay_summary",
    "empirical_pmf",
    "hub_delay_pmf",
]

DEFAULT_MAX_DELAY: Final = 60
"""Longest delay kept, in days. Longer delays are dropped (about 0.3%)."""

HUB_DOUBLE_CENSORED_FROM: Final = dt.date(2026, 11, 1)
"""First release whose imputation delay fit is double interval censored.

The 2026-11-01 release was published after hub PR 88 and the earlier
releases in that commit are unchanged, so they keep the exact fit.
"""

HUB_GENGAMMA: Final = {
    "exact": (5.12406648030701, 0.97184390713028, 1.07646963898928),
    "double_censored": (4.99016267662049, 0.936895070027868, 0.715805359543864),
}
"""The hub's generalised gamma delay fits in hours, as `(mu, sigma, Q)`.

flexsurv's parameterisation, refitted to the hub's 2026-10-17 delays by
`scripts/hub_imputation_fit.R`.
"""


def empirical_pmf(
    delays: Sequence[int] | NDArray[np.int64] | pd.Series,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> NDArray[np.float64]:
    """Return the empirical daily pmf of a set of delays.

    Negative delays and delays above `max_delay` are dropped.

    Parameters
    ----------
    delays : array-like of int
        Delays in days.
    max_delay : int
        Longest delay kept.

    Returns
    -------
    numpy.ndarray
        Probabilities for delays 0 to `max_delay`. All zero if no delays
        are left.
    """
    d = np.asarray(delays, dtype=np.int64)
    d = d[(d >= 0) & (d <= max_delay)]
    counts = np.bincount(d, minlength=max_delay + 1).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def delay_summary(
    delays: pd.DataFrame, by: Sequence[str] = ("vintage", "location")
) -> pd.DataFrame:
    """Summarise delays by group: count, mean, median and 90th centile.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data (see `bdbvhubrenewal.data.tidy_delays`).
    by : sequence of str
        Columns to group by.

    Returns
    -------
    pandas.DataFrame
        The group columns plus `n`, `mean`, `median` and `q90`.
    """
    grp = delays.groupby(list(by))["delay"]
    out = grp.agg(
        n="size",
        mean="mean",
        median="median",
        q90=lambda s: float(np.quantile(s, 0.9)),
    )
    return out.reset_index()


class DelayFit(Protocol):
    """A fitted delay distribution that may vary by province and week."""

    def pmf(self, location: str, onset_week: str) -> NDArray[np.float64]:
        """Return the daily delay pmf for a province and onset week.

        Parameters
        ----------
        location : str
            Province code.
        onset_week : str
            The Saturday ending the onset week, ISO format.

        Returns
        -------
        numpy.ndarray
            Probabilities for delays 0 to the fit's maximum delay.
        """
        ...


class DelayModel(Protocol):
    """Something that fits delays to the line data of one vintage."""

    def fit(self, delays: pd.DataFrame, vintage: str) -> DelayFit:
        """Fit to the delay line data published at a vintage.

        Parameters
        ----------
        delays : pandas.DataFrame
            Delay line data with columns `location`, `onset_week` and
            `delay`, for one vintage.
        vintage : str
            The vintage date, used for right truncation.

        Returns
        -------
        DelayFit
            The fitted delays.
        """
        ...


@dataclass(frozen=True)
class EmpiricalDelayFit:
    """Empirical delay pmfs by province and onset week.

    Attributes
    ----------
    pooled : numpy.ndarray
        The pmf over all provinces and weeks.
    by_cell : dict
        `(location, onset_week)` to pmf.
    max_delay : int
        Longest delay.
    """

    pooled: NDArray[np.float64]
    by_cell: dict[tuple[str, str], NDArray[np.float64]] = field(
        default_factory=dict
    )
    max_delay: int = DEFAULT_MAX_DELAY

    def pmf(self, location: str, onset_week: str) -> NDArray[np.float64]:
        """Return the pmf for a cell, or the pooled pmf if it is not known.

        Parameters
        ----------
        location : str
            Province code.
        onset_week : str
            The Saturday ending the onset week.

        Returns
        -------
        numpy.ndarray
            Probabilities for delays 0 to `max_delay`.
        """
        return self.by_cell.get((location, onset_week), self.pooled)

    def cdf(self, location: str, onset_week: str) -> NDArray[np.float64]:
        """Return the cumulative probabilities for a cell.

        Parameters
        ----------
        location : str
            Province code.
        onset_week : str
            The Saturday ending the onset week.

        Returns
        -------
        numpy.ndarray
            `P(delay <= d)` for d from 0 to `max_delay`.
        """
        return np.cumsum(self.pmf(location, onset_week))

    def quantile(self, q: float) -> int:
        """Return a quantile of the pooled delay, in whole days.

        Parameters
        ----------
        q : float
            Probability in (0, 1].

        Returns
        -------
        int
            The smallest delay whose pooled cdf reaches `q`.
        """
        return int(np.searchsorted(np.cumsum(self.pooled), q))


@dataclass(frozen=True)
class EmpiricalDelayModel:
    r"""Rolling-window empirical delays with shrinkage to the pooled pmf.

    For province $p$ and onset week $w$, with $n_{pw}$ delays in the
    window and their empirical pmf $\hat f_{pw}$, the estimate is

    $$
    \tilde f_{pw} = \frac{n_{pw} \hat f_{pw} + k \hat f_{w}}{n_{pw} + k},
    $$

    where $\hat f_w$ is the pmf pooled over provinces in the same window
    and $k$ is `prior_weight`.
    This is the posterior mean under a Dirichlet prior centred on the
    pooled pmf.
    With no change over time or place it reduces to the pooled pmf.

    Attributes
    ----------
    window_weeks : int
        Width of the rolling window of onset weeks, centred where
        possible. Use a large number for a static delay.
    prior_weight : float
        Pseudo-count $k$ pulling each province to the pooled pmf.
    max_delay : int
        Longest delay kept.
    """

    window_weeks: int = 5
    prior_weight: float = 50.0
    max_delay: int = DEFAULT_MAX_DELAY

    def fit(self, delays: pd.DataFrame, vintage: str) -> EmpiricalDelayFit:  # noqa: ARG002
        """Fit to the delay line data of one vintage.

        Parameters
        ----------
        delays : pandas.DataFrame
            Delay line data with `location`, `onset_week` and `delay`.
        vintage : str
            Not used: this estimator ignores right truncation.

        Returns
        -------
        EmpiricalDelayFit
            Pooled and per-cell pmfs.
        """
        pooled = empirical_pmf(delays["delay"], self.max_delay)
        weeks = sorted(set(delays["onset_week"].astype(str)))
        half = self.window_weeks // 2
        by_cell: dict[tuple[str, str], NDArray[np.float64]] = {}
        week_idx = {w: i for i, w in enumerate(weeks)}
        idx = delays["onset_week"].astype(str).map(week_idx).to_numpy()
        for i, week in enumerate(weeks):
            in_window = np.abs(idx - i) <= half
            window = delays.loc[in_window]
            pooled_w = empirical_pmf(window["delay"], self.max_delay)
            n_w = int(pooled_w.sum() > 0)
            centre = pooled_w if n_w else pooled
            for loc, rows in window.groupby("location"):
                counts = empirical_pmf(rows["delay"], self.max_delay)
                n = float(
                    (
                        (rows["delay"] >= 0) & (rows["delay"] <= self.max_delay)
                    ).sum()
                )
                by_cell[(str(loc), week)] = (
                    n * counts + self.prior_weight * centre
                ) / (n + self.prior_weight)
        return EmpiricalDelayFit(pooled, by_cell, self.max_delay)


def hub_delay_pmf(
    release: dt.date | str,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> NDArray[np.float64]:
    r"""Return the daily pmf of the hub's imputed onset delay at a release.

    The hub puts the onset of a case without one at report hour
    $U \sim \mathrm{Unif}(0, 24)$ minus a delay $X$ in hours drawn from a
    generalised gamma (`auxiliary-data/imputation-details.md`), so the
    onset is $k = \lceil (X - U) / 24 \rceil$ days before the report.
    With $F$ the cdf of $X / 24$,

    $$\hat f(k) = \int_0^1 \{F(k + u) - F(k - 1 + u)\}\, du,$$

    the double interval censored pmf, here by quadrature.
    The generalised gamma is fitted to the delays published at 2026-10-17
    as exact before `HUB_DOUBLE_CENSORED_FROM` and as double interval
    censored from then on (hub issue 75), with the parameters in
    `HUB_GENGAMMA`. Truncated at `max_delay` and renormalised.

    Parameters
    ----------
    release : datetime.date or str
        The release date.
    max_delay : int
        Longest delay kept.

    Returns
    -------
    numpy.ndarray
        $\hat f$ on days 0 to `max_delay`.
    """
    fit = (
        "double_censored"
        if as_date(release) >= HUB_DOUBLE_CENSORED_FROM
        else "exact"
    )
    mu, sigma, q = HUB_GENGAMMA[fit]

    def cdf(days: float) -> float:
        # flexsurv's pgengamma for Q > 0, which both fits have.
        if days <= 0:
            return 0.0
        w = (np.log(24 * days) - mu) / sigma
        return float(special.gammainc(q**-2, q**-2 * np.exp(q * w)))

    def mass(k: int) -> float:
        return float(
            integrate.quad(lambda u: cdf(k + u) - cdf(k - 1 + u), 0, 1)[0]
        )

    pmf = np.array([mass(k) for k in range(max_delay + 1)], dtype=float)
    return np.asarray(pmf / pmf.sum(), dtype=float)
