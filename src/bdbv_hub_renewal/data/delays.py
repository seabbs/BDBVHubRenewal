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
from `ParametricDelay.pmf`.
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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn

__all__ = [
    "DEFAULT_MAX_DELAY",
    "DelayFit",
    "DelayModel",
    "EmpiricalDelayFit",
    "EmpiricalDelayModel",
    "ParametricDelay",
    "delay_summary",
    "empirical_pmf",
    "hub_delay_pmf",
]

DEFAULT_MAX_DELAY: Final = 60
"""Longest delay kept, in days. Longer delays are dropped (about 0.3%)."""

Family = Literal["gamma", "lognormal", "weibull"]


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
        Delay line data (see `bdbv_hub_renewal.data.tidy_delays`).
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


@dataclass(frozen=True)
class ParametricDelay:
    """A continuous delay discretised to days with primary censoring.

    Attributes
    ----------
    family : {"gamma", "lognormal", "weibull"}
        The distribution family.
    mean : float
        Mean delay in days.
    sd : float
        Standard deviation in days.
    """

    family: Family
    mean: float
    sd: float

    def _dist(self) -> stats.rv_continuous:
        m, s = self.mean, self.sd
        if m <= 0 or s <= 0:
            raise ValueError("Mean and SD must be positive.")
        if self.family == "gamma":
            shape = (m / s) ** 2
            return stats.gamma(a=shape, scale=s**2 / m)
        if self.family == "lognormal":
            sdlog = np.sqrt(np.log1p((s / m) ** 2))
            return stats.lognorm(
                s=sdlog, scale=np.exp(np.log(m) - sdlog**2 / 2)
            )
        if self.family == "weibull":
            cv = s / m

            def cv_gap(k: float) -> float:
                ratio = gamma_fn(1 + 2 / k) / gamma_fn(1 + 1 / k) ** 2
                return float(np.sqrt(ratio - 1) - cv)

            k = float(brentq(cv_gap, 0.05, 50.0))
            return stats.weibull_min(c=k, scale=m / gamma_fn(1 + 1 / k))
        raise ValueError(f"Unknown family {self.family!r}.")

    def pmf(
        self, max_delay: int = DEFAULT_MAX_DELAY, primary_window: int = 1
    ) -> NDArray[np.float64]:
        """Return the censored daily pmf.

        The onset is uniform over a window of `primary_window` days
        (1 for a known onset day, 7 for a known onset week) and the
        report is censored to its day.
        The pmf is truncated at `max_delay` and renormalised.

        Parameters
        ----------
        max_delay : int
            Longest delay.
        primary_window : int
            Width of the onset window in days.

        Returns
        -------
        numpy.ndarray
            Probabilities for delays 0 to `max_delay`.
        """
        dist = self._dist()
        # Integrate the onset position over a fine grid within its window.
        u = (np.arange(200) + 0.5) / 200 * primary_window
        d = np.arange(max_delay + 1)[:, None]
        upper = np.clip(d + 1 - u, 0, None)
        lower = np.clip(d - u, 0, None)
        raw = (dist.cdf(upper) - dist.cdf(lower)).mean(axis=1)
        probs = np.asarray(raw, dtype=np.float64)
        return np.asarray(probs / probs.sum(), dtype=np.float64)


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
