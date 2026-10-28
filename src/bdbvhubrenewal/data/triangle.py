# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Reporting triangles and the baselinenowcast method.

A port of the core of the R package
[baselinenowcast](https://github.com/epinowcast/baselinenowcast).

`reporting_triangle` counts the known-onset cases of the hub's delay file
by onset week $W$ (rows) and delay $d$ in days (columns).
A case is taken to be published on day $W + d$: its onset mid-week,
$W - 3$, plus its delay, plus a 3-day publication lag.
So cell $(W, d)$ is observed at vintage $T$ if $W + d \le T$ and is
missing otherwise.

`chainladder_fill` fills the missing cells column by column.
For column $d$, with $A$ the rows where it is observed,
$$
\hat n(W, d) = \frac{\sum_{W' \in A} n(W', d)}{\sum_{W' \in A}
\sum_{d' < d} n(W', d')} \sum_{d' < d} \hat n(W, d'),
$$
and `delay_pmf` is the column sums of the filled triangle over its total.

`retrospective_errors` repeats this on the triangle as it was at earlier
vintages, and compares what each past nowcast expected to be added to
each open row with what was added by now. `fit_nb_size` fits a negative
binomial size to those errors for each horizon, so that the counts still
to come are drawn as $\text{NegBin}$ with the point nowcast as mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize_scalar

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.delays import DEFAULT_MAX_DELAY

__all__ = [
    "chainladder_fill",
    "delay_pmf",
    "fit_nb_size",
    "reporting_triangle",
    "retrospective_errors",
]


def reporting_triangle(
    delays: pd.DataFrame,
    vintage: str,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> tuple[list[str], NDArray[np.float64]]:
    """Count known-onset cases by onset week and delay, pooled over places.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data with `vintage`, `onset_week` and `delay` columns.
        Rows of `vintage` are used.
    vintage : str
        The delay vintage, $T$.
    max_delay : int
        Last delay column. Longer delays are counted in it.

    Returns
    -------
    weeks : list of str
        Every onset week from the first in the data to the last ending on
        or before $T$.
    triangle : numpy.ndarray
        Counts, one row per week and one column per delay, missing where
        $W + d > T$.
    """
    rows = delays.loc[delays["vintage"] == vintage]
    end = pd.Timestamp(as_date(vintage))
    onset = pd.to_datetime(rows["onset_week"])
    rows, onset = rows.loc[(onset <= end).to_numpy()], onset[onset <= end]
    if rows.empty:
        return [], np.zeros((0, max_delay + 1))
    ends = pd.date_range(onset.min(), onset.max(), freq="7D")
    weeks = [w.date().isoformat() for w in ends]
    row = ((onset - ends[0]).dt.days // 7).to_numpy()
    col = rows["delay"].to_numpy(dtype=np.int64).clip(0, max_delay)
    tri = np.zeros((len(weeks), max_delay + 1))
    np.add.at(tri, (row, col), 1.0)
    seen = (end - ends).days.to_numpy()
    unseen = np.arange(max_delay + 1)[None, :] > seen[:, None]
    return weeks, np.where(unseen, np.nan, tri)


def chainladder_fill(triangle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fill the missing cells of a reporting triangle by chain ladder.

    Parameters
    ----------
    triangle : numpy.ndarray
        Counts by row and delay, missing in the bottom right. Needs at
        least one complete row.

    Returns
    -------
    numpy.ndarray
        The triangle with its missing cells filled with expectations.
    """
    out = triangle.copy()
    for col in range(1, triangle.shape[1]):
        missing = np.isnan(triangle[:, col])
        if not missing.any():
            continue
        top = ~missing
        factor = np.nansum(triangle[top, col]) / max(
            np.nansum(triangle[top, :col]), 1.0
        )
        out[missing, col] = factor * out[missing, :col].sum(axis=1)
    return out


def delay_pmf(triangle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the delay pmf of a reporting triangle, after chain ladder.

    Parameters
    ----------
    triangle : numpy.ndarray
        Counts by row and delay, as for `chainladder_fill`.

    Returns
    -------
    numpy.ndarray
        Probabilities for each delay column.
    """
    filled = chainladder_fill(triangle)
    return np.asarray(filled.sum(axis=0) / filled.sum(), dtype=np.float64)


def retrospective_errors(
    triangle: NDArray[np.float64],
    n_retro: int,
    n_history: int,
    step: int = 7,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compare past point nowcasts with what has been added since.

    For each of the last `n_retro` rows, the triangle is cut back to what
    was observed then: rows up to it, and in each row `step` fewer delay
    columns per row cut. A chain ladder on its last `n_history` rows
    predicts the cells observed now but not then.

    Parameters
    ----------
    triangle : numpy.ndarray
        Counts by row and delay at the vintage.
    n_retro : int
        Past nowcasts to make, one per row before the vintage.
    n_history : int
        Rows each chain ladder uses.
    step : int
        Delay columns between rows: 7 for weekly rows and daily delays.

    Returns
    -------
    obs, pred : numpy.ndarray
        Added and predicted counts, one row per past nowcast with a
        complete row in its history, one column per horizon (0 for the
        latest row then). Horizons are the rows open at the vintage.
    """
    n_rows, n_cols = triangle.shape
    horizons = int(np.isnan(triangle).any(axis=1).sum())
    observed = ~np.isnan(triangle)
    n_seen = observed.sum(axis=1)
    obs_rows, pred_rows = [], []
    for j in range(1, n_retro + 1):
        last = n_rows - 1 - j
        if last - n_history + 1 < 0:
            break
        # Delay columns observed in each row when this row was the latest.
        age = (last - np.arange(last + 1)) * step + (n_seen[-1] - 1)
        then = np.arange(n_cols)[None, :] <= age[:, None]
        cut = triangle[: last + 1]
        cut = np.where(then, cut, np.nan)
        hist = cut[-n_history:]
        if not (~np.isnan(hist)).all(axis=1).any():
            continue
        filled = chainladder_fill(hist)
        o, p = np.zeros(horizons), np.zeros(horizons)
        for h in range(min(horizons, n_history)):
            r = last - h
            new = ~then[r] & observed[r]
            o[h] = triangle[r, new].sum()
            p[h] = filled[n_history - 1 - h, new].sum()
        obs_rows.append(o)
        pred_rows.append(p)
    shape = (len(obs_rows), horizons)
    return (
        np.asarray(obs_rows, dtype=np.float64).reshape(shape),
        np.asarray(pred_rows, dtype=np.float64).reshape(shape),
    )


def fit_nb_size(
    obs: NDArray[np.float64], pred: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fit a negative binomial size to observed against predicted counts.

    Each column (horizon) gets its own size by maximum likelihood, with
    the predictions as means. Entries with a zero prediction are left
    out. A column with none gets an infinite size (Poisson).

    Parameters
    ----------
    obs, pred : numpy.ndarray
        Observed and predicted counts, one column per horizon.

    Returns
    -------
    numpy.ndarray
        One size per column, between $e^{-5}$ and $e^{12}$.
    """
    sizes = np.full(obs.shape[1], np.inf)
    for h in range(obs.shape[1]):
        keep = pred[:, h] > 0
        if not keep.any():
            continue
        y, mu = obs[keep, h], pred[keep, h]

        def nll(log_size: float, y: NDArray[np.float64] = y,
                mu: NDArray[np.float64] = mu) -> float:  # fmt: skip
            size = np.exp(log_size)
            return float(
                -stats.nbinom.logpmf(y, size, size / (size + mu)).sum()
            )

        fit = minimize_scalar(nll, bounds=(-5.0, 12.0), method="bounded")
        sizes[h] = float(np.exp(fit.x))
    return sizes
