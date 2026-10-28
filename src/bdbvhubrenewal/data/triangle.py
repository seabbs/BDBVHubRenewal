# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Reporting triangles and the baselinenowcast method.

A port of the core of the R package
[baselinenowcast](https://github.com/epinowcast/baselinenowcast).

`reporting_triangle` counts the known-onset cases of the hub's delay file
by onset week $W$ (rows) and delay $d$ in days (columns).
`first_publication_delays` first replaces each case's delay by the days
from the end of its onset week to the first delay file it is in, so a
batch published late counts as a long delay, which the chain ladder and
its retrospective uncertainty then learn.
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
`apply_delay` then fills each open row from that pmf $p$, with $F$ its
cdf and $x$ the row's sum over earlier columns, filled as it goes,
$$
\hat n(W, d) = \frac{x + 1 - F(d - 1)}{F(d - 1)}\, p(d),
$$
as baselinenowcast's `apply_delay` does.

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
    "apply_delay",
    "chainladder_fill",
    "delay_pmf",
    "first_publication_delays",
    "fit_nb_size",
    "preprocess_negative_values",
    "reporting_triangle",
    "retrospective_errors",
]


def first_publication_delays(
    history: pd.DataFrame,
    vintage: str,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> pd.DataFrame:
    r"""Return delay rows with each delay counted to first publication.

    Rows are matched across the delay files on or before `vintage` by
    province, onset week and delay. A row in the first file keeps its
    delay $d$. A row first in a later file, absent from the file before
    it dated $V$, was published after $V$, so its delay becomes
    $\max(d, V + 1 - W)$ days for onset week $W$. Rows whose delay is then
    over `max_delay` are dropped: those are mostly onsets recorded late
    for cases already counted as imputed, which leave the total unchanged.

    Parameters
    ----------
    history : pandas.DataFrame
        Delay rows of one or more files, with `vintage`, `location`,
        `onset_week` and `delay` columns.
    vintage : str
        The forecast date. Later files are not used.
    max_delay : int
        Longest delay kept, in days.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage` (the latest file on or before `vintage`),
        `location`, `onset_week` and `delay`.
    """
    key = ["location", "onset_week", "delay"]
    files = sorted(str(v) for v in history["vintage"].unique() if v <= vintage)
    counts = [
        history.loc[history["vintage"] == f].groupby(key).size() for f in files
    ]
    final = counts[-1]
    seen = pd.Series(0, index=final.index)
    parts = []
    for i, count in enumerate(counts):
        now = count.reindex(final.index, fill_value=0).clip(upper=final)
        new = (now - seen).clip(lower=0)
        new = new.loc[new > 0]
        seen = seen.combine(now, max)
        if new.empty:
            continue
        rows = new.index.to_frame(index=False).loc[
            np.repeat(np.arange(len(new)), new.to_numpy())
        ]
        if i > 0:
            week = pd.to_datetime(rows["onset_week"])
            gap = pd.Timestamp(files[i - 1]) + pd.Timedelta(days=1) - week
            rows["delay"] = np.maximum(rows["delay"], gap.dt.days)
        parts.append(rows)
    out = pd.concat(parts, ignore_index=True)
    out = out.loc[out["delay"] <= max_delay].reset_index(drop=True)
    return pd.DataFrame(out.assign(vintage=files[-1])[["vintage", *key]])


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

    Raises
    ------
    ValueError
        If the triangle has no cases.
    """
    filled = chainladder_fill(triangle)
    total = filled.sum()
    if not total > 0:
        raise ValueError("The triangle has no cases to estimate a delay from.")
    return np.asarray(filled.sum(axis=0) / total, dtype=np.float64)


def preprocess_negative_values(
    triangle: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Move negative counts to earlier delays, as baselinenowcast does.

    In each row, from the longest delay back, a negative count is set to
    0 and added to the next earlier observed delay. Counts in the delay
    file are never negative, so this only matters for corrected data.

    Parameters
    ----------
    triangle : numpy.ndarray
        Counts by row and delay, missing where unobserved.

    Returns
    -------
    numpy.ndarray
        The triangle with no negative counts.
    """
    out = triangle.copy()
    for row in out:
        carry = 0.0
        for col in range(row.size - 1, -1, -1):
            if np.isnan(row[col]):
                continue
            value = row[col] + carry
            row[col], carry = (0.0, value) if value < 0 else (value, 0.0)
    return out


def apply_delay(
    triangle: NDArray[np.float64], pmf: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fill the missing cells of a reporting triangle from a delay pmf.

    Parameters
    ----------
    triangle : numpy.ndarray
        Counts by row and delay, missing in the bottom right.
    pmf : numpy.ndarray
        Delay pmf, one entry per column.

    Returns
    -------
    numpy.ndarray
        The triangle with its missing cells filled with expectations.
    """
    out = triangle.copy()
    cdf = np.cumsum(pmf)
    for col in range(1, triangle.shape[1]):
        missing = np.isnan(out[:, col])
        if not missing.any():
            continue
        x = out[missing, :col].sum(axis=1)
        out[missing, col] = (x + 1 - cdf[col - 1]) / cdf[col - 1] * pmf[col]
    return out


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
        filled = apply_delay(hist, delay_pmf(hist))
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

    Each column (horizon) gets its own size by maximum likelihood on
    0.1 to 1000, with the predictions as means, as baselinenowcast's
    `fit_nb` does. Entries with a zero prediction are left out, unless
    one has a count, which gives 1000 as in baselinenowcast. A column
    with no prediction gets an infinite size (Poisson).

    Parameters
    ----------
    obs, pred : numpy.ndarray
        Observed and predicted counts, one column per horizon.

    Returns
    -------
    numpy.ndarray
        One size per column.
    """
    sizes = np.full(obs.shape[1], np.inf)
    for h in range(obs.shape[1]):
        keep = pred[:, h] > 0
        if not keep.any():
            continue
        if (obs[~keep, h] > 0).any():
            # A count where none was predicted makes the likelihood zero for
            # every size, and baselinenowcast's optimiser then stops at the
            # upper bound.
            sizes[h] = 1000.0
            continue
        y, mu = obs[keep, h], pred[keep, h]

        def nll(size: float, y: NDArray[np.float64] = y,
                mu: NDArray[np.float64] = mu) -> float:  # fmt: skip
            return float(
                -stats.nbinom.logpmf(y, size, size / (size + mu)).sum()
            )

        fit = minimize_scalar(nll, bounds=(0.1, 1000.0), method="bounded")
        sizes[h] = float(fit.x)
    return sizes
