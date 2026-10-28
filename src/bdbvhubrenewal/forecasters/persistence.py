# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Persistence baseline.

For each province, take the last week judged complete, $c$, and carry its
count $y_c$ forward on the log scale with random walk noise:

$$
\log(1 + \lambda_{c+h}) = \log(1 + y_c) + \sum_{j=1}^{h} \epsilon_j,
\quad \epsilon_j \sim \mathcal{N}(0, \sigma^2),
$$

$$
\tilde y_{c+h} \sim \text{Poisson}(\lambda_{c+h}), \quad h \ge 1.
$$

$\sigma$ is the standard deviation of the weekly changes in
$\log(1 + y)$ over the last `window` complete weeks, floored at
`min_sd`.
Weeks up to $c$ are returned as published (a point mass).
The median is flat and the spread grows with the horizon.
Each sample is one joint trajectory across weeks.

A week is judged complete if the vintage date is at least `lag_days`
after its midpoint.
`lag_days` is the `completeness` quantile of the pooled delay published
with the vintage, or `default_lag_days` if no delays were published.
Recent onset weeks only show the delays reported so far, so this lag is
biased short.

For each later date $T$ in `task.report_by`, the counts reported by $T$
are predicted by thinning each sample of week $w$ with the share of a
week reported $T - w$ days after it ends
(`bdbvhubrenewal.data.completeness.reported_share`, from the delays
published with the vintage), and taking at least the count already
published for that week.
There are no such rows when the vintage has no delay data.

`PersistenceNow` (`persistencenow`) first nowcasts the recent weeks, with
the baselinenowcast method (`bdbvhubrenewal.data.triangle`).
A chain ladder on the reporting triangle of the hub's delay file, pooled
over provinces, gives for each open onset week $w$ the ratio $r(w)$ of
the cases still to come to those reported.
Past nowcasts of the same triangle give a negative binomial size
$\phi_h$ for each horizon $h$.
The national cases still to come, $U(w) \sim \text{NegBin}$ with mean
$r(w) \sum_p Y_p(w)$ and size $\phi_h$, are split over places in
proportion to their published totals $Y_p(w)$, known plus imputed.
The hub re-imputes missing onsets at each vintage, so totals also move
both ways. Each place adds $E_p(w) = A - B$, with $A, B \sim
\text{Poisson}(\kappa I_p(w) / 2)$ and $I_p(w)$ its imputed count, so
the variance is $\kappa I_p(w)$, with $\kappa$ from the revisions of
settled weeks between past vintages (`revision_variance`). So

$$
C_p(w) = \max\{Y_p(w) + U_p(w) + E_p(w),\ K_p(w)\},
$$

with $K_p(w)$ the known onsets, and $C_p(w)$ can fall below $Y_p(w)$.
The last week with a share reported $1 / \{1 + r(w)\} \ge$ `min_share`
takes the place of the last complete week, and each sample carries its
own nowcast forward as above.
The spread of the walk at $h$ steps is scaled by $\lambda_h$, from the
errors of the same walk run from each past vintage against the current
data (`walk_errors`, `horizon_scales`), so only data available at the
vintage are used.
Later weeks are at least the known onsets.
If the delay data have no complete onset week or fewer than `min_delays`
cases in the last `n_history` weeks, or give fewer than `min_retro` past
nowcasts, it is `Persistence`.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.completeness import reported_share
from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.delays import EmpiricalDelayModel
from bdbvhubrenewal.data.triangle import (
    chainladder_fill,
    delay_pmf,
    fit_nb_size,
    reporting_triangle,
    retrospective_errors,
)
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    register_model,
)

__all__ = [
    "Persistence",
    "PersistenceNow",
    "completeness_lag",
    "last_complete_week",
    "nowcast_draws",
]


def completeness_lag(
    delays: pd.DataFrame | None,
    vintage: str,
    quantile: float = 0.9,
    default: int = 10,
) -> int:
    """Return the days after a week's midpoint at which it counts as complete.

    Parameters
    ----------
    delays : pandas.DataFrame or None
        Delay line data published with the vintage.
    vintage : str
        The vintage date.
    quantile : float
        Share of cases that must have been reported.
    default : int
        Lag used when no delays are available.

    Returns
    -------
    int
        The lag in days.
    """
    if delays is None or delays.empty:
        return default
    fit = EmpiricalDelayModel().fit(delays, vintage)
    return fit.quantile(quantile)


def last_complete_week(vintage: str, lag_days: int) -> dt.date:
    """Return the last week whose midpoint is at least `lag_days` old.

    Parameters
    ----------
    vintage : str
        The vintage date.
    lag_days : int
        Days from a week's midpoint (its Wednesday) to completeness.

    Returns
    -------
    datetime.date
        The Saturday ending that week.
    """
    # Midpoint of week ending s is s - 3; need s - 3 + lag <= vintage.
    latest = as_date(vintage) - dt.timedelta(days=lag_days - 3)
    return latest - dt.timedelta(days=(latest.weekday() - 5) % 7)


@dataclass
class Persistence:
    """Carry the last complete week forward with growing noise.

    Attributes
    ----------
    name : str
        Model name.
    window : int
        Complete weeks used to estimate the week-to-week spread.
    min_sd : float
        Floor on the log-scale weekly spread.
    completeness : float
        Delay quantile defining a complete week.
    default_lag_days : int
        Completeness lag when a vintage has no delay data.
    """

    name: str = "persistence"
    window: int = 8
    min_sd: float = 0.1
    completeness: float = 0.9
    default_lag_days: int = 10

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return persistence trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.
        """
        rng = task.rng()
        lag = completeness_lag(
            task.delays,
            task.vintage,
            self.completeness,
            self.default_lag_days,
        )
        last = last_complete_week(task.vintage, lag).isoformat()
        weeks = list(task.target_end_dates)
        n = task.n_samples
        frames = []
        for loc in task.locations:
            series = (
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .sort_index()
            )
            complete = series.loc[series.index <= last]
            level = float(complete.iloc[-1]) if len(complete) else 0.0
            sd = self._spread(complete.to_numpy(dtype=float))
            values = np.empty((n, len(weeks)))
            log_level = np.full(n, np.log1p(level))
            for j, week in enumerate(weeks):
                if week <= last:
                    values[:, j] = float(series.get(week, 0))
                    continue
                log_level, values[:, j] = _step(log_level, sd, rng)
            frames.append(
                pd.DataFrame(
                    {
                        "location": loc,
                        "target_end_date": np.tile(weeks, n),
                        "sample": np.repeat(np.arange(1, n + 1), len(weeks)),
                        "value": values.reshape(-1),
                    }
                )
            )
        final = pd.concat(frames, ignore_index=True)[list(SAMPLE_COLUMNS)]
        if not task.report_by:
            return final
        return pd.concat(
            [final.assign(as_of=np.nan), *self._reported_by(task, final, rng)],
            ignore_index=True,
        )

    def _spread(self, complete: np.ndarray) -> float:
        """Return the SD of the last weekly log changes, floored."""
        recent = np.log1p(complete[-self.window :])
        diffs = np.diff(recent)
        sd = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0
        return max(sd, self.min_sd)

    def _reported_by(
        self,
        task: ForecastTask,
        final: pd.DataFrame,
        rng: np.random.Generator,
    ) -> list[pd.DataFrame]:
        """Thin the final samples to the counts reported by each later date."""
        if task.delays is None or task.delays.empty:
            return []
        delay_vintage = str(task.delays["vintage"].max())
        seen = task.target.set_index(["location", "target_end_date"])["total"]
        floor = np.array(
            [
                seen.get((loc, week), 0)
                for loc, week in zip(
                    final["location"], final["target_end_date"], strict=True
                )
            ],
            dtype=float,
        )
        ends = pd.to_datetime(final["target_end_date"])
        out = []
        for date in task.report_by:
            days = (pd.Timestamp(date) - ends).dt.days.to_numpy(dtype=np.int64)
            due = days >= 0
            share = reported_share(task.delays, delay_vintage, days[due])
            counts = final["value"].to_numpy()[due].astype(np.int64)
            thinned = rng.binomial(counts, share)
            value = np.maximum(thinned, floor[due])
            out.append(final.loc[due].assign(value=value, as_of=date))
        return out


register_model(
    "persistence",
    Persistence,
    "Last complete week carried forward, with log-scale random walk noise.",
    baseline=True,
)


def _step(
    log_level: np.ndarray, sd: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Take one random walk step on the log scale and draw Poisson counts."""
    log_level = log_level + rng.normal(0.0, sd, size=log_level.size)
    return log_level, rng.poisson(np.expm1(np.maximum(log_level, 0)))


def nowcast_draws(
    observed: np.ndarray,
    ratio: np.ndarray,
    size: np.ndarray,
    n: int,
    rng: np.random.Generator,
    known: np.ndarray | None = None,
    revision: float = 0.0,
) -> np.ndarray:
    r"""Draw final totals from published totals and point nowcast ratios.

    The national count still to come in week $w$ is negative binomial
    with mean $r(w) \sum_p Y_p(w)$ and size $\phi(w)$, or Poisson where
    $\phi$ is infinite. It is split over places multinomially in
    proportion to $Y_p(w)$. With `known` and `revision` $\kappa > 0$,
    each place also adds the difference of two Poisson draws with mean
    $\kappa (Y_p - K_p) / 2$, and is kept at least $K_p$.

    Parameters
    ----------
    observed : numpy.ndarray
        Published totals $Y_p(w)$, one row per place, one column per week.
    ratio : numpy.ndarray
        Cases still to come per case reported, $r(w)$.
    size : numpy.ndarray
        Negative binomial size $\phi(w)$.
    n : int
        Number of draws.
    rng : numpy.random.Generator
        Random generator.
    known : numpy.ndarray, optional
        Known onsets $K_p(w)$, shaped as `observed`.
    revision : float
        $\kappa$, the revision variance per imputed case.

    Returns
    -------
    numpy.ndarray
        Final totals, shaped `(n, places, weeks)`.
    """
    obs = np.asarray(observed, dtype=float)
    total = obs.sum(axis=0)
    mean = total * ratio
    out = np.broadcast_to(obs, (n, *obs.shape)).copy()
    for j in np.flatnonzero(mean > 0):
        if np.isfinite(size[j]):
            phi = size[j]
            added = rng.negative_binomial(phi, phi / (phi + mean[j]), size=n)
        else:
            added = rng.poisson(mean[j], size=n)
        out[:, :, j] += rng.multinomial(added, obs[:, j] / total[j])
    if known is not None and revision > 0:
        lam = revision * np.clip(obs - known, 0, None) / 2
        shape = out.shape
        out += rng.poisson(lam, size=shape) - rng.poisson(lam, size=shape)
        out = np.maximum(out, known)
    return out


def revision_variance(
    history: pd.DataFrame, vintage: str, min_age: int = 63
) -> float:
    r"""Return $\kappa$, the variance of total revisions per imputed case.

    Over each pair of consecutive vintages up to `vintage`, provinces
    and weeks that ended at least `min_age` days before the earlier
    vintage, whose cases are all reported, give
    $\kappa = \sum \Delta^2 / \sum I$, with $\Delta$ the change in the
    total and $I$ the imputed count at the earlier vintage.

    Parameters
    ----------
    history : pandas.DataFrame
        Target rows of several vintages.
    vintage : str
        The forecast vintage. Later vintages are not used.
    min_age : int
        Days from the end of a week to the earlier vintage.

    Returns
    -------
    float
        $\kappa$, or 0 with no pair of vintages.
    """
    rows = history.loc[
        (history["vintage"] <= vintage) & (history["location"] != NATIONAL)
    ]
    key = ["location", "target_end_date"]
    num = den = 0.0
    vintages = sorted(str(v) for v in rows["vintage"].unique())
    for a, b in itertools.pairwise(vintages):
        old = rows.loc[rows["vintage"] == a].set_index(key)
        new = rows.loc[rows["vintage"] == b].set_index(key)
        both = old.join(new[["total"]], rsuffix="_new", how="inner")
        end = pd.to_datetime(both.index.get_level_values("target_end_date"))
        settled = np.asarray((pd.Timestamp(a) - end).days >= min_age)
        both = both.loc[settled]
        num += float(((both["total_new"] - both["total"]) ** 2).sum())
        den += float(both["imputed"].sum())
    return num / den if den > 0 else 0.0


def horizon_scales(errors: list[np.ndarray], prior: float = 10.0) -> np.ndarray:
    r"""Return the scale of the walk's spread at each horizon.

    $\lambda_h = \sqrt{(\sum z^2 + k) / (n_h + k)}$ over the $n_h$
    standardised errors $z$ at horizon $h$, with $k$ = `prior`, so few
    errors leave $\lambda_h$ near 1. A horizon with none takes the scale
    of the one before.

    Parameters
    ----------
    errors : list of numpy.ndarray
        Standardised errors, one array per horizon from 1.
    prior : float
        Weight of $\lambda = 1$, in errors.

    Returns
    -------
    numpy.ndarray
        $\lambda_h$, one per horizon.
    """
    out = np.ones(len(errors))
    for h, z in enumerate(errors):
        if z.size == 0:
            out[h] = out[h - 1] if h > 0 else 1.0
            continue
        out[h] = np.sqrt((np.sum(z**2) + prior) / (z.size + prior))
    return out


@dataclass
class PersistenceNow(Persistence):
    """Persistence carried forward from a nowcast of the recent weeks.

    Attributes
    ----------
    name : str
        Model name.
    min_share : float
        Smallest share reported for a week's nowcast to be carried
        forward. The last such week is.
    n_history : int
        Onset weeks of the reporting triangle each chain ladder uses.
    n_retro : int
        Past nowcasts to fit the negative binomial sizes to.
    min_retro : int
        Fewest past nowcasts to nowcast at all.
    min_delays : int
        Fewest delays in the chain ladder's onset weeks to nowcast at all.

    See `Persistence` for the other attributes.
    """

    name: str = "persistencenow"
    min_share: float = 0.5
    n_history: int = 14
    n_retro: int = 8
    min_retro: int = 3
    min_delays: int = 50

    def _delay(
        self, task: ForecastTask
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the triangle's weeks, ratios, sizes and delay cdf, or None.

        None means too little delay data to nowcast.
        """
        if task.delays is None or task.delays.empty:
            return None
        delay_vintage = str(task.delays["vintage"].max())
        weeks, tri = reporting_triangle(task.delays, delay_vintage)
        hist = tri[-self.n_history :]
        complete = (~np.isnan(hist)).all(axis=1).any()
        enough = np.nansum(hist) >= self.min_delays
        if len(hist) < self.n_history or not complete or not enough:
            return None
        obs, pred = retrospective_errors(tri, self.n_retro, self.n_history)
        if obs.shape[0] < self.min_retro:
            return None
        filled = chainladder_fill(hist)
        reported = np.nansum(hist, axis=1)
        to_come = filled.sum(axis=1) - reported
        ratio = np.where(reported > 0, to_come / np.maximum(reported, 1), 0)
        cdf = np.cumsum(delay_pmf(hist))
        return weeks[-self.n_history :], ratio, fit_nb_size(obs, pred), cdf

    def walk_errors(self, task: ForecastTask) -> list[np.ndarray]:
        r"""Return the standardised errors of the walk from past vintages.

        From each earlier vintage in `task.target_history`, each province
        with a nowcast level $L = Y / s$ of at least 10 at its last week
        with share $s \ge$ `min_share` is carried forward $h$ weeks, and
        $z = \{\log(1 + y) - \log(1 + L)\} / (\sigma \sqrt{h})$ against
        the count $y$ in the task's vintage, for weeks with at least 95%
        reported then. The share comes from the task's delays.

        Parameters
        ----------
        task : ForecastTask
            The task. Only data available at its vintage are used.

        Returns
        -------
        list of numpy.ndarray
            One array of errors per horizon from 1 to 8, or an empty list
            without history or delays.
        """
        fit = self._delay(task)
        history = task.target_history
        if fit is None or history is None:
            return []
        cdf = fit[3]
        rows = history.loc[history["location"] != NATIONAL]
        wide = rows.pivot_table(
            index=["vintage", "location"],
            columns="target_end_date",
            values="total",
        )
        if task.vintage not in wide.index.get_level_values("vintage"):
            return []
        now = wide.loc[task.vintage]
        now_age = (
            pd.Timestamp(task.vintage) - pd.to_datetime(now.columns)
        ).days
        now_ok = _share(cdf, now_age.to_numpy()) >= 0.95
        errors: list[list[float]] = [[] for _ in range(8)]
        for vintage in sorted(set(wide.index.get_level_values("vintage"))):
            if vintage >= task.vintage:
                continue
            past = wide.loc[vintage].dropna(axis=1, how="all").fillna(0.0)
            ends = pd.to_datetime(past.columns)
            share = _share(cdf, (pd.Timestamp(vintage) - ends).days.to_numpy())
            anchor = np.flatnonzero(share >= self.min_share)
            if anchor.size == 0:
                continue
            c = int(anchor[-1])
            for loc, values in past.iterrows():
                level = values.to_numpy(dtype=float)[: c + 1] / share[: c + 1]
                if level[-1] < 10 or loc not in now.index:
                    continue
                sd = self._spread(level)
                for h in range(1, 9):
                    week = (ends[c] + pd.Timedelta(days=7 * h)).date()
                    col = now.columns.get_indexer(pd.Index([week.isoformat()]))[
                        0
                    ]
                    if col < 0 or not now_ok[col]:
                        break
                    truth = now.loc[loc].iloc[col]
                    if np.isnan(truth):
                        break
                    z = (np.log1p(truth) - np.log1p(level[-1])) / (
                        sd * np.sqrt(h)
                    )
                    errors[h - 1].append(float(z))
        return [np.asarray(e) for e in errors]

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return persistence trajectories from a nowcast for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`,
            and `as_of` if the task asks for `report_by`.
        """
        fit = self._delay(task)
        if fit is None:
            return super().forecast(task)
        tri_weeks, ratio_tri, sizes, cdf = fit
        weeks = list(task.target_end_dates)
        ends = pd.date_range(
            min(task.target["target_end_date"].min(), weeks[0]),
            weeks[-1],
            freq="7D",
        )
        grid = [d.date().isoformat() for d in ends]
        # Ratio and size of each grid week; weeks before the history are
        # complete, weeks after the delay vintage are not nowcast.
        ratio, size = np.zeros(len(grid)), np.full(len(grid), np.inf)
        share = np.where(np.array(grid) < tri_weeks[0], 1.0, 0.0)
        for i, week in enumerate(tri_weeks):
            if week in grid:
                j = grid.index(week)
                h = len(tri_weeks) - 1 - i
                ratio[j] = ratio_tri[i]
                size[j] = sizes[h] if h < sizes.size else np.inf
                share[j] = 1.0 / (1.0 + ratio_tri[i])
        anchor = np.flatnonzero(share >= self.min_share)
        if anchor.size == 0:
            return super().forecast(task)
        last = int(anchor[-1])
        ratio[last + 1 :] = 0.0
        rng = task.rng()
        n = task.n_samples
        cols = [grid.index(w) for w in weeks]

        def wide(column: str) -> np.ndarray:
            return np.array(
                [
                    task.target.loc[task.target["location"] == loc]
                    .set_index("target_end_date")[column]
                    .reindex(grid, fill_value=0)
                    .to_numpy(dtype=float)
                    for loc in task.locations
                ]
            )

        seen, known = wide("total"), wide("observed")
        history = (
            task.target
            if task.target_history is None
            else (task.target_history)
        )
        kappa = revision_variance(history, task.vintage)
        totals = nowcast_draws(seen, ratio, size, n, rng, known, kappa)
        scales = horizon_scales(self.walk_errors(task))
        steps = len(grid) - 1 - last
        lam = np.concatenate(
            [scales, np.full(max(steps - scales.size, 0), scales[-1])]
            if scales.size
            else [np.ones(steps)]
        )
        # Cumulative walk variance lambda_h^2 h sigma^2, in unit steps.
        cum = lam[:steps] ** 2 * np.arange(1, steps + 1)
        inc = np.sqrt(np.maximum(np.diff(np.r_[0.0, cum]), 0.01))
        for p in range(len(task.locations)):
            point = seen[p, : last + 1] * (1 + ratio[: last + 1])
            sd = self._spread(point)
            log_level = np.log1p(totals[:, p, last])
            for k, j in enumerate(range(last + 1, len(grid))):
                log_level, draw = _step(log_level, sd * inc[k], rng)
                totals[:, p, j] = np.maximum(draw, known[p, j])
        assert task.delays is not None
        age = (pd.Timestamp(str(task.delays["vintage"].max())) - ends).days
        s_v = _share(cdf, age.to_numpy())
        final, as_of = [], []
        for p, loc in enumerate(task.locations):
            final.append(_frame(loc, weeks, totals[:, p, cols]))
            diff = totals[:, p] - seen[p]
            for date in task.report_by:
                later = (pd.Timestamp(date) - ends).days.to_numpy()
                s_t = _share(cdf, later)
                q = np.clip((s_t - s_v) / np.maximum(1 - s_v, 1e-12), 0, 1)
                q = np.where(s_v >= 1.0, 1.0, q)
                up = rng.binomial(np.maximum(diff, 0).astype(np.int64), q)
                counts = seen[p] + up + np.minimum(diff, 0)
                due = [c for c in cols if grid[c] <= date]
                as_of.append(
                    _frame(loc, [grid[c] for c in due], counts[:, due]).assign(
                        as_of=date
                    )
                )
        out = pd.concat(final, ignore_index=True)[list(SAMPLE_COLUMNS)]
        if not task.report_by:
            return out
        return pd.concat([out.assign(as_of=np.nan), *as_of], ignore_index=True)


def _share(cdf: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Share of a week's cases published `age` days after it ends."""
    return np.where(age < 0, 0.0, cdf[np.clip(age, 0, cdf.size - 1)])


def _frame(loc: str, weeks: list[str], values: np.ndarray) -> pd.DataFrame:
    """Return sample rows for one location, one column of `values` a week."""
    n = values.shape[0]
    return pd.DataFrame(
        {
            "location": loc,
            "target_end_date": np.tile(weeks, n),
            "sample": np.repeat(np.arange(1, n + 1), len(weeks)),
            "value": values.reshape(-1),
        }
    )


register_model(
    "persistencenow",
    PersistenceNow,
    "Persistence carried forward from a nowcast of the recent weeks.",
)
