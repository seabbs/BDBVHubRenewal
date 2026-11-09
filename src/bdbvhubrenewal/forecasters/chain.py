# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Daily chain ladder on release snapshots, with its uncertainty.

Counts $N_r(t)$ are the daily onsets of release $r$, on the onset clock.
Release $r$ has cut-off $c_r$, and day $t$ is $a = c_r - t$ days old.
A pair of consecutive releases $(r, r + 1)$ has step
$s = c_{r+1} - c_r$ days and is in window when $s \le D$.

**Delay.** Releases are batched reports of one daily delay.
A day of age $a$ at release $r$ grows by $\exp \sum_{j=a}^{a+s-1} g(j)$
to release $r + 1$, with $g(j) \ge 0$ the daily log development factor
and $g(j) = 0$ for $j \ge D$.
For each gap $s$ and age $a$, the ratio of sums
$\sum N_{r+1}(t) / \sum N_r(t)$, over the in-window pairs of that gap
and the days of age $a$ that both releases observe, gives one equation
for $\sum_{j=a}^{a+s-1} g(j)$ on the log scale, weighted by the
denominator.
$g$ is the non-negative least squares fit of all of them
(`log_factors`).
Cells a release does not observe are skipped, not filled.
A place uses its own $g(a)$ when the denominator at age $a$, summed over
the pairs, is at least 10 cases, and otherwise its parent's (zone,
province, country).
The country uses its own.
Only onset days within `Settings.history_delay` (63 for $D = 42$) of the
latest cut-off are summed, and downward revisions are first moved onto
earlier releases of the day, as baselinenowcast does.

**Nowcast.** At the latest release $o$, day $t$ of age $a$ grows to final
by $G(a) = \exp \sum_{j=a}^{D-1} g(j)$.
Each hub week is a row of a weekly reporting triangle, with a column every
7 days, and the mean over its observed days of $1 / G$ at each column is
the share of the week reported by it.
baselinenowcast's `apply_delay` fills the row from the week's reported
count (`week_nowcast`), with its $+1$, so a week with nothing reported yet
still expects some cases.

**Uncertainty.** A release $r < o$ with an in-window pair at $r$ nowcasts
the count expected at $o$ for days up to $c_r$, with the same weekly fill
from $c_r$ to $c_o$ and the $g$ of $r$.
Its new reports and removals by weeks back fit the observation model
(`obs`).

**Introductions.** A place with no case in the latest `walk_weeks` weeks
is silent, and its weekly probability of a first case is pooled over its
siblings (`introductions.introduction_rates`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bdbvhubrenewal.baselinenowcast import (
    fill_row,
    growth,
    log_factors,
    pair_sums,
    preprocess_negative_values,
    window_pairs,
)
from bdbvhubrenewal.data.snapshots import SnapshotArrays, week_sum, weekly_sums
from bdbvhubrenewal.forecasters.base import ModelUnavailableError
from bdbvhubrenewal.forecasters.introductions import (
    introduction_rates,
    silent_places,
)
from bdbvhubrenewal.forecasters.obs import fit_sizes, rate
from bdbvhubrenewal.forecasters.pooling import (
    MIN_ERRORS,
    MIN_LEVEL,
    pool_matrix,
    source_index,
)

__all__ = [
    "DelayFit",
    "DelayFn",
    "Errors",
    "MissingSnapshots",
    "NoDelayPair",
    "Origin",
    "Settings",
    "build_origin",
    "delay_summary",
    "delay_sums",
    "final_growth",
    "levels",
    "own_ages",
    "retro_errors",
    "walk_changes",
    "week_nowcast",
    "weeks_back",
]

log = logging.getLogger(__name__)


class MissingSnapshots(ModelUnavailableError):  # noqa: N818
    """The task has no daily release snapshots."""


class NoDelayPair(ModelUnavailableError):  # noqa: N818
    """No pair of releases has a step within the delay window."""


class NoCompleteWeek(ModelUnavailableError):  # noqa: N818
    """No week of a place reaches the completeness that starts the walk."""


@dataclass(frozen=True)
class Settings:
    """Settings shared by the random walk models.

    Attributes
    ----------
    max_delay : int
        $D$, the days a count may still grow, baselinenowcast's
        `max_delay`.
    min_level : float
        Cases a week for a place to use its own spreads, and cases in the
        pooled denominator of an age for it to use its own delay factor.
    min_errors : int
        Errors at a horizon to fit the observation model.
    level_weeks : tuple of int
        First and last weeks back averaged for a place's level.
    completeness : float
        Share of a week's final count that must be reported for it to
        start the random walk.
    walk_weeks : int
        Weeks, ending at the start week, whose changes set the spread.
    clock_shift : int
        Days added to the ages of the origin's days, for the sensitivity
        of the nowcast to the onset clock. Ages below 0 count as 0. The
        delay and the retrospective errors are not moved.
    start_share : float
        Share of a week reported for `randomwalknowcast` to start
        its walk from the week's nowcast.
    start_count : float
        Cases reported in a week for `randomwalknowcast` to start its walk
        from the week's nowcast.
    filter_noise : str
        Variance of a week's observation of the walk's level in
        `randomwalknowcast`'s filter. `"poisson"` is $1 / (F + y)$.
        `"nowcast"` adds the variance of the nowcast's count still to
        come, negative binomial with the fitted size, on the log scale,
        so a week mostly still to come is a weak observation.
    """

    max_delay: int = 42
    min_level: float = MIN_LEVEL
    min_errors: int = MIN_ERRORS
    level_weeks: tuple[int, int] = (3, 6)
    completeness: float = 0.95
    walk_weeks: int = 9
    clock_shift: int = 0
    start_share: float = 0.5
    start_count: float = 5.0
    filter_noise: str = "nowcast"

    @property
    def history_delay(self) -> int:
        """Onset days used for the delay, baselinenowcast's default.

        R uses `floor(0.5 * round(3 * D))` reference times for the delay,
        at least `D + 1`, and the rest for the uncertainty.
        """
        n_used = round(3 * self.max_delay)
        return max(n_used // 2, self.max_delay + 1)


@dataclass
class DelayFit:
    """The daily delay at a release, as each place uses it.

    Attributes
    ----------
    pairs : numpy.ndarray
        Earlier and later release of each in-window pair, `[P, 2]`.
    g : numpy.ndarray
        Daily log development factor $g(a)$ of each place, `[L, D]`. A
        place uses its own at an age when the denominator there, summed
        over the pairs, is at least `Settings.min_level` cases, and
        otherwise its parent's.
    level : numpy.ndarray
        Cases a week of each place, `[L]`.
    source : numpy.ndarray
        Index of the estimate each place uses for its walk spread, `[L]`,
        as for `source_index`.
    own : numpy.ndarray
        Whether each place uses its own factor at each age, `[L, D]`. The
        country always does.
    """

    pairs: np.ndarray
    g: np.ndarray
    level: np.ndarray
    source: np.ndarray
    own: np.ndarray


def weeks_back(snap: SnapshotArrays, r: int, max_delay: int) -> int:
    r"""Return the number of weeks back from the cut-off week with young days.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index.
    max_delay : int
        $D$.

    Returns
    -------
    int
        $K$: weeks back $k = 0, \dots, K - 1$ hold a day younger than
        $D$.
    """
    youngest = max(int(snap.cut[r]) - max_delay + 1, 0)
    return snap.cut_week(r) - int(snap.week[youngest]) + 1


def levels(snap: SnapshotArrays, r: int, settings: Settings) -> np.ndarray:
    """Return mean weekly counts over the level weeks at release `r`."""
    weekly = weekly_sums(snap, r)
    wa = snap.cut_week(r)
    lo, hi = settings.level_weeks
    cols = [wa - k for k in range(lo, hi + 1)]
    vals = [weekly[:, c] if c >= 0 else np.zeros(weekly.shape[0]) for c in cols]
    return np.mean(vals, axis=0)


def _pairs_at(snap: SnapshotArrays, r: int, max_delay: int) -> np.ndarray:
    """Return the in-window pairs among releases up to `r`."""
    keep, _ = window_pairs(snap.cut[: r + 1], max_delay)
    return np.stack([keep, keep + 1], axis=1).astype(np.int64)


def _delay_counts(snap: SnapshotArrays, r: int) -> np.ndarray:
    """Return counts with downward revisions moved onto earlier releases.

    For each day the increments between its observed releases up to `r`
    go through baselinenowcast's `preprocess_negative_values`, so a
    negative revision is set to zero and taken from the earlier
    increments. The cumulative counts are then rebuilt, and never fall
    from one release to the next. Releases after `r` are left as they
    are, and are not read by a fit at `r`.
    """
    c = np.asarray(snap.counts, dtype=float)
    n_loc, _, n_day = c.shape
    obs = snap.observed[: r + 1]
    cum = c[:, : r + 1]
    prev = np.concatenate(
        [np.zeros_like(cum[:, :1]), cum[:, :-1] * obs[None, :-1]], axis=1
    )
    inc = np.where(obs[None], cum - prev, 0.0)
    rows = inc.transpose(0, 2, 1).reshape(n_loc * n_day, r + 1)
    mask = np.tile(obs.T, (n_loc, 1))
    fixed = np.asarray(preprocess_negative_values(rows, mask, validate=False))
    fixed = fixed.reshape(n_loc, n_day, r + 1).transpose(0, 2, 1)
    out = c.copy()
    out[:, : r + 1] = np.where(
        obs[None], np.cumsum(fixed, axis=1), c[:, : r + 1]
    )
    return out


def delay_sums(
    snap: SnapshotArrays, r: int, settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the pair sums the delay is fitted to at release `r`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index. Pairs of later releases are not used.
    settings : Settings
        Settings.

    Returns
    -------
    tuple of numpy.ndarray or None
        The in-window pairs `[P, 2]`, `num` and `den` from `pair_sums`
        `[L, P, D]`, and each pair's gap `[P]`, or None with no
        in-window pair.
    """
    pairs = _pairs_at(snap, r, settings.max_delay)
    if pairs.size == 0:
        return None
    counts = _delay_counts(snap, r)
    day = np.arange(snap.week.size)
    window = day > int(snap.cut[r]) - settings.history_delay
    observed = snap.observed & window[None, :]
    gaps = snap.cut[pairs[:, 1]] - snap.cut[pairs[:, 0]]
    num, den = pair_sums(
        jnp.asarray(counts),
        jnp.asarray(observed),
        jnp.asarray(snap.cut),
        jnp.asarray(pairs[:, 0]),
        settings.max_delay,
    )
    return pairs, np.asarray(num), np.asarray(den), gaps


def own_ages(
    den: np.ndarray, parent: np.ndarray, settings: Settings
) -> np.ndarray:
    """Return where each place's own data fit its delay, by age.

    Parameters
    ----------
    den : numpy.ndarray
        Denominators from `delay_sums`, `[L, P, D]`.
    parent : numpy.ndarray
        Parent of each place, -1 for the country, `[L]`.
    settings : Settings
        Settings.

    Returns
    -------
    numpy.ndarray
        `[L, D]`, true where the denominator at an age, summed over the
        pairs, is at least `Settings.min_level` cases. The country is
        always true.
    """
    own = den.sum(axis=1) >= settings.min_level
    own[parent < 0] = True
    return np.asarray(own)


def fit_delay(
    snap: SnapshotArrays, r: int, settings: Settings
) -> DelayFit | None:
    """Estimate the delay factors from the pairs up to release `r`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        Release index. Pairs of later releases are not used.
    settings : Settings
        Settings.

    Returns
    -------
    DelayFit or None
        The factors, or None with no in-window pair.
    """
    sums = delay_sums(snap, r, settings)
    if sums is None:
        return None
    pairs, num, den, gaps = sums
    g = log_factors(num, den, gaps)
    own = own_ages(den, snap.parent, settings)
    # Nodes are sorted by code length, so a parent precedes its children.
    for n in range(g.shape[0]):
        p = snap.parent[n]
        if p >= 0:
            g[n] = np.where(own[n], g[n], g[p])
    level = levels(snap, r, settings)
    source = source_index(level, snap.parent, min_level=settings.min_level)
    return DelayFit(pairs, g, level, source, own)


def final_growth(
    snap: SnapshotArrays, o: int, fit: DelayFit, settings: Settings
) -> np.ndarray:
    """Return the growth to final of each place and day at release `o`.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    fit : DelayFit
        The delay at the origin.
    settings : Settings
        Settings.

    Returns
    -------
    numpy.ndarray
        $G(a)$ at each day's age, shifted by `Settings.clock_shift`,
        `[L, T]`.
    """
    age = snap.cut[o] - np.arange(snap.week.size) + settings.clock_shift
    return growth(fit.g, age, settings.max_delay)


def delay_summary(origin: Origin, model: str, vintage: str) -> pd.DataFrame:
    """Return the share of a day's final count reported by each age.

    The share at age $a$ is $1 / G(a)$, from the delay fitted at the
    origin, for every place, in the form of the renewal models' delay
    summaries (`bdbvhubrenewal.forecasters.renewal.fit_delays`).

    Parameters
    ----------
    origin : Origin
        The fit.
    model : str
        Model name.
    vintage : str
        The origin.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `vintage`, `location`, `kind` (`"cumulative"`),
        `x` (age in days), `quantity` (`"reported_share"`) and `q50`.
    """
    ages = np.arange(origin.settings.max_delay + 1)
    share = 1.0 / growth(origin.fit.g, ages, origin.settings.max_delay)
    places = origin.snap.locations
    return pd.DataFrame(
        {
            "model": model,
            "vintage": vintage,
            "location": np.repeat(places, ages.size),
            "kind": "cumulative",
            "x": np.tile(ages, len(places)),
            "quantity": "reported_share",
            "q50": share.ravel(),
        }
    )


@dataclass
class Errors:
    """Retrospective nowcasts by release, place and weeks back.

    Attributes
    ----------
    added, removed : numpy.ndarray
        Rise and fall of each week's count between each release and the
        origin, `[R, L, K]`.
    remaining : numpy.ndarray
        Nowcast of the count still to come, `[R, L, K]`.
    reported : numpy.ndarray
        Count reported at each release, `[R, L, K]`.
    valid : numpy.ndarray
        Whether each cell is an error, `[R, L, K]`.
    releases : list of str
        Releases that gave errors.
    """

    added: np.ndarray
    removed: np.ndarray
    remaining: np.ndarray
    reported: np.ndarray
    valid: np.ndarray
    releases: list[str] = field(default_factory=list)


def week_nowcast(
    snap: SnapshotArrays, r: int, g: np.ndarray, age: np.ndarray, gap: int
) -> np.ndarray:
    """Return baselinenowcast's count of each hub week `gap` days after `r`.

    Each hub week is a reference time of a weekly reporting triangle,
    with a column every 7 days from the cut-off of release `r` and a last
    one at `gap`.
    The share of the week reported by a column is the mean, over the
    week's days observed at `r`, of their growth from that column to
    `gap` inverted.
    R's `apply_delay` fills each row from these shares (`fill_row`).

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r : int
        The release.
    g : numpy.ndarray
        Daily log factors at `r`, `[L, D]`.
    age : numpy.ndarray
        Age of each day at `r`, `[T]`.
    gap : int
        Days from the cut-off of `r` to the count nowcast. `max_delay`
        or more gives the final count.

    Returns
    -------
    numpy.ndarray
        `[L, W]`, 0 for weeks with no day observed at `r`.
    """
    seen = np.broadcast_to(snap.observed[r], snap.counts[:, r, :].shape)
    x = week_sum(snap, np.where(seen, snap.counts[:, r, :], 0.0))
    n = week_sum(snap, seen.astype(float))
    steps = np.r_[np.arange(0, gap, 7), gap]
    share = np.stack(
        [
            week_sum(
                snap, np.where(seen, 1.0 / growth(g, age + s, age + gap), 0)
            )
            for s in steps
        ],
        axis=-1,
    )
    cdf = np.where(n[..., None] > 0, share / np.maximum(n, 1)[..., None], 1.0)
    return np.where(n > 0, fill_row(x, cdf), 0.0)


DelayFn = Callable[[SnapshotArrays, int, Settings], DelayFit | None]
"""Fits the delay at a release, as `fit_delay`."""


def retro_errors(
    snap: SnapshotArrays, o: int, settings: Settings, delay: DelayFn = fit_delay
) -> Errors:
    """Collect the errors of nowcasts made at earlier releases.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    settings : Settings
        Settings.
    delay : callable
        Fits the delay at each earlier release.

    Returns
    -------
    Errors
        Sums over the releases that give errors.
    """
    n_loc = len(snap.locations)
    k_n = weeks_back(snap, o, settings.max_delay)
    cells: list[tuple[np.ndarray, ...]] = []
    releases: list[str] = []
    for r in range(1, o):
        fit = delay(snap, r, settings)
        if fit is None:
            continue
        age = snap.cut[r] - np.arange(snap.week.size)
        ahead = week_nowcast(snap, r, fit.g, age, snap.cut[o] - snap.cut[r])
        wa, k_r = snap.cut_week(r), weeks_back(snap, r, settings.max_delay)
        cols = wa - np.arange(k_n)
        take = (cols >= 0) & (np.arange(k_n) < k_r)
        safe = np.clip(cols, 0, None)
        obs = weekly_sums(snap, r)[:, safe]
        pred = ahead[:, safe]
        # The week's change to the origin, on days up to the cut-off of r.
        change = weekly_sums(snap, o, upto=int(snap.cut[r]))[:, safe] - obs
        added, removed = np.maximum(change, 0.0), np.maximum(-change, 0.0)
        valid = take[None, :] & ((obs > 0) | (pred > 0))
        cells.append((added, removed, pred - obs, obs, valid))
        releases.append(snap.releases[r])
    c = np.array(cells, dtype=float).reshape(-1, 5, n_loc, k_n)
    return Errors(c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4] > 0, releases)


@dataclass
class Origin:
    r"""Everything the random walk models need at one release.

    Attributes
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    settings : Settings
        Settings.
    fit : DelayFit
        Delay factors at the origin.
    growth : numpy.ndarray
        $G$ for each place and day, `[L, T]`.
    reported : numpy.ndarray
        Counts reported at the origin by hub week, `[L, W]`.
    point : numpy.ndarray
        Nowcast mean by hub week, `[L, W]`.
    complete : numpy.ndarray
        Estimated share reported, $\pi$, `[L, W]`.
    start : numpy.ndarray
        Week index where `randomwalk` starts, `[L]`.
    sigma : numpy.ndarray
        Spread of the walk, `[L]`.
    sigma_source : numpy.ndarray
        Index of the estimate behind each spread, `[L]`, as for
        `source_index`.
    silent : numpy.ndarray
        Places with no case reported in the latest `walk_weeks` weeks,
        `[L]`.
    intro : numpy.ndarray
        Weekly probability of a first case after `walk_weeks` silent
        weeks, pooled over the place's siblings
        (`introduction_rates`), `[L]`.
    errors : Errors or None
        Retrospective errors, or None where they were not asked for.
    size : numpy.ndarray or None
        Negative binomial size $\phi_k$ of the new reports of each place
        by weeks back from the cut-off week, `[L, K]`.
    removal : numpy.ndarray or None
        Removal rate $b_k$, `[L, K]`.
    obs_n : numpy.ndarray or None
        Errors behind each fit, `[L, K]`.
    obs_source : numpy.ndarray or None
        Index of the estimate behind each fit, `[L, K]`.
    """

    snap: SnapshotArrays
    o: int
    settings: Settings
    fit: DelayFit
    growth: np.ndarray
    reported: np.ndarray
    point: np.ndarray
    complete: np.ndarray
    start: np.ndarray
    sigma: np.ndarray
    sigma_source: np.ndarray
    silent: np.ndarray
    intro: np.ndarray
    errors: Errors | None = None
    size: np.ndarray | None = None
    removal: np.ndarray | None = None
    obs_n: np.ndarray | None = None
    obs_source: np.ndarray | None = None

    @property
    def start_nowcast(self) -> int:
        """Latest week whose seven days are all on or before the cut-off."""
        sat = np.array(
            [
                np.flatnonzero(self.snap.week == w).max()
                for w in range(self.snap.n_weeks)
            ]
        )
        full = np.flatnonzero(sat <= self.snap.cut[self.o])
        return int(full.max()) if full.size else 0


def build_origin(
    snap: SnapshotArrays,
    settings: Settings,
    errors: bool,
    delay: DelayFn = fit_delay,
) -> Origin:
    """Fit the delay and the walk spread at the latest release.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays. The origin is the latest release.
    settings : Settings
        Settings.
    errors : bool
        Whether to fit the observation model.
    delay : callable
        Fits the delay at a release.

    Returns
    -------
    Origin
        The fit.

    Raises
    ------
    NoDelayPair
        If no pair of releases has a step within the delay window.
    """
    o = len(snap.releases) - 1
    fit = delay(snap, o, settings)
    if fit is None:
        raise NoDelayPair(
            f"No pair of releases up to {snap.releases[o]} has a step of at "
            f"most {settings.max_delay} days."
        )
    g = final_growth(snap, o, fit, settings)
    reported = weekly_sums(snap, o)
    age = snap.cut[o] - np.arange(snap.week.size) + settings.clock_shift
    point = week_nowcast(snap, o, fit.g, age, settings.max_delay)
    inv = np.where(snap.observed[o][None, :], 1.0 / g, 0.0)
    complete = week_sum(snap, inv) / 7.0
    good = complete >= settings.completeness
    start = np.where(
        good.any(axis=1), snap.n_weeks - 1 - np.argmax(good[:, ::-1], axis=1), 0
    )
    own, pooled = _walk_spreads(reported, start, snap, fit.level, settings)
    both = np.concatenate([own, pooled])
    sigma_source = fit.source
    sigma = both[sigma_source]
    k = settings.walk_weeks
    silent = silent_places(reported, snap.cut_week(o), k)
    intro = introduction_rates(reported, start, snap.parent, k)
    origin = Origin(
        snap, o, settings, fit, g, reported, point, complete, start, sigma,
        sigma_source, silent, intro,
    )  # fmt: skip
    if errors:
        _add_spread(origin)
    return origin


def walk_changes(
    reported: np.ndarray, start: np.ndarray, settings: Settings
) -> list[np.ndarray]:
    """Return each place's weekly changes of log(1 + y) up to its start week.

    Parameters
    ----------
    reported : numpy.ndarray
        Weekly reported counts, `[L, W]`.
    start : numpy.ndarray
        Start week of each place, `[L]`.
    settings : Settings
        Settings, for `walk_weeks`.

    Returns
    -------
    list of numpy.ndarray
        The changes of each place over the `walk_weeks` weeks ending at
        its start week.
    """
    diffs = []
    for n in range(reported.shape[0]):
        lo = max(int(start[n]) - settings.walk_weeks + 1, 0)
        diffs.append(np.diff(np.log1p(reported[n, lo : int(start[n]) + 1])))
    return diffs


def _walk_spreads(
    reported: np.ndarray,
    start: np.ndarray,
    snap: SnapshotArrays,
    level: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each place's own and pooled spread of the weekly log changes."""
    n_loc = reported.shape[0]
    diffs = walk_changes(reported, start, settings)

    def sd(x: np.ndarray) -> float:
        return float(np.std(x, ddof=1)) if x.size >= 2 else 0.0

    own = np.array([sd(d) for d in diffs])
    pool = pool_matrix(level, snap.parent, settings.min_level)
    pooled = np.array(
        [
            sd(np.concatenate([diffs[m] for m in np.flatnonzero(pool[n])]))
            for n in range(n_loc)
        ]
    )
    return own, pooled


def _add_spread(origin: Origin) -> None:
    """Fit the observation model by weeks back.

    The size follows the one rule. Each place's own estimate uses its
    errors, and its pooled estimate the errors of the places in its
    pool. The size is fitted only for the estimates some place uses.
    The removal rate is each place's own, so a place with no falls has
    none.
    """
    snap, settings = origin.snap, origin.settings
    errors = retro_errors(snap, origin.o, settings)
    n_loc = len(snap.locations)
    pool = pool_matrix(origin.fit.level, snap.parent, settings.min_level)
    member = np.concatenate([np.eye(n_loc), pool]) > 0  # [2L, L]
    # A cell with nothing nowcast to come says nothing about the size.
    fit = errors.valid & (errors.remaining > 0)
    n = np.einsum("el,rlk->ek", member, fit, dtype=int)
    removed = (errors.removed * errors.valid).sum(axis=0)
    reported = (errors.reported * errors.valid).sum(axis=0)
    source = source_index(
        origin.fit.level,
        snap.parent,
        n[:n_loc],
        n[n_loc:],
        settings.min_level,
        settings.min_errors,
    )
    size = np.zeros(n.shape)
    for e in np.unique(source):
        keep = fit & member[e][None, :, None]
        k_n = keep.shape[-1]
        size[e] = fit_sizes(
            np.where(keep, errors.added, 0.0).reshape(-1, k_n),
            np.where(keep, errors.remaining, 0.0).reshape(-1, k_n),
            settings.min_errors,
        )
    k = np.arange(n.shape[1])[None, :]
    origin.errors = errors
    origin.size = size[source, k]
    origin.removal = rate(removed, reported)
    origin.obs_n, origin.obs_source = n[source, k], source
