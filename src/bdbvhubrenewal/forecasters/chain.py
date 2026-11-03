# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Daily chain ladder on release snapshots, with its uncertainty.

Counts $N_r(t)$ are the daily onsets of release $r$, on the onset clock.
Release $r$ has cut-off $c_r$, and day $t$ is $a = c_r - t$ days old.
A pair of consecutive releases $(r, r + 1)$ has step
$s = c_{r+1} - c_r$ days and is in window when $s \le D$.

**Delay.** For step $s$ and age $x = 0, \dots, D - 1$,

$$
F_s(x) = \frac{\sum N_{r+1}(t)}{\sum N_r(t)},
$$

summed over all in-window pairs and over days of age $x$ that both
releases observe.
Each pair is one step at its own exact ages, and the pooled factor is a
ratio of sums, not a mean of ratios.
Cells a release does not observe are skipped, not filled.
A place uses its own factor at an age when the denominator, summed over
the pooled pairs, is at least 10 cases, and otherwise its parent's factor
at that age (zone, province, country).
The country uses its own factor.
Only onset days within `Settings.history_delay` (63 for $D = 42$) of the
latest cut-off are summed, and downward revisions are first moved onto
earlier releases of the day, as baselinenowcast does.
A zero denominator takes baselinenowcast's rule, so that
$F_s(x) = 1 + (\sum N_{r+1}(t) - \sum N_r(t)) / \max(\sum N_r(t), 1)$.

**Nowcast.** At the latest release $o$, with $s_o$ the step of its latest
in-window pair, day $t$ is chained over the actual release gap,

$$
\mu(t) = N_o(t) \prod_{m=0}^{M-1} F_{s_o}(a + m s_o),
$$

with $M$ the fewest steps taking the age to $D$ or more.
Every day with data is nowcast.
Daily nowcasts are summed to hub weeks.

**Uncertainty.** A release $r < o$ with an in-window pair at $r$ nowcasts
the count expected at $o$ for days up to $c_r$.
It chains over the gap $c_o - c_r$ only, using the nearest earlier pair's
factor for each remaining part of the gap.
Only plans within `max_mismatch` days of the gap are kept.
Its new reports and removals by weeks back fit the observation model
(`obs`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from bdbvhubrenewal.baselinenowcast import (
    chain_growth,
    exact_plan,
    pair_factors,
    pair_sums,
    preprocess_negative_values,
    window_pairs,
)
from bdbvhubrenewal.data.snapshots import SnapshotArrays, week_sum, weekly_sums
from bdbvhubrenewal.forecasters.base import ModelUnavailableError
from bdbvhubrenewal.forecasters.obs import fit_sizes, rate
from bdbvhubrenewal.forecasters.pooling import (
    MIN_ERRORS,
    MIN_LEVEL,
    pool_matrix,
    source_index,
)

__all__ = [
    "MissingSnapshots",
    "NoDelayPair",
    "Origin",
    "Settings",
    "build_origin",
    "final_plan",
    "retro_plan",
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
    max_mismatch : int
        Days by which a retrospective plan may differ from the gap.
    completeness : float
        Share of a week's final count that must be reported for it to
        start the random walk.
    walk_weeks : int
        Weeks, ending at the start week, whose changes set the spread.
    clock_shift : int
        Days added to the ages of the origin's days, for the sensitivity
        of the nowcast to the onset clock. Ages below 0 count as 0. The
        delay factors and the retrospective errors are not moved.
    """

    max_delay: int = 42
    min_level: float = MIN_LEVEL
    min_errors: int = MIN_ERRORS
    level_weeks: tuple[int, int] = (3, 6)
    max_mismatch: int = 1
    completeness: float = 0.95
    walk_weeks: int = 9
    clock_shift: int = 0

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
    """Delay factors at a release, as each place uses them.

    Attributes
    ----------
    pairs : numpy.ndarray
        Earlier and later release of each in-window pair, `[P, 2]`.
    steps : numpy.ndarray
        Distinct steps in days, `[S]`.
    pair_step : numpy.ndarray
        Index in `steps` of each pair, `[P]`.
    factors : numpy.ndarray
        $F$ for each place, `[L, S, D]`. Every step pools all the
        in-window pairs, so the steps share one factor. A place uses its
        own factor at an age when the pooled denominator there is at
        least `Settings.min_level` cases, and otherwise its parent's.
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
    steps: np.ndarray
    pair_step: np.ndarray
    factors: np.ndarray
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


class ChainPlan(NamedTuple):
    """Chain steps for each onset day, padded to a common length.

    Attributes
    ----------
    step : numpy.ndarray
        Slot of the step size, `[T, M]`.
    age : numpy.ndarray
        Age in days at each step, `[T, M]`.
    valid : numpy.ndarray
        Whether the step is used, `[T, M]`.
    """

    step: np.ndarray
    age: np.ndarray
    valid: np.ndarray


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
    pairs = _pairs_at(snap, r, settings.max_delay)
    if pairs.size == 0:
        return None
    counts = _delay_counts(snap, r)
    day = np.arange(snap.week.size)
    window = day > int(snap.cut[r]) - settings.history_delay
    observed = snap.observed & window[None, :]
    gaps = snap.cut[pairs[:, 1]] - snap.cut[pairs[:, 0]]
    steps = np.unique(gaps)
    pair_step = np.searchsorted(steps, gaps)
    num, den = pair_sums(
        jnp.asarray(counts),
        jnp.asarray(observed),
        jnp.asarray(snap.cut),
        jnp.asarray(pairs[:, 0]),
        settings.max_delay,
    )
    every = jnp.ones((len(pairs), len(steps)))
    f = np.array(pair_factors(num, den, every))
    # With every pair in every step, the denominator is the same for all.
    own = np.array(den).sum(axis=1) >= settings.min_level
    own[snap.parent < 0] = True
    # Nodes are sorted by code length, so a parent precedes its children.
    for n in range(f.shape[0]):
        p = snap.parent[n]
        if p >= 0:
            f[n] = np.where(own[n][None, :], f[n], f[p])
    level = levels(snap, r, settings)
    source = source_index(level, snap.parent, min_level=settings.min_level)
    return DelayFit(pairs, steps, pair_step, f, level, source, own)


def final_plan(
    snap: SnapshotArrays, o: int, fit: DelayFit, settings: Settings
) -> ChainPlan:
    """Chain each day over the steps of the latest pair, with no tail.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    fit : DelayFit
        Factors at the origin.
    settings : Settings
        Settings.

    Returns
    -------
    ChainPlan
        For each day, the steps from its age until the age reaches $D$.
    """
    last = len(fit.pairs) - 1
    s = int(fit.steps[fit.pair_step[last]])
    age0 = snap.cut[o] - np.arange(snap.week.size)
    seen = age0 >= 0
    shifted = np.maximum(age0[seen] + settings.clock_shift, 0)
    slot, age, mask = exact_plan(
        shifted, s, int(fit.pair_step[last]), settings.max_delay
    )
    m = slot.shape[1]
    step = np.zeros((age0.size, m), dtype=np.int64)
    ages = np.zeros((age0.size, m), dtype=np.int64)
    valid = np.zeros((age0.size, m), dtype=bool)
    step[seen], ages[seen], valid[seen] = slot, age, mask
    return ChainPlan(step, ages, valid)


def retro_plan(
    snap: SnapshotArrays, r: int, o: int, fit: DelayFit, settings: Settings
) -> tuple[ChainPlan, int]:
    """Chain each day from release `r` over the gap to release `o`.

    Each part of the remaining gap uses the pair whose step is nearest
    (ties to the latest pair). Ages are never interpolated.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    r, o : int
        The earlier release and the origin.
    fit : DelayFit
        Factors at release `r`.
    settings : Settings
        Settings.

    Returns
    -------
    tuple
        The plan, and the days by which its steps differ from the gap.
    """
    d = settings.max_delay
    gap = int(snap.cut[o] - snap.cut[r])
    sizes = snap.cut[fit.pairs[:, 1]] - snap.cut[fit.pairs[:, 0]]
    recency = r - fit.pairs[:, 0]
    seq: list[int] = []
    left = gap
    while left > 0:
        j = int(np.argmin(np.abs(sizes - left) + 1e-3 * recency))
        seq.append(j)
        left -= int(sizes[j])
    age0 = snap.cut[r] - np.arange(snap.week.size)
    taken = np.concatenate([[0], np.cumsum(sizes[seq])])[: len(seq)]
    age = age0[:, None] + taken[None, :]
    valid = (age0 >= 0)[:, None] & (age < d)
    step = np.broadcast_to(fit.pair_step[seq], age.shape)
    mismatch = abs(int(sizes[seq].sum()) - gap)
    plan = ChainPlan(step, age, valid)
    return plan, mismatch


def grow(fit: DelayFit, plan: ChainPlan) -> np.ndarray:
    """Return growth `[L, T]`."""
    return np.array(
        chain_growth(fit.factors, (plan.step, plan.age, plan.valid))
    )


@dataclass
class Errors:
    """Retrospective nowcasts by release, place and weeks back.

    Attributes
    ----------
    added, removed : numpy.ndarray
        New reports and removals between each release and the origin,
        `[R, L, K]`.
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


def retro_errors(snap: SnapshotArrays, o: int, settings: Settings) -> Errors:
    """Collect the errors of nowcasts made at earlier releases.

    Parameters
    ----------
    snap : SnapshotArrays
        The arrays.
    o : int
        The origin release.
    settings : Settings
        Settings.

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
        fit = fit_delay(snap, r, settings)
        if fit is None:
            continue
        plan, mismatch = retro_plan(snap, r, o, fit, settings)
        if mismatch > settings.max_mismatch:
            continue
        counts = snap.counts[:, r, :]
        g = grow(fit, plan)
        wa, k_r = snap.cut_week(r), weeks_back(snap, r, settings.max_delay)
        cols = wa - np.arange(k_n)
        take = (cols >= 0) & (np.arange(k_n) < k_r)
        safe = np.clip(cols, 0, None)
        obs = weekly_sums(snap, r)[:, safe]
        pred = week_sum(snap, counts * g)[:, safe]
        upto = np.arange(snap.week.size) <= snap.cut[r]
        change = (snap.counts[:, o, :] - counts) * upto
        added = week_sum(snap, np.maximum(change, 0.0))[:, safe]
        removed = week_sum(snap, np.maximum(-change, 0.0))[:, safe]
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
    snap: SnapshotArrays, settings: Settings, errors: bool
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
    fit = fit_delay(snap, o, settings)
    if fit is None:
        raise NoDelayPair(
            f"No pair of releases up to {snap.releases[o]} has a step of at "
            f"most {settings.max_delay} days."
        )
    plan = final_plan(snap, o, fit, settings)
    counts = snap.counts[:, o, :]
    g = grow(fit, plan)
    reported = weekly_sums(snap, o)
    point = week_sum(snap, counts * g)
    # A chain with a zero factor revises the day to nothing, so it counts
    # as fully reported.
    share = 1.0 / np.where(g > 0, g, 1.0)
    inv = np.where(snap.observed[o][None, :], share, 0.0)
    complete = week_sum(snap, inv) / 7.0
    good = complete >= settings.completeness
    start = np.where(
        good.any(axis=1), snap.n_weeks - 1 - np.argmax(good[:, ::-1], axis=1), 0
    )
    own, pooled = _walk_spreads(reported, start, snap, fit.level, settings)
    both = np.concatenate([own, pooled])
    sigma_source = fit.source
    sigma = both[sigma_source]
    origin = Origin(
        snap, o, settings, fit, g, reported, point, complete, start, sigma,
        sigma_source,
    )  # fmt: skip
    if errors:
        _add_spread(origin)
    return origin


def _walk_spreads(
    reported: np.ndarray,
    start: np.ndarray,
    snap: SnapshotArrays,
    level: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each place's own and pooled spread of the weekly log changes."""
    n_loc = reported.shape[0]
    diffs = []
    for n in range(n_loc):
        lo = max(int(start[n]) - settings.walk_weeks + 1, 0)
        x = np.log1p(reported[n, lo : int(start[n]) + 1])
        diffs.append(np.diff(x))

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
    """Fit the observation model by weeks back, with the one rule.

    Each place's own estimate uses its errors, and its pooled estimate
    the errors of the places in its pool. The size is fitted only for
    the estimates some place uses.
    """
    snap, settings = origin.snap, origin.settings
    errors = retro_errors(snap, origin.o, settings)
    n_loc = len(snap.locations)
    pool = pool_matrix(origin.fit.level, snap.parent, settings.min_level)
    member = np.concatenate([np.eye(n_loc), pool]) > 0  # [2L, L]
    # A cell with nothing nowcast to come says nothing about the size.
    fit = errors.valid & (errors.remaining > 0)
    n = np.einsum("el,rlk->ek", member, fit, dtype=int)
    removed = np.einsum("el,rlk->ek", member, errors.removed * errors.valid)
    reported = np.einsum("el,rlk->ek", member, errors.reported * errors.valid)
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
    origin.removal = rate(removed, reported)[source, k]
    origin.obs_n, origin.obs_source = n[source, k], source
