# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""`randomwalknowcast` on known onsets, scaled up for missing onsets.

Experimental.
`RandomWalkNowcastKnown` (`rwnc_known`) is `randomwalknowcast` run on the
onsets known at each release, the daily table's `observed` column.
Its chain ladder delay, observation model and walk are all fitted to
known onsets, so the delay is not shaped by the hub's imputation, and a
case that moves from imputed to known between releases is new growth of
the known count rather than growth at one age and shrinkage at another.

The hub's target is known plus imputed onsets.
Each place's trajectories of known onsets are scaled up by its ratio

$$
\rho = \frac{\sum_w (y^{obs}_w + y^{imp}_w)}{\sum_w y^{obs}_w}
$$

at the origin release, over the hub weeks `share_weeks` back from the
cut-off week. By default these are weeks 6 to 9 back, at or past the
42-day delay, so their known counts are final for the chain ladder.
A place uses its own ratio where it has at least `Settings.min_level`
known onsets a week over those weeks, and otherwise the pooled ratio of
its nearest parent that does (`pooling`), as the delay and spread do.
The ratio is at least 1, and 1 where no place has a known onset.

Given $K$ known onsets in a sample, the cases without an onset are drawn
as $M \sim \mathrm{NegBin}(K, 1 / \rho)$, the failures before $K$
successes, which has mean $K (\rho - 1)$.
This is the count of cases without an onset when each case lacks one
independently with probability $1 - 1 / \rho$.
The uncertainty in $\rho$ itself is not propagated.
Each data week is then floored at the total count the hub reports.

`delay_summary`, which maps final counts to what later releases show in
the release backtest, comes from the chain ladder on the hub's total
counts, as for `randomwalknowcast`, because the samples are of the total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.snapshots import snapshot_arrays, weekly_sums
from bdbvhubrenewal.forecasters.base import ForecastTask, register_model
from bdbvhubrenewal.forecasters.chain import build_origin, delay_summary
from bdbvhubrenewal.forecasters.pooling import (
    MIN_LEVEL,
    pool_matrix,
    source_index,
)
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalkNowcast,
    _start_weeks,
    frame,
)

__all__ = ["RandomWalkNowcastKnown", "add_missing", "missing_ratio"]


def missing_ratio(
    known: np.ndarray,
    total: np.ndarray,
    parent: np.ndarray,
    min_level: float = MIN_LEVEL,
) -> np.ndarray:
    """Return each place's ratio of total to known onsets.

    Parameters
    ----------
    known, total : numpy.ndarray
        Weekly known and total (known plus imputed) onsets over the weeks
        the ratio uses, `[L, W]`.
    parent : numpy.ndarray
        Index of each place's parent, or -1, `[L]`.
    min_level : float
        Known onsets a week for a place to use its own ratio.

    Returns
    -------
    numpy.ndarray
        `[L]`, each place's own ratio where its mean weekly known count is
        at least `min_level`, and otherwise the pooled ratio of its nearest
        parent that qualifies (`source_index`). At least 1, and 1 where
        the ratio has no known onset.
    """
    k, t = known.sum(axis=1), total.sum(axis=1)
    level = known.mean(axis=1) if known.shape[1] else np.zeros(len(k))
    pool = pool_matrix(level, parent, min_level)
    num = np.concatenate([t, pool @ t])
    den = np.concatenate([k, pool @ k])
    both = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    source = source_index(level, parent, min_level=min_level)
    return np.asarray(np.maximum(both[source], 1.0))


def add_missing(
    rng: np.random.Generator, known: np.ndarray, ratio: np.ndarray
) -> np.ndarray:
    r"""Return known onset samples plus drawn cases without an onset.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random numbers.
    known : numpy.ndarray
        Known onset samples `[S, L, H]`, rounded and floored at 0 first.
    ratio : numpy.ndarray
        Ratio of total to known onsets of each place, `[L]`, at least 1.

    Returns
    -------
    numpy.ndarray
        $K + M$ with $M \sim \mathrm{NegBin}(K, 1 / \rho)$, `[S, L, H]`.
    """
    k = np.maximum(np.round(known), 0.0)
    p = np.broadcast_to((1.0 / ratio)[None, :, None], k.shape)
    drawn = rng.negative_binomial(np.maximum(k, 1.0), p)
    return k + np.where(k > 0, drawn, 0.0)


@dataclass
class RandomWalkNowcastKnown(RandomWalkNowcast):
    """`randomwalknowcast` on known onsets, scaled up to the total.

    Experimental. See the module docstring.

    Attributes
    ----------
    name : str
        Model name.
    share_weeks : tuple of int
        First and last weeks back from the cut-off week whose total and
        known counts set the ratio.
    """

    name: str = "rwnc_known"
    share_weeks: tuple[int, int] = (6, 9)
    onsets: ClassVar[tuple[str, ...]] = ("observed",)

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return scaled up trajectories from a known onset nowcast.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `daily_history` holds the
            release snapshots.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no snapshots, or no pair of releases within
            the delay window.
        """
        origin = self.origin_at(task, errors=True)
        known, weeks = self.trajectories(origin, task, origin.point)
        assert task.daily_history is not None
        total_snap = snapshot_arrays(
            task.daily_history, task.vintage, task.locations
        )
        total = weekly_sums(total_snap, origin.o)
        cw = origin.snap.cut_week(origin.o)
        lo, hi = self.share_weeks
        cols = [cw - k for k in range(lo, hi + 1) if cw - k >= 0]
        ratio = missing_ratio(
            origin.reported[:, cols],
            total[:, cols],
            origin.snap.parent,
            origin.settings.min_level,
        )
        values = add_missing(task.rng(self.name), known, ratio)
        n_data = origin.snap.n_weeks
        values[:, :, :n_data] = np.maximum(values[:, :, :n_data], total[None])
        total_origin = build_origin(total_snap, self.settings, errors=False)
        self.delay_summary = delay_summary(
            total_origin, self.name, task.vintage
        )
        self._diagnostics(origin, task, _start_weeks(origin))
        for row, i in zip(
            self.fit_diagnostics, self.rows(origin, task), strict=True
        ):
            row["missing_ratio"] = float(ratio[i])
        return frame(task, weeks, values, self.rows(origin, task))


register_model(
    "rwnc_known",
    RandomWalkNowcastKnown,
    "Experimental: randomwalknowcast on known onsets, scaled up for "
    "cases without an onset.",
)
