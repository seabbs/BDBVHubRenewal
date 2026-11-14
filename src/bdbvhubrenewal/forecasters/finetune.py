# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Training blocks for fine-tuning Chronos-2 on simulated outbreaks.

See the Foundation models methods page for the training data.
`scripts/finetune_simulate.R` simulates each outbreak's reporting
triangle with epinowcast. `releases` turns a triangle into the hub
target's weekly or daily releases, and `training_block` cuts those
releases at one release and lays them out with `foundation.hub_triangle`
and the ragged layout's block, so lags, $D$ and the cut-off match
inference. The block's
target runs past the cut-off with the outbreak's final counts, which is
what Chronos-2 learns to forecast.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.dates import as_date, epiweek_end
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.foundation import hub_triangle, layout_block

__all__ = [
    "AHEAD_WEEKS",
    "final_counts",
    "releases",
    "training_block",
    "weekly_cutoffs",
]

AHEAD_WEEKS: Final = 4
"""Weeks after the cut-off's week to the last week forecast, as for a
forecast's last horizon."""


def releases(
    counts: pd.DataFrame, source: str, cutoffs: Sequence[str]
) -> pd.DataFrame:
    """Return the hub target of each release of a simulated outbreak.

    A release with cut-off $c$ counts every case reported by $c$. The
    country is the sum of the places, and a single place is the country.

    Parameters
    ----------
    counts : pandas.DataFrame
        One outbreak's reporting triangle: `location`, `reference_date`
        (onset day or week), `report_date` and `count`.
    source : str
        `"weekly"` (rows by onset week) or `"daily"` (by onset day).
    cutoffs : sequence of str
        The releases' cut-offs, ISO format.

    Returns
    -------
    pandas.DataFrame
        Rows `hub_triangle` reads: `vintage`, `cutoff`, `location`, and
        `target_end_date` and `total` (weekly) or `onset_date`, `observed`
        and `imputed` (daily).
    """
    counts = counts.assign(
        reference_date=pd.to_datetime(counts["reference_date"]),
        report_date=pd.to_datetime(counts["report_date"]),
    )
    country = counts.assign(location=NATIONAL)
    if len(counts["location"].unique()) > 1:
        counts = pd.concat([country, counts], ignore_index=True)
    else:
        counts = country
    if source == "weekly":
        days = counts["reference_date"].dt.day_of_week
        counts["reference_date"] += pd.to_timedelta((5 - days) % 7, unit="D")
    frames = []
    for cutoff in cutoffs:
        seen = counts.loc[counts["report_date"] <= pd.Timestamp(cutoff)]
        rows = seen.groupby(["location", "reference_date"], as_index=False)[
            "count"
        ].sum()
        rows.insert(0, "vintage", cutoff)
        rows.insert(1, "cutoff", cutoff)
        frames.append(rows)
    out = pd.DataFrame(pd.concat(frames, ignore_index=True))
    out["reference_date"] = out["reference_date"].dt.strftime("%Y-%m-%d")
    if source == "weekly":
        return out.rename(
            columns={"reference_date": "target_end_date", "count": "total"}
        )
    out["imputed"] = 0
    return out.rename(
        columns={"reference_date": "onset_date", "count": "observed"}
    )


def final_counts(
    counts: pd.DataFrame,
    source: str,
    places: Sequence[str],
    times: Sequence[str],
) -> np.ndarray:
    """Return each place's count of each step with every report in.

    Parameters
    ----------
    counts : pandas.DataFrame
        As `releases`.
    source : str
        `"weekly"` or `"daily"`.
    places : sequence of str
        Places, the country first if there are several.
    times : sequence of str
        Steps of the grid.

    Returns
    -------
    numpy.ndarray
        `[L, T]`.
    """
    last = str(pd.to_datetime(counts["report_date"]).max().date())
    rows = releases(counts, source, [last])
    tri = hub_triangle(rows, source, places, times[-1], max_lag=1)
    at = {t: i for i, t in enumerate(tri.times)}
    out = np.zeros((len(places), len(times)))
    for j, t in enumerate(times):
        if t in at:
            out[:, j] = tri.final[:, at[t]]
    return out


def training_block(
    counts: pd.DataFrame,
    source: str,
    places: Sequence[str],
    cutoffs: Sequence[str],
    max_lag: int,
    ahead_weeks: int = AHEAD_WEEKS,
) -> dict[str, Any]:
    """Return one training block, cut at the last of `cutoffs`.

    The block is the ragged layout's group of series at the cut-off, from
    `hub_triangle` and the layout's block, with the target extended over
    the horizon by the outbreak's final counts.

    Parameters
    ----------
    counts : pandas.DataFrame
        As `releases`. Every onset step up to the horizon's end must be
        complete in it.
    source : str
        `"weekly"` or `"daily"`.
    places : sequence of str
        The places of the block, the country first if there are several.
    cutoffs : sequence of str
        Cut-offs of the releases up to the cut, ISO format.
    max_lag : int
        Release lag $D$ in the source's steps.
    ahead_weeks : int
        Weeks after the cut-off's week to the last step.

    Returns
    -------
    dict
        Chronos-2's prepared input: `context` `[V, S + H]`, the targets
        first and then the lag series, `future_covariates` `[V, H]`,
        `n_targets`, `n_covariates` and `n_future_covariates`, and the
        `start` $S$ and `horizon` $H$.
    """
    cut = cutoffs[-1]
    end = epiweek_end(cut) + dt.timedelta(weeks=ahead_weeks)
    rows = releases(counts, source, cutoffs)
    tri = hub_triangle(rows, source, places, end.isoformat(), max_lag=max_lag)
    start = tri.n_context
    horizon = len(tri.times) - start
    group = layout_block(tri, "ragged", places, start)
    truth = final_counts(counts, source, places, tri.times)
    names = list(group["past_covariates"])
    lagged = np.stack(
        [
            np.concatenate(
                [group["past_covariates"][n], group["future_covariates"][n]]
            )
            for n in names
        ]
    )
    target = np.concatenate([group["target"], truth[:, start:]], axis=1)
    context = np.concatenate([target, lagged]).astype(np.float32)
    return {
        "context": context,
        "future_covariates": context[:, start:].copy(),
        "n_targets": len(places),
        "n_covariates": len(names),
        "n_future_covariates": len(names),
        "start": start,
        "horizon": horizon,
    }


def weekly_cutoffs(first: str, last: str, weekday: int) -> list[str]:
    """Return weekly cut-offs on a weekday (Monday 0) from `first` to `last`.

    Parameters
    ----------
    first, last : str
        The range, ISO format.
    weekday : int
        Day of the week of each cut-off.

    Returns
    -------
    list of str
    """
    day = as_date(first)
    day += dt.timedelta(days=(weekday - day.weekday()) % 7)
    out = []
    while day <= as_date(last):
        out.append(day.isoformat())
        day += dt.timedelta(days=7)
    return out
