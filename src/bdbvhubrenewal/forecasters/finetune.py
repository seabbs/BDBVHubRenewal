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
from bdbvhubrenewal.forecasters.foundation import (
    HubTriangle,
    hub_triangle,
    layout_block,
)

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
    counts: pd.DataFrame,
    source: str,
    cutoffs: Sequence[str],
    totals: bool = False,
    country: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the hub target of each release of a simulated outbreak.

    A release with cut-off $c$ counts every case reported by $c$. The
    country is the sum of the places, and a single place is the country.
    With `totals`, a place `P` whose strata are `P-<stratum>` also has its
    own total, as a province has the total of its zones, and an age group
    `A<k>` has its total across places. Every total is summed from the
    same counts, by onset and report date.

    Parameters
    ----------
    counts : pandas.DataFrame
        One outbreak's reporting triangle: `location`, `reference_date`
        (onset day or week), `report_date` and `count`.
    source : str
        `"weekly"` (rows by onset week) or `"daily"` (by onset day).
    cutoffs : sequence of str
        The releases' cut-offs, ISO format.
    totals : bool
        Add each place's total of its strata.
    country : pandas.DataFrame, optional
        The counts the country is summed from, as `counts`, when it is not
        the sum of the places (`country_counts`).

    Returns
    -------
    pandas.DataFrame
        Rows `hub_triangle` reads: `vintage`, `cutoff`, `location`, and
        `target_end_date` and `total` (weekly) or `onset_date`, `observed`
        and `imputed` (daily).
    """

    def dated(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            reference_date=pd.to_datetime(frame["reference_date"]),
            report_date=pd.to_datetime(frame["report_date"]),
        )

    counts = dated(counts)
    country = dated(counts if country is None else country).assign(
        location=NATIONAL
    )
    if len(counts["location"].unique()) > 1:
        frames = [country, counts]
        if totals:
            parts = counts["location"].str.split("-")
            nested = counts["location"].str.contains("-")
            frames.append(counts.loc[nested].assign(location=parts.str[0]))
            # Strata that are age groups also total across places.
            ages = nested & parts.str[1].str.fullmatch(r"A\d+").fillna(False)
            frames.append(counts.loc[ages].assign(location=parts.str[1]))
        counts = pd.concat(frames, ignore_index=True)
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
        frames.append(pd.DataFrame(rows))
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
    totals: bool = False,
    country: pd.DataFrame | None = None,
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
    totals : bool
        As `releases`.
    country : pandas.DataFrame, optional
        As `releases`.

    Returns
    -------
    numpy.ndarray
        `[L, T]`.
    """
    reported = pd.to_datetime(counts["report_date"])
    if country is not None:
        reported = pd.concat([reported, pd.to_datetime(country["report_date"])])
    last = str(reported.max().date())
    rows = releases(counts, source, [last], totals, country)
    tri = hub_triangle(rows, source, places, times[-1], max_lag=1)
    at = {t: i for i, t in enumerate(tri.times)}
    out = np.zeros((len(places), len(times)))
    for j, t in enumerate(times):
        if t in at:
            out[:, j] = tri.final[:, at[t]]
    return out


BINNED: Final = {"3day": 3}
"""Training-only sources summed from the daily triangle, by days per
step."""


def bin_ends(tri: HubTriangle, k: int) -> list[int]:
    """Return the last day of each `k`-day bin of a daily triangle.

    Bins end on the day of the latest release's cut-off and every `k` days
    either side of it, and only whole bins are kept.
    """
    first = tri.last % k
    if first < k - 1:
        first += k
    return list(range(first, len(tri.times), k))


def _sum_bins(x: np.ndarray, ends: list[int], k: int) -> np.ndarray:
    """Return `x` `[L, T]` summed over each `k`-day bin ending in `ends`."""
    return np.stack([x[:, e - k + 1 : e + 1].sum(axis=1) for e in ends], 1)


def aggregate_triangle(tri: HubTriangle, k: int, lags: int) -> HubTriangle:
    """Return a daily triangle in steps of `k` days, with `lags` lags.

    A bin's count at lag $d$ is the release $kd$ days after its last day,
    summed over its days: day $e - i$ of a bin ending on day $e$ is at
    daily lag $kd + i$ in that release. It is missing where no release
    had its cut-off then.

    Parameters
    ----------
    tri : HubTriangle
        A daily triangle with at least `k * lags` daily lags.
    k : int
        Days per step.
    lags : int
        Lags of the result.

    Returns
    -------
    HubTriangle
    """
    ends = bin_ends(tri, k)
    partial = np.stack(
        [
            np.stack(
                [
                    sum(tri.partial[:, e - i, k * d + i] for i in range(k))
                    for d in range(lags)
                ],
                axis=-1,
            )
            for e in ends
        ],
        axis=1,
    )
    return HubTriangle(
        times=[tri.times[e] for e in ends],
        final=_sum_bins(tri.final, ends, k),
        partial=partial,
        last=ends.index(tri.last),
    )


PARENTS: Final = ("exact", "unallocated", "late_parent")
"""How a block's country relates to its places (`country_counts`)."""


def country_counts(
    counts: pd.DataFrame, parent: str, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the counts the places and the country are made from.

    `exact`: both the outbreak's, so the country is the sum of its places.
    `unallocated`: a share of 5% to 30% of cases has no place, so it is
    in the country's count only, as cases of unknown province or age.
    `late_parent`: the country is reported through its own stream, 3 to
    14 days after its places.

    Raises
    ------
    ValueError
        If `parent` is none of `PARENTS`.
    """
    if parent == "exact":
        return counts, counts
    if parent == "unallocated":
        keep = 1 - rng.uniform(0.05, 0.3)
        kept = rng.binomial(counts["count"].to_numpy(dtype=int), keep)
        return counts.assign(count=kept), counts
    if parent == "late_parent":
        days = pd.Timedelta(days=int(rng.integers(3, 15)))
        later = pd.to_datetime(counts["report_date"]) + days
        return counts, counts.assign(report_date=later)
    raise ValueError(f"parent {parent!r} is none of {PARENTS}")


def training_block(
    counts: pd.DataFrame,
    source: str,
    places: Sequence[str],
    cutoffs: Sequence[str],
    max_lag: int,
    ahead_weeks: int = AHEAD_WEEKS,
    totals: bool = False,
    parent: str = "exact",
    rng: np.random.Generator | None = None,
) -> dict[str, Any] | None:
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
    totals : bool
        As `releases`, so `places` may hold a place's total and its strata.
    parent : str
        How the country relates to its places (`country_counts`), drawn
        with `rng`.
    rng : numpy.random.Generator, optional
        Draws the `parent` cases; a fixed seed by default.

    Returns
    -------
    dict
        Chronos-2's prepared input: `context` `[V, S + H]`, the targets
        first and then the lag series, `future_covariates` `[V, H]`,
        `n_targets`, `n_covariates` and `n_future_covariates`, and the
        `start` $S$ and `horizon` $H$. None if no release up to the cut
        has a row for any of `places`, as for strata with no cases, or, for a
        binned source, if no whole bin ends at the cut-off.
    """
    cut = cutoffs[-1]
    end = epiweek_end(cut) + dt.timedelta(weeks=ahead_weeks)
    counts, country = country_counts(
        counts, parent, rng or np.random.default_rng(0)
    )
    rows = releases(counts, source, cutoffs, totals, country)
    if not rows["location"].isin(places).any():
        return None
    if source in BINNED:
        k = BINNED[source]
        rows = releases(counts, "daily", cutoffs, totals, country)
        daily = hub_triangle(
            rows, "daily", places, end.isoformat(), max_lag=k * max_lag
        )
        if daily.last < k - 1:
            return None
        tri = aggregate_triangle(daily, k, max_lag)
        days = final_counts(
            counts, "daily", places, daily.times, totals, country
        )
        truth = _sum_bins(days, bin_ends(daily, k), k)
    else:
        tri = hub_triangle(
            rows, source, places, end.isoformat(), max_lag=max_lag
        )
        truth = final_counts(counts, source, places, tri.times, totals, country)
    start = tri.n_context
    horizon = len(tri.times) - start
    group = layout_block(tri, "ragged", places, start)
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
