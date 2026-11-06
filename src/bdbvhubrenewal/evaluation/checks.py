# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Plausibility checks of the backtest forecasts, scored.

Each forecast of a place from an origin $o$ gets a place type and a
flag, both decided from the counts reported at $o$, and its scores are
then averaged by each.
Week $w$ is the latest week ending on or before $o$, and $c(t)$ the
count reported at $o$ for the week ending $t$, 0 where none is.

**Place type.** From the largest count $m$ of the `RECENT_WEEKS` weeks up
to $w$, the random walk's walk window: silent if $m = 0$, small if
$m \le$ `SMALL_MAX`, larger otherwise.

**Flag.** From the forecast of the reference week $w + 1$, with median
$q_{50}$ and 95% quantile $q_{95}$.
The level is $\ell = \max\{(c(w - 3) + c(w - 2)) / 2,\ c(w - 1)\}$, the
mean of the two latest weeks at least two weeks old, or the last full
week's count where that is higher, as reported counts only grow.
The recent maximum is $r = \max(1, \max_{w - 4 \le t \le w} c(t))$.

- Collapse: $\ell \ge 3$ and $q_{50} \le \ell / 3$.
- Blow-up: $q_{50} > 3r$ and $q_{50} - r \ge 5$, or $q_{95} > 10r$ and
  $q_{95} - r \ge 20$.

These are the rules of the 2026-11-07 review (issues #49 to #51), set
so they pick out the zones found implausible by eye.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "RECENT_WEEKS",
    "SMALL_MAX",
    "check_summary",
    "flags",
    "place_types",
]

RECENT_WEEKS = 9
"""Weeks up to the origin whose counts set the place type."""

SMALL_MAX = 5
"""Largest recent weekly count of a small place."""

_METRICS = {"wis": "WIS", "log_wis": "log WIS", "coverage_90": "90% coverage"}


def _last_week(origin: pd.Series) -> pd.Series:
    """Return the Saturday on or before each origin."""
    day = pd.to_datetime(origin)
    return day - pd.to_timedelta((day.dt.weekday - 5) % 7, unit="D")


def _by_week(counts: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    """Return `c(w - k)` for k = 0 to `RECENT_WEEKS - 1`, wide by k."""
    keys = origins[["origin", "location"]].drop_duplicates()
    keys = keys.assign(week=_last_week(keys["origin"]))
    lags = pd.DataFrame({"k": range(RECENT_WEEKS)})
    grid = keys.merge(lags, how="cross")
    grid["target_end_date"] = (
        grid["week"] - pd.to_timedelta(7 * grid["k"], unit="D")
    ).dt.strftime("%Y-%m-%d")
    reported = counts.rename(columns={"vintage": "origin"})[
        ["origin", "location", "target_end_date", "total"]
    ]
    grid = grid.merge(
        reported, on=["origin", "location", "target_end_date"], how="left"
    )
    grid["total"] = grid["total"].fillna(0.0)
    return grid.pivot_table(
        index=["origin", "location"], columns="k", values="total"
    )


def place_types(counts: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    """Return the place type of each place at each origin.

    Parameters
    ----------
    counts : pandas.DataFrame
        Weekly counts as each vintage reports them: `vintage`,
        `location`, `target_end_date` and `total`. Missing weeks are 0.
    origins : pandas.DataFrame
        The `origin` and `location` pairs to type.

    Returns
    -------
    pandas.DataFrame
        Columns `origin`, `location` and `place_type`, one of `silent`,
        `small` and `larger`.
    """
    top = _by_week(counts, origins).max(axis=1)
    kind = pd.Series("larger", index=top.index)
    kind[top <= SMALL_MAX] = "small"
    kind[top == 0] = "silent"
    return kind.rename("place_type").reset_index()


def flags(counts: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    """Return the flag of each forecast of the reference week.

    Parameters
    ----------
    counts : pandas.DataFrame
        As for `place_types`.
    quantiles : pandas.DataFrame
        `model`, `origin`, `location`, `q50` and `q95` of each forecast
        of the reference week.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `origin`, `location` and `flag`, one of
        `collapse`, `blow-up` and `none`.
    """
    wide = _by_week(counts, quantiles)
    level = pd.concat([(wide[3] + wide[2]) / 2, wide[1]], axis=1).max(axis=1)
    recent = wide[[0, 1, 2, 3, 4]].max(axis=1).clip(lower=1.0)
    out = quantiles.join(
        pd.DataFrame({"level": level, "recent": recent}),
        on=["origin", "location"],
    )
    q50, q95 = out["q50"], out["q95"]
    r, lv = out["recent"], out["level"]
    blow = ((q50 > 3 * r) & (q50 - r >= 5)) | ((q95 > 10 * r) & (q95 - r >= 20))
    collapse = (lv >= 3) & (q50 <= lv / 3)
    out["flag"] = "none"
    out.loc[blow, "flag"] = "blow-up"
    out.loc[collapse, "flag"] = "collapse"
    return out[["model", "origin", "location", "flag"]]


def check_summary(
    scores: pd.DataFrame, types: pd.DataFrame, flagged: pd.DataFrame
) -> pd.DataFrame:
    """Average the scores by scale, model and place type, and by flag.

    Parameters
    ----------
    scores : pandas.DataFrame
        One row per scored forecast, with `model`, `origin`, `location`,
        `scale` and the `wis`, `log_wis` and `coverage_90` columns.
    types : pandas.DataFrame
        From `place_types`.
    flagged : pandas.DataFrame
        From `flags`.

    Returns
    -------
    pandas.DataFrame
        Columns `scale`, `split` (`place type` or `flag`), `group`,
        `model`, `n` and the mean of each metric.
    """
    keyed = scores.merge(types, on=["origin", "location"], how="left").merge(
        flagged, on=["model", "origin", "location"], how="left"
    )
    parts = []
    for split, col in (("place type", "place_type"), ("flag", "flag")):
        grp = keyed.groupby(["scale", col, "model"])
        part = grp[list(_METRICS)].mean()
        part.insert(0, "n", grp.size())
        parts.append(
            part.reset_index()
            .rename(columns={col: "group"})
            .assign(split=split)
        )
    out = pd.concat(parts, ignore_index=True).rename(columns=_METRICS)
    return out[["scale", "split", "group", "model", "n", *_METRICS.values()]]
