"""Which nowcasts the figures show: places, outbreaks and origins.

The main panel shows one series: the national one for a case study, and
for the synthetic study one outbreak drawn with the workflow's seed. The
SI figure shows the other places of a case study (`other_places`), or
outbreaks drawn to cover the levels of chosen grid factors
(`pick_outbreaks`). `pick_origins` draws the origins shown, far enough
apart that their nowcasts do not overlap. Every draw is seeded, so the
figures pick the same series and origins on every run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from paperfigs.factors import factor_label
from paperfigs.schema import DEFAULT_STRATUM

__all__ = [
    "PLACE_RULE",
    "other_places",
    "pick_origins",
    "pick_outbreaks",
    "pick_places",
    "place_models",
]

PLACE_RULE: Final = ("largest", "median", "small", "rising", "falling")
"""How the SI picks places to draw, one per entry, each a new place."""


def place_models(models: Sequence[str], wanted: Sequence[str]) -> list[str]:
    """Return the wanted models that have forecasts, in `wanted`'s order."""
    have = set(models)
    return [m for m in wanted if m in have]


def pick_origins(
    forecasts: pd.DataFrame, n: int, seed: int
) -> list[pd.Timestamp]:
    """Draw `n` origins whose nowcast windows do not overlap.

    Candidates are every origin a window's length after the last one
    kept, from the first.
    """
    origins = np.sort(forecasts["origin_date"].unique())
    days = forecasts["target_date"] - forecasts["origin_date"]
    span = days.max() - days.min() + pd.Timedelta(days=1)
    candidates, last = [], None
    for origin in origins:
        if last is None or pd.Timestamp(origin) - last >= span:
            candidates.append(pd.Timestamp(origin))
            last = pd.Timestamp(origin)
    rng = np.random.default_rng(seed)
    k = min(n, len(candidates))
    return sorted(rng.choice(candidates, size=k, replace=False))


def other_places(
    forecasts: pd.DataFrame, national: str
) -> list[tuple[str, str, str]]:
    """Return every place below the national one.

    Subnational locations one level down come first, then the national
    strata, each as `(location, stratum, title)`.
    """
    keys = forecasts[["location", "stratum"]].drop_duplicates()
    depth = national.count("-") + 1
    below = keys.loc[
        (keys["location"] != national)
        & (keys["location"].str.count("-") <= depth)
        & (keys["stratum"] == DEFAULT_STRATUM)
    ]
    strata = keys.loc[
        (keys["location"] == national) & (keys["stratum"] != DEFAULT_STRATUM)
    ]
    places = [(loc, DEFAULT_STRATUM, loc) for loc in sorted(below["location"])]
    return places + [
        (national, st, f"{national}, {st}") for st in sorted(strata["stratum"])
    ]


def pick_outbreaks(
    cells: pd.DataFrame, factors: Sequence[str], n: int, seed: int
) -> list[tuple[str, str, str]]:
    """Draw `n` outbreaks that cover the levels of `factors` between them.

    Parameters
    ----------
    cells : pandas.DataFrame
        `location` and each location's recorded cell. Locations ending
        `-total` are an outbreak's total and are preferred when present.
    factors : sequence of str
        Grid factors whose levels the outbreaks should cover.
    n : int
        Outbreaks to draw.
    seed : int
        Random seed.

    Returns
    -------
    list of tuple of str
        `(location, stratum, title)`, the title naming the cell.
    """
    totals = cells.loc[cells["location"].str.endswith("-total")]
    pool = totals if len(totals) else cells
    factors = [f for f in factors if f in pool.columns]
    rng = np.random.default_rng(seed)
    order = pool.iloc[rng.permutation(len(pool))]
    covered: set[tuple[str, object]] = set()
    chosen: list[int] = []
    for i, row in order.iterrows():
        if len(chosen) == n:
            break
        new = {(f, row[f]) for f in factors} - covered
        if new:
            chosen.append(i)
            covered |= new
    rest = [i for i in order.index if i not in chosen]
    chosen += rest[: n - len(chosen)]
    return [
        (
            pool.loc[i, "location"],
            DEFAULT_STRATUM,
            "; ".join(
                f"{factor_label(f)}: {str(pool.loc[i, f]).replace('_', ' ')}"
                for f in factors
            ),
        )
        for i in chosen
    ]


def pick_places(
    truth: pd.DataFrame,
    places: Sequence[tuple[str, str, str]],
    origin: pd.Timestamp,
    rule: Sequence[str] = PLACE_RULE,
    seed: int = 1,
    weeks: int = 4,
) -> list[tuple[str, str, str]]:
    """Pick places by a rule: largest, median, small, rising, falling.

    Each place's level is its mean truth over the `weeks` weeks before
    `origin`, and its trend the log ratio of the last of those weeks to
    the first. Each entry of `rule` picks a place not yet picked:
    `largest`, `median` and `small` (the smallest with any cases) by
    level, `rising` and `falling` by trend. Ties are broken in an order
    drawn with `seed`.

    Returns
    -------
    list of tuple of str
        `(location, stratum, title)`, the title naming why it was picked.
    """
    start = pd.Timestamp(origin) - pd.Timedelta(weeks=weeks)
    window = truth.loc[truth["target_date"].between(start, origin)].dropna(
        subset=["truth"]
    )
    rows = []
    for location, stratum, title in places:
        one = window.loc[
            (window["location"] == location) & (window["stratum"] == stratum)
        ].sort_values("target_date")
        if one.empty:
            continue
        first, last = one["truth"].iloc[0], one["truth"].iloc[-1]
        rows.append(
            {
                "place": (location, stratum, title),
                "level": float(one["truth"].mean()),
                "trend": float(np.log((last + 1) / (first + 1))),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return []
    rng = np.random.default_rng(seed)
    table = table.iloc[rng.permutation(len(table))].reset_index(drop=True)
    chosen: list[tuple[str, str, str]] = []
    out = []
    for reason in rule:
        left = table.loc[~table["place"].isin(chosen)]
        if left.empty:
            break
        if reason == "largest":
            i = left["level"].idxmax()
        elif reason == "small":
            some = left.loc[left["level"] > 0]
            i = (some if len(some) else left)["level"].idxmin()
        elif reason == "median":
            mid = left["level"].median()
            i = (left["level"] - mid).abs().idxmin()
        elif reason == "rising":
            i = left["trend"].idxmax()
        elif reason == "falling":
            i = left["trend"].idxmin()
        else:
            raise ValueError(f"Unknown place rule {reason!r}.")
        location, stratum, title = table["place"][i]
        chosen.append((location, stratum, title))
        out.append((location, stratum, f"{title} ({reason})"))
    return out
