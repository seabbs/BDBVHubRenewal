"""Relative scores by scenario factor, for the synthetic study.

Each synthetic location (an outbreak's place) has the values of the
simulation grid factors it was drawn with, in a table with a `location`
column and one column per factor. Numeric factors are binned into
tertiles so each level holds a similar number of locations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from paperfigs.scores import relative_scores

__all__ = [
    "ALWAYS_SHOWN",
    "REDUNDANT_PAIRS",
    "bin_factors",
    "factor_label",
    "redundant_factors",
    "relative_by_factor",
    "top_factors",
]

ALWAYS_SHOWN: Final = ("delay_q99",)
"""Factors the main panel always shows: long delays should make
nowcasting harder, so the 99% delay is worth seeing whatever its spread."""

REDUNDANT_PAIRS: Final = (("delay_q99", "delay_mean"),)
"""(kept, dropped) pairs: the second leaves the main panel when its rank
correlation with the first is above the threshold."""


def factor_label(name: str) -> str:
    """Return a factor's display name: underscores become spaces."""
    return name.replace("_", " ").capitalize()


def _range_label(lo: float, hi: float) -> str:
    return f"{lo:.3g} to {hi:.3g}"


def bin_factors(factors: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Turn factor values into labelled levels.

    Numeric factors with more than four values are cut at their quantiles
    into `n_bins` levels labelled by their range. Other factors become
    strings. `n_places`, if present, becomes `places` (`single` or
    `multi`), and `report_days` becomes `reporting` (`daily` or
    `weekdays`).

    Parameters
    ----------
    factors : pandas.DataFrame
        `location` and one column per factor.
    n_bins : int
        Levels for a numeric factor.

    Returns
    -------
    pandas.DataFrame
        `location` and one string column per factor.
    """
    out = factors.copy()
    if "n_places" in out.columns:
        out["places"] = np.where(out["n_places"] == 1, "single", "multi")
        out = out.drop(columns="n_places")
    if "report_days" in out.columns:
        # 7 is daily reporting; fewer is reports on some weekdays only.
        days = out["report_days"]
        out["reporting"] = np.where(days == 7, "daily", "weekdays")
        out = out.drop(columns="report_days")
    for col in out.columns.drop("location"):
        values = out[col]
        numeric = pd.api.types.is_numeric_dtype(values)
        if numeric and values.nunique() > 4:
            bins = pd.qcut(values, n_bins, duplicates="drop")
            labels = {
                iv: _range_label(
                    values[bins == iv].min(), values[bins == iv].max()
                )
                for iv in bins.cat.categories
            }
            out[col] = bins.map(labels).astype(str)
        else:
            out[col] = values.astype(str)
    return out


def relative_by_factor(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    baseline: str | None = None,
) -> pd.DataFrame:
    """Return relative scores for each level of each factor.

    Parameters
    ----------
    scores : pandas.DataFrame
        `score_forecasts` output for one study.
    factors : pandas.DataFrame
        Output of `bin_factors`.
    baseline : str, optional
        As for `relative_scores`.

    Returns
    -------
    pandas.DataFrame
        `factor`, `level`, `model`, `rel_score`, `rel_log_score` and `n`.
    """
    sc = scores.merge(factors, on="location", how="inner")
    parts = []
    for col in factors.columns.drop("location"):
        rel = relative_scores(sc, [col], baseline)
        parts.append(rel.rename(columns={col: "level"}).assign(factor=col))
    out = pd.concat(parts, ignore_index=True)
    first = ["factor", "level", "model", "rel_score", "rel_log_score", "n"]
    return out.loc[:, first]


def redundant_factors(
    factors: pd.DataFrame, threshold: float = 0.9
) -> list[str]:
    """Return factors that repeat another too closely for the main panel.

    For each pair in `REDUNDANT_PAIRS` present in `factors`, the second is
    redundant when its Spearman rank correlation with the first, over
    locations, is above `threshold`.

    Parameters
    ----------
    factors : pandas.DataFrame
        Raw factor values by `location`, before `bin_factors`.
    threshold : float
        Rank correlation above which the second factor is dropped.

    Returns
    -------
    list of str
        Factors to leave out of the main panel.
    """
    out = []
    for kept, dropped in REDUNDANT_PAIRS:
        if {kept, dropped} <= set(factors.columns):
            rho = factors[kept].corr(factors[dropped], method="spearman")
            if rho > threshold:
                out.append(dropped)
    return out


def top_factors(
    rel: pd.DataFrame,
    n: int = 4,
    col: str = "rel_score",
    always: Sequence[str] = (),
) -> list[str]:
    """Return the `n` factors whose levels spread the relative score most.

    Spread is the range of the log relative score over a factor's levels,
    averaged over models.

    Parameters
    ----------
    rel : pandas.DataFrame
        Output of `relative_by_factor`.
    n : int
        Factors to keep.
    col : str
        Score column.
    always : sequence of str
        Factors to keep first whatever their spread, if present.

    Returns
    -------
    list of str
        The `always` factors present, then the others, most spread first.
    """
    logs = rel.assign(_log=np.log(rel[col]))
    g = logs.groupby(["factor", "model"])["_log"]
    spread = g.max() - g.min()
    order = spread.groupby("factor").mean().sort_values(ascending=False)
    kept = [f for f in always if f in order.index]
    rest = [f for f in order.index if f not in kept]
    return (kept + rest)[:n]
