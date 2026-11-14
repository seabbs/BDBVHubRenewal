"""Split a study's forecasts into a development and a held-out half.

The German study tunes on nowcast dates before a split date and holds out
the rest for the paper. The split date is the workflow's
(`heldout.germany_from` in `workflow/config.yaml`). Scores are shown for
both halves, never the held-out half alone.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = [
    "PERIODS",
    "PERIOD_ALPHA",
    "add_period",
]

PERIODS: Final = ("development", "held out")
"""Period labels, in time order."""

PERIOD_ALPHA: Final = {"development": 0.45, "held out": 1.0}
"""Shade of each period: the development half is lighter."""


def add_period(df: pd.DataFrame, split: pd.Timestamp | None) -> pd.DataFrame:
    """Label rows `development` before `split` and `held out` from it.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows with an `origin_date`.
    split : pandas.Timestamp or None
        First held-out origin date. None leaves `df` unchanged.

    Returns
    -------
    pandas.DataFrame
        `df` with a `period` column when `split` is given.
    """
    if split is None:
        return df
    held = df["origin_date"] >= pd.Timestamp(split)
    return df.assign(period=held.map({False: PERIODS[0], True: PERIODS[1]}))
