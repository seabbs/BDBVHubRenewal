"""Split a study's forecasts into a development and a held-out half.

The German study tunes on nowcast dates before a split date and holds out
the rest for the paper. The split is a per-study setting, so any study can
use it. Scores are shown for both halves, never the held-out half alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pandas as pd

__all__ = [
    "PERIODS",
    "PERIOD_ALPHA",
    "STUDY_SPLITS",
    "add_period",
    "read_sidecar",
    "study_split",
]

PERIODS: Final = ("development", "held out")
"""Period labels, in time order."""

PERIOD_ALPHA: Final = {"development": 0.45, "held out": 1.0}
"""Shade of each period: the development half is lighter."""

STUDY_SPLITS: Final[dict[str, pd.Timestamp]] = {
    "germany": pd.Timestamp("2022-02-10"),
}
"""First held-out origin date of each split study.

Germany develops on nowcast dates 2021-11-22 to 2022-02-09 (80 dates) and
holds out 2022-02-10 to 2022-04-29 (79 dates). A study's sidecar, where it
gives one, takes precedence (see `study_split`).
"""


def read_sidecar(path: Path | str) -> dict[str, Any]:
    """Read a forecast table's JSON sidecar, such as paper-forecasts.json.

    Parameters
    ----------
    path : pathlib.Path or str
        The sidecar file.

    Returns
    -------
    dict
        Its fields, such as `study`, `split_date` and `baseline`.
    """
    return json.loads(Path(path).read_text())


def study_split(
    study: str, sidecar: dict[str, Any] | None = None
) -> pd.Timestamp | None:
    """Return a study's split date: the sidecar's, else `STUDY_SPLITS`.

    Parameters
    ----------
    study : str
        The study.
    sidecar : dict, optional
        Output of `read_sidecar` for the study's forecast table.

    Returns
    -------
    pandas.Timestamp or None
        First held-out origin date, or None if the study is not split.
    """
    if sidecar and sidecar.get("split_date"):
        return pd.Timestamp(sidecar["split_date"])
    return STUDY_SPLITS.get(study)


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
