"""The forecast and observation tables every study writes.

The columns and rules are set out in `paper/schema.md`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "FORECAST_COLUMNS",
    "FORECAST_KEYS",
    "OBSERVATION_COLUMNS",
    "STUDIES",
    "validate_forecasts",
    "validate_observations",
]

STUDIES: Final = ("synthetic", "germany", "ebola")
"""The three studies, in the order the paper reports them."""

FORECAST_KEYS: Final = (
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
)
"""Columns that identify one forecast."""

FORECAST_COLUMNS: Final = (
    *FORECAST_KEYS,
    "quantile_level",
    "value",
    "truth",
)
"""Columns of the forecast table, in order."""

OBSERVATION_COLUMNS: Final = (
    "study",
    "location",
    "target_date",
    "as_of_date",
    "value",
)
"""Columns of the observation table, in order."""


def _check_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    extra = [c for c in df.columns if c not in columns]
    if extra:
        raise ValueError(f"Unexpected extra columns: {', '.join(extra)}.")


def _check_studies(df: pd.DataFrame) -> None:
    bad = sorted(set(df["study"]) - set(STUDIES))
    if bad:
        raise ValueError(f"Unknown study values: {', '.join(map(str, bad))}.")


def validate_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Check a forecast table against the schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Forecasts in the long schema of `paper/schema.md`.

    Returns
    -------
    pandas.DataFrame
        The table with columns in schema order, dates as timestamps and
        `horizon` as an integer.

    Raises
    ------
    ValueError
        If a rule in `paper/schema.md` is broken.
    """
    _check_columns(df, FORECAST_COLUMNS)
    out = df.loc[:, list(FORECAST_COLUMNS)].copy()
    _check_studies(out)
    for col in ("origin_date", "target_date"):
        out[col] = pd.to_datetime(out[col])
    out["horizon"] = out["horizon"].astype(int)
    lvl = out["quantile_level"].to_numpy(dtype=float)
    if not ((lvl > 0) & (lvl < 1)).all():
        raise ValueError("Every quantile_level must be in (0, 1).")
    if (out["value"].to_numpy(dtype=float) < 0).any():
        raise ValueError("Every value must be at least zero.")
    keys = list(FORECAST_KEYS)
    if out.duplicated([*keys, "quantile_level"]).any():
        raise ValueError("A forecast has duplicate quantile levels.")
    groups = out.groupby(keys, sort=False, dropna=False)
    truth = groups["truth"]
    spread = truth.transform("max") - truth.transform("min")
    if (spread > 0).any():
        raise ValueError("truth differs between rows of one forecast.")
    has_median = (
        out.assign(_median=np.isclose(lvl, 0.5))
        .groupby(keys, sort=False, dropna=False)["_median"]
        .any()
    )
    if not has_median.all():
        raise ValueError("Every forecast needs the quantile level 0.5.")
    ordered = out.sort_values([*keys, "quantile_level"])
    step = ordered.groupby(keys, sort=False, dropna=False)["value"].diff()
    if (step < -1e-9).any():
        raise ValueError("Quantiles decrease with level in some forecast.")
    return out


def validate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Check an observation table against the schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations as of each date, as in `paper/schema.md`.

    Returns
    -------
    pandas.DataFrame
        The table with columns in schema order and dates as timestamps.

    Raises
    ------
    ValueError
        If a column is missing or extra, a study is unknown, a value is
        negative or a count appears twice for one as-of date.
    """
    _check_columns(df, OBSERVATION_COLUMNS)
    out = df.loc[:, list(OBSERVATION_COLUMNS)].copy()
    _check_studies(out)
    for col in ("target_date", "as_of_date"):
        out[col] = pd.to_datetime(out[col])
    if (out["value"].to_numpy(dtype=float) < 0).any():
        raise ValueError("Every value must be at least zero.")
    if out.duplicated(["study", "location", "target_date", "as_of_date"]).any():
        raise ValueError("An observation appears twice for one as_of_date.")
    return out
