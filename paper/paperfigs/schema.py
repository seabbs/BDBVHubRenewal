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
    "sample_id",
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
        The table with columns in schema order, dates as timestamps,
        `horizon` as an integer and `sample_id` as a nullable integer.

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
    out["quantile_level"] = pd.to_numeric(out["quantile_level"]).astype(float)
    out["sample_id"] = pd.to_numeric(out["sample_id"]).astype("Int64")
    out["value"] = out["value"].astype(float)
    out["truth"] = pd.to_numeric(out["truth"]).astype(float)
    is_q = out["quantile_level"].notna().to_numpy()
    is_s = out["sample_id"].notna().to_numpy()
    if (is_q == is_s).any():
        raise ValueError(
            "Each row needs exactly one of quantile_level or sample_id."
        )
    keys = list(FORECAST_KEYS)
    form = pd.Series(is_q, index=out.index).groupby(
        [out[k] for k in keys], sort=False
    )
    if (form.transform("min") != form.transform("max")).any():
        raise ValueError("Every row of a forecast must use one form.")
    if (out["value"].to_numpy() < 0).any():
        raise ValueError("Every value must be at least zero.")
    truth = out.groupby(keys, sort=False)["truth"]
    if (truth.transform("max") - truth.transform("min") > 0).any():
        raise ValueError("truth differs between rows of one forecast.")
    if out.loc[is_s].duplicated([*keys, "sample_id"]).any():
        raise ValueError("A forecast has duplicate sample_id values.")
    _check_quantiles(out.loc[is_q], keys)
    return out


def _check_quantiles(q: pd.DataFrame, keys: list[str]) -> None:
    if q.empty:
        return
    lvl = q["quantile_level"].to_numpy()
    if not ((lvl > 0) & (lvl < 1)).all():
        raise ValueError("Every quantile_level must be in (0, 1).")
    if q.duplicated([*keys, "quantile_level"]).any():
        raise ValueError("A forecast has duplicate quantile levels.")
    has_median = (
        q.assign(_median=np.isclose(lvl, 0.5))
        .groupby(keys, sort=False)["_median"]
        .any()
    )
    if not has_median.all():
        raise ValueError("Every quantile forecast needs the level 0.5.")
    ordered = q.sort_values([*keys, "quantile_level"])
    step = ordered.groupby(keys, sort=False)["value"].diff()
    if (step < -1e-9).any():
        raise ValueError("Quantiles decrease with level in some forecast.")


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
