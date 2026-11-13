"""Rolling truth: the data as reported D days after each target date.

As in the German nowcast hub evaluation (Wolffram et al. 2023), a target
is scored against the count in the latest release made on or before
`target_date + D` days. The truth is missing until a release on or after
that date exists, since a later release could still change it.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["attach_truth", "rolling_truth"]

_KEYS = ["study", "location", "target_date"]


def rolling_truth(observations: pd.DataFrame, delay_days: int) -> pd.DataFrame:
    """Return the count reported `delay_days` after each target date.

    Parameters
    ----------
    observations : pandas.DataFrame
        Observations in the schema of `paper/schema.md`, one row per
        target date and release (`as_of_date`).
    delay_days : int
        D, the days after the target date at which the data are taken.

    Returns
    -------
    pandas.DataFrame
        `study`, `location`, `target_date` and `truth`, with `truth`
        missing where no release on or after `target_date + D` exists.

    Raises
    ------
    ValueError
        If `delay_days` is negative.
    """
    if delay_days < 0:
        raise ValueError("delay_days must be at least zero.")
    obs = observations.copy()
    obs["cutoff"] = obs["target_date"] + pd.Timedelta(days=delay_days)
    latest = obs.groupby("study")["as_of_date"].transform("max")
    known = obs.loc[obs["as_of_date"] <= obs["cutoff"]]
    known = known.sort_values("as_of_date").drop_duplicates(_KEYS, keep="last")
    targets = obs.assign(final=latest).drop_duplicates(_KEYS)
    out = targets.loc[:, [*_KEYS, "cutoff", "final"]].merge(
        known.loc[:, [*_KEYS, "value"]], on=_KEYS, how="left"
    )
    out["truth"] = out["value"].where(out["final"] >= out["cutoff"])
    return out.loc[:, [*_KEYS, "truth"]].sort_values(_KEYS, ignore_index=True)


def attach_truth(forecasts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Replace the forecasts' `truth` with the rolling truth.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts in the schema.
    truth : pandas.DataFrame
        Output of `rolling_truth`.

    Returns
    -------
    pandas.DataFrame
        The forecasts with `truth` from `truth`, missing where it has no
        row, and columns in their original order.
    """
    out = forecasts.drop(columns="truth").merge(truth, on=_KEYS, how="left")
    return out.loc[:, list(forecasts.columns)]
