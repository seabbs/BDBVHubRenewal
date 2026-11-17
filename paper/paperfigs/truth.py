"""Rolling truth: the data as reported D days after each target date.

As in the German nowcast hub evaluation (Wolffram et al. 2023), a target
is scored against the count in the latest release made on or before
`target_date + D` days. The truth is missing until a release on or after
that date exists, since a later release could still change it.
"""

from __future__ import annotations

import pandas as pd

from paperfigs.schema import DEFAULT_STRATUM

__all__ = ["attach_truth", "rolling_truth"]

_KEYS = ["study", "location", "stratum", "target_date"]


def rolling_truth(
    observations: pd.DataFrame,
    delay_days: int | pd.Series,
    first_after: bool = False,
) -> pd.DataFrame:
    """Return the count reported `delay_days` after each target date.

    The truth is the latest release on or before `target_date + D`, as the
    German nowcast hub took it, or with `first_after` the first release on
    or after it, as the BDBV release backtest takes it for releases that
    come at irregular gaps.

    Parameters
    ----------
    observations : pandas.DataFrame
        Observations in the schema of `paper/schema.md`, one row per
        target date and release (`as_of_date`).
    delay_days : int or pandas.Series
        D, the days after the target date at which the data are taken.
        A series indexed by `location` gives each location its own D, as
        for the synthetic outbreaks (their 98% reporting delay).
    first_after : bool
        Take the first release on or after `target_date + D`.

    Returns
    -------
    pandas.DataFrame
        `study`, `location`, `stratum`, `target_date` and `truth`, with
        `truth` missing where the location has no release on or after
        `target_date + D`.

    Raises
    ------
    ValueError
        If a delay is negative, or a per-location delay misses a location.
    """
    obs = observations.copy()
    if "stratum" not in obs.columns:
        obs["stratum"] = DEFAULT_STRATUM
    if isinstance(delay_days, pd.Series):
        missing = sorted(set(obs["location"]) - set(delay_days.index))
        if missing:
            raise ValueError(f"No delay_days for: {', '.join(missing)}.")
        days = obs["location"].map(delay_days).astype(int)
    else:
        days = pd.Series(delay_days, index=obs.index)
    if (days < 0).any():
        raise ValueError("delay_days must be at least zero.")
    obs["cutoff"] = obs["target_date"] + pd.to_timedelta(days, unit="D")
    # A location's releases may stop before another's.
    latest = obs.groupby(["study", "location"])["as_of_date"].transform("max")
    if first_after:
        known = obs.loc[obs["as_of_date"] >= obs["cutoff"]]
        known = known.sort_values("as_of_date").drop_duplicates(_KEYS)
    else:
        known = obs.loc[obs["as_of_date"] <= obs["cutoff"]]
        known = known.sort_values("as_of_date").drop_duplicates(
            _KEYS, keep="last"
        )
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
    fc = forecasts.drop(columns="truth")
    if "stratum" not in fc.columns:
        fc = fc.assign(stratum=DEFAULT_STRATUM)
    out = fc.merge(truth, on=_KEYS, how="left")
    return out.loc[:, list(forecasts.columns)]
