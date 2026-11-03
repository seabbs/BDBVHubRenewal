# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Release cut-offs.

`cutoff_lags` gives the days from a release to its data cut-off by
weekday, for simulated releases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cutoff_lags"]


def cutoff_lags(daily: pd.DataFrame) -> dict[int, int]:
    """Days from release to data cut-off, by release weekday.

    The cut-off of a real release is its latest onset day. A weekday with
    no real release takes the median over the releases.

    Parameters
    ----------
    daily : pandas.DataFrame
        Real daily rows with `vintage` and `onset_date`
        (`bdbvhubrenewal.data.tidy_daily`).

    Returns
    -------
    dict of int to int
        Lag in days for every weekday, 0 (Monday) to 6.
    """
    last = pd.to_datetime(daily.groupby("vintage")["onset_date"].max())
    lag = (
        pd.to_datetime(pd.Series(last.index, index=last.index)) - last
    ).dt.days
    lag = lag.sort_index()
    weekday = pd.to_datetime(
        pd.Series(lag.index, index=lag.index)
    ).dt.day_of_week
    seen = lag.groupby(weekday).last()
    fill = int(np.median(lag.to_numpy()))
    return {w: int(seen.get(w, fill)) for w in range(7)}
