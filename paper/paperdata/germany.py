"""Load the German nowcast benchmark (Wolffram et al. 2023) for the paper.

`bdbvhubrenewal germany-backtest`, `germany-teams` and `germany-paper`
write our models' and the hub teams' nowcasts as the paper's long table,
`paper-forecasts.csv.gz` (`bdbvhubrenewal.evaluation.germany`).
`forecasts` reads one or more of them. `observations` gives the 7-day sums
as each day's data gave them, from the hub's reporting triangle, with
age groups as strata of Germany as in the forecasts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from bdbvhubrenewal.data.germany import GERMANY, STATES, hub_triangle
from bdbvhubrenewal.evaluation.germany import asof_sums

__all__ = ["forecasts", "observations"]


def forecasts(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read paper tables of the benchmark, with `truth` left empty.

    The workflow attaches the truth from the observations.
    """
    parts = [pd.read_csv(p, dtype={"stratum": str}) for p in paths]
    out = pd.concat(parts, ignore_index=True)
    for col in ("origin_date", "target_date"):
        out[col] = pd.to_datetime(out[col])
    return out.assign(truth=float("nan"))


def observations(
    triangle: str | Path, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    """Return the 7-day sums as the data of each day gave them.

    Parameters
    ----------
    triangle : str or pathlib.Path
        The hub's reporting triangle file.
    first, last : pandas.Timestamp
        The first and last data dates (`as_of_date`) to keep.

    Returns
    -------
    pandas.DataFrame
        Observations in the paper's schema, for sums up to eight weeks
        before each data date.
    """
    dates = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d")
    sums = asof_sums(hub_triangle(triangle), list(dates))
    ages = ~sums["location"].isin({GERMANY, *STATES})
    return pd.DataFrame(
        {
            "study": "germany",
            "location": sums["location"].where(~ages, GERMANY),
            "stratum": sums["location"].where(ages, "all"),
            "target_date": pd.to_datetime(sums["target_end_date"]),
            "as_of_date": pd.to_datetime(sums["origin"]),
            "value": sums["total"].astype(float),
        }
    ).reset_index(drop=True)
