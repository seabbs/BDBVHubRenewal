"""Load the synthetic study from the synthetic runner's outputs.

The runner of #192 forecasts the simulated outbreaks of a set of seeds
and writes, beside its forecasts `<stem>.csv.gz` in the paper's schema,
`<stem>-observations.csv` (every weekly release and the final count) and
`<stem>-factors.csv` (each location's grid cell and `delay_q98`, its
outbreak's 98% reporting delay in days). A location is
`<seed>-<simulation>-<draw>_<stratum>`, so it carries its seed.

`missing` says which outputs are absent, and `load` returns the
observations and the per-location table of the run's seeds, with D the
98% delay.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from paperpipe.config import seed_of

__all__ = ["load", "missing", "outputs"]


def outputs(forecasts: str | Path) -> tuple[Path, Path, Path]:
    """Return the runner's forecasts, observations and factors files."""
    forecasts = Path(forecasts)
    stem = forecasts.name.removesuffix(".gz").removesuffix(".csv")
    folder = forecasts.parent
    return (
        forecasts,
        folder / f"{stem}-observations.csv",
        folder / f"{stem}-factors.csv",
    )


def missing(forecasts: str | Path) -> str | None:
    """Return which of the runner's outputs are absent, or None."""
    absent = [str(p) for p in outputs(forecasts) if not p.exists()]
    return (
        f"no synthetic runner output ({', '.join(absent)})" if absent else None
    )


def _of_seeds(df: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    keep = df["location"].map(seed_of).isin(set(seeds))
    return df.loc[keep].reset_index(drop=True)


def load(
    forecasts: str | Path, seeds: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the observations and per-location table of `seeds`.

    Parameters
    ----------
    forecasts : str or pathlib.Path
        The runner's forecasts file; its observations and factors sit
        beside it.
    seeds : sequence of int
        Simulation seeds to keep.

    Returns
    -------
    tuple of pandas.DataFrame
        Observations in the schema, and `location`, `delay_days` and the
        grid cell of each location.
    """
    _, obs_file, factors_file = outputs(forecasts)
    obs = pd.read_csv(obs_file, parse_dates=["target_date", "as_of_date"])
    factors = pd.read_csv(factors_file)
    locations = _of_seeds(factors, seeds)
    locations.insert(1, "delay_days", locations["delay_q98"].astype(int))
    return _of_seeds(obs, seeds), locations
