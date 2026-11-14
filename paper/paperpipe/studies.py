"""Each study's data loader and forecast runner, as the workflow calls them.

`load_data` returns a study's observations as of each release and a table
with one row per location: `location`, `delay_days` (D, the days after a
target date at which its truth is taken) and, for the synthetic study, the
scenario factors. `run_forecasts` returns the models' forecasts in the
schema of `paper/schema.md`.

Both take the run's data limits. `seeds` picks the synthetic outbreaks,
and a synthetic location code carries its seed (`synthetic_location`).
`before` keeps German nowcast and target dates before that date.

Ebola reads a clone of the BDBV Modeling Hub (`hub`) through
`paperdata.ebola`: the hub's releases, every team's samples and,
optionally, our stored forecasts (`candidates`) and the fit store
(`store`). D is the hub's 98% reporting delay. Without a hub, Ebola uses
dummy data.

The synthetic and German studies run models that are not merged yet
(#192), so they always use dummy data from `paperfigs.dummy`
(`DUMMY_STUDIES`), one dummy study per synthetic seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from paperdata import ebola
from paperfigs.dummy import dummy_factors, dummy_study
from paperpipe.config import synthetic_location

__all__ = ["DUMMY_EBOLA_DELAY", "DUMMY_STUDIES", "load_data", "run_forecasts"]

DUMMY_STUDIES: Final = ("synthetic", "germany")
"""Studies whose loaders need unmerged models, so run on dummy data."""

DUMMY_EBOLA_DELAY: Final = 21
"""D for the dummy Ebola data, in days."""


def _relabel(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.assign(
        location=[synthetic_location(seed, loc) for loc in df["location"]]
    )


def _before(
    df: pd.DataFrame, column: str, before: pd.Timestamp | None
) -> pd.DataFrame:
    if before is None:
        return df
    return df.loc[df[column] < pd.Timestamp(before)].reset_index(drop=True)


def load_data(
    study: str,
    seeds: Sequence[int] = (),
    before: pd.Timestamp | None = None,
    delay_days: int | None = None,
    hub: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a study's observations and its per-location table.

    Parameters
    ----------
    study : str
        `synthetic`, `germany` or `ebola`.
    seeds : sequence of int
        Synthetic seeds to load.
    before : pandas.Timestamp, optional
        For Germany, keep target dates before this date.
    delay_days : int, optional
        For Germany, D for every location. For Ebola, D overrides the
        hub's 98% reporting delay.
    hub : str or pathlib.Path, optional
        For Ebola, a clone of the hub.

    Returns
    -------
    tuple of pandas.DataFrame
        Observations in the schema, and `location`, `delay_days` and any
        scenario factors.
    """
    if study == "synthetic":
        obs, locs = [], []
        for seed in seeds:
            _, one = dummy_study(study, seed=seed, n_samples=1)
            factors = dummy_factors(one, seed=seed)
            obs.append(_relabel(one, seed))
            locs.append(_relabel(factors, seed))
        locations = pd.concat(locs, ignore_index=True)
        locations.insert(1, "delay_days", locations["delay_q98"].astype(int))
        return pd.concat(obs, ignore_index=True), locations
    if study == "ebola" and hub is not None:
        obs = ebola.observations(hub)
        if delay_days is None:
            delay_days = ebola.reporting_delay(hub)
        if delay_days is None:
            msg = f"No age reaches {ebola.DELAY_LEVEL:.0%} of the count."
            raise ValueError(msg)
        locations = pd.DataFrame(
            {
                "location": sorted(obs["location"].unique()),
                "delay_days": delay_days,
            }
        )
        return obs, locations
    _, obs = dummy_study(study, n_samples=1)
    if study == "germany":
        obs = _before(obs, "target_date", before)
        delay = delay_days
    else:
        delay = DUMMY_EBOLA_DELAY if delay_days is None else delay_days
    locations = pd.DataFrame(
        {"location": sorted(obs["location"].unique()), "delay_days": delay}
    )
    return obs, locations


def run_forecasts(
    study: str,
    seeds: Sequence[int] = (),
    before: pd.Timestamp | None = None,
    n_samples: int = 100,
    hub: str | Path | None = None,
    candidates: Sequence[str | Path] = (),
    store: str | Path | None = None,
) -> pd.DataFrame:
    """Return every model's forecasts for a study, in the schema.

    Parameters
    ----------
    study : str
        `synthetic`, `germany` or `ebola`.
    seeds : sequence of int
        Synthetic seeds to forecast.
    before : pandas.Timestamp, optional
        For Germany, keep nowcast dates before this date.
    n_samples : int
        Samples per forecast, for dummy studies scored on samples.
    hub : str or pathlib.Path, optional
        For Ebola, a clone of the hub.
    candidates : sequence of str or pathlib.Path
        For Ebola, files of our stored forecasts.
    store : str or pathlib.Path, optional
        For Ebola, a clone holding the fit store branch.

    Returns
    -------
    pandas.DataFrame
        Forecasts in the schema.
    """
    if study == "synthetic":
        parts = [
            _relabel(dummy_study(study, seed=s, n_samples=n_samples)[0], s)
            for s in seeds
        ]
        return pd.concat(parts, ignore_index=True)
    if study == "ebola" and hub is not None:
        # The truth is attached later, from the observations.
        fc = ebola.load(hub, candidates=candidates, store=store, delay_days=0)
        return fc.assign(truth=float("nan"))
    fc, _ = dummy_study(study, n_samples=n_samples)
    if study == "germany":
        fc = _before(fc, "origin_date", before)
    return fc
