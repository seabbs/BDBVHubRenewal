"""Load the held-out simulated outbreaks for the paper.

The simulator (`scripts/finetune_simulate.R`, #192) writes each seed's
outbreaks to `counts-<seed>.csv.gz`, one row per place, reference date and
report date with the count reported that day, and each simulation's cell
of the grid to `settings-<seed>.csv`. An outbreak `<seed>-<index>-<draw>`
is one observation draw of simulation `<seed>-<index>`.

`load` gives the observations in the paper's schema: weekly onsets by the
Saturday ending their week, as reported by each Saturday, for each place
and their total. Each is a location `s<seed>-<outbreak>-<place>`, with
the cell of its simulation and D, the 98% point of its outbreak's
reporting delay in days. `missing` says which seeds have no simulations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from paperpipe.config import synthetic_location

__all__ = ["DELAY_LEVEL", "TOTAL", "load", "missing"]

DELAY_LEVEL: Final = 0.98
"""Share of an outbreak's counts reported by its truth delay, D."""

TOTAL: Final = "total"
"""Place name of an outbreak's total over its places."""


def _files(sims: Path, seed: int) -> tuple[Path, Path]:
    return sims / f"counts-{seed}.csv.gz", sims / f"settings-{seed}.csv"


def missing(sims: str | Path, seeds: Sequence[int]) -> str | None:
    """Return which seeds lack simulations in `sims`, or None."""
    absent = [
        str(s)
        for s in seeds
        if not all(f.exists() for f in _files(Path(sims), s))
    ]
    if not absent:
        return None
    return f"no simulations for seeds {', '.join(absent)} in {sims}"


def _saturday(dates: pd.Series) -> pd.Series:
    return dates + pd.to_timedelta((5 - dates.dt.day_of_week) % 7, unit="D")


def _with_total(counts: pd.DataFrame) -> pd.DataFrame:
    keys = ["outbreak", "reference_date", "report_date"]
    total = counts.groupby(keys, as_index=False)["count"].sum()
    return pd.concat([counts, total.assign(location=TOTAL)], ignore_index=True)


def _observations(counts: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = counts.assign(
        target_date=_saturday(counts["reference_date"]),
        as_of_date=_saturday(counts["report_date"]),
    )
    keys = ["outbreak", "location", "target_date"]
    weekly = counts.groupby([*keys, "as_of_date"], as_index=False)[
        "count"
    ].sum()
    weekly = weekly.sort_values("as_of_date")
    weekly["value"] = weekly.groupby(keys)["count"].cumsum().astype(float)
    # Every Saturday from a week's end to the outbreak's last release.
    last = weekly.groupby("outbreak")["as_of_date"].transform("max")
    rows = [
        (o, loc, t, a, v)
        for (o, loc, t), g in weekly.groupby(keys)
        for a, v in _carried(g, t, last.loc[g.index].iloc[0])
    ]
    obs = pd.DataFrame(
        rows,
        columns=["outbreak", "place", "target_date", "as_of_date", "value"],
    )
    location = [
        synthetic_location(seed, f"{o.split('-', 1)[1]}-{p}")
        for o, p in zip(obs["outbreak"], obs["place"], strict=True)
    ]
    return obs.assign(study="synthetic", location=location)


def _carried(
    group: pd.DataFrame, target: pd.Timestamp, last: pd.Timestamp
) -> list[tuple[pd.Timestamp, float]]:
    """Return a week's count as of each Saturday from its end to `last`."""
    saturdays = pd.date_range(target, last, freq="7D")
    value = group.set_index("as_of_date")["value"]
    value = value.reindex(saturdays.union(value.index)).ffill().fillna(0.0)
    return list(value.loc[saturdays].items())


def _delay_q98(counts: pd.DataFrame) -> pd.Series:
    """Each outbreak's 98% reporting delay in days, weighted by count."""
    delay = (counts["report_date"] - counts["reference_date"]).dt.days
    out = {}
    for outbreak, idx in counts.groupby("outbreak").groups.items():
        d = delay.loc[idx].to_numpy()
        w = counts.loc[idx, "count"].to_numpy(float)
        order = np.argsort(d)
        share = np.cumsum(w[order]) / w.sum()
        out[outbreak] = int(d[order][np.searchsorted(share, DELAY_LEVEL)])
    return pd.Series(out, name="delay_days")


def load(
    sims: str | Path, seeds: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the observations and per-location table of `seeds`.

    Parameters
    ----------
    sims : str or pathlib.Path
        The folder of `counts-<seed>.csv.gz` and `settings-<seed>.csv`.
    seeds : sequence of int
        Simulation seeds to load.

    Returns
    -------
    tuple of pandas.DataFrame
        Observations in the schema, and `location`, `delay_days` and the
        recorded cell of each location's simulation.
    """
    obs, locs = [], []
    for seed in seeds:
        counts_file, settings_file = _files(Path(sims), seed)
        counts = pd.read_csv(
            counts_file,
            dtype={"outbreak": str},
            parse_dates=["reference_date", "report_date"],
        )
        counts = _with_total(counts)
        settings = pd.read_csv(settings_file, dtype={"outbreak": str})
        one = _observations(counts, seed)
        sim = one["outbreak"].str.rsplit("-", n=1).str[0]
        # r2 outbreaks have no draw suffix and are their own simulation.
        sim = sim.where(sim.isin(settings["outbreak"]), one["outbreak"])
        table = (
            one.assign(simulation=sim)[["location", "outbreak", "simulation"]]
            .drop_duplicates("location")
            .merge(_delay_q98(counts), left_on="outbreak", right_index=True)
            .merge(settings, left_on="simulation", right_on="outbreak")
        )
        locs.append(
            table.drop(
                columns=["outbreak_x", "outbreak_y", "simulation"],
                errors="ignore",
            )
        )
        obs.append(one)
    observations = pd.concat(obs, ignore_index=True)[
        ["study", "location", "target_date", "as_of_date", "value"]
    ]
    return observations, pd.concat(locs, ignore_index=True)
