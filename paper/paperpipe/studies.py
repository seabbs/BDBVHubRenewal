"""Each study's data loader and forecast runner, as the workflow calls them.

`load_data` returns a study's observations as of each release and a table
with one row per location: `location`, `delay_days` (D, the days after a
target date at which its truth is taken) and, for the synthetic study, the
scenario factors. `run_forecasts` returns the models' forecasts in the
schema of `paper/schema.md`.

Both take the run's data limits. `seeds` picks the synthetic outbreaks,
and a synthetic location code carries its seed (`synthetic_location`).
`before` keeps German nowcast and target dates before that date, and
German releases up to D days after the last of them, which their truth
needs (`paperpipe.config.last_release`).

`Sources` says where the real data are. `source_of` gives `real` for a
study whose sources are set and usable, and `dummy` otherwise, when both
functions return dummy data from `paperfigs.dummy`:

- Germany reads the benchmark's paper tables and reporting triangle
  (`paperdata.germany`).
- Ebola reads the release backtest's samples from past releases and the
  releases from main's data, with D and the truth as the backtest sets
  them (`paperdata.ebola`). Once the hub's baseline has a forecast with
  known truth, the hub's forecasts join from a clone of the hub and
  scores are relative to it; until then they are relative to
  `randomwalknowcast` (`study_baseline`).
- The synthetic study reads the synthetic runner's forecasts of the
  simulated outbreaks and its observations and factors beside them
  (`paperdata.synthetic`). Its dummy data are one dummy study,
  for the run's first seed, so a dummy run stays small.

`missing` says why a study cannot use its real data. A development run
may then use dummy data; a paper run fails.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import pandas as pd

from paperdata import ebola, germany, synthetic
from paperfigs.dummy import dummy_factors, dummy_study
from paperpipe.config import last_release, seed_of, synthetic_location

__all__ = [
    "DUMMY_EBOLA_DELAY",
    "NO_SOURCES",
    "Sources",
    "load_data",
    "missing",
    "run_forecasts",
    "source_of",
    "study_baseline",
]

DUMMY_EBOLA_DELAY: Final = 21
"""D for the dummy Ebola data, in days."""


@dataclass(frozen=True)
class Sources:
    """Where each study's real data are; unset means dummy data.

    Attributes
    ----------
    synthetic_forecasts : pathlib.Path, optional
        The synthetic runner's forecasts, with its observations and
        factors beside them (`paperdata.synthetic`).
    germany_forecasts : tuple of pathlib.Path
        The benchmark's paper tables (`paper-forecasts.csv.gz`).
    germany_triangle : pathlib.Path, optional
        The hub's reporting triangle.
    hub : pathlib.Path, optional
        A clone of the Ebola hub.
    ebola_data : pathlib.Path, optional
        Main's data folder, whose `target.csv` holds every release.
    ebola_config : pathlib.Path, optional
        The backtest config, which sets D (`config/backtest.toml`).
    ebola_backtest : pathlib.Path, optional
        The release backtest's samples (`results/forecasts.csv.gz`).
    ebola_extra : tuple of pathlib.Path
        Further models' release-backtest samples in the same layout.
    candidates : tuple of pathlib.Path
        Files of our stored Ebola forecasts.
    store : pathlib.Path, optional
        A clone holding the fit store branch `store_ref`.
    store_ref : str
        The fit store branch.
    """

    synthetic_forecasts: Path | None = None
    germany_forecasts: tuple[Path, ...] = ()
    germany_triangle: Path | None = None
    hub: Path | None = None
    ebola_data: Path | None = None
    ebola_config: Path | None = None
    ebola_backtest: Path | None = None
    ebola_extra: tuple[Path, ...] = ()
    candidates: tuple[Path, ...] = ()
    store: Path | None = None
    store_ref: str = "origin/fit-store"


log = logging.getLogger(__name__)

NO_SOURCES: Final = Sources()
"""No real data: every study runs on dummy data."""


def _has_ref(repo: Path, ref: str) -> bool:
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", ref],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@cache
def _release_delay(data: Path, config: Path) -> int:
    return ebola.truth_delay(data, config)


@cache
def _latest_release(data: Path) -> pd.Timestamp:
    return ebola.release_observations(data)["as_of_date"].max()


def _ebola_delay(sources: Sources, delay_days: int | None) -> int:
    if delay_days is not None:
        return delay_days
    assert sources.ebola_data is not None
    assert sources.ebola_config is not None
    return _release_delay(sources.ebola_data, sources.ebola_config)


@cache
def _baseline_scorable(hub: Path, data: Path, delay_days: int) -> bool:
    """Whether the hub's baseline has a target with known truth."""
    samples = ebola.hub_samples(hub)
    base = samples.loc[samples["model"] == ebola.HUB_BASELINE]
    due = base["target_date"] + pd.Timedelta(days=delay_days)
    return bool((due <= _latest_release(data)).any())


def missing(
    study: str,
    sources: Sources,
    delay_days: int | None = None,
) -> str | None:
    """Return why a study cannot use its real data, or None if it can.

    Parameters
    ----------
    study : str
        `synthetic`, `germany` or `ebola`.
    sources : Sources
        Where the real data are.
    delay_days : int, optional
        For Ebola, D in place of the release backtest's.

    Returns
    -------
    str or None
        What is missing, or None.
    """
    if study == "synthetic":
        if sources.synthetic_forecasts is None:
            return "no synthetic runner output is set (`synthetic.forecasts`)"
        return synthetic.missing(sources.synthetic_forecasts)
    if study == "germany":
        if not sources.germany_forecasts or sources.germany_triangle is None:
            return "no German benchmark tables or triangle are set"
        return None
    if (
        sources.ebola_data is None
        or not (sources.ebola_data / "target.csv").exists()
    ):
        return "no release data are set (`ebola.data`)"
    if sources.ebola_backtest is None or not sources.ebola_backtest.exists():
        return "no release backtest is set (`ebola.backtest`)"
    if delay_days is None:
        try:
            _ebola_delay(sources, delay_days)
        except ValueError as err:
            return f"the release backtest has no truth delay yet ({err})"
    return None


def study_baseline(
    study: str, sources: Sources, delay_days: int | None = None
) -> str | None:
    """Return the model a study's scores are relative to, if not the usual.

    Ebola scores are relative to the hub's baseline once it has a
    forecast with known truth, and to `randomwalknowcast` until then.
    """
    if study != "ebola":
        return None
    if sources.hub is not None and sources.ebola_data is not None:
        delay = _ebola_delay(sources, delay_days)
        if _baseline_scorable(sources.hub, sources.ebola_data, delay):
            return None
    return "randomwalknowcast"


def source_of(
    study: str,
    sources: Sources,
    delay_days: int | None = None,
) -> str:
    """Return `real` if a study's real data are set and usable, else `dummy`.

    See `missing` for the parameters and why a study is on dummy data.
    """
    return "dummy" if missing(study, sources, delay_days) else "real"


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


def _locations(obs: pd.DataFrame, delay_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"location": sorted(obs["location"].unique()), "delay_days": delay_days}
    )


def _synthetic_forecasts(
    sources: Sources, seeds: Sequence[int]
) -> pd.DataFrame:
    """Return the runner's forecasts of the run's seeds, `truth` empty."""
    assert sources.synthetic_forecasts is not None
    fc = pd.read_csv(sources.synthetic_forecasts)
    for col in ("origin_date", "target_date"):
        fc[col] = pd.to_datetime(fc[col])
    keep = fc["location"].map(seed_of).isin(set(seeds))
    return fc.loc[keep].assign(truth=float("nan")).reset_index(drop=True)


def _germany_forecasts(
    sources: Sources, before: pd.Timestamp | None
) -> pd.DataFrame:
    """Return the benchmark's forecasts, failing on any from `before`."""
    fc = germany.forecasts(sources.germany_forecasts)
    if before is not None and (fc["origin_date"] >= before).any():
        raise ValueError(
            f"German forecast tables hold nowcast dates from {before:%Y-%m-%d}"
            ", which this run must not read."
        )
    return fc


def _ebola_forecasts(sources: Sources, with_hub: bool) -> pd.DataFrame:
    """Return the release backtest's samples, with the hub's once scorable.

    The backtest's table holds our models' forecasts from every past
    release, the stored fits' among them; `ebola_extra` adds further
    models' samples in its layout. The hub's forecasts join once its
    baseline can be scored.
    """
    assert sources.ebola_backtest is not None
    parts = [ebola.backtest_samples(sources.ebola_backtest)]
    parts += [ebola.backtest_samples(p) for p in sources.ebola_extra]
    if with_hub and sources.hub is not None:
        store = sources.store
        if store is not None and not _has_ref(store, sources.store_ref):
            log.warning(
                "%s has no %s; the Ebola study reads no stored fits.",
                store,
                sources.store_ref,
            )
            store = None
        parts.append(
            ebola.load(
                sources.hub,
                candidates=sources.candidates,
                store=store,
                store_ref=sources.store_ref,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _germany_data(
    sources: Sources, before: pd.Timestamp | None, delay_days: int
) -> pd.DataFrame:
    assert sources.germany_triangle is not None
    fc = _germany_forecasts(sources, before)
    # Releases up to D days after the last date give the last truths.
    last = (
        fc["origin_date"].max() + pd.Timedelta(days=delay_days)
        if before is None
        else last_release(before, delay_days)
    )
    obs = germany.observations(
        sources.germany_triangle, fc["origin_date"].min(), last
    )
    return _development_releases(obs, before, delay_days)


def _development_releases(
    obs: pd.DataFrame, before: pd.Timestamp | None, delay_days: int
) -> pd.DataFrame:
    """Keep target dates before `before` and releases its truth needs."""
    if before is None:
        return obs
    obs = _before(obs, "target_date", before)
    last = last_release(before, delay_days)
    return obs.loc[obs["as_of_date"] <= last].reset_index(drop=True)


def load_data(
    study: str,
    seeds: Sequence[int] = (),
    before: pd.Timestamp | None = None,
    delay_days: int | None = None,
    sources: Sources = NO_SOURCES,
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
        D for every location. Germany needs it. For Ebola it replaces the
        hub's 98% reporting delay.
    sources : Sources
        Where the real data are; a study without them gets dummy data.

    Returns
    -------
    tuple of pandas.DataFrame
        Observations in the schema, and `location`, `delay_days` and any
        scenario factors.

    Raises
    ------
    ValueError
        If Germany has no `delay_days`.
    """
    if study == "synthetic" and source_of(study, sources) == "real":
        assert sources.synthetic_forecasts is not None
        return synthetic.load(sources.synthetic_forecasts, seeds)
    if study == "synthetic":
        obs, locs = [], []
        for seed in seeds[:1]:
            _, one = dummy_study(study, seed=seed, n_samples=1)
            factors = dummy_factors(one, seed=seed)
            obs.append(_relabel(one, seed))
            locs.append(_relabel(factors, seed))
        locations = pd.concat(locs, ignore_index=True)
        locations.insert(1, "delay_days", locations["delay_q98"].astype(int))
        return pd.concat(obs, ignore_index=True), locations
    if study == "germany" and delay_days is None:
        raise ValueError("Germany needs `delay_days`, its truth delay.")
    real = source_of(study, sources, delay_days) == "real"
    if study == "germany" and real:
        assert delay_days is not None
        obs = _germany_data(sources, before, delay_days)
        return obs, _locations(obs, delay_days)
    if study == "ebola" and real:
        assert sources.ebola_data is not None
        obs = ebola.release_observations(sources.ebola_data)
        return obs, _locations(obs, _ebola_delay(sources, delay_days))
    _, obs = dummy_study(study, n_samples=1)
    if study == "germany":
        assert delay_days is not None
        obs = _development_releases(obs, before, delay_days)
    delay = delay_days if delay_days is not None else DUMMY_EBOLA_DELAY
    return obs, _locations(obs, delay)


def run_forecasts(
    study: str,
    seeds: Sequence[int] = (),
    before: pd.Timestamp | None = None,
    n_samples: int = 100,
    sources: Sources = NO_SOURCES,
    source: str = "dummy",
    baseline: str | None = None,
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
    sources : Sources
        Where the real data are.
    source : str
        `real` or `dummy`, as `source_of` gave for the study's data.
    baseline : str, optional
        The baseline `study_baseline` gave, if not the usual one. Ebola
        leaves the hub's forecasts out while it is set.

    Returns
    -------
    pandas.DataFrame
        Forecasts in the schema, with `truth` left for the workflow.
    """
    if study == "synthetic" and source == "real":
        return _synthetic_forecasts(sources, seeds)
    if study == "synthetic":
        parts = [
            _relabel(dummy_study(study, seed=s, n_samples=n_samples)[0], s)
            for s in seeds[:1]
        ]
        return pd.concat(parts, ignore_index=True)
    if study == "germany" and source == "real":
        return _germany_forecasts(sources, before)
    if study == "ebola" and source == "real":
        return _ebola_forecasts(sources, with_hub=baseline is None)
    fc, _ = dummy_study(study, n_samples=n_samples)
    if study == "germany":
        fc = _before(fc, "origin_date", before)
    return fc
