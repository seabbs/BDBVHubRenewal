"""Run settings of the workflow, and the guard that keeps held-out data out.

The workflow config (`workflow/config.yaml`) names the held-out sets in its
`heldout` block: the German nowcast dates from `germany_from` on, and the
synthetic outbreaks of `synthetic_seeds`. `run_settings` turns the config
into the settings of one run. A `development` run gets the development
seeds and only the German dates before `germany_from`. A `paper` run gets
the held-out seeds and every German date, split at `germany_from`.
`check_no_heldout` fails if a development forecast table holds held-out
rows.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

__all__ = [
    "MODES",
    "RunSettings",
    "check_no_heldout",
    "check_no_heldout_releases",
    "last_release",
    "run_settings",
    "seed_of",
    "synthetic_location",
]

MODES: Final = ("development", "paper")
"""Run modes: `development` leaves out the held-out sets, `paper` uses them."""

_SEED: Final = re.compile(r"^s(\d+)-")


@dataclass(frozen=True)
class RunSettings:
    """Data one run may use.

    Attributes
    ----------
    mode : str
        One of `MODES`.
    synthetic_seeds : tuple of int
        Seeds of the synthetic outbreaks to evaluate.
    germany_before : pandas.Timestamp or None
        German nowcast dates and target dates are kept before this date.
        None keeps every date.
    germany_split : pandas.Timestamp or None
        First held-out German nowcast date, to split scores by half.
        None when the run has only the development half.
    heldout_seeds : tuple of int
        Seeds a development run must not use.
    heldout_from : pandas.Timestamp
        First held-out German nowcast date.
    """

    mode: str
    synthetic_seeds: tuple[int, ...]
    germany_before: pd.Timestamp | None
    germany_split: pd.Timestamp | None
    heldout_seeds: tuple[int, ...]
    heldout_from: pd.Timestamp


def _entry(block: Mapping[str, Any], name: str, key: str) -> Any:
    if key not in block:
        raise ValueError(f"The config's `{name}` block needs `{key}`.")
    return block[key]


def run_settings(config: Mapping[str, Any], mode: str) -> RunSettings:
    """Return the settings of a run in `mode`.

    Parameters
    ----------
    config : mapping
        The workflow config, with `development` and `heldout` blocks.
    mode : str
        One of `MODES`.

    Returns
    -------
    RunSettings
        The data the run may use.

    Raises
    ------
    ValueError
        If the mode is unknown, an entry is missing, a seed list is empty,
        or a development seed is also held out.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; use one of {MODES}.")
    held = config.get("heldout", {})
    dev = config.get("development", {})
    held_seeds = tuple(
        int(s) for s in _entry(held, "heldout", "synthetic_seeds")
    )
    held_from = pd.Timestamp(_entry(held, "heldout", "germany_from"))
    dev_seeds = tuple(
        int(s) for s in _entry(dev, "development", "synthetic_seeds")
    )
    if not dev_seeds or not held_seeds:
        raise ValueError("Each run needs at least one synthetic seed.")
    shared = sorted(set(dev_seeds) & set(held_seeds))
    if shared:
        seeds = ", ".join(str(s) for s in shared)
        raise ValueError(f"Seeds both development and held out: {seeds}.")
    if mode == "development":
        return RunSettings(
            mode, dev_seeds, held_from, None, held_seeds, held_from
        )
    return RunSettings(mode, held_seeds, None, held_from, held_seeds, held_from)


def synthetic_location(seed: int, name: str) -> str:
    """Return a synthetic location code that carries its simulation seed."""
    return f"s{seed}-{name}"


def seed_of(location: str) -> int | None:
    """Return the seed of a synthetic location code, or None."""
    m = _SEED.match(location)
    return int(m.group(1)) if m else None


def last_release(split: pd.Timestamp, delay_days: int) -> pd.Timestamp:
    """Return the last release a development run reads.

    The truth of the last development target date, the day before `split`,
    is its count as reported `delay_days` later.
    """
    return (
        pd.Timestamp(split)
        - pd.Timedelta(days=1)
        + pd.Timedelta(days=delay_days)
    )


def check_no_heldout_releases(
    observations: pd.DataFrame, settings: RunSettings, delay_days: int
) -> None:
    """Fail if a development run's German data go past `last_release`.

    Parameters
    ----------
    observations : pandas.DataFrame
        Observations with `study`, `target_date` and `as_of_date`.
    settings : RunSettings
        The run's settings. Paper runs pass unchecked.
    delay_days : int
        D for Germany.

    Raises
    ------
    ValueError
        If a German target date is held out or a release is later than
        the development truth needs.
    """
    if settings.mode != "development":
        return
    germany = observations.loc[observations["study"] == "germany"]
    last = last_release(settings.heldout_from, delay_days)
    late = (pd.to_datetime(germany["as_of_date"]) > last) | (
        pd.to_datetime(germany["target_date"]) >= settings.heldout_from
    )
    if late.any():
        raise ValueError(
            f"{int(late.sum())} German observations are held out: target "
            f"dates from {settings.heldout_from:%Y-%m-%d} or releases after "
            f"{last:%Y-%m-%d}."
        )


def check_no_heldout(forecasts: pd.DataFrame, settings: RunSettings) -> None:
    """Fail if a development run's forecasts hold held-out rows.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts with `study`, `location` and `origin_date`.
    settings : RunSettings
        The run's settings. Paper runs pass unchecked.

    Raises
    ------
    ValueError
        If a German origin is on or after the held-out date, or a synthetic
        location comes from a held-out seed.
    """
    if settings.mode != "development":
        return
    germany = forecasts.loc[forecasts["study"] == "germany"]
    late = germany.loc[
        pd.to_datetime(germany["origin_date"]) >= settings.heldout_from
    ]
    if not late.empty:
        raise ValueError(
            f"{len(late)} German rows are from held-out dates "
            f"(from {settings.heldout_from:%Y-%m-%d})."
        )
    synthetic = forecasts.loc[forecasts["study"] == "synthetic"]
    seeds = synthetic["location"].map(seed_of)
    bad = sorted(set(seeds.dropna().astype(int)) & set(settings.heldout_seeds))
    if bad:
        raise ValueError(f"Synthetic rows from held-out seeds: {bad}.")
