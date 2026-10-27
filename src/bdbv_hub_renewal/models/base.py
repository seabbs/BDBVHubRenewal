# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The forecast task, the model interface and the model registry.

A model is anything with a `name` and a `forecast(task)` method that
returns sample trajectories for the task's provinces and weeks.
The harness adds the national total, writes hub files, scores and plots.
So adding a model variant is one class and one `register_model` call.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..dates import as_date

__all__ = [
    "NATIONAL",
    "SAMPLE_COLUMNS",
    "ForecastModel",
    "ForecastTask",
    "ModelSpec",
    "add_national",
    "check_samples",
    "get_model",
    "list_models",
    "register_model",
]

NATIONAL = "CD"
"""Location code of the whole country."""

SAMPLE_COLUMNS = ("location", "target_end_date", "sample", "value")
"""Columns of a model's sample output."""


@dataclass(frozen=True)
class ForecastTask:
    """Everything a model sees for one forecast.

    Attributes
    ----------
    vintage : str
        The `as_of` date of the data, ISO format.
    reference_date : datetime.date
        The round's reference date.
    kind : str
        `"real"`, `"reconstructed"` or `"simulated"`.
    target : pandas.DataFrame
        The target table rows of this vintage only (columns as in
        `bdbv_hub_renewal.data.tidy_target`), for the country and
        provinces.
    delays : pandas.DataFrame or None
        Delay line data available at this vintage (see
        `bdbv_hub_renewal.data.tidy_delays`), or None if none was published.
    locations : tuple of str
        Provinces to forecast.
    target_end_dates : tuple of str
        Weeks to forecast, ISO Saturdays.
    n_samples : int
        Number of sample trajectories.
    seed : int
        Seed for the model's random numbers.
    mask : tuple of str
        Provinces whose counts the model must not fit to, for the
        leave-one-province-out check. They stay in `locations`, so they
        are still forecast, and in `target`, so a model can keep them in
        its latent process (for spatial coupling) while dropping them from
        its likelihood. Only models with `supports_mask = True` are given
        a non-empty mask.
    """

    vintage: str
    reference_date: dt.date
    kind: str
    target: pd.DataFrame
    delays: pd.DataFrame | None
    locations: tuple[str, ...]
    target_end_dates: tuple[str, ...]
    n_samples: int = 100
    seed: int = 0
    mask: tuple[str, ...] = ()

    def rng(self) -> np.random.Generator:
        """Return a random generator seeded from the task.

        Returns
        -------
        numpy.random.Generator
            Seeded from `seed` and the vintage, so tasks differ.
        """
        offset = as_date(self.vintage).toordinal()
        return np.random.default_rng([self.seed, offset])


@runtime_checkable
class ForecastModel(Protocol):
    """A forecasting model.

    Attributes
    ----------
    name : str
        Short name, used in tables and plots and as the hub model
        abbreviation (letters, digits and underscores, under 16
        characters).
    """

    name: str

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return sample trajectories for a task.

        Parameters
        ----------
        task : ForecastTask
            The data and the weeks to forecast.

        Returns
        -------
        pandas.DataFrame
            Columns `SAMPLE_COLUMNS`: one row per province, week and
            sample, with samples numbered 1 to `task.n_samples`. A sample
            number is one joint trajectory across weeks and provinces.
        """
        ...


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    Attributes
    ----------
    name : str
        Registry key.
    factory : callable
        Builds the model with no arguments.
    description : str
        One line for tables and docs.
    baseline : bool
        Whether this is the reference model for relative scores.
    fitted : bool
        Whether the model is fitted by MCMC for each task, so is slow.
        The docs report and the smoke tests run only models that are not.
    """

    name: str
    factory: Callable[[], ForecastModel]
    description: str
    baseline: bool = False
    fitted: bool = False


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    factory: Callable[[], ForecastModel],
    description: str,
    *,
    baseline: bool = False,
    fitted: bool = False,
    replace: bool = False,
) -> ModelSpec:
    """Add a model to the registry used by the backtest, CLI and docs.

    Parameters
    ----------
    name : str
        Registry key and model name.
    factory : callable
        Builds the model with no arguments.
    description : str
        One line describing the model.
    baseline : bool
        Mark the model as the baseline for relative skill.
    fitted : bool
        Mark the model as fitted by MCMC for each task.
    replace : bool
        Allow replacing an existing entry.

    Returns
    -------
    ModelSpec
        The registered entry.

    Raises
    ------
    ValueError
        If `name` is already registered and `replace` is False.
    """
    if name in _REGISTRY and not replace:
        raise ValueError(f"Model {name!r} is already registered.")
    spec = ModelSpec(name, factory, description, baseline, fitted)
    _REGISTRY[name] = spec
    return spec


def get_model(name: str) -> ForecastModel:
    """Build a registered model by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    ForecastModel
        A new instance.

    Raises
    ------
    KeyError
        If no model has that name.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No model {name!r}. Known: {sorted(_REGISTRY)}.")
    return _REGISTRY[name].factory()


def list_models() -> list[ModelSpec]:
    """List registered models, baseline first.

    Returns
    -------
    list of ModelSpec
        The registry entries.
    """
    return sorted(_REGISTRY.values(), key=lambda s: (not s.baseline, s.name))


def check_samples(samples: pd.DataFrame, task: ForecastTask) -> None:
    """Check a model's output against its task.

    Parameters
    ----------
    samples : pandas.DataFrame
        Output of `ForecastModel.forecast`.
    task : ForecastTask
        The task it answers.

    Raises
    ------
    ValueError
        If columns, locations, weeks or sample numbers do not match, or
        any value is negative or missing.
    """
    if tuple(samples.columns) != SAMPLE_COLUMNS:
        raise ValueError(f"Columns must be {SAMPLE_COLUMNS}.")
    if set(samples["location"]) != set(task.locations):
        raise ValueError("Locations do not match the task.")
    if set(samples["target_end_date"]) != set(task.target_end_dates):
        raise ValueError("Weeks do not match the task.")
    want = set(range(1, task.n_samples + 1))
    if set(samples["sample"]) != want:
        raise ValueError(f"Samples must be numbered 1 to {task.n_samples}.")
    expected = len(task.locations) * len(task.target_end_dates) * len(want)
    if len(samples) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(samples)}.")
    if samples["value"].isna().any() or (samples["value"] < 0).any():
        raise ValueError("Values must be non-negative and not missing.")


def add_national(samples: pd.DataFrame) -> pd.DataFrame:
    """Add the national total as the sum of provinces by sample.

    Parameters
    ----------
    samples : pandas.DataFrame
        Province samples with columns `SAMPLE_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        The input with `CD` rows appended.
    """
    provinces = samples.loc[samples["location"] != NATIONAL]
    national = provinces.groupby(["target_end_date", "sample"], as_index=False)[
        "value"
    ].sum()
    national.insert(0, "location", NATIONAL)
    return pd.concat([provinces, national[list(SAMPLE_COLUMNS)]]).reset_index(
        drop=True
    )
