# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fit configurations for the automated submission workflow.

The submission CI (`ci/`, see `ci/INTEGRATION.md`) fits each candidate of
its fallback chain with `bdbv-hub-renewal fit --config <name>` and then
forecasts from the saved fit with `bdbv-hub-renewal nowcast --config <name>
--fit <dir>`. A fit configuration says how to do both.

The chain names `main` (the full model, the `renewal` variant), with
`bvd` as a shadow. Both run the PyRenew renewal model through
`bdbv_hub_renewal.forecasters.renewal.fit_renewal`, which samples, draws the
trajectories and writes `ppc.json`. `persistence` needs no fit.

A fit function gets the forecast task, the sampler settings from
`ci/policy.toml` and the fit directory. It writes whatever the forecast
needs into the directory and returns the `diagnostics.json` record. It
may append progress lines to `progress.log`. The forecast function gets
the task and the directory and returns province samples (columns
`SAMPLE_COLUMNS`), optionally with an `unassigned` frame for a national
component not assigned to any province.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

import pandas as pd

from bdbv_hub_renewal.forecasters.base import (
    ForecastTask,
    check_samples,
    get_model,
)

__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "FitConfig",
    "FitResult",
    "Forecast",
    "get_config",
    "list_configs",
    "no_sampler_diagnostics",
    "persistence_config",
    "read_fit",
    "register_config",
    "run_fit",
    "run_forecast",
]

DIAGNOSTICS_SCHEMA_VERSION: Final = 1
"""Version of the `diagnostics.json` format in `ci/INTEGRATION.md`."""

FIT_RECORD: Final = "fit.json"
"""File in the fit directory recording what was fitted."""


@dataclass(frozen=True)
class Forecast:
    """Samples from a fit, before the hub table is made.

    Attributes
    ----------
    samples : pandas.DataFrame
        Province samples, columns `SAMPLE_COLUMNS`.
    unassigned : pandas.DataFrame or None
        A modelled national component not assigned to any province, with
        columns `target_end_date`, `sample` and `value`, or None.
    modelled : tuple of str or None
        The provinces the model reports, for `national.json`. None means
        every province of the task.
    national : pandas.DataFrame or None
        National samples from a model that fits the nation on its own
        (`target_end_date`, `sample`, `value`). They are submitted as they
        are, not as the sum of the provinces.
    """

    samples: pd.DataFrame
    unassigned: pd.DataFrame | None = None
    modelled: tuple[str, ...] | None = None
    national: pd.DataFrame | None = None


FitFn = Callable[[ForecastTask, Mapping[str, Any], Path], dict[str, Any]]
ForecastFn = Callable[[ForecastTask, Path], Forecast]


@dataclass(frozen=True)
class FitConfig:
    """A named way to fit a model and forecast from the fit.

    Attributes
    ----------
    name : str
        The name the CI passes as `--config`.
    description : str
        One line for logs and docs.
    fit : callable
        `fit(task, sampler, fit_dir) -> diagnostics record`.
    forecast : callable
        `forecast(task, fit_dir) -> Forecast`.
    stand_in : str or None
        If set, this configuration stands in for one not built yet, and
        the text says so, as a fragment that reads after "Stand-in:".
        It is recorded in `fit.json` and in the diagnostics, and printed
        by the CLI.
    """

    name: str
    description: str
    fit: FitFn
    forecast: ForecastFn
    stand_in: str | None = None


_CONFIGS: dict[str, FitConfig] = {}


def register_config(config: FitConfig, *, replace: bool = False) -> FitConfig:
    """Add a fit configuration.

    Parameters
    ----------
    config : FitConfig
        The configuration.
    replace : bool
        Allow replacing one of the same name, as the model integration
        does for the `main` stand-in.

    Returns
    -------
    FitConfig
        The registered configuration.

    Raises
    ------
    ValueError
        If the name is taken and `replace` is False.
    """
    if config.name in _CONFIGS and not replace:
        raise ValueError(f"Fit config {config.name!r} is already registered.")
    _CONFIGS[config.name] = config
    return config


def get_config(name: str) -> FitConfig:
    """Return a registered fit configuration.

    Parameters
    ----------
    name : str
        The configuration name.

    Returns
    -------
    FitConfig
        The configuration.

    Raises
    ------
    KeyError
        If there is none of that name.
    """
    if name not in _CONFIGS:
        raise KeyError(f"No fit config {name!r}. Known: {sorted(_CONFIGS)}.")
    return _CONFIGS[name]


def list_configs() -> list[FitConfig]:
    """List the registered fit configurations by name.

    Returns
    -------
    list of FitConfig
        The configurations.
    """
    return [_CONFIGS[k] for k in sorted(_CONFIGS)]


def no_sampler_diagnostics(
    model: str, config: str, reason: str
) -> dict[str, Any]:
    """Return the diagnostics record of a fit that has no sampler.

    The CI's convergence gate reads `sampler: none` as not applicable,
    which neither passes nor blocks a candidate.

    Parameters
    ----------
    model : str
        The model that was run.
    config : str
        The fit configuration.
    reason : str
        Why convergence does not apply.

    Returns
    -------
    dict
        The `diagnostics.json` record.
    """
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "model": model,
        "config": config,
        "sampler": "none",
        "status": "not-applicable",
        "reason": reason,
    }


def _ppc_not_available(model: str, config: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "model": model,
        "config": config,
        "reason": reason,
    }


def persistence_config(
    name: str, stand_in: str | None = None, model: str = "persistence"
) -> FitConfig:
    """Build a fit configuration that runs a registered sample-free model.

    There is nothing to fit. `fit` records the task and writes
    not-applicable diagnostics and posterior predictive checks, and
    `forecast` runs the model on the task.

    Parameters
    ----------
    name : str
        The configuration name.
    stand_in : str, optional
        Set when the configuration stands in for one not built yet.
    model : str
        The registered model to run.

    Returns
    -------
    FitConfig
        The configuration.
    """
    reason = f"{model} has no sampler and no posterior"

    def fit(
        task: ForecastTask, sampler: Mapping[str, Any], fit_dir: Path
    ) -> dict[str, Any]:
        del task, sampler
        _write_json(
            fit_dir / "ppc.json", _ppc_not_available(model, name, reason)
        )
        diag = no_sampler_diagnostics(model, name, reason)
        if stand_in:
            diag["stand_in"] = stand_in
        return diag

    def forecast(task: ForecastTask, fit_dir: Path) -> Forecast:
        del fit_dir
        return Forecast(get_model(model).forecast(task))

    return FitConfig(
        name=name,
        description=f"{model} model, nothing to fit",
        fit=fit,
        forecast=forecast,
        stand_in=stand_in,
    )


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


@dataclass(frozen=True)
class FitResult:
    """What `run_fit` wrote.

    Attributes
    ----------
    fit_dir : pathlib.Path
        The fit directory.
    diagnostics : dict
        The `diagnostics.json` record.
    """

    fit_dir: Path
    diagnostics: dict[str, Any]


def run_fit(
    config: FitConfig,
    task: ForecastTask,
    sampler: Mapping[str, Any],
    fit_dir: Path | str,
    *,
    model: str,
) -> FitResult:
    """Fit a configuration and write the fit directory.

    Writes `fit.json` (the configuration, model, vintage, reference date,
    sampler settings and any stand-in note), `diagnostics.json` and
    `progress.log`. The diagnostics are written whether or not the fit
    converged; the CI's gate judges them.

    Parameters
    ----------
    config : FitConfig
        The configuration.
    task : ForecastTask
        The forecast task built from the vintage.
    sampler : mapping
        Sampler settings, the candidate's `sampler` table in
        `ci/policy.toml`.
    fit_dir : pathlib.Path or str
        Where to write.
    model : str
        The model name the CI passed (`--model`), recorded as is.

    Returns
    -------
    FitResult
        The directory and the diagnostics.
    """
    out = Path(fit_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.log"
    with progress.open("a") as fh:
        fh.write(f"config={config.name} vintage={task.vintage} start\n")
    start = time.monotonic()
    diag = config.fit(task, sampler, out)
    diag.setdefault("schema_version", DIAGNOSTICS_SCHEMA_VERSION)
    diag.setdefault("config", config.name)
    diag.setdefault("runtime_seconds", round(time.monotonic() - start, 3))
    _write_json(out / "diagnostics.json", diag)
    _write_json(
        out / FIT_RECORD,
        {
            "model": model,
            "config": config.name,
            "description": config.description,
            "stand_in": config.stand_in,
            "vintage": task.vintage,
            "reference_date": task.reference_date.isoformat(),
            "sampler": dict(sampler),
        },
    )
    with progress.open("a") as fh:
        fh.write(f"config={config.name} done\n")
    return FitResult(out, diag)


def read_fit(fit_dir: Path | str) -> dict[str, Any]:
    """Read the `fit.json` record of a fit directory.

    Parameters
    ----------
    fit_dir : pathlib.Path or str
        The fit directory.

    Returns
    -------
    dict
        The record written by `run_fit`.

    Raises
    ------
    FileNotFoundError
        If the directory holds no fit.
    """
    path = Path(fit_dir) / FIT_RECORD
    if not path.exists():
        raise FileNotFoundError(f"No fit in {fit_dir} ({FIT_RECORD} missing).")
    record: dict[str, Any] = json.loads(path.read_text())
    return record


def run_forecast(
    config: FitConfig, task: ForecastTask, fit_dir: Path | str | None
) -> Forecast:
    """Forecast from a fit, checking the fit matches the task.

    Parameters
    ----------
    config : FitConfig
        The configuration.
    task : ForecastTask
        The forecast task.
    fit_dir : pathlib.Path or str, optional
        The fit directory written by `run_fit`. Required unless the
        configuration needs no fit, which the persistence stand-ins do not.

    Returns
    -------
    Forecast
        Province samples and any unassigned national component.

    Raises
    ------
    ValueError
        If the fit is of another configuration or vintage.
    """
    path = Path(fit_dir) if fit_dir is not None else Path()
    if fit_dir is not None:
        record = read_fit(path)
        if record["config"] != config.name:
            raise ValueError(
                f"The fit in {fit_dir} is of config {record['config']!r}, "
                f"not {config.name!r}."
            )
        if record["vintage"] != task.vintage:
            raise ValueError(
                f"The fit in {fit_dir} is of vintage {record['vintage']}, "
                f"not {task.vintage}."
            )
    result = config.forecast(task, path)
    check_samples(result.samples, task)
    return result


register_config(persistence_config("persistence"))


def _renewal_config(name: str) -> FitConfig:
    from bdbv_hub_renewal.forecasters.renewal import (
        FIT_VARIANTS,
        VARIANTS,
        fit_renewal,
        forecast_renewal,
    )

    variant = FIT_VARIANTS[name]
    return FitConfig(
        name=name,
        description=VARIANTS[variant].description,
        fit=partial(fit_renewal, variant=variant, config_name=name),
        forecast=forecast_renewal,
    )


for _name in ("main", "bvd"):
    register_config(_renewal_config(_name))
