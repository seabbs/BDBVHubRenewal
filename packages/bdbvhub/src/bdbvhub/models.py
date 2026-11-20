# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The model registry, with every built-in model registered.

The models live in their own packages: the random walks in
`nowcastbaselines.randomwalk`, Chronos-2 in `nowcastfoundation.forecaster`,
and the renewal variants in `nowcastrenewal.forecaster` and
`nowcastrenewal.zones`. The registry itself is `nowcastutils.base`.

Importing this module registers the built-in models, so look models up
here (`get_model`, `list_models`).
To add a variant, write a class with `name` and `forecast(task)`, call
`register_model` in its module, then import that module here.
"""

from nowcastbaselines.randomwalk import RandomWalk, RandomWalkNowcast
from nowcastfoundation.forecaster import FoundationModel
from nowcastrenewal.forecaster import VARIANTS, RenewalModel
from nowcastrenewal.zones import ZoneRenewalModel
from nowcastutils.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastModel,
    ForecastTask,
    ModelSpec,
    ModelUnavailableError,
    add_national,
    check_samples,
    get_model,
    list_models,
    register_model,
)

__all__ = [
    "NATIONAL",
    "SAMPLE_COLUMNS",
    "VARIANTS",
    "ForecastModel",
    "ForecastTask",
    "FoundationModel",
    "ModelSpec",
    "ModelUnavailableError",
    "RandomWalk",
    "RandomWalkNowcast",
    "RenewalModel",
    "ZoneRenewalModel",
    "add_national",
    "check_samples",
    "get_model",
    "list_models",
    "register_model",
]
