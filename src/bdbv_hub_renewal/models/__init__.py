# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Forecast models and the model registry.

Importing this package registers the built-in models.
To add a variant, write a class with `name` and `forecast(task)` and call
`register_model` in its module, then import that module here.
"""

from bdbv_hub_renewal.models.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastModel,
    ForecastTask,
    ModelSpec,
    add_national,
    check_samples,
    get_model,
    list_models,
    register_model,
)
from bdbv_hub_renewal.models.persistence import (
    Persistence,
    completeness_lag,
    last_complete_week,
)
from bdbv_hub_renewal.models.renewal import VARIANTS, RenewalModel

__all__ = [
    "NATIONAL",
    "SAMPLE_COLUMNS",
    "VARIANTS",
    "ForecastModel",
    "ForecastTask",
    "ModelSpec",
    "Persistence",
    "RenewalModel",
    "add_national",
    "check_samples",
    "completeness_lag",
    "get_model",
    "last_complete_week",
    "list_models",
    "register_model",
]
