# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Forecasters: the interface every model meets, the registry and the models.

- `base`: `ForecastTask`, the `ForecastModel` protocol and the registry.
- `persistence`: the persistence baseline.
- `renewal`: the renewal variants, the adapter from a `ForecastTask` to
  the PyRenew model in `bdbvhubrenewal.model`.

Importing this package registers the built-in models.
To add a variant, write a class with `name` and `forecast(task)` and call
`register_model` in its module, then import that module here.
"""

from bdbvhubrenewal.forecasters.base import (
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
from bdbvhubrenewal.forecasters.persistence import (
    Persistence,
    completeness_lag,
    last_complete_week,
)
from bdbvhubrenewal.forecasters.renewal import VARIANTS, RenewalModel

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
