# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Forecasters: the interface every model meets, the registry and the models.

- `base`: `ForecastTask`, the `ForecastModel` protocol and the registry.
- `randomwalk`: the random walk baseline, and random walk on a nowcast,
  with `chain` (the daily chain ladder), `pooling`, `obs` and `walk`.
- `pprandomwalk`: a generative random walk nowcast, the walk and the
  delay fitted jointly with partial pooling (numpyro).
- `renewal`: the renewal variants, the adapter from a `ForecastTask` to
  the PyRenew model in `bdbvhubrenewal.model`.
- `foundation`: the pretrained Chronos-2 model on two layouts of the hub
  target's releases.
- `zones`: health-zone forecasts split from any model's province samples.

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
    ModelUnavailableError,
    add_national,
    check_samples,
    get_model,
    list_models,
    register_model,
)
from bdbvhubrenewal.forecasters.foundation import FoundationModel
from bdbvhubrenewal.forecasters.pprandomwalk import PPRandomWalkNowcast
from bdbvhubrenewal.forecasters.randomwalk import (
    RandomWalk,
    RandomWalkNowcast,
)
from bdbvhubrenewal.forecasters.renewal import VARIANTS, RenewalModel

__all__ = [
    "NATIONAL",
    "SAMPLE_COLUMNS",
    "VARIANTS",
    "ForecastModel",
    "ForecastTask",
    "FoundationModel",
    "ModelSpec",
    "ModelUnavailableError",
    "PPRandomWalkNowcast",
    "RandomWalk",
    "RandomWalkNowcast",
    "RenewalModel",
    "add_national",
    "check_samples",
    "get_model",
    "list_models",
    "register_model",
]
