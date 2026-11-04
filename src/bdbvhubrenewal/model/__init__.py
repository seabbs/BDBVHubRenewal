# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The province renewal model, built on PyRenew (JAX, NumPyro).

- `temporal`: a non-centred random walk or AR(1) temporal process.
- `pooled`: pooled group values and sum-to-zero group deviations.
- `renewal`: the coupled renewal with importation (`Importation`),
  seeding and depletion.
- `censored`: double interval censored daily pmfs of continuous delays.
- `hazard_delay`: a delay pmf that varies with the reference day.
- `delays`: a fixed time-varying delay.
- `observation`: the known and imputed onset streams, the missing-onset
  fraction, the hub's imputation and importation-only zeros.
- `composition_noise`: multinomial and Dirichlet-multinomial noise for
  counts by category given totals (the delay counts).
- `rt_observation`: external Rt estimates as a tempered
  pseudo-observation (the BVDOutbreakSize term).
- `model`: `ModelConfig`, the configurations, building, fitting and
  forecasting.
- `inputs`: model inputs (`Vintage`) from the tidy hub tables.

The package imports without JAX until this subpackage is imported.
"""

from .censored import censored_pmf
from .delays import (
    FixedTimeVarying,
)
from .hazard_delay import HazardDelay, hazard_to_pmf
from .model import (
    BVD,
    MAIN,
    ModelConfig,
    Vintage,
    build_model,
    fit,
    forecast_trajectories,
    generation_interval,
    incubation_period,
    model_args,
)
from .observation import (
    MissingOnsetFraction,
    OnsetStreams,
    onset_expectations,
)
from .pooled import PooledInitialValue, pooled_walk
from .renewal import (
    Importation,
    MetapopulationInfections,
    coupled_renewal,
    mixing_matrix,
)
from .rt_observation import RtPseudoObservation
from .temporal import NonCentredRandomWalk

__all__ = [
    "BVD",
    "MAIN",
    "FixedTimeVarying",
    "HazardDelay",
    "Importation",
    "MetapopulationInfections",
    "MissingOnsetFraction",
    "ModelConfig",
    "NonCentredRandomWalk",
    "OnsetStreams",
    "PooledInitialValue",
    "RtPseudoObservation",
    "Vintage",
    "build_model",
    "censored_pmf",
    "coupled_renewal",
    "fit",
    "forecast_trajectories",
    "generation_interval",
    "hazard_to_pmf",
    "incubation_period",
    "mixing_matrix",
    "model_args",
    "onset_expectations",
    "pooled_walk",
]
