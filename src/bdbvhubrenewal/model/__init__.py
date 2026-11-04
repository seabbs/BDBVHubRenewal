# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The province renewal model, built on PyRenew (JAX, NumPyro).

- `rt`: non-centred random walks, pooled initial values and sum-to-zero
  province deviations.
- `renewal`: the coupled renewal with importation (`Importation`),
  seeding and depletion.
- `delays`: the generation interval and the onset-to-report delay.
- `observation`: the known and imputed onset streams, the hub's
  imputation and importation-only zeros.
- `composition_noise`: multinomial and Dirichlet-multinomial noise for
  counts by category given totals (the delay counts).
- `rt_observation`: external Rt estimates as a tempered
  pseudo-observation (the BVDOutbreakSize term).
- `model`: `ModelConfig`, the configurations, building, fitting and
  forecasting.
- `inputs`: model inputs (`Vintage`) from the tidy hub tables.

The package imports without JAX until this subpackage is imported.
"""

from .delays import (
    FixedTimeVarying,
    HazardDelay,
    censored_pmf,
    hazard_to_pmf,
)
from .model import (
    BVD,
    MAIN,
    ModelConfig,
    Vintage,
    build_model,
    fit,
    forecast_trajectories,
    generation_interval,
    model_args,
)
from .observation import OnsetStreams, onset_expectations
from .renewal import (
    Importation,
    MetapopulationInfections,
    coupled_renewal,
    mixing_matrix,
)
from .rt import (
    MissingOnsetFraction,
    NonCentredRandomWalk,
    PooledInitialValue,
    pooled_walk,
)
from .rt_observation import RtPseudoObservation

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
    "mixing_matrix",
    "model_args",
    "onset_expectations",
    "pooled_walk",
]
