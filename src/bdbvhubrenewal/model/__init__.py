# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The province renewal model, built on PyRenew (JAX, NumPyro).

- `rt`: random walks on knots, pooled initial values and sum-to-zero
  province deviations.
- `renewal`: the coupled renewal with importation, seeding and depletion.
- `delays`: the generation interval and the onset-to-report delay.
- `observation`: the known and imputed onset streams, the hub's
  imputation, importation-only zeros and the BVDOutbreakSize term.
- `model`: `ModelConfig`, the configurations, building, fitting and
  forecasting.
- `inputs`: model inputs (`Vintage`) from the tidy hub tables.

The package imports without JAX until this subpackage is imported.
"""

from .delays import (
    FixedTimeVarying,
    HazardDelay,
    ParametricDelay,
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
from .observation import OnsetStreams, RtPseudoObservation, onset_expectations
from .renewal import (
    LearntKernel,
    MetapopulationInfections,
    coupled_renewal,
    flow_pair_basis,
    importation_flow_factor,
    mixing_matrix,
)
from .rt import (
    MissingOnsetFraction,
    NonCentredRandomWalk,
    PooledInitialValue,
    pooled_walk,
)

__all__ = [
    "BVD",
    "MAIN",
    "FixedTimeVarying",
    "HazardDelay",
    "LearntKernel",
    "MetapopulationInfections",
    "MissingOnsetFraction",
    "ModelConfig",
    "NonCentredRandomWalk",
    "OnsetStreams",
    "ParametricDelay",
    "PooledInitialValue",
    "RtPseudoObservation",
    "Vintage",
    "build_model",
    "censored_pmf",
    "coupled_renewal",
    "fit",
    "flow_pair_basis",
    "forecast_trajectories",
    "generation_interval",
    "hazard_to_pmf",
    "importation_flow_factor",
    "mixing_matrix",
    "model_args",
    "onset_expectations",
    "pooled_walk",
]
