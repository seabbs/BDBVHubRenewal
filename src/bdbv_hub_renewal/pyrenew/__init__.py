# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The province renewal model on PyRenew (JAX).

The modules here are synced from the prototype by
`scripts/sync_pyrenew.py`, so the final model drops in by re-running it.
The repository's adapter is `bdbv_hub_renewal.models.renewal`, which
imports this subpackage lazily: `import bdbv_hub_renewal` works without
the `model` extra.
"""

from .latent import adjacency_kernel
from .model import (
    FALLBACK,
    MAIN,
    ModelConfig,
    Vintage,
    build_model,
    fit,
    forecast_trajectories,
)

__all__ = [
    "FALLBACK",
    "MAIN",
    "ModelConfig",
    "Vintage",
    "adjacency_kernel",
    "build_model",
    "fit",
    "forecast_trajectories",
]
