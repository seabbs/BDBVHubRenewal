"""JAX port of R baselinenowcast 0.2.2 with one extension.

Reporting triangles are ``(triangle, mask)`` pairs.
``mask`` is True where R has an observed cell and False where R has
NA.
The extension (E1) skips unobserved cells in the chain ladder ratio
of sums for release snapshots (``snapshots.py``).
"""

from . import _x64  # noqa: F401  (enables float64 first)
from .delay import apply_delay, estimate_and_apply_delay, estimate_delay
from .nowcast import baselinenowcast
from .sample import (
    combine_obs_with_pred,
    sample_nb,
    sample_nowcast,
    sample_nowcasts,
    sample_prediction,
)
from .snapshots import (
    chain_growth,
    exact_plan,
    hub_week_matrix,
    hub_weeks,
    pair_factors,
    pair_sums,
    window_pairs,
)
from .triangle import (
    allocate_reference_times,
    apply_reporting_structure,
    get_reporting_structure,
    identity,
    preprocess_negative_values,
    rolling_sum,
    row_sums,
    truncate_to_rows,
)
from .uncertainty import (
    estimate_uncertainty,
    estimate_uncertainty_retro,
    fit_by_horizon,
    fit_nb,
)

__all__ = [
    "allocate_reference_times",
    "apply_delay",
    "apply_reporting_structure",
    "baselinenowcast",
    "chain_growth",
    "combine_obs_with_pred",
    "estimate_and_apply_delay",
    "estimate_delay",
    "estimate_uncertainty",
    "estimate_uncertainty_retro",
    "exact_plan",
    "fit_by_horizon",
    "fit_nb",
    "get_reporting_structure",
    "hub_week_matrix",
    "hub_weeks",
    "identity",
    "pair_factors",
    "pair_sums",
    "preprocess_negative_values",
    "rolling_sum",
    "row_sums",
    "sample_nb",
    "sample_nowcast",
    "sample_nowcasts",
    "sample_prediction",
    "truncate_to_rows",
    "window_pairs",
]
