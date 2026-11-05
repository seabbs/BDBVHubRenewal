"""JAX port of R baselinenowcast 0.2.2 with one extension.

Reporting triangles are ``(triangle, mask)`` pairs.
``mask`` is True where R has an observed cell and False where R has
NA.
The extension (E1) skips unobserved cells in the chain ladder ratio
of sums for release snapshots (``snapshots.py``).
"""

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
    age_blocks,
    fill_row,
    growth,
    hub_week_matrix,
    hub_weeks,
    log_factors,
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
    "age_blocks",
    "allocate_reference_times",
    "apply_delay",
    "apply_reporting_structure",
    "baselinenowcast",
    "combine_obs_with_pred",
    "estimate_and_apply_delay",
    "estimate_delay",
    "estimate_uncertainty",
    "estimate_uncertainty_retro",
    "fill_row",
    "fit_by_horizon",
    "fit_nb",
    "get_reporting_structure",
    "growth",
    "hub_week_matrix",
    "hub_weeks",
    "identity",
    "log_factors",
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
