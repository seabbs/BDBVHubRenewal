"""JAX port of R baselinenowcast 0.2.2.

Reporting triangles are ``(triangle, mask)`` pairs.
``mask`` is True where R has an observed cell and False where R has
NA.
See ``docs/methods/baselines.md`` for the methods.
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
from .snapshots import hub_week_matrix, hub_weeks
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
    "combine_obs_with_pred",
    "estimate_and_apply_delay",
    "estimate_delay",
    "estimate_uncertainty",
    "estimate_uncertainty_retro",
    "fit_by_horizon",
    "fit_nb",
    "get_reporting_structure",
    "hub_week_matrix",
    "hub_weeks",
    "identity",
    "preprocess_negative_values",
    "rolling_sum",
    "row_sums",
    "sample_nb",
    "sample_nowcast",
    "sample_nowcasts",
    "sample_prediction",
    "truncate_to_rows",
]
