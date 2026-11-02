"""The ``baselinenowcast`` entry point for a reporting triangle.

Ports the ``reporting_triangle`` method of R baselinenowcast 0.2.2
(``baselinenowcast.R:122``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax.numpy as jnp
from jax import Array

from ._checks import coerce, validate_reporting_triangle
from .delay import apply_delay, estimate_delay
from .sample import Sampler, sample_nb, sample_nowcasts
from .triangle import (
    allocate_reference_times,
    identity,
    preprocess_negative_values,
    row_sums,
)
from .uncertainty import (
    DelayAggregator,
    RefTimeAggregator,
    estimate_uncertainty_retro,
    fit_by_horizon,
)


def baselinenowcast(
    triangle: Any,
    mask: Any,
    key: Array | None = None,
    scale_factor: float = 3,
    prop_delay: float = 0.5,
    output_type: Literal["samples", "point"] = "samples",
    draws: int = 1000,
    uncertainty_model: Callable[..., Array] = fit_by_horizon,
    uncertainty_sampler: Sampler = sample_nb,
    delay_pmf: Any = None,
    uncertainty_params: Any = None,
    preprocess: Callable[..., Array] | None = preprocess_negative_values,
    structure: Any = None,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
    validate: bool = True,
) -> Array:
    """Nowcast a reporting triangle.

    Returns the point nowcast per reference time for
    ``output_type="point"`` (row sums of the filled triangle) and
    draws of shape ``(draws, n_aggregated)`` for ``"samples"``, which
    needs ``key``.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if output_type not in ("samples", "point"):
        raise ValueError("`output_type` must be 'samples' or 'point'")
    n_req = 2 if output_type == "samples" else 0
    n_history_delay, n_retro = allocate_reference_times(
        m,
        scale_factor=scale_factor,
        prop_delay=prop_delay,
        n_min_retro_nowcasts=n_req,
        validate=False,
    )
    processed = values
    if preprocess is not None:
        processed = preprocess(values, m, validate=False)
    if delay_pmf is None:
        delay_pmf = estimate_delay(
            processed, m, n=n_history_delay, validate=False
        )
    pt = apply_delay(processed, m, delay_pmf, validate=False)
    if output_type == "point":
        return jnp.sum(pt, axis=1)
    if key is None:
        raise ValueError("`key` is required for output_type='samples'")
    if uncertainty_params is None:
        uncertainty_params = estimate_uncertainty_retro(
            processed,
            m,
            n_history_delay=n_history_delay,
            n_retrospective_nowcasts=n_retro,
            structure=structure,
            uncertainty_model=uncertainty_model,
            ref_time_aggregator=ref_time_aggregator,
            delay_aggregator=delay_aggregator,
            validate=False,
        )
    return sample_nowcasts(
        key,
        pt,
        processed,
        m,
        uncertainty_params,
        draws=draws,
        uncertainty_sampler=uncertainty_sampler,
        ref_time_aggregator=ref_time_aggregator,
        delay_aggregator=delay_aggregator,
    )
