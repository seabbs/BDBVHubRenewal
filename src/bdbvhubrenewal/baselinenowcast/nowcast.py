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

    Estimates the delay and its uncertainty from the triangle, fills
    the unobserved cells and, for samples, draws the counts still to
    be reported.

    Parameters
    ----------
    triangle : array_like
        Reporting triangle, reference times by delays.
    mask : array_like of bool
        True where a cell is observed.
    key : jax.Array, optional
        Random key. Needed for ``output_type="samples"``.
    scale_factor, prop_delay : float
        Passed to `allocate_reference_times`.
    output_type : {"samples", "point"}, default "samples"
        Draws, or the point nowcast.
    draws : int, default 1000
        Number of draws.
    uncertainty_model : callable, default fit_by_horizon
        Fits the uncertainty parameters.
    uncertainty_sampler : Sampler, default sample_nb
        Draws counts given a mean and the parameters.
    delay_pmf, uncertainty_params : array_like, optional
        Use these instead of estimating them.
    preprocess : callable or None, default preprocess_negative_values
        Applied to the triangle first.
    structure : int or array_like, optional
        Reporting structure. Defaults to `get_reporting_structure`.
    ref_time_aggregator : RefTimeAggregator, default identity
        Aggregates reference times, such as `rolling_sum`.
    delay_aggregator : DelayAggregator, default row_sums
        Sums over delays.
    validate : bool, default True
        Check the triangle and mask first.

    Returns
    -------
    jax.Array
        For ``"point"``, the point nowcast per reference time (row sums
        of the filled triangle). For ``"samples"``, draws of shape
        ``(draws, n_aggregated)``.

    Examples
    --------
    >>> import jax
    >>> import numpy as np
    >>> from bdbvhubrenewal.baselinenowcast import baselinenowcast
    >>> rng = np.random.default_rng(1)
    >>> triangle = rng.poisson(10, size=(20, 4)).astype(float)
    >>> mask = np.add.outer(np.arange(20), np.arange(4)) < 20
    >>> draws = baselinenowcast(
    ...     triangle * mask, mask, key=jax.random.key(1), draws=100
    ... )
    >>> draws.shape
    (100, 20)
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
