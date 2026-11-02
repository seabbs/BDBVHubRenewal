"""Sampling nowcasts.

Ports of R baselinenowcast 0.2.2 ``sample_distrib.R`` and
``sample_predictions.R``.
Randomness takes a ``jax.random`` key.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._checks import coerce, validate_reporting_triangle
from .triangle import identity, row_sums
from .uncertainty import DelayAggregator, RefTimeAggregator

Sampler = Callable[[Array, Array, Array], Array]


def sample_nb(key: Array, pred: Any, uncertainty_params: Any) -> Array:
    """Negative binomial draws with mean ``pred`` and size ``params``.

    R sample_distrib.R:18 (``rnbinom(size, mu)``), drawn as a gamma
    mixed Poisson.
    ``uncertainty_params`` has length 1 or matches ``pred``.
    """
    mu = jnp.asarray(pred)
    size = jnp.asarray(uncertainty_params, dtype=mu.dtype)
    if size.ndim > 0 and size.size not in (1, mu.size):
        raise ValueError(
            "`uncertainty_params` must have length 1 or match length(pred)."
        )
    size = size.reshape(()) if size.size == 1 else size.reshape(mu.shape)
    k_gamma, k_pois = jax.random.split(key)
    lam = jax.random.gamma(k_gamma, size, mu.shape, dtype=mu.dtype) * mu / size
    return jax.random.poisson(k_pois, lam).astype(mu.dtype)


class _Prepared:
    """Deterministic parts of a nowcast draw."""

    def __init__(
        self,
        mean_pred: Array,
        params_use: Array,
        obs_counts: Array,
        n_agg: int,
    ) -> None:
        self.mean_pred = mean_pred
        self.params_use = params_use
        self.obs_counts = obs_counts
        self.n_agg = n_agg


def _prepare(
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    ref_time_aggregator: RefTimeAggregator,
    delay_aggregator: DelayAggregator,
    validate: bool = True,
) -> _Prepared:
    """Aggregate and split the nowcast (R sample_prediction, :55)."""
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    nc = jnp.asarray(point_nowcast_matrix)
    params = jnp.atleast_1d(jnp.asarray(uncertainty_params, dtype=nc.dtype))
    if params.shape[0] > nc.shape[0]:
        raise ValueError(
            "Vector of uncertainty parameters is greater than the number of "
            "reference times in `point_nowcast_matrix`."
        )
    if nc.shape != values.shape:
        raise ValueError(
            "`point_nowcast_matrix` and `reporting_triangle` must have the "
            "same shape."
        )
    # .extract_predictions (sample_predictions.R:138)
    # The check needs concrete values, so it is skipped under tracing.
    traced = isinstance(nc, jax.core.Tracer) or isinstance(
        values, jax.core.Tracer
    )
    if not traced and not bool(
        jnp.all(jnp.where(jnp.asarray(m), nc == values, True))
    ):
        raise ValueError(
            "`reporting_triangle` is not a subset of `point_nowcast_matrix`."
        )
    agg_nc, _ = ref_time_aggregator(nc, np.ones(nc.shape, dtype=bool))
    agg_rt, agg_m = ref_time_aggregator(values, m)
    n_agg = agg_rt.shape[0]
    n_horizons = int(np.sum(~agg_m.all(axis=1)))
    if params.shape[0] < n_horizons:
        raise ValueError(
            "Vector of uncertainty parameters is less than the number of "
            "horizons in the `reporting_triangle`."
        )
    pred_cells = jnp.where(jnp.asarray(~agg_m), agg_nc, 0.0)
    mean_pred_all = delay_aggregator(pred_cells)
    if mean_pred_all.shape != (n_agg,):
        raise ValueError("`delay_aggregator` must return a vector.")
    obs_counts = delay_aggregator(agg_rt)
    if n_horizons == 0:
        return _Prepared(mean_pred_all[:0], params[:0], obs_counts, n_agg)
    # tail(params, n_horizons), reversed (sample_predictions.R:106)
    params_use = params[-n_horizons:][::-1]
    return _Prepared(
        mean_pred_all[n_agg - n_horizons :], params_use, obs_counts, n_agg
    )


def _draw(key: Array, prep: _Prepared, sampler: Sampler) -> Array:
    """One padded prediction draw, zero for fully observed rows."""
    n_h = prep.mean_pred.shape[0]
    if n_h == 0:
        return jnp.zeros((prep.n_agg,), prep.obs_counts.dtype)
    drawn = sampler(key, prep.mean_pred, prep.params_use)
    return jnp.concatenate([jnp.zeros((prep.n_agg - n_h,), drawn.dtype), drawn])


def sample_prediction(
    key: Array,
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    uncertainty_sampler: Sampler = sample_nb,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """One draw of the counts still to be reported (R sample_predictions.R:55).

    Returns a vector over aggregated reference times with zeros for
    rows that are fully observed.
    The last ``n`` of the uncertainty parameters are used, as R's
    ``tail``, most recent horizon first.
    """
    prep = _prepare(
        point_nowcast_matrix,
        triangle,
        mask,
        uncertainty_params,
        ref_time_aggregator,
        delay_aggregator,
    )
    return _draw(key, prep, uncertainty_sampler)


def combine_obs_with_pred(
    predicted_counts: Any,
    triangle: Any,
    mask: Any,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """Observed totals plus predicted counts (R sample_predictions.R:189)."""
    values, m = coerce(triangle, mask)
    agg, _ = ref_time_aggregator(values, m)
    return delay_aggregator(agg) + jnp.asarray(predicted_counts)


def sample_nowcast(
    key: Array,
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    uncertainty_sampler: Sampler = sample_nb,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """One nowcast draw over aggregated reference times (R :312)."""
    prep = _prepare(
        point_nowcast_matrix,
        triangle,
        mask,
        uncertainty_params,
        ref_time_aggregator,
        delay_aggregator,
    )
    return prep.obs_counts + _draw(key, prep, uncertainty_sampler)


def sample_nowcasts(
    key: Array,
    point_nowcast_matrix: Any,
    triangle: Any,
    mask: Any,
    uncertainty_params: Any,
    draws: int = 1000,
    uncertainty_sampler: Sampler = sample_nb,
    ref_time_aggregator: RefTimeAggregator = identity,
    delay_aggregator: DelayAggregator = row_sums,
) -> Array:
    """Draws of the nowcast, shape ``(draws, n_aggregated)`` (R :372).

    Draws are ``vmap``ped over split keys.
    With a rolling aggregator the leading rows R pads with NA are
    dropped, so the columns are the last ``n_aggregated`` reference
    times.
    """
    if draws < 1:
        raise ValueError("`draws` must be at least 1")
    prep = _prepare(
        point_nowcast_matrix,
        triangle,
        mask,
        uncertainty_params,
        ref_time_aggregator,
        delay_aggregator,
    )
    keys = jax.random.split(key, draws)
    drawn = jax.vmap(lambda k: _draw(k, prep, uncertainty_sampler))(keys)
    return prep.obs_counts[None, :] + drawn
