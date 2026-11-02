"""Delay estimation and point nowcasts.

Ports of R baselinenowcast 0.2.2 ``estimate_delay.R``,
``apply_delay.R`` and ``estimate_and_apply_delay.R``.
The column loops are ``lax.scan`` with running row sums as the carry.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ._checks import (
    assert_integerish,
    coerce,
    n_unobserved_rows,
    validate_reporting_triangle,
)


@jax.jit
def _estimate_kernel(values: Array, mask: Array) -> Array:
    """Chain ladder fill and pmf (R .chainladder_fill_triangle, .calculate_pmf).

    ``rs`` holds the row sums of the filled columns to the left.
    The multiplication factor is the observed column total over the
    left row sums of the rows that observe the column.
    """
    zero = jnp.zeros((), values.dtype)

    def step(rs: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        col, m = x
        top = jnp.sum(jnp.where(m, col, zero))
        left = jnp.sum(jnp.where(m, rs, zero))
        mult = top / jnp.maximum(left, 1.0)
        filled = jnp.where(m, col, mult * rs)
        return rs + filled, filled

    _, rest = lax.scan(step, values[:, 0], (values[:, 1:].T, mask[:, 1:].T))
    colsum = jnp.concatenate([values[:, :1].sum(axis=0), rest.sum(axis=1)])
    return colsum / colsum.sum()


@jax.jit
def _apply_kernel(values: Array, mask: Array, pmf: Array) -> Array:
    """Fill unobserved cells (R .calc_expectation, apply_delay.R:160)."""
    cdf = jnp.cumsum(pmf)

    def step(
        rs: Array, x: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        col, m, p, cdf_prev = x
        expected = (rs + 1 - cdf_prev) / cdf_prev * p
        filled = jnp.where(m, col, expected)
        return rs + filled, filled

    xs = (values[:, 1:].T, mask[:, 1:].T, pmf[1:], cdf[:-1])
    _, rest = lax.scan(step, values[:, 0], xs)
    return jnp.concatenate([values[:, :1], rest.T], axis=1)


def _validate_for_delay_estimation(
    values: Array, mask: np.ndarray, n: int
) -> None:
    """R ``.validate_for_delay_estimation`` (estimate_delay.R:89)."""
    assert_integerish(n, "n")
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError(
            "Invalid reporting triangle structure. Each column must have "
            "at least one non-NA value."
        )
    n_rows = mask.shape[0]
    if n_rows < n:
        raise ValueError(
            f"Reporting triangle has {n_rows} reference times but n = {n} "
            "was requested."
        )
    if n < 1:
        raise ValueError(
            "Insufficient `n`, must be greater than or equal to 1."
        )
    last = slice(n_rows - n, n_rows)
    if not np.any(mask[last].all(axis=1)):
        raise ValueError(
            "The rows used for delay estimation in the reporting triangle "
            "must contain at least one row with no missing observations."
        )
    last_row = mask[-1]
    if not last_row[0]:
        raise ValueError(
            "The last row of the reporting triangle contains an NA in the "
            "first column. There must be at least an entry for the first "
            "delay at the latest reference time."
        )
    if not last_row.all() and not isinstance(values, jax.core.Tracer):
        first_na = int(np.argmin(last_row))
        lhs = np.asarray(values)[last][:, :first_na]
        if np.all(lhs == 0):
            raise ValueError(
                "The values for the recent reference times and delays only "
                "contain 0s, so the chain ladder fill and delay pmf would "
                "be invalid. Consider increasing `n`."
            )


def estimate_delay(
    triangle: Any,
    mask: Any,
    n: int | None = None,
    validate: bool = True,
) -> Array:
    """Estimate the delay pmf from the last ``n`` rows.

    R estimate_delay.R:51.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if n is None:
        n = m.shape[0]
    _validate_for_delay_estimation(values, m, int(n))
    n = int(n)
    return _estimate_kernel(values[-n:], jnp.asarray(m[-n:]))


def _validate_delay_and_triangle(mask: np.ndarray, pmf: Array) -> None:
    """R ``.validate_delay_and_triangle`` (apply_delay.R:109).

    Checks on values are skipped under ``vmap`` and ``jit``, where
    they are not concrete.
    """
    if mask.shape[1] != pmf.shape[0]:
        raise ValueError(
            "Length of the delay PMF is not the same as the number of "
            "delays in the triangle to be nowcasted."
        )
    if not mask[:, 0].all():
        raise ValueError("The first delay must be observed in every row.")
    if isinstance(pmf, jax.core.Tracer):
        return
    if not mask[-1, 1] and float(pmf[0]) == 0:
        raise ValueError(
            "Value of delay PMF at delay = 0 is 0, and the latest reference "
            "time only contains a value at delay = 0. There is insufficient "
            "information to generate a point nowcast for the latest "
            "reference time."
        )
    if float(pmf[0]) < 0:
        raise ValueError(
            "First entry of delay PMF (delay = 0) is negative "
            f"({float(pmf[0])})."
        )


def apply_delay(
    triangle: Any,
    mask: Any,
    delay_pmf: Any,
    validate: bool = True,
) -> Array:
    """Fill the unobserved cells with expected counts (R apply_delay.R:43).

    Rows are independent, so only the rows with a missing cell are
    passed to the kernel.
    The result has no missing cells.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    pmf = jnp.asarray(delay_pmf, dtype=values.dtype)
    _validate_delay_and_triangle(m, pmf)
    k = n_unobserved_rows(m)
    if k == 0:
        raise ValueError(
            "`reporting_triangle` doesn't contain any missing values, "
            "there is nothing to nowcast."
        )
    n_rows = m.shape[0]
    filled = _apply_kernel(
        values[n_rows - k :], jnp.asarray(m[n_rows - k :]), pmf
    )
    return jnp.concatenate([values[: n_rows - k], filled], axis=0)


def estimate_and_apply_delay(
    triangle: Any,
    mask: Any,
    n: int | None = None,
    delay_pmf: Any = None,
    validate: bool = True,
) -> Array:
    """Estimate the delay (unless given) and nowcast.

    R estimate_and_apply_delay.R:44.
    """
    values, m = coerce(triangle, mask)
    if validate:
        validate_reporting_triangle(m)
    if n is None:
        n = m.shape[0]
    if n > m.shape[0]:
        raise ValueError(
            "The number of observations (rows) specified for delay "
            "estimation must be less than or equal to the number of rows "
            "of the reporting triangle."
        )
    if not np.any(m[m.shape[0] - n :].all(axis=1)):
        raise ValueError(
            "The rows used for delay estimation in the reporting triangle "
            "must contain at least one row with no missing observations."
        )
    if delay_pmf is None:
        delay_pmf = estimate_delay(values, m, n=n, validate=False)
    return apply_delay(values, m, delay_pmf, validate=False)
