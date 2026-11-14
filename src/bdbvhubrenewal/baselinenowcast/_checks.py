"""Input coercion and validation shared by the baselinenowcast port.

R marks unobserved cells with NA.
Here every reporting triangle is a pair ``(triangle, mask)`` with
``mask`` True where a cell is observed.
Values under False are set to zero and never read.
Validation runs eagerly on concrete arrays, outside ``jit``, and raises
``ValueError`` under the same conditions as R's ``cli_abort`` calls.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

#: Tolerance of checkmate::assert_integerish (sqrt of machine epsilon).
INTEGERISH_TOL = float(np.sqrt(np.finfo(np.float64).eps))


def coerce(triangle: object, mask: object) -> tuple[Array, np.ndarray]:
    """Return ``(values, mask)`` with values under False set to zero.

    The mask is a NumPy array because it drives shapes and validation.
    """
    values = jnp.asarray(triangle)
    m = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("`triangle` must be a matrix")
    if values.shape != m.shape:
        raise ValueError("`triangle` and `mask` must have the same shape")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.result_type(float))
    return jnp.where(jnp.asarray(m), values, 0.0), m


def integerish(x: object, lower: float | None = None) -> bool:
    """Whether every element is within tolerance of an integer."""
    a = np.asarray(x, dtype=float)
    ok = bool(np.all(np.abs(a - np.round(a)) <= INTEGERISH_TOL))
    if lower is not None:
        ok = ok and bool(np.all(a >= lower))
    return ok


def assert_integerish(x: object, name: str, lower: float | None = None) -> None:
    """Raise ValueError unless ``x`` is integer valued (and >= lower)."""
    if not integerish(x, lower):
        bound = "" if lower is None else f" and at least {lower}"
        raise ValueError(f"`{name}` must be integer valued{bound}")


def na_pattern_valid(mask: np.ndarray) -> bool:
    """R's ``.check_na_pattern``: unobserved cells sit bottom right.

    Every row and every column must have its unobserved cells at the end.
    """
    rows_ok = not np.any(~mask[:, :-1] & mask[:, 1:])
    cols_ok = not np.any(~mask[:-1, :] & mask[1:, :])
    return bool(rows_ok and cols_ok)


def validate_reporting_triangle(mask: np.ndarray) -> None:
    """R's ``validate_reporting_triangle`` for a mask.

    The matrix cannot be all NA and NAs must sit bottom right.
    """
    if not mask.any():
        raise ValueError("Matrix cannot be all NA")
    if not mask.all() and not na_pattern_valid(mask):
        raise ValueError(
            "Invalid reporting triangle structure: NA values should only "
            "appear in the bottom right portion of the triangle"
        )


def n_unobserved_rows(mask: np.ndarray) -> int:
    """Rows with at least one unobserved cell (R: sum(is.na(rowSums)))."""
    return int(np.sum(~mask.all(axis=1)))
