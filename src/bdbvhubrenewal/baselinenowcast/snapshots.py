"""Snapshot data extension (E1) and the hub week aggregator.

R's ``estimate_delay`` needs a report in every delay cell of the
triangle or a right truncation.
Release snapshots are published on a few days only.
The extension E1 skips unobserved cells in the chain ladder ratio of
sums instead of filling them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

Aggregator = Callable[[Array, np.ndarray], tuple[Array, np.ndarray]]


def hub_week_matrix(
    first_onset: Any, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrix of hub weeks (weeks by days) and the Saturday labels.

    Hub weeks run Sunday to Saturday and carry the Saturday's date.
    A first week that starts before ``first_onset`` is dropped.
    The last week keeps only the days present, so the cut-off week is
    partial unless the last day is a Saturday.
    """
    start = np.datetime64(first_onset, "D")
    days = start + np.arange(n_days)
    # numpy weekday: 1970-01-01 was a Thursday; Sunday = 0 below.
    dow = (days.astype(int) + 4) % 7
    sat = days + (6 - dow)
    first_sunday = start + ((7 - ((start.astype(int) + 4) % 7)) % 7)
    keep_from = int((first_sunday - start).astype(int))
    saturdays = np.unique(sat[keep_from:])
    w = (sat[None, :] == saturdays[:, None]).astype(float)
    w[:, :keep_from] = 0.0
    return w, saturdays


def hub_weeks(first_onset: Any) -> Aggregator:
    """Build an aggregator summing onset days into hub weeks.

    The returned callable has the ``ref_time_aggregator`` signature.
    It builds the week matrix from the number of rows of the matrix it
    is given, whose last row is that matrix's cut-off day.
    An aggregated cell is unobserved if any of its inputs is.
    """

    def aggregate(values: Array, mask: np.ndarray) -> tuple[Array, np.ndarray]:
        v = jnp.asarray(values)
        m = np.asarray(mask, dtype=bool)
        w, _ = hub_week_matrix(first_onset, v.shape[0])
        wj = jnp.asarray(w, dtype=v.dtype)
        out = wj @ jnp.where(jnp.asarray(m), v, 0.0)
        ok = (w @ (~m).astype(float)) == 0
        return jnp.where(jnp.asarray(ok), out, 0.0), ok

    return aggregate
