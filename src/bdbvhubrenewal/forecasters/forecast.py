# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""A fit's samples, before any hub table is made."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["Forecast"]


@dataclass(frozen=True)
class Forecast:
    """Samples from a fit, before the hub table is made.

    Attributes
    ----------
    samples : pandas.DataFrame
        Province samples, columns `SAMPLE_COLUMNS`.
    unassigned : pandas.DataFrame or None
        A modelled national component not assigned to any province, with
        columns `target_end_date`, `sample` and `value`, or None.
    modelled : tuple of str or None
        The provinces the model reports, for `national.json`. None means
        every province of the task.
    """

    samples: pd.DataFrame
    unassigned: pd.DataFrame | None = None
    modelled: tuple[str, ...] | None = None
