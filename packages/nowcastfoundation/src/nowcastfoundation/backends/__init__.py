# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The foundation models the forecasts run on.

A backend predicts quantiles of groups of series. `chronos2` is the
first (Chronos-2); `nowcastfoundation.forecaster` lays the hub target's
releases out as groups and reads the forecasts back.
"""

from collections.abc import Mapping
from typing import Any

__all__ = ["Series"]

Series = Mapping[str, Any]
"""One group for Chronos-2: `target` `[V, S]`, and optionally
`past_covariates` (name to `[S]`) and `future_covariates` (name to
`[H]`, NaN where unknown)."""
