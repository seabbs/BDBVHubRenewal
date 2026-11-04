# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Score metrics of the overall figures and the score summary's JSON writer.

`docs/javascripts/scoresummary.js` draws the summaries with
`docs/javascripts/scorechart.js`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "ENERGY_METRICS",
    "OVERALL_METRICS",
    "write_score_chart",
]

OVERALL_METRICS: Mapping[str, tuple[str, float | None]] = {
    "rel_wis": ("Relative WIS", 1.0),
    "rel_log_wis": ("Relative WIS on the log scale", 1.0),
    "coverage_50": ("50% coverage", 0.5),
    "coverage_90": ("90% coverage", 0.9),
    "bias": ("Bias", 0.0),
}
"""Columns of `relative_wis` in the overall figures, with each label and
the line of a perfect or baseline score. Bias is in -1 to 1."""

ENERGY_METRICS: Mapping[str, tuple[str, float | None]] = {
    "rel_energy": ("Relative energy score", 1.0),
    "rel_log_energy": ("Relative energy score, log scale", 1.0),
}
"""Columns of `relative_energy` in the overall figures. The energy score
is one score per scale, so it has no per-place or per-horizon view."""


def write_score_chart(path: Path, data: Mapping[str, object]) -> None:
    """Write the chart data as compact JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
