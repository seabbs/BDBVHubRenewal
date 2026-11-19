"""Colours from the paper figures, for the dashboard.

A reference configured with `paper_model` takes that model's name and
colour from the shared map (`bdbvhubrenewal.evaluation.labels`, through
`paperfigs.style`), so a model is named and drawn alike on the dashboard
and in the paper; WIS parts stack and fill as the paper's do
(`labels.WIS_PART_STACK`, `labels.WIS_PART_FILLS`). `paper/` is put on
the path here, as the paper's workflow does. The nowcast panels keep the
laptop dashboard's own drawing (ribbons, medians, truth line).
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).parents[3] / "paper"
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from paperfigs import scores, style  # noqa: E402

from bdbvhubrenewal.evaluation import labels  # noqa: E402


def model_colour(model: str) -> str:
    """Return a model's colour in the paper figures."""
    return style.model_style(model, [model])[0]


def model_dash(model: str) -> str:
    """Return a model's line style in the paper figures (`solid`, ...)."""
    return style.model_style(model, [model])[1]


def model_label(model: str) -> str:
    """Return a model's display name (`labels.model_label`)."""
    return labels.model_label(model)


WIS_PARTS: tuple[str, ...] = tuple(scores.WIS_PARTS)
"""WIS's components, in the order the paper's bars stack them."""

WIS_STACK: tuple[str, ...] = tuple(labels.WIS_PART_STACK)
"""WIS parts in the order they stack, from the bottom or left."""


def part_fill(part: str) -> str:
    """Return a WIS part's fill: `solid`, `hatch-up` or `hatch-down`."""
    return str(labels.WIS_PART_FILLS[part])
