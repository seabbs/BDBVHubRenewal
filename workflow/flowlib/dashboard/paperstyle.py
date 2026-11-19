"""Colours from the paper figures, for the dashboard.

A reference configured with `paper_model` takes that model's name and
colour from the shared map (`bdbvhubrenewal.evaluation.labels`, through
`paperfigs.style`), so a model is named and drawn alike on the dashboard
and in the paper; the WIS bars shade
the model's colour by part as the paper's do. `paper/` is put on
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
    return labels.MODEL_STYLES.get(model, ("", "solid"))[1]


def model_label(model: str) -> str:
    """Return a model's display name (`labels.model_label`)."""
    return labels.model_label(model)


WIS_PARTS: tuple[str, ...] = tuple(scores.WIS_PARTS)
"""WIS's components, in the order the paper's bars stack them."""

INK_TEXT: str = style.INK["text"]
"""The paper's text ink, whose shades key the parts in a legend."""


WIS_STACK: tuple[str, ...] = tuple(labels.WIS_PART_STACK)
"""WIS parts in the order they stack, from the bottom or left."""


def part_fill(part: str) -> str:
    """Return a WIS part's fill: `solid`, `hatch-up` or `hatch-down`."""
    return str(labels.WIS_PART_FILLS[part])


def part_share(part: str) -> float:
    """Return the share of its model's colour a WIS part's segment takes."""
    return float(labels.WIS_PART_SHADES[part])


def part_colour(colour: str, part: str) -> str:
    """Return a WIS part's bar colour: its model's colour, shaded.

    The share of the colour each part takes, the rest white, is the
    paper's (`labels.WIS_PART_SHADES`).
    """
    from matplotlib.colors import to_hex, to_rgb

    share = part_share(part)
    try:
        rgb = to_rgb(colour)
    except ValueError:
        # A CSS colour (a variable) matplotlib cannot read stays as it is.
        return colour
    return to_hex(tuple(share * c + (1 - share) for c in rgb))
