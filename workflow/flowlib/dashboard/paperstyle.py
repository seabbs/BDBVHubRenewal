"""Colours from the paper figures, for the dashboard.

A reference configured with `paper_model` takes that model's colour in
the paper figures (`paperfigs.style`), so randomwalknowcast is drawn in
the same colour on the dashboard and in the paper; the WIS bars take the
paper's component colours. `paper/` is put on
the path here, as the paper's workflow does. The nowcast panels keep the
laptop dashboard's own drawing (ribbons, medians, truth line).
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).parents[3] / "paper"
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from paperfigs import style  # noqa: E402


def model_colour(model: str) -> str:
    """Return a model's colour in the paper figures."""
    return style.model_style(model, [model])[0]


def wis_colour(part: str) -> str:
    """Return a WIS component's colour in the paper figures."""
    return style.wis_colour(part)


WIS_PARTS: tuple[str, ...] = tuple(style.WIS_PARTS)
"""WIS's components, in the order the paper's bars stack them."""
