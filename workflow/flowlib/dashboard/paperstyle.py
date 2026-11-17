"""The paper figures' nowcast styling, for the dashboard.

The dashboard's German nowcasts and the paper's static nowcast panels
(`paperfigs.plots.nowcast_panels`) share their quantile levels, their
50% and 95% bands, the truth and data inks, and the reference models'
colours (`paperfigs.style`). Runs keep the colours they are configured
with. `paper/` is put on the path here, as the paper's workflow does.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PAPER = Path(__file__).parents[3] / "paper"
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from paperfigs import plots, style  # noqa: E402

LEVELS: tuple[float, ...] = tuple(plots.NOWCAST_LEVELS)
"""The quantile levels the panels draw: the median, 50% and 95% bands."""

BANDS: tuple[tuple[float, float, float], ...] = (
    (0.025, 0.975, 0.15),
    (0.25, 0.75, 0.3),
)
"""Each band's lower and upper level and its opacity. Must match the
inline `bands` of `paperfigs.plots.nowcast_panels` (a test checks)."""

INK: dict[str, str] = dict(plots._PANEL_INK)
"""Truth, data, text and grid colours of the nowcast panels."""


def model_colour(model: str) -> str:
    """Return a model's colour in the paper figures."""
    return style.model_style(model, [model])[0]


def page_style() -> dict[str, Any]:
    """Return the styling the dashboard page draws nowcasts with."""
    return {
        "levels": list(LEVELS),
        "bands": [list(b) for b in BANDS],
        "ink": INK,
    }
