"""Colours, dashes and the matplotlib theme for the paper figures.

The ink colours and categorical palette are those of BDBVHubRenewal's
`reporting/plots.py`, copied so the paper does not import the package.
Each model has a fixed colour and dash. A fine-tuned foundation model is
solid and its zero-shot version takes the same colour, dashed (hatched in
bars, hollow as points).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Final

import matplotlib as mpl

__all__ = [
    "INK",
    "MODEL_ORDER",
    "bar_style",
    "line_style",
    "model_order",
    "model_style",
    "point_style",
    "theme",
]

INK: Final[dict[str, str]] = {
    "ink": "#0b0b0b",
    "text": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "data": "#0b0b0b",
    "asof": "#b4b2a9",
}
"""Ink colours of the light theme; `asof` is data as of the origin."""

_PALETTE: Final = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)

_GREYS: Final = ("#3d3c39", "#a3a198", "#c3c2b7")

_DASHES: Final = {"solid": "-", "dashed": (0, (5, 2.5))}

_STYLE: Final[dict[str, tuple[str, str]]] = {
    "hub_baseline": ("#5f5e5a", "solid"),
    "BDBVHub-baseline": ("#5f5e5a", "solid"),
    "randomwalknowcast": (_PALETTE[1], "solid"),
    "randomwalk": (_PALETTE[7], "solid"),
    "fm_chronos_daily_ragged_ft": (_PALETTE[0], "solid"),
    "fm_chronos_daily_ragged": (_PALETTE[0], "dashed"),
    "fm_chronos_weekly_ragged_ft": (_PALETTE[6], "solid"),
    "fm_chronos_weekly_ragged": (_PALETTE[6], "dashed"),
    "epinowcast": (_PALETTE[2], "solid"),
    "hub_ensemble": (_PALETTE[3], "solid"),
    "hub_team_a": (_PALETTE[4], "solid"),
    "hub_team_b": (_PALETTE[5], "solid"),
    "hub_team_c": ("#898781", "solid"),
}

MODEL_ORDER: Final = tuple(_STYLE)
"""Order models are listed in: baselines, foundation models, others."""


def model_order(models: Sequence[str]) -> list[str]:
    """Sort models into `MODEL_ORDER`, unknown models last by name.

    Parameters
    ----------
    models : sequence of str
        Models to sort.

    Returns
    -------
    list of str
        The unique models in display order.
    """
    known = [m for m in MODEL_ORDER if m in set(models)]
    return known + sorted(set(models) - set(MODEL_ORDER))


def model_style(name: str, models: Sequence[str]) -> tuple[str, str]:
    """Return a model's colour and dash name (`solid` or `dashed`).

    Known models have a fixed style. Others take greys in their order in
    `models`, solid and then dashed.

    Parameters
    ----------
    name : str
        The model.
    models : sequence of str
        All models shown.

    Returns
    -------
    tuple of str
        Hex colour and dash name.
    """
    if name in _STYLE:
        return _STYLE[name]
    others = [m for m in model_order(models) if m not in _STYLE]
    i = others.index(name) if name in others else 0
    dash = "solid" if (i // len(_GREYS)) % 2 == 0 else "dashed"
    return _GREYS[i % len(_GREYS)], dash


def line_style(name: str, models: Sequence[str]) -> dict[str, Any]:
    """Return line keywords for a model."""
    colour, dash = model_style(name, models)
    return {"color": colour, "linestyle": _DASHES[dash]}


def bar_style(name: str, models: Sequence[str]) -> dict[str, Any]:
    """Return bar keywords: filled when solid, hatched when dashed."""
    colour, dash = model_style(name, models)
    if dash == "solid":
        return {"color": colour}
    return {"facecolor": "white", "edgecolor": colour, "hatch": "////"}


def point_style(name: str, models: Sequence[str]) -> dict[str, Any]:
    """Return marker keywords: filled when solid, hollow when dashed."""
    colour, dash = model_style(name, models)
    face = colour if dash == "solid" else "white"
    return {"color": colour, "markerfacecolor": face}


@contextmanager
def theme() -> Iterator[dict[str, str]]:
    """Apply the paper theme to figures made inside the block.

    Yields
    ------
    dict
        The ink colours.
    """
    params: dict[str, Any] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["text"],
        "axes.titlecolor": INK["ink"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["text"],
        "ytick.labelcolor": INK["text"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK["text"],
        "text.color": INK["text"],
        "font.size": 9,
        "lines.linewidth": 1.6,
        "hatch.linewidth": 0.8,
    }
    with mpl.rc_context(params):
        yield INK
