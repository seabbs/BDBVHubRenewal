"""Colours, dashes and the matplotlib theme for the paper figures.

The ink colours are those of bdbv-hub-modelling's `reporting/plots.py`.
Each model's colour, dash and display name are the shared ones of
`bdbvhubrenewal.evaluation.labels`, which the dashboard and site use too.
The Chronos variants of one family share a colour: fine-tuned on ragged
data is solid, zero-shot on ragged data dashed (hollow circles), and
zero-shot on complete data dotted (hollow squares). Bars are filled with
the model's colour, and the model is named on the axis in that colour.
They are listed complete, ragged, fine-tuned so each step reads left to
right.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Final

import matplotlib as mpl
from bdbvhubrenewal.evaluation.labels import MODEL_STYLES, model_label
from matplotlib.artist import Artist

__all__ = [
    "INK",
    "MODEL_ORDER",
    "bar_style",
    "line_style",
    "model_order",
    "model_style",
    "point_style",
    "shared_legend",
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

_DASHES: Final = {
    "solid": "-",
    "dashed": (0, (5, 2.5)),
    "dotted": (0, (1.2, 1.8)),
}

_MARKERS: Final = {"solid": "o", "dashed": "o", "dotted": "s"}

_STYLE: Final[dict[str, tuple[str, str]]] = {
    **MODEL_STYLES,
    # Stand-ins for hub teams in the dummy studies.
    "hub_team_a": (_PALETTE[4], "solid"),
    "hub_team_b": (_PALETTE[5], "solid"),
    "hub_team_c": ("#898781", "solid"),
}

MODEL_ORDER: Final = tuple(_STYLE)
"""Order models are listed in: baselines, foundation models, others."""


def _model_rank(label: str) -> int | None:
    """Rank of the model a legend label names, or None if it names none.

    A label names a model when it starts with the model's display name,
    so checkpoints ("..., 2,000 outbreaks") and notes ("... (90% made)")
    keep their model's place.
    """
    names = sorted(
        ((model_label(m), i) for i, m in enumerate(MODEL_ORDER)),
        key=lambda x: -len(x[0]),
    )
    return next((i for name, i in names if label.startswith(name)), None)


def shared_legend(
    fig: Any,
    axes: Any,
    extra: Sequence[Artist] = (),
    ncol: int | None = None,
    **kwargs: Any,
) -> None:
    """Draw a figure's one legend, above its panels.

    Entries are gathered from every panel in `axes`, then `extra`, once
    per label. Labels naming a model come first, in `MODEL_ORDER`; the
    rest follow in the order drawn.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure, laid out with `layout="constrained"`.
    axes : Axes or iterable of Axes
        The panels whose labelled artists form the legend.
    extra : sequence of Artist
        Further handles with labels, such as WIS parts.
    ncol : int, optional
        Columns, by default up to six.
    **kwargs
        Passed to `Figure.legend`.
    """
    panels = [axes] if hasattr(axes, "get_legend_handles_labels") else axes
    entries: dict[str, Artist] = {}
    for ax in getattr(panels, "flat", panels):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    for handle in extra:
        entries.setdefault(handle.get_label(), handle)
    drawn = list(entries)
    ranked = sorted(
        (label for label in drawn if _model_rank(label) is not None),
        key=lambda label: (_model_rank(label), drawn.index(label)),
    )
    labels = ranked + [label for label in drawn if label not in ranked]
    fig.legend(
        [entries[label] for label in labels],
        labels,
        loc="outside upper center",
        ncol=ncol or min(len(labels), 6) or 1,
        frameon=False,
        **kwargs,
    )


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
    """Return a model's colour and dash name (`solid`, `dashed` or `dotted`).

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
    """Return bar keywords: every model's bar filled with its colour."""
    colour, _ = model_style(name, models)
    return {"color": colour}


def point_style(name: str, models: Sequence[str]) -> dict[str, Any]:
    """Return marker keywords.

    Solid models are filled circles, dashed hollow circles and dotted
    hollow squares.
    """
    colour, dash = model_style(name, models)
    face = colour if dash == "solid" else "white"
    return {"color": colour, "markerfacecolor": face, "marker": _MARKERS[dash]}


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
