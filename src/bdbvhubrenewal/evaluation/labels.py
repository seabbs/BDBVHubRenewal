# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Display names, colours and line styles of the models.

The paper, SI, dashboard and site show models by these names and colours,
never by their ids. Data keep the ids; names apply when drawing.

- `MODEL_LABELS`: each model id's display name.
- `MODEL_STYLES`: each model id's colour and dash (`solid`, `dashed` or
  `dotted`). Bars have no dashes, so the main models each have their
  own colour; the dashes keep fine-tuned solid, zero-shot with delays
  dotted and zero-shot on complete data dashed in line plots.
- `WIS_PART_SHADES`: how much of a model's colour each WIS part takes in
  a stacked bar, the rest white, in the order drawn.
- `WIS_PART_STACK`, `WIS_PART_FILLS`: the order WIS parts stack in and
  how each fills (solid or hatched), so parts differ by more than shade.
- `model_label`: the display name of any id, including checkpoints
  (`<id>@<outbreaks>`), models fine-tuned on real data (`<id>_real`) and
  hub teams, which keep their own names.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "MODEL_LABELS",
    "MODEL_STYLES",
    "WIS_PART_FILLS",
    "WIS_PART_SHADES",
    "WIS_PART_STACK",
    "model_label",
    "tint",
]

MODEL_LABELS: Final[dict[str, str]] = {
    "fm_chronos_daily_ragged_ft": "Fine-tuned (delays)",
    "fm_chronos_daily_ragged": "Zero-shot (delays)",
    "fm_chronos_daily_complete": "Zero-shot (complete data only)",
    "randomwalknowcast": "Random walk nowcast",
    "randomwalk": "Random walk",
    "fm_chronos_weekly_ragged_ft": "Fine-tuned (delays, weekly)",
    "fm_chronos_weekly_ragged": "Zero-shot (delays, weekly)",
    "fm_chronos_weekly_complete": "Zero-shot (complete data only, weekly)",
    "KIT-frozen_baseline": "Hub baseline (KIT)",
    "BDBVHub-baseline": "Hub baseline",
    "hub_ensemble": "Hub ensemble",
    "epinowcast": "epinowcast",
}
"""Display name of each model id."""

_BLUE: Final = "#2a78d6"
_GREEN: Final = "#2aa769"
_VIOLET: Final = "#9146a4"
_BROWN: Final = "#9c5916"
_PURPLE: Final = "#6250d6"
_GREY: Final = "#5f5e5a"

MODEL_STYLES: Final[dict[str, tuple[str, str]]] = {
    "KIT-frozen_baseline": (_GREY, "solid"),
    "BDBVHub-baseline": (_GREY, "solid"),
    "randomwalknowcast": ("#eb6834", "solid"),
    "randomwalk": ("#e34948", "solid"),
    "fm_chronos_daily_complete": (_VIOLET, "dashed"),
    "fm_chronos_daily_ragged": (_GREEN, "dotted"),
    "fm_chronos_daily_ragged_ft": (_BLUE, "solid"),
    "fm_chronos_weekly_complete": (_PURPLE, "dashed"),
    "fm_chronos_weekly_ragged": (_PURPLE, "dotted"),
    "fm_chronos_weekly_ragged_ft": (_PURPLE, "solid"),
    "epinowcast": (_BROWN, "solid"),
    "hub_ensemble": ("#eda100", "solid"),
}
"""Colour (hex, light theme) and dash of each model id, in display order:
baselines, foundation models, others.

Zero-shot with delays is green and zero-shot on complete data only
purple (Sam, 19 Nov), chosen with the dataviz palette validator: against
Fine-tuned (delays) and Random walk nowcast every pair is at least 8 apart
(OKLab Delta E x 100) under simulated deuteranopia and protanopia and 15
under normal vision, on light and dark backgrounds. Every pair of
distinct colours differs by a CIEDE2000 of 5 or more under normal vision
and the three colour-blindness simulations, and the main models' colours
have a contrast of 3 or more on light and dark backgrounds
(tests/evaluation/test_labels.py)."""

WIS_PART_SHADES: Final[dict[str, float]] = {
    "dispersion": 1.0,
    "overprediction": 0.6,
    "underprediction": 0.3,
}
"""Share of a model's colour in each WIS part's segment, dispersion first
and darkest, as in Wolffram et al. (2023)."""

WIS_PART_STACK: Final = ("underprediction", "dispersion", "overprediction")
"""Order of WIS parts in a stacked bar, from the bottom (or left) up."""

WIS_PART_FILLS: Final[dict[str, str]] = {
    "underprediction": "hatch-down",
    "dispersion": "solid",
    "overprediction": "hatch-up",
}
"""How each WIS part fills its segment in its model's colour, so parts
differ by more than shade (Sam, 19 Nov): dispersion solid,
overprediction hatched rising to the right, underprediction hatched
falling to the right."""

_CHECKPOINT: Final = "@"
_REAL: Final = "_real"


def model_label(model: str) -> str:
    """Return a model's display name.

    Parameters
    ----------
    model : str
        A model id. A checkpoint `<id>@<outbreaks>` is named after its
        model with the outbreaks it was trained on, and `<id>_real` as
        fine-tuned on real data where `<id>`'s name has a variant in
        brackets. Ids without a name, such as hub teams, are returned
        unchanged.

    Returns
    -------
    str
        The display name.
    """
    name, sep, size = model.rpartition(_CHECKPOINT)
    if sep and size.isdigit():
        return f"{model_label(name)}, {int(size):,} outbreaks"
    base = MODEL_LABELS.get(model.removesuffix(_REAL), "")
    if model.endswith(_REAL) and " (" in base:
        return f"Fine-tuned on real data ({base.split(' (', 1)[1]}"
    return MODEL_LABELS.get(model, model)


def tint(colour: str, share: float) -> str:
    """Return `share` of a hex `colour` mixed with white, as hex."""
    rgb = [int(colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(share * c + (1 - share) * 255) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)
