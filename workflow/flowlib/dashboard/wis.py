"""Mean WIS as stacked bars of its components (Wolffram et al. 2023).

One card per study and level with a bar per checkpoint and reference
(`scores.wis_bars`): dispersion, underprediction and overprediction,
which sum to the WIS, in the paper's colours (`paperstyle`). Drawn here
as SVG, so the page needs no script for it.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import paperstyle, svg

PARTS = paperstyle.WIS_PARTS
"""The components, in the order they stack."""

NOTE = (
    "Mean WIS on the forecasts the models were scored on together; the "
    "three parts sum to it (Bracher et al. 2021). * scored on other "
    "forecasts than most of its card's bars, so not comparable with them."
)


def _usual(bars: Sequence[Mapping[str, Any]]) -> int:
    """Return the number of cells most of a card's bars were scored on."""
    counts = Counter(b.get("cells", 0) for b in bars)
    return max(counts, key=lambda c: (counts[c], c))


def card(title: str, bars: Sequence[Mapping[str, Any]], info: str = "") -> str:
    """Return one card of stacked WIS bars, with a legend of the parts."""
    esc = html.escape
    left, right, top, row, width = 170, 56, 6, 22, 620
    height = top + row * len(bars) + 24
    hi = max((b["wis"] for b in bars), default=1.0) or 1.0
    ticks = [t for t in svg.nice(0, hi, 4) if t <= hi * 1.05]
    span = max(ticks[-1], hi)

    def sx(v: float) -> float:
        return left + v / span * (width - left - right)

    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    base = top + row * len(bars)
    for t in ticks:
        o.append(
            f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" '
            f'y2="{base}" class=grid /><text x="{sx(t):.1f}" '
            f'y="{base + 14}" text-anchor=middle class=ax>{t:g}</text>'
        )
    usual = _usual(bars)
    for i, b in enumerate(bars):
        y = top + row * i + 4
        h = row - 8
        other = b.get("cells", 0) != usual
        label = esc(b["label"]) + (" *" if other else "")
        o.append(
            f'<rect x="4" y="{y + 3}" width="10" height="10" rx="2" '
            f'fill="{esc(b["colour"])}" /><text x="20" y="{y + 12}" '
            f"class=lab>{label}</text>"
        )
        x = 0.0
        for part in PARTS:
            v = b["parts"][part]
            x0, x1 = sx(x), sx(x + v)
            x += v
            if x1 - x0 <= 0:
                continue
            # A 2 px surface gap between the parts.
            w = max(x1 - x0 - 2, 0.5)
            o.append(
                f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{h}" '
                f'rx="2" fill="{paperstyle.wis_colour(part)}"><title>'
                f"{label}: {part} {v:.3g} of {b['wis']:.3g}</title></rect>"
            )
        o.append(
            f'<text x="{sx(b["wis"]) + 4:.1f}" y="{y + 12}" '
            f"class=lab>{b['wis']:.3g}</text>"
        )
    o.append("</svg>")
    leg = "".join(
        f'<span><i style="background:{paperstyle.wis_colour(p)}"></i>{p}</span>'
        for p in PARTS
    )
    return (
        f"<div class=card><h3>{esc(title)}{info}</h3>"
        f"<div class=leg>{leg}</div>{''.join(o)}</div>"
    )


def section(
    bars: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    view: Mapping[str, Any],
    tip: Callable[[str], str] = lambda _: "",
) -> str:
    """Return a card per study and level, or nothing without bars.

    Cards follow the `dashboard` config's study and level order
    (`view`), which also names the studies.
    """
    if not bars:
        return ""
    studies = dict(view.get("studies") or {})
    levels = list(view.get("levels") or [])

    def rank(key: tuple[str, str]) -> tuple[int, str, int, str]:
        s, lv = key
        si = list(studies).index(s) if s in studies else len(studies)
        li = levels.index(lv) if lv in levels else len(levels)
        return si, s, li, lv

    cards = "".join(
        card(f"{studies.get(s, s)}, {lv}", bars[(s, lv)], tip("wisbars"))
        for s, lv in sorted(bars, key=rank)
    )
    return f"<div class=plots2>{cards}</div><p class=mut>{NOTE}</p>"
