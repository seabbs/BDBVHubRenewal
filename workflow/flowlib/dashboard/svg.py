"""Small SVG line charts for the dashboard page (standard library only)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"]
"""CSS colours of series that name none (sources, in the loss plots)."""

DASHES = {
    "dash": ("5 3", "dashed"),
    "dot": ("1.5 3", "dotted"),
    "dd": ("8 3 1.5 3", "dash-dot"),
}
"""Dash patterns and the legend's name for each."""


def nice(lo: float, hi: float, n: int = 5) -> list[float]:
    """Return round tick values covering `lo` to `hi`, about `n` steps."""
    if hi <= lo:
        hi = lo + 1
    step = (hi - lo) / n
    mag = 10 ** int(f"{step:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= step:
            step = m * mag
            break
    v = int(lo // step) * step
    ticks = []
    while v <= hi + step:
        ticks.append(v)
        v += step
    return ticks


def rolling(rows: Sequence[tuple[float, float]], k: int) -> list[tuple]:
    """Return each point with the mean of it and the `k - 1` before it."""
    out, acc = [], []
    for x, y in rows:
        acc.append(y)
        if len(acc) > k:
            acc.pop(0)
        out.append((x, sum(acc) / len(acc)))
    return out


def chart(
    title: str,
    series: Sequence[tuple[Any, ...]],
    marks: Sequence[tuple[float, str]] = (),
    ylog: bool = False,
    width: int = 620,
    height: int = 300,
    xticks: Sequence[tuple[float, str]] | None = None,
    hlines: Sequence[float] = (),
    xlabel: str = "step",
    note: str = "",
) -> str:
    """Return a card with an SVG line chart and its legend.

    Each series is (label, colour, points, kind[, dash]): `kind` is
    `line`, `faint` (a line at low opacity), `dots` or `pts` (a line
    with labelled points); `dash` is a key of `DASHES`. `xticks` puts x
    on a log2 axis with those ticks; `marks` are labelled vertical lines
    and `hlines` horizontal ones.
    """
    pts = [p for s in series for p in s[2] if p[1] is not None]
    if not pts:
        return f"<div class=card><h3>{title}</h3><p class=mut>no data</p></div>"

    def f(y: float) -> float:
        return math.log10(max(y, 1e-9)) if ylog else y

    xs = [p[0] for p in pts]
    ys = [f(p[1]) for p in pts]
    x0, x1 = 0.0, max(max(xs), 1)
    if xticks:
        x0, x1 = math.log2(xticks[0][0]), math.log2(xticks[-1][0])
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or 0.1
    y0, y1 = y0 - pad, y1 + pad
    left, right, top, bottom = 52, 12, 10, 34
    xm: Callable[[float], float] = math.log2 if xticks else (lambda x: x)

    def sx(x: float) -> float:
        return left + (xm(x) - x0) / (x1 - x0) * (width - left - right)

    def sy(y: float) -> float:
        return top + (1 - (f(y) - y0) / (y1 - y0)) * (height - top - bottom)

    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for t in nice(y0, y1):
        if y0 <= t <= y1:
            yy = top + (1 - (t - y0) / (y1 - y0)) * (height - top - bottom)
            lab = f"{10**t:.3g}" if ylog else f"{t:.3g}"
            o.append(
                f'<line x1="{left}" x2="{width - right}" y1="{yy:.1f}" '
                f'y2="{yy:.1f}" class=grid /><text x="{left - 6}" '
                f'y="{yy + 4:.1f}" text-anchor=end class=ax>{lab}</text>'
            )
    ticks = xticks or [(t, f"{t:g}") for t in nice(x0, x1) if x0 <= t <= x1]
    for t, lab in ticks:
        xx = sx(t)
        o.append(
            f'<text x="{xx:.1f}" y="{height - 14}" text-anchor=middle '
            f"class=ax>{lab}</text>"
        )
        if xticks:
            o.append(
                f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" '
                f'y2="{height - bottom}" class=grid />'
            )
    o.append(
        f'<text x="{(left + width - right) / 2}" y="{height - 1}" '
        f"text-anchor=middle class=ax>{xlabel}</text>"
    )
    for hv in hlines:
        if y0 <= f(hv) <= y1:
            o.append(
                f'<line x1="{left}" x2="{width - right}" y1="{sy(hv):.1f}" '
                f'y2="{sy(hv):.1f}" class=mark />'
            )
    for m, lab in marks:
        if not xticks and x0 <= m <= x1:
            o.append(
                f'<line x1="{sx(m):.1f}" x2="{sx(m):.1f}" y1="{top}" '
                f'y2="{height - bottom}" class=mark /><text '
                f'x="{sx(m) + 2:.1f}" y="{top + 10}" class=ax>{lab}</text>'
            )
    for s in series:
        lab, col, p, kind = s[:4]
        dash = f' stroke-dasharray="{DASHES[s[4]][0]}"' if len(s) > 4 else ""
        op = ' opacity=".4"' if kind == "faint" else ""
        p = [q for q in p if q[1] is not None]
        if kind == "dots":
            for x, y in p:
                o.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.8" '
                    f'fill="{col}" opacity=".35" />'
                )
            continue
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p)
        if kind == "pts":
            o.append(
                f'<polyline points="{d}" fill=none stroke="{col}" '
                f"stroke-width=1.6{dash} />"
            )
            for x, y in p:
                o.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r=2.4 '
                    f'fill="{col}"><title>{lab} {xlabel} {x}: {y:.4f}'
                    f"</title></circle>"
                )
        else:
            o.append(
                f'<polyline points="{d}" fill=none stroke="{col}" '
                f"stroke-width=1.6{dash}{op} />"
            )
    o.append("</svg>")
    leg = "".join(
        f'<span><i style="background:{s[1]}'
        f'{";opacity:.4" if s[3] == "faint" else ""}"></i>{s[0]}'
        f"{' (' + DASHES[s[4]][1] + ')' if len(s) > 4 else ''}</span>"
        for s in series
    )
    return (
        f"<div class=card><h3>{title}</h3>{''.join(o)}"
        f"<div class=leg>{leg}</div>{note}</div>"
    )
