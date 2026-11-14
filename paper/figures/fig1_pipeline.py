# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Draw Figure 1, a schematic of the fine-tuning pipeline.

Writes `fig1-pipeline.svg` next to this script, and `fig1-pipeline.pdf`
and, with `--png`, a 200 dpi `fig1-pipeline.png` with `rsvg-convert`
(librsvg). Run from the repo root:

    uv run python paper/figures/fig1_pipeline.py [--png]

The panels are:

- A, the assumptions in the epinowcast simulations
  (`scripts/finetune_simulate.R`), one box per stage;
- B, the reporting triangle laid out as ragged series for Chronos-2
  (`forecasters.foundation.layout_block`);
- C, a training block cut from a simulated release history
  (`forecasters.finetune.training_block`) and the LoRA adapters;
- D, forecasting with the fine-tuned model
  (`forecasters.foundation.chronos_sampler`).

Units are points at the final size, 6.5 inches wide, so font sizes are
the printed sizes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fig1-pipeline"
WIDTH = 468.0

FONT = "Helvetica, Arial, sans-serif"
FG = "#1a1e27"
MUTED = "#4f5666"
LINE = "#b9c0cd"
PANEL = "#eceff4"
CELL = "#d9dee8"
ACCENT = "#2a56c6"
ACCENT_SOFT = "#dde6fa"
COV = "#9a5b07"
COV_SOFT = "#f5e5c8"
HATCH = "#a3aabb"
TRUTH = "#1b7f5f"
TRUTH_SOFT = "#d4eee3"

# Body text, small notes and headings, in points.
BODY = 7.5
SMALL = 7.0
HEAD = 8.0
TITLE = 9.0


class Svg:
    """An SVG document built from primitives in points."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        """Append raw SVG."""
        self.parts.append(s)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        sw: float = 0.6,
        rx: float = 1.5,
        dash: str | None = None,
    ) -> None:
        """Draw a rectangle."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        s: str | list[str],
        size: float = BODY,
        fill: str = FG,
        anchor: str = "start",
        weight: str = "normal",
        lead: float = 1.25,
        italic: bool = False,
    ) -> None:
        """Write text, one line per item of a list."""
        lines = [s] if isinstance(s, str) else s
        style = ' font-style="italic"' if italic else ""
        spans = "".join(
            f'<tspan x="{x:.2f}" dy="{0 if i == 0 else size * lead:.2f}">'
            f"{_escape(line)}</tspan>"
            for i, line in enumerate(lines)
        )
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" xml:space="preserve"{style}>'
            f"{spans}</text>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = MUTED,
        sw: float = 0.8,
        dash: str | None = None,
        arrow: str | None = None,
    ) -> None:
        """Draw a line, with an arrowhead marker id at its end."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{arrow})"' if arrow else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'
        )

    def path(
        self,
        d: str,
        stroke: str = MUTED,
        sw: float = 0.8,
        fill: str = "none",
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        """Draw a path."""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" '
            f'fill="{fill}" opacity="{opacity}"{da}/>'
        )

    def poly(self, pts: list[tuple[float, float]], **kw: object) -> None:
        """Draw a polyline through points."""
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.path(d, **kw)  # type: ignore[arg-type]

    def label(self, x: float, y: float, letter: str, title: str) -> None:
        """Write a panel's letter and title."""
        self.text(x, y, letter, size=11, weight="bold")
        self.text(x + 12, y, title, size=TITLE, weight="bold")

    def render(self, height: float) -> str:
        """Return the document."""
        defs = (
            "<defs>"
            '<pattern id="hatch" width="3.5" height="3.5" '
            'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="3.5" height="3.5" fill="{PANEL}"/>'
            f'<line x1="0" y1="0" x2="0" y2="3.5" stroke="{HATCH}" '
            'stroke-width="0.9"/></pattern>'
            '<marker id="ah" viewBox="0 0 6 6" refX="5.5" refY="3" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 z" fill="{MUTED}"/></marker>'
            '<marker id="aha" viewBox="0 0 6 6" refX="5.5" refY="3" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 z" fill="{ACCENT}"/></marker>'
            "</defs>"
        )
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}pt" height="{height}pt" '
            f'viewBox="0 0 {WIDTH} {height}" font-family="{FONT}">'
            f"{defs}"
            f'<rect width="{WIDTH}" height="{height}" fill="white"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )


def _escape(s: str) -> str:
    """Escape text for XML."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell(g: Svg, x: float, y: float, w: float, h: float, kind: str) -> None:
    """Draw one cell of a triangle or block, styled by what it holds.

    `kind` is `final` (a complete count), `lag` (a count reported so
    far), `future` (to forecast), `truth` (a final count the training
    loss scores), `none` (no release at that lag) or `unseen` (not yet
    reported).
    """
    if kind == "final":
        g.rect(x, y, w, h, fill=CELL, stroke="white", sw=0.5)
    elif kind == "lag":
        g.rect(x, y, w, h, fill=COV_SOFT, stroke=COV, sw=0.6)
    elif kind == "future":
        g.rect(
            x, y, w, h, fill=ACCENT_SOFT, stroke=ACCENT, sw=0.6, dash="2 1.2"
        )
    elif kind == "truth":
        g.rect(x, y, w, h, fill=TRUTH_SOFT, stroke=TRUTH, sw=0.8)
    elif kind == "none":
        g.rect(x, y, w, h, fill="url(#hatch)", stroke="white", sw=0.5)
    elif kind == "unseen":
        g.rect(x, y, w, h, fill="none", stroke=LINE, sw=0.5, dash="1.5 1.5")


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, the simulations' generative chain."""
    g.label(0, oy + 9, "A", "Simulated outbreaks")
    widths = [96.0, 84.0, 96.0, 84.0, 72.0]
    gap = (WIDTH - sum(widths)) / (len(widths) - 1)
    top = oy + 18
    stages = [
        ("Latent process", ["renewal \u00b7 random walks", "growth steps"]),
        ("Expected onsets", ["places \u00b7 ages", "places \u00d7 ages"]),
        ("Reporting delay", ["parametric \u00b7 hazard", "drift \u00b7 steps"]),
        ("Report days", ["weekday effects", "set weekdays"]),
        ("Observation", ["NB2 \u00b7 NB1", "Poisson"]),
    ]
    height = 66.0
    x = 0.0
    for i, (w, (head, words)) in enumerate(zip(widths, stages, strict=True)):
        g.rect(x, top, w, height, fill="white", stroke=LINE, sw=0.7, rx=3)
        g.rect(x, top, w, 30, fill=PANEL, stroke="none", rx=3)
        g.rect(x, top + 26, w, 4, fill=PANEL, stroke="none", rx=0)
        _glyph(g, i, x + w / 2, top + 14)
        g.text(
            x + w / 2,
            top + 40,
            head,
            size=HEAD,
            weight="bold",
            anchor="middle",
        )
        g.text(
            x + w / 2,
            top + 51,
            words,
            size=SMALL,
            anchor="middle",
            fill=MUTED,
            lead=1.2,
        )
        if i < len(widths) - 1:
            g.line(
                x + w + 1.5,
                top + 15,
                x + w + gap - 1.5,
                top + 15,
                arrow="ah",
            )
        x += w + gap
    return top + height


def _curve(
    cx: float, base: float, width: float, peak: float, skew: float = 0.0
) -> list[tuple[float, float]]:
    """Return points of a smooth epidemic curve over `width` points."""
    pts = []
    for k in range(31):
        u = k / 30
        centre = 0.5 + skew
        v = 2.718 ** (-(((u - centre) / 0.2) ** 2))
        pts.append((cx - width / 2 + u * width, base - peak * v))
    return pts


def _area(
    g: Svg, pts: list[tuple[float, float]], base: float, **kw: object
) -> None:
    """Fill the area under a curve down to `base`."""
    d = f"M{pts[0][0]:.2f},{base:.2f} " + " ".join(
        f"L{x:.2f},{y:.2f}" for x, y in pts
    )
    d += f" L{pts[-1][0]:.2f},{base:.2f} Z"
    g.path(d, **kw)  # type: ignore[arg-type]


def _glyph(g: Svg, i: int, cx: float, cy: float) -> None:
    """Draw the icon of stage `i` of panel A: what the stage produces."""
    base = cy + 9
    if i == 0:
        # A growth-rate path, with a step change, about zero (dashed).
        g.line(
            cx - 24, cy + 1, cx + 24, cy + 1, stroke=LINE, sw=0.6, dash="2 1.5"
        )
        pts = [
            (cx - 24 + k * 2.4, cy - 4 - 2.5 * (k % 3 == 1)) for k in range(9)
        ]
        pts += [
            (cx - 2.4 + k * 2.4, cy + 5 - 1.5 * (k % 2)) for k in range(1, 12)
        ]
        g.poly(pts, stroke=ACCENT, sw=1.1)
    elif i == 1:
        # Expected onsets: stacked epidemic curves of three strata.
        for peak, skew, op in [
            (16, -0.08, 0.25),
            (11, 0.04, 0.45),
            (6, 0.12, 0.9),
        ]:
            pts = _curve(cx, base, 46, peak, skew)
            _area(g, pts, base, stroke="none", fill=ACCENT, opacity=op)
        g.line(cx - 24, base, cx + 24, base, stroke=MUTED, sw=0.6)
    elif i == 2:
        # A delay distribution as bars, and its drifted shape (dashed).
        probs = [0.15, 0.55, 0.85, 0.7, 0.48, 0.3, 0.18, 0.1, 0.05]
        for k, p in enumerate(probs):
            g.rect(
                cx - 20 + k * 4.6, base - 17 * p, 3.4, 17 * p, fill=ACCENT, rx=0
            )
        shifted = [
            (cx - 18.3 + (k + 1.6) * 4.6, base - 17 * p)
            for k, p in enumerate(probs)
        ]
        g.poly(shifted, stroke=FG, sw=0.8, dash="1.8 1.4")
        g.line(cx - 22, base, cx + 24, base, stroke=MUTED, sw=0.6)
    elif i == 3:
        # Reports by day of the week, with two days of no reports.
        for k, h in enumerate([12, 8, 0, 9, 15, 0, 5]):
            if h:
                g.rect(cx - 16 + k * 4.8, base - h, 3.4, h, fill=ACCENT, rx=0)
        g.line(cx - 18, base, cx + 18, base, stroke=MUTED, sw=0.6)
    else:
        # Observed counts scattered about their expectation.
        pts = _curve(cx, base, 44, 15, 0.0)
        g.poly(pts, stroke=MUTED, sw=0.8)
        noise = [1.5, -2.5, 2, -1, 3, -3, 0.5, 2.5, -2, 1, -1.5]
        for k, e in enumerate(noise):
            x, y = pts[k * 3]
            g.add(
                f'<circle cx="{x:.2f}" cy="{min(y + e, base):.2f}" r="1.2" '
                f'fill="{ACCENT}"/>'
            )
        g.line(cx - 23, base, cx + 23, base, stroke=MUTED, sw=0.6)


def _block(
    g: Svg,
    bx: float,
    by: float,
    cols: int,
    bw: float,
    rh: float,
    kinds: list[list[str]],
) -> float:
    """Draw a block of series from cell kinds by row; return its bottom."""
    y = by
    for k, row in enumerate(kinds):
        if k == 1:
            y += 2
        for c in range(cols):
            cell(g, bx + c * bw, y, bw, rh, row[c])
        y += rh + 1
    return y - 1


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, the reporting triangle as ragged series."""
    g.label(0, oy + 9, "B", "The reporting triangle as ragged series")
    top = oy + 22
    # A toy triangle: onset steps 1 to 8, releases at steps 4 to 8, and
    # lag series 0 to 3, so the target is complete up to E = 4.
    n_on, first_rel, n_lag = 8, 4, 4
    releases = list(range(first_rel, n_on + 1))
    e_step = n_on - n_lag
    cw, ch = 15.0, 10.0
    tx, ty = 24.0, top + 14
    g.text(
        tx + cw * len(releases) / 2,
        top + 4,
        "release \u2192",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    g.add(
        f'<text x="{tx - 6:.2f}" y="{ty + 30:.2f}" font-size="{SMALL}" '
        f'fill="{MUTED}" text-anchor="middle" '
        f'transform="rotate(-90 {tx - 6:.2f} {ty + 30:.2f})">'
        "onset \u2192</text>"
    )
    for t in range(1, n_on + 1):
        y = ty + (t - 1) * ch
        for j, r in enumerate(releases):
            x = tx + j * cw
            if t > r:
                cell(g, x, y, cw, ch, "unseen")
                continue
            lag = r - t
            cell(g, x, y, cw, ch, "final" if lag >= n_lag else "lag")
            if lag == 0:
                g.rect(
                    x + 0.4,
                    y + 0.4,
                    cw - 0.8,
                    ch - 0.8,
                    stroke=FG,
                    sw=1.0,
                    rx=1,
                )
    hi = e_step + 1
    g.rect(
        tx + (hi - first_rel) * cw - 0.8,
        ty + (hi - 1) * ch - 0.8,
        (n_on - hi + 1) * cw + 1.6,
        ch + 1.6,
        stroke=ACCENT,
        sw=1.3,
    )
    ey = ty + e_step * ch
    g.line(tx - 2, ey, tx + cw * len(releases) + 3, ey, stroke=ACCENT, sw=1.1)
    g.text(
        tx + cw * len(releases) + 6,
        ey + 3,
        "E",
        size=BODY,
        weight="bold",
        fill=ACCENT,
    )
    tri_bottom = ty + n_on * ch

    # The arrow to the ragged layout.
    ax = tx + cw * len(releases) + 16
    g.line(ax, ty + 40, ax + 22, ty + 40, arrow="ah")

    # The ragged block of two places: each a target and its lag series.
    bx = 196.0
    by = ty + 2
    cols = n_on + 2
    bw = 17.0
    rh = 8.5

    def kinds() -> list[list[str]]:
        rows = [["final" if c + 1 <= e_step else "future" for c in range(cols)]]
        for d in range(n_lag):
            lo, hi_ = first_rel - d, n_on - d
            rows.append(
                ["lag" if lo <= c + 1 <= hi_ else "none" for c in range(cols)]
            )
        return rows

    bottoms = []
    starts = []
    y0 = by
    for p in (1, 2):
        starts.append(y0)
        bottom = _block(g, bx, y0, cols, bw, rh, kinds())
        g.text(
            bx - 4,
            y0 + rh - 1.5,
            "target",
            size=SMALL,
            anchor="end",
            fill=ACCENT,
        )
        lag_mid = (y0 + rh + 2 + bottom) / 2
        g.text(
            bx - 4,
            lag_mid + 2.5,
            "lag 0\u20133",
            size=SMALL,
            anchor="end",
            fill=COV,
        )
        g.line(bx - 2, y0 + rh + 2.5, bx - 2, bottom - 0.5, stroke=COV, sw=0.6)
        g.text(
            bx - 36,
            (y0 + bottom) / 2 + 2.5,
            f"place {p}",
            size=SMALL,
            anchor="end",
            weight="bold",
        )
        bottoms.append(bottom)
        y0 = bottom + 7
    bottom = bottoms[-1]
    # Column headings.
    for lo, hi_, name in [
        (1, e_step, "context"),
        (e_step + 1, n_on, "nowcast"),
        (n_on + 1, cols, "forecast"),
    ]:
        g.text(
            bx + (lo - 1 + hi_) / 2 * bw,
            by - 4,
            name,
            size=SMALL,
            anchor="middle",
            fill=MUTED,
        )
    # E, the end of the targets' context, and the latest release.
    exx = bx + e_step * bw
    g.line(exx, by - 2, exx, bottom + 3, stroke=ACCENT, sw=1.1)
    g.text(
        exx,
        bottom + 11,
        "E",
        size=BODY,
        weight="bold",
        fill=ACCENT,
        anchor="middle",
    )
    rx_ = bx + n_on * bw
    g.line(rx_, by - 2, rx_, bottom + 3, stroke=FG, sw=0.8, dash="2.5 2")
    # The highlighted onset step of place 1, read down its column.
    g.rect(
        bx + (hi - 1) * bw - 0.8,
        starts[0] + rh + 1.2,
        bw + 1.6,
        n_lag * (rh + 1) + 1.6,
        stroke=ACCENT,
        sw=1.3,
    )
    # Known-future inputs, beside place 1's lag rows.
    g.text(
        bx + cols * bw + 5,
        starts[0] + 24,
        ["known", "future"],
        size=SMALL,
        fill=COV,
    )
    # One-row legend.
    lgy = max(tri_bottom, bottom + 11) + 12
    items = [
        ("final", "complete"),
        ("lag", "reported so far"),
        ("future", "to forecast"),
        ("none", "no release"),
        ("unseen", "not yet reported"),
    ]
    x = 0.0
    for kind, name in items:
        cell(g, x, lgy - 6.5, 9, 7.5, kind)
        g.text(x + 12, lgy, name, size=SMALL)
        x += 12 + 3.6 * len(name) + 16
    return lgy + 3


def panel_c(g: Svg, oy: float) -> float:
    """Draw panel C, a training block and the LoRA adapters."""
    g.label(0, oy + 9, "C", "Training blocks and LoRA fine-tuning")
    top = oy + 22
    # The training block, cut at a random cut-off.
    cw, rh = 12.0, 9.0
    n_ctx, n_now, n_fc = 8, 4, 3
    cols = n_ctx + n_now + n_fc
    bx = 34.0
    by = top + 18
    rows = [["final" if c < n_ctx else "truth" for c in range(cols)]]
    for d in range(n_now):
        rows.append(
            [
                "lag" if d <= c < n_ctx + n_now - d else "none"
                for c in range(cols)
            ]
        )
    bottom = by + (n_now + 1) * (rh + 1) + 1
    # A stack of blocks behind, for many outbreaks and cut-offs.
    for k in (2, 1):
        g.rect(
            bx - 2 + 3 * k,
            by - 2 - 3 * k,
            cols * cw + 4,
            bottom - by + 4,
            fill="white",
            stroke=LINE,
            sw=0.6,
            rx=2,
        )
    _block(g, bx, by, cols, cw, rh, rows)
    g.text(
        bx - 4, by + rh - 1.5, "target", size=SMALL, anchor="end", fill=ACCENT
    )
    g.text(
        bx - 4,
        (by + rh + 2 + bottom) / 2 + 2.5,
        "lag 0\u20133",
        size=SMALL,
        anchor="end",
        fill=COV,
    )
    g.text(
        bx + n_ctx * cw / 2,
        by - 10,
        "context",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    g.text(
        bx + (n_ctx + cols) / 2 * cw,
        by - 10,
        "final counts",
        size=SMALL,
        anchor="middle",
        fill=TRUTH,
        weight="bold",
    )
    exx = bx + n_ctx * cw
    g.line(exx, by - 2, exx, bottom + 3, stroke=ACCENT, sw=1.1)
    g.text(
        exx,
        bottom + 11,
        "E",
        size=BODY,
        weight="bold",
        fill=ACCENT,
        anchor="middle",
    )
    cut = bx + (n_ctx + n_now) * cw
    g.line(cut, by - 2, cut, bottom + 3, stroke=FG, sw=0.8, dash="2.5 2")
    g.text(cut, bottom + 11, "cut-off", size=SMALL, anchor="middle")
    left_bottom = bottom + 12

    # LoRA: the frozen model with small trained adapters.
    mx = 268.0
    mw, aw = 104.0, 24.0
    y0 = top
    g.rect(mx, y0, mw, 13, fill=CELL, rx=2)
    g.text(
        mx + mw / 2, y0 + 9.3, "input embedding", size=SMALL, anchor="middle"
    )
    g.rect(
        mx - 4,
        y0 + 17,
        mw + aw + 12,
        41,
        fill="none",
        stroke=LINE,
        sw=0.7,
        dash="3 2",
        rx=3,
    )
    g.rect(mx, y0 + 21, mw, 17, fill=CELL, rx=2)
    g.text(
        mx + mw / 2,
        y0 + 32.3,
        "time and group attention",
        size=SMALL,
        anchor="middle",
    )
    g.rect(mx, y0 + 41, mw, 13, fill=CELL, rx=2)
    g.text(mx + mw / 2, y0 + 50.3, "feed-forward", size=SMALL, anchor="middle")
    g.text(mx + mw + aw + 11, y0 + 40, "\u00d7 12", size=SMALL, fill=MUTED)
    g.rect(mx, y0 + 62, mw, 13, fill=CELL, rx=2)
    g.text(mx + mw / 2, y0 + 71.3, "output layer", size=SMALL, anchor="middle")
    for yy, hh in ((y0 + 21, 17), (y0 + 62, 13)):
        g.rect(
            mx + mw + 4,
            yy,
            aw,
            hh,
            fill=ACCENT_SOFT,
            stroke=ACCENT,
            sw=0.9,
            rx=2,
        )
        g.text(
            mx + mw + 4 + aw / 2,
            yy + hh / 2 + 2.5,
            "B\u00b7A",
            size=SMALL,
            anchor="middle",
            fill=ACCENT,
            weight="bold",
        )
    lgy = y0 + 88
    g.rect(mx, lgy - 6.5, 9, 7.5, fill=CELL, rx=1)
    g.text(mx + 12, lgy, "frozen", size=SMALL)
    g.rect(
        mx + 50,
        lgy - 6.5,
        9,
        7.5,
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        sw=0.8,
        rx=1,
    )
    g.text(mx + 62, lgy, "trained", size=SMALL)
    # The arrow from the blocks to the model.
    g.line(bx + cols * cw + 12, y0 + 38, mx - 10, y0 + 38, arrow="ah")
    return max(left_bottom, lgy + 2)


def panel_d(g: Svg, oy: float) -> float:
    """Draw panel D, forecasting with the fine-tuned model."""
    g.label(0, oy + 9, "D", "Forecasting with the fine-tuned model")
    top = oy + 18
    h = 56.0
    gap = 18.0
    widths = [86.0, 86.0, 112.0]
    widths.append(WIDTH - sum(widths) - 3 * gap)
    xs = [0.0]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    labels = [
        "latest data",
        "Chronos-2",
        "quantiles \u2192 draws",
        "nowcast and forecast",
    ]
    for k, (x, w, lab) in enumerate(zip(xs, widths, labels, strict=True)):
        g.rect(
            x,
            top,
            w,
            h,
            fill=PANEL if k == 1 else "white",
            stroke=LINE,
            sw=0.7,
            rx=3,
        )
        g.text(
            x + w / 2, top + 11, lab, size=SMALL, anchor="middle", weight="bold"
        )
        if k:
            g.line(x - gap + 2, top + h / 2, x - 2, top + h / 2, arrow="ah")
    # Stage 1: the latest data as a ragged block.
    cw, rh = 7.0, 5.0
    rows = [["final" if c < 5 else "future" for c in range(10)]]
    for d in range(4):
        rows.append(
            [
                "lag" if 1 - d <= c < 9 - d and c >= d else "none"
                for c in range(10)
            ]
        )
    _block(g, xs[0] + (widths[0] - 70) / 2, top + 19, 10, cw, rh, rows)
    # Stage 2: the model and its adapter.
    x2, w2 = xs[1], widths[1]
    g.rect(
        x2 + w2 / 2 - 22,
        top + 24,
        44,
        15,
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        sw=0.9,
        rx=2,
    )
    g.text(
        x2 + w2 / 2,
        top + 34.5,
        "+ adapter",
        size=SMALL,
        anchor="middle",
        fill=ACCENT,
        weight="bold",
    )
    # Stage 3: a quantile function read at a uniform level.
    x3, w3 = xs[2], widths[2]
    px, py, pw, ph = x3 + 26, top + 17, w3 - 40, 28.0
    g.line(px, py + ph, px + pw, py + ph, stroke=MUTED, sw=0.6)
    g.line(px, py, px, py + ph, stroke=MUTED, sw=0.6)
    g.text(px + pw + 2, py + ph + 2.5, "u", size=SMALL, fill=MUTED)
    g.text(px - 3, py + 6, "Q(u)", size=SMALL, anchor="end", fill=MUTED)
    levels = [
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
    ]
    pts = []
    for u in levels:
        q = 0.5 + 0.32 * ((u - 0.5) / 0.5) ** 3 + 0.16 * (u - 0.5)
        pts.append((px + u * pw, py + ph - (0.08 + 0.84 * q) * ph))
    g.poly(pts, stroke=ACCENT, sw=1.1)
    for x, y in pts:
        g.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.1" fill="{ACCENT}"/>')
    ux, uy = pts[levels.index(0.7)]
    g.line(ux, py + ph, ux, uy, stroke=FG, sw=0.7, dash="1.5 1.5")
    g.line(ux, uy, px, uy, stroke=FG, sw=0.7, dash="1.5 1.5", arrow="ah")
    # Stage 4: trajectories.
    x4, w4 = xs[3], widths[3]
    fx, fy, fw, fh = x4 + 8, top + 16, w4 - 16, 34.0
    n, now = 14, 8
    step = fw / (n - 1)
    base = [12, 13, 15, 18, 20, 24, 27, 30, 32, 34, 35, 35, 34, 33]
    reported = [12, 13, 15, 18, 19, 21, 18, 11]

    def yv(v: float) -> float:
        return fy + fh - v / 52 * fh

    for wdt, op in [(1.6, 0.18), (0.8, 0.32)]:
        up, lo = [], []
        for k in range(3, n):
            s = 0 if k < 4 else (k - 3) * wdt
            up.append((fx + k * step, yv(base[k] + s)))
            lo.append((fx + k * step, yv(base[k] - s * 0.8)))
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in up)
        d += " L" + " L".join(f"{x:.2f},{y:.2f}" for x, y in lo[::-1])
        g.path(d + " Z", stroke="none", fill=ACCENT, opacity=op)
    g.poly(
        [(fx + k * step, yv(base[k])) for k in range(3, n)],
        stroke=ACCENT,
        sw=1.0,
    )
    g.poly([(fx + k * step, yv(base[k])) for k in range(4)], stroke=FG, sw=0.9)
    for k in range(now):
        g.add(
            f'<circle cx="{fx + k * step:.2f}" cy="{yv(reported[k]):.2f}" '
            f'r="1.3" fill="{COV}"/>'
        )
    nx = fx + (now - 0.5) * step
    g.line(nx, fy - 1, nx, fy + fh, stroke=FG, sw=0.7, dash="2 1.5")
    return top + h


def build() -> str:
    """Return Figure 1 as SVG."""
    g = Svg()
    y = panel_a(g, 0.0)
    y = panel_b(g, y + 10)
    y = panel_c(g, y + 10)
    y = panel_d(g, y + 10)
    return g.render(round(y + 3))


def main() -> None:
    """Write the SVG, PDF and optionally PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="Also write PNG.")
    args = parser.parse_args()
    svg = OUT.with_suffix(".svg")
    svg.write_text(build())
    rsvg = shutil.which("rsvg-convert")
    if rsvg is None:
        msg = "rsvg-convert (librsvg) is needed to write the PDF and PNG."
        raise FileNotFoundError(msg)
    subprocess.run(
        [rsvg, "-f", "pdf", "-o", str(OUT.with_suffix(".pdf")), str(svg)],
        check=True,
    )
    if args.png:
        subprocess.run(
            [
                rsvg,
                "-d",
                "200",
                "-p",
                "200",
                "-o",
                str(OUT.with_suffix(".png")),
                str(svg),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
