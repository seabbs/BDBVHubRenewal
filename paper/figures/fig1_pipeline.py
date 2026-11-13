# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Draw Figure 1, a schematic of the fine-tuning pipeline.

Writes `fig1-pipeline.svg` next to this script, and `fig1-pipeline.pdf`
and, with `--png`, a 200 dpi `fig1-pipeline.png` with `rsvg-convert`
(librsvg). Run from the repo root:

    uv run python paper/figures/fig1_pipeline.py [--png]

The panels are:

- A, the generative chain of the epinowcast simulations
  (`scripts/finetune_simulate.R`), with the options drawn at each stage;
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
        g.rect(x, y, w, h, fill=ACCENT_SOFT, stroke=ACCENT, sw=0.8)
    elif kind == "none":
        g.rect(x, y, w, h, fill="url(#hatch)", stroke="white", sw=0.5)
    elif kind == "unseen":
        g.rect(x, y, w, h, fill="none", stroke=LINE, sw=0.5, dash="1.5 1.5")


def panel_a(g: Svg, oy: float) -> float:
    """Draw panel A, the simulations' generative chain."""
    g.label(
        0, oy + 9, "A", "Simulated outbreaks: the epinowcast generative model"
    )
    widths = [98.0, 80.0, 102.0, 84.0, 64.0]
    gap = (WIDTH - sum(widths)) / (len(widths) - 1)
    top = oy + 20
    stages = [
        (
            "Latent process",
            [
                "renewal",
                "renewal with incubation",
                "growth random walk",
                "log-incidence|random walk",
                "growth with steps",
                "smooth growth",
            ],
            "Rt or growth paths, some",
            ["with drift and steps"],
        ),
        (
            "Expected onsets",
            [
                "one series",
                "several places",
                "age groups",
                "places by ages",
            ],
            "strata share growth;",
            ["some stay quiet"],
        ),
        (
            "Onset-to-report delay",
            [
                "parametric: lognormal,|gamma or exponential",
                "and/or a non-parametric|hazard",
            ],
            "stable, or with drift, steps",
            ["and wander; stratum effects"],
        ),
        (
            "Reporting",
            [
                "day-of-week effects",
                "holidays",
                "week effects",
                "every day or on|set weekdays",
            ],
            "the report days",
            ["may change once"],
        ),
        (
            "Observation",
            ["NB2", "NB1", "Poisson"],
            "noise on the",
            ["reported counts"],
        ),
    ]
    n_lines = max(sum(o.count("|") + 1 for o in st[1]) for st in stages)
    note_y = top + 44 + n_lines * BODY * 1.2 + 4
    height = note_y - top + 2 * SMALL * 1.25 + 6
    x = 0.0
    for i, (w, (head, opts, note1, note2)) in enumerate(
        zip(widths, stages, strict=True)
    ):
        g.rect(x, top, w, height, fill="white", stroke=LINE, sw=0.7, rx=3)
        g.rect(x, top, w, 34, fill=PANEL, stroke="none", rx=3)
        g.rect(x, top + 30, w, 4, fill=PANEL, stroke="none", rx=0)
        _glyph(g, i, x + w / 2, top + 13)
        g.text(
            x + w / 2, top + 30, head, size=HEAD, weight="bold", anchor="middle"
        )
        lines = [
            ("\u2022 " if k == 0 else "   ") + part
            for opt in opts
            for k, part in enumerate(opt.split("|"))
        ]
        g.text(x + 4, top + 44, lines, size=BODY, lead=1.2)
        g.text(
            x + 4,
            note_y,
            [note1, *note2],
            size=SMALL,
            fill=MUTED,
            italic=True,
        )
        if i < len(widths) - 1:
            g.line(
                x + w + 1.5, top + 17, x + w + gap - 1.5, top + 17, arrow="ah"
            )
        x += w + gap
    yb = top + height + 10
    g.text(
        0,
        yb,
        "Each outbreak gives a daily reporting triangle by stratum "
        "(count by onset day and report day), also summed to weeks, "
        "with the total over strata.",
        size=SMALL,
        fill=MUTED,
    )
    return yb + 6


def _glyph(g: Svg, i: int, cx: float, cy: float) -> None:
    """Draw the small icon of stage `i` of panel A, centred on cx, cy."""
    if i == 0:
        # A latent path: growth, a step down and decline.
        pts = [
            (cx - 22 + k * 2.2, cy + 7 - 13 * (k / 10) ** 1.6)
            for k in range(11)
        ]
        pts += [
            (cx + 2 + k * 2.2, cy - 6 + 12 * (k / 9) ** 0.8) for k in range(10)
        ]
        g.poly(pts, stroke=ACCENT, sw=1.1)
    elif i == 1:
        # Onsets by stratum: three small bar series.
        for s, (dy, scale) in enumerate([(-6, 0.6), (0, 1.0), (6, 0.4)]):
            for k in range(9):
                h = scale * 4.5 * (0.4 + (k / 8) ** 1.3)
                g.rect(
                    cx - 18 + k * 4,
                    cy + dy + 2 - h,
                    3,
                    h,
                    fill=[MUTED, ACCENT, HATCH][s],
                    rx=0,
                )
    elif i == 2:
        # A delay distribution and a drifted copy.
        def pmf(shift: float) -> list[tuple[float, float]]:
            out = []
            for k in range(25):
                t = k / 24 * 5
                v = t * 2.7 ** (-(t - shift) * 1.4) if t > shift else 0.0
                out.append((cx - 20 + k * 1.7, cy + 7 - 17 * min(v, 0.75)))
            return out

        g.line(cx - 21, cy + 7, cx + 22, cy + 7, stroke=LINE, sw=0.6)
        g.poly(pmf(0.0), stroke=ACCENT, sw=1.1)
        g.poly(pmf(0.9), stroke=MUTED, sw=0.9, dash="2 1.5")
    elif i == 3:
        # Report-day effects: a week of reports with two silent days.
        for k, h in enumerate([10, 7, 0, 8, 12, 0, 4]):
            if h:
                g.rect(cx - 17 + k * 5, cy + 7 - h, 3.6, h, fill=ACCENT, rx=0)
            else:
                g.line(
                    cx - 17 + k * 5,
                    cy + 7,
                    cx - 13.4 + k * 5,
                    cy + 7,
                    stroke=MUTED,
                    sw=0.8,
                )
    else:
        # Counts scattered around their expectation.
        g.poly(
            [(cx - 18 + k * 4, cy + 5 - k * 1.1) for k in range(10)],
            stroke=MUTED,
            sw=0.8,
        )
        for k, e in enumerate([1.5, -2.5, 2, -1, 3, -3, 0.5, 2.5, -2, 1]):
            g.add(
                f'<circle cx="{cx - 18 + k * 4:.2f}" '
                f'cy="{cy + 5 - k * 1.1 + e:.2f}" r="1.2" fill="{ACCENT}"/>'
            )


def panel_b(g: Svg, oy: float) -> float:
    """Draw panel B, the reporting triangle as ragged series."""
    g.label(
        0, oy + 9, "B", "The reporting triangle as ragged series for Chronos-2"
    )
    top = oy + 22
    # A toy triangle: onset steps 1 to 8, releases at steps 4 to 8, and
    # lag series 0 to 3, so the target is complete up to E = 4.
    n_on, first_rel, n_lag = 8, 4, 4
    releases = list(range(first_rel, n_on + 1))
    e_step = n_on - n_lag
    cw, ch = 16.0, 11.0
    # The triangle, rows onset steps, columns releases old to new.
    tx = 26.0
    ty = top + 22
    g.text(
        tx + cw * len(releases) / 2,
        top + 8,
        "release, old \u2192 new",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    g.text(tx - 4, ty - 3, "onset", size=SMALL, anchor="end", fill=MUTED)
    for j in range(len(releases)):
        g.text(
            tx + j * cw + cw / 2,
            ty - 3,
            f"R{j + 1}",
            size=SMALL,
            anchor="middle",
            fill=MUTED,
        )
    for t in range(1, n_on + 1):
        y = ty + (t - 1) * ch
        g.text(tx - 4, y + ch - 3, str(t), size=SMALL, anchor="end", fill=MUTED)
        for j, r in enumerate(releases):
            x = tx + j * cw
            if t > r:
                cell(g, x, y, cw, ch, "unseen")
                continue
            lag = r - t
            kind = "final" if lag >= n_lag else "lag"
            cell(g, x, y, cw, ch, kind)
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
    # The highlighted onset step, read along its row.
    hi = e_step + 1
    hy = ty + (hi - 1) * ch
    g.rect(
        tx + (hi - first_rel) * cw - 0.8,
        hy - 0.8,
        (n_on - hi + 1) * cw + 1.6,
        ch + 1.6,
        stroke=ACCENT,
        sw=1.3,
    )
    ey = ty + e_step * ch
    g.line(tx - 14, ey, tx + cw * len(releases) + 3, ey, stroke=ACCENT, sw=1.1)
    g.text(
        tx - 20,
        ey + 3,
        "E",
        size=BODY,
        weight="bold",
        fill=ACCENT,
        anchor="middle",
    )
    ly = ty + n_on * ch + 9
    g.text(
        tx - 20,
        ly,
        "outlined: lag 0, newest in each release",
        size=SMALL,
        fill=MUTED,
    )

    # The arrow to the ragged layout.
    ax = tx + cw * len(releases) + 8
    g.text(ax + 15, ty + 33, "by lag", size=SMALL, anchor="middle", fill=MUTED)
    g.line(ax, ty + 38, ax + 30, ty + 38, arrow="ah")

    # The ragged block: rows the target and lag series, columns onset
    # steps and two forecast steps.
    bx = ax + 72
    by = ty
    cols = n_on + 2
    bw = 15.0
    rows = ["target (final)", *[f"lag {d}" for d in range(n_lag)]]
    for k, name in enumerate(rows):
        y = by + k * (ch + 2) + (3 if k else 0)
        g.text(
            bx - 4,
            y + ch - 3,
            name,
            size=SMALL,
            anchor="end",
            fill=ACCENT if k == 0 else COV,
        )
        for c in range(cols):
            t = c + 1
            x = bx + c * bw
            if k == 0:
                kind = "final" if t <= e_step else "future"
            else:
                d = k - 1
                kind = "lag" if first_rel - d <= t <= n_on - d else "none"
            cell(g, x, y, bw, ch, kind)
            if k == 0 and kind == "future":
                g.text(
                    x + bw / 2,
                    y + ch - 3,
                    "?",
                    size=SMALL,
                    anchor="middle",
                    fill=ACCENT,
                    weight="bold",
                )
    # Column headings: context, nowcast and forecast steps.
    hy0 = by - 13
    for lo, hi_, name in [
        (1, e_step, "context"),
        (e_step + 1, n_on, "nowcast"),
        (n_on + 1, cols, "forecast"),
    ]:
        g.text(
            bx + (lo - 1 + hi_) / 2 * bw,
            hy0,
            name,
            size=SMALL,
            anchor="middle",
            fill=MUTED,
        )
    for c in range(cols):
        lab = str(c + 1) if c < n_on else f"+{c - n_on + 1}"
        g.text(
            bx + c * bw + bw / 2,
            by - 3,
            lab,
            size=SMALL,
            anchor="middle",
            fill=MUTED,
        )
    bottom = by + n_lag * (ch + 2) + 3 + ch
    # E, the end of the target's context, and the latest release.
    exx = bx + e_step * bw
    g.line(exx, by - 4, exx, bottom + 3, stroke=ACCENT, sw=1.1)
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
    g.line(rx_, by - 4, rx_, bottom + 3, stroke=FG, sw=0.8, dash="2.5 2")
    g.text(rx_, bottom + 11, "latest release", size=SMALL, anchor="middle")
    # The highlighted onset step, read down its column.
    g.rect(
        bx + (hi - 1) * bw - 0.8,
        by + ch + 2,
        bw + 1.6,
        n_lag * (ch + 2) + 1.6,
        stroke=ACCENT,
        sw=1.3,
    )
    # Notes on the right.
    nx = bx + cols * bw + 6
    g.text(
        nx,
        by + ch - 3,
        ["forecast from E,", "nowcast steps too"],
        size=SMALL,
        fill=ACCENT,
    )
    g.text(
        nx,
        by + 2 * ch + 14,
        [
            "each place's counts",
            "reported so far enter",
            "the forecast window as",
            "known-future inputs",
        ],
        size=SMALL,
        fill=COV,
    )
    # Legend.
    lx = bx - 30
    lgy = bottom + 20
    items = [
        ("final", "complete count"),
        ("lag", "count reported so far"),
        ("future", "to forecast"),
        ("none", "no release at that lag"),
    ]
    for k, (kind, name) in enumerate(items):
        x = lx + (k % 2) * 118
        y = lgy + (k // 2) * 11
        cell(g, x, y - 6.5, 9, 7.5, kind)
        g.text(x + 12, y, name, size=SMALL)
    return max(ly + 2, lgy + 14)


def panel_c(g: Svg, oy: float) -> float:
    """Draw panel C, a training block and the LoRA adapters."""
    g.label(0, oy + 9, "C", "Training blocks and LoRA fine-tuning")
    top = oy + 22
    # The training block: a cut-off from a simulated release history.
    g.text(
        0,
        top + 4,
        "One block cut from a simulated release history",
        size=BODY,
        weight="bold",
    )
    cw, ch = 11.5, 10.0
    n_ctx, n_now, n_fc = 8, 4, 3
    cols = n_ctx + n_now + n_fc
    bx = 40.0
    by = top + 24
    for lo, hi_, name, col in [
        (0, n_ctx, "context", MUTED),
        (n_ctx, cols, "future, from the final counts", ACCENT),
    ]:
        g.text(
            bx + (lo + hi_) / 2 * cw,
            by - 6,
            name,
            size=SMALL,
            anchor="middle",
            fill=col,
        )
    rows = ["target", *[f"lag {d}" for d in range(n_now)]]
    for k, name in enumerate(rows):
        y = by + k * (ch + 2) + (3 if k else 0)
        g.text(
            bx - 4,
            y + ch - 2.5,
            name,
            size=SMALL,
            anchor="end",
            fill=ACCENT if k == 0 else COV,
        )
        for c in range(cols):
            x = bx + c * cw
            if k == 0:
                kind = "final" if c < n_ctx else "truth"
            else:
                d = k - 1
                kind = "lag" if d <= c < n_ctx + n_now - d else "none"
            cell(g, x, y, cw, ch, kind)
    bottom = by + n_now * (ch + 2) + 3 + ch
    exx = bx + n_ctx * cw
    g.line(exx, by - 3, exx, bottom + 3, stroke=ACCENT, sw=1.1)
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
    g.line(cut, by - 3, cut, bottom + 3, stroke=FG, sw=0.8, dash="2.5 2")
    g.text(cut, bottom + 11, "random cut-off", size=SMALL, anchor="middle")
    g.text(
        bx - 36,
        bottom + 25,
        [
            "Each outbreak gives several weekly and daily blocks, each with",
            "a random cut-off, delay D and set of places. Daily blocks keep",
            "weekly releases, so their lag series have a value once a week.",
            "Checkpoints are trained on nested sets of simulated outbreaks.",
        ],
        size=SMALL,
        fill=MUTED,
    )
    left_bottom = bottom + 25 + 3 * SMALL * 1.25 + 4

    # LoRA: the frozen model with small trained adapters.
    mx = 232.0
    g.text(
        mx,
        top + 4,
        "LoRA: a frozen model, small trained adapters",
        size=BODY,
        weight="bold",
    )
    mw = 96.0
    aw = 22.0
    y0 = top + 14
    g.rect(mx, y0, mw, 15, fill=CELL, rx=2)
    g.text(
        mx + mw / 2, y0 + 10.5, "input embedding", size=SMALL, anchor="middle"
    )
    g.rect(
        mx - 4,
        y0 + 21,
        mw + aw + 12,
        47,
        fill="none",
        stroke=LINE,
        sw=0.7,
        dash="3 2",
        rx=3,
    )
    g.rect(mx, y0 + 25, mw, 20, fill=CELL, rx=2)
    g.text(
        mx + mw / 2,
        y0 + 33,
        ["time and group attention", "q, k, v, o"],
        size=SMALL,
        anchor="middle",
        lead=1.15,
    )
    g.rect(mx, y0 + 49, mw, 15, fill=CELL, rx=2)
    g.text(mx + mw / 2, y0 + 59.5, "feed-forward", size=SMALL, anchor="middle")
    g.text(mx + mw + aw + 11, y0 + 50, "\u00d7 12", size=SMALL, fill=MUTED)
    g.rect(mx, y0 + 74, mw, 15, fill=CELL, rx=2)
    g.text(mx + mw / 2, y0 + 84.5, "output layer", size=SMALL, anchor="middle")
    for yy, hh in ((y0 + 25, 20), (y0 + 74, 15)):
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
    g.text(
        mx,
        y0 + 100,
        "y = Wx + (\u03b1/r) B(Ax), rank r = 8, \u03b1 = 16",
        size=SMALL,
        fill=FG,
    )
    # Legend.
    lgy = y0 + 112
    g.rect(mx, lgy - 6.5, 9, 7.5, fill=CELL, rx=1)
    g.text(mx + 12, lgy, "frozen, as pretrained", size=SMALL)
    g.rect(mx, lgy + 4.5, 9, 7.5, fill=ACCENT_SOFT, stroke=ACCENT, sw=0.8, rx=1)
    g.text(mx + 12, lgy + 11, "trained on the simulations", size=SMALL)
    # Why it is fast.
    wx = mx + mw + aw + 30
    g.text(wx, y0 + 6, "Why it is fast", size=BODY, weight="bold")
    g.text(
        wx,
        y0 + 18,
        [
            "about 1.2M of 120M",
            "weights trained (1%)",
        ],
        size=SMALL,
        fill=ACCENT,
        weight="bold",
    )
    g.text(
        wx,
        y0 + 40,
        [
            "\u2022 no gradients or",
            "  optimiser state for",
            "  the frozen backbone",
            "\u2022 adapter about 5 MB,",
            "  model about 0.5 GB",
            "\u2022 trained once on a",
            "  laptop GPU, reused",
            "  for every forecast",
        ],
        size=SMALL,
        lead=1.22,
    )
    return max(left_bottom, lgy + 16)


def panel_d(g: Svg, oy: float) -> float:
    """Draw panel D, forecasting with the fine-tuned model."""
    g.label(0, oy + 9, "D", "Forecasting with the fine-tuned model")
    top = oy + 24
    h = 62.0
    # Stage 1: the latest release as a ragged block.
    w1 = 80.0
    g.rect(0, top, w1, h, fill="white", stroke=LINE, sw=0.7, rx=3)
    g.text(
        w1 / 2,
        top + 10,
        ["latest data", "as a ragged block"],
        size=SMALL,
        anchor="middle",
        weight="bold",
    )
    cw, ch = 6.5, 5.0
    gx, gy = 10.0, top + 26
    for k in range(5):
        for c in range(10):
            if k == 0:
                kind = "final" if c < 5 else "future"
            else:
                d = k - 1
                kind = "lag" if 1 - d <= c < 9 - d else "none"
            y = gy + k * (ch + 1) + (1.5 if k else 0)
            if kind == "future":
                g.rect(
                    gx + c * cw,
                    y,
                    cw,
                    ch,
                    fill=ACCENT_SOFT,
                    stroke=ACCENT,
                    sw=0.4,
                    rx=0.5,
                )
            elif kind == "lag":
                g.rect(
                    gx + c * cw,
                    y,
                    cw,
                    ch,
                    fill=COV_SOFT,
                    stroke=COV,
                    sw=0.4,
                    rx=0.5,
                )
            elif kind == "final":
                g.rect(
                    gx + c * cw,
                    y,
                    cw,
                    ch,
                    fill=CELL,
                    stroke="white",
                    sw=0.4,
                    rx=0.5,
                )
            else:
                g.rect(
                    gx + c * cw,
                    y,
                    cw,
                    ch,
                    fill="url(#hatch)",
                    stroke="white",
                    sw=0.4,
                    rx=0.5,
                )
    gap = 18.0
    # Stage 2: the model.
    x2 = w1 + gap
    w2 = 88.0
    g.line(w1 + 2, top + h / 2, x2 - 2, top + h / 2, arrow="ah")
    g.rect(x2, top, w2, h, fill=PANEL, stroke=LINE, sw=0.7, rx=3)
    g.text(
        x2 + w2 / 2,
        top + 14,
        "Chronos-2",
        size=HEAD,
        anchor="middle",
        weight="bold",
    )
    g.rect(
        x2 + w2 / 2 - 24,
        top + 21,
        48,
        14,
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        sw=0.9,
        rx=2,
    )
    g.text(
        x2 + w2 / 2,
        top + 31,
        "+ adapter",
        size=SMALL,
        anchor="middle",
        fill=ACCENT,
        weight="bold",
    )
    g.text(
        x2 + w2 / 2,
        top + 47,
        ["one forward pass,", "no refitting"],
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    # Stage 3: quantiles, read at uniform draws.
    x3 = x2 + w2 + gap
    w3 = 138.0
    g.line(x2 + w2 + 2, top + h / 2, x3 - 2, top + h / 2, arrow="ah")
    g.rect(x3, top, w3, h, fill="white", stroke=LINE, sw=0.7, rx=3)
    g.text(x3 + 5, top + 10, "quantiles to draws", size=SMALL, weight="bold")
    px, py, pw, ph = x3 + 20, top + 15, 52.0, 35.0
    g.line(px, py + ph, px + pw, py + ph, stroke=MUTED, sw=0.6)
    g.line(px, py, px, py + ph, stroke=MUTED, sw=0.6)
    g.text(
        px + pw / 2,
        py + ph + 8,
        "level u",
        size=SMALL,
        anchor="middle",
        fill=MUTED,
    )
    g.text(px - 3, py + 7, "Q(u)", size=SMALL, anchor="end", fill=MUTED)
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
    u0 = 0.7
    i0 = levels.index(u0)
    ux, uy = pts[i0]
    g.line(ux, py + ph, ux, uy, stroke=FG, sw=0.7, dash="1.5 1.5")
    g.line(ux, uy, px, uy, stroke=FG, sw=0.7, dash="1.5 1.5", arrow="ah")
    g.text(
        px + pw + 5,
        py + 8,
        [
            "draw u ~ U(0, 1),",
            "read Q(u);",
            "round, and keep",
            "at least the count",
            "reported so far",
        ],
        size=SMALL,
        fill=FG,
        lead=1.2,
    )
    # Stage 4: trajectories.
    x4 = x3 + w3 + gap
    w4 = WIDTH - x4
    g.line(x3 + w3 + 2, top + h / 2, x4 - 2, top + h / 2, arrow="ah")
    g.rect(x4, top, w4, h, fill="white", stroke=LINE, sw=0.7, rx=3)
    g.text(x4 + 5, top + 10, "nowcast and forecast", size=SMALL, weight="bold")
    fx, fy, fw, fh = x4 + 8, top + 15, w4 - 16, 36.0
    n = 14
    now = 8
    step = fw / (n - 1)
    base = [12, 13, 15, 18, 20, 24, 27, 30, 32, 34, 35, 35, 34, 33]
    reported = [12, 13, 15, 18, 19, 21, 18, 11, 0, 0, 0, 0, 0, 0]

    def yv(v: float) -> float:
        return fy + fh - v / 52 * fh

    def band(k: int, wdt: float) -> tuple[float, float]:
        s = 0 if k < 4 else (k - 3) * wdt
        return base[k] - s * 0.8, base[k] + s

    for wdt, op in [(1.6, 0.18), (0.8, 0.32)]:
        upper = [(fx + k * step, yv(band(k, wdt)[1])) for k in range(3, n)]
        lower = [(fx + k * step, yv(band(k, wdt)[0])) for k in range(3, n)]
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in upper)
        d += " L" + " L".join(f"{x:.2f},{y:.2f}" for x, y in lower[::-1])
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
    g.line(nx, fy - 2, nx, fy + fh, stroke=FG, sw=0.7, dash="2 1.5")
    g.text(nx - 2, fy + 4, "now", size=SMALL, anchor="end", fill=FG)
    g.text(
        fx,
        fy + fh + 8,
        "dots: reported so far",
        size=SMALL,
        fill=MUTED,
    )
    return top + h + 4


def build() -> str:
    """Return Figure 1 as SVG."""
    g = Svg()
    y = panel_a(g, 0.0)
    y = panel_b(g, y + 7)
    y = panel_c(g, y + 7)
    y = panel_d(g, y + 7)
    return g.render(round(y + 4))


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
