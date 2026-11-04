# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Interactive score charts: the data `docs/javascripts/scorechart.js` draws.

One chart shows one summary table, with a tab per metric and a checkbox
per model. Its data are a compact JSON file next to the figures, and the
page holds a `div` the script fills. Colours are the static figures'
(`model_colour`), in light and dark, and follow the site theme.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.reporting.plots import model_colour

__all__ = [
    "OVERALL_METRICS",
    "SCORE_METRICS",
    "score_chart_data",
    "score_chart_html",
    "write_score_chart",
]

OVERALL_METRICS: Mapping[str, tuple[str, float | None]] = {
    "rel_wis": ("Relative WIS", 1.0),
    "rel_log_wis": ("Relative WIS on the log scale", 1.0),
    "coverage_50": ("50% coverage", 0.5),
    "coverage_90": ("90% coverage", 0.9),
    "bias": ("Bias", 0.0),
}
"""Columns of `relative_wis` in the overall figures, with each label and
the line of a perfect or baseline score. Bias is in -1 to 1."""

SCORE_METRICS: Mapping[str, tuple[str, float | None]] = {
    "wis": ("WIS", None),
    "log_wis": ("WIS on the log scale", None),
    "coverage_50": ("50% coverage", 0.5),
    "coverage_90": ("90% coverage", 0.9),
    "bias": ("Bias", 0.0),
}
"""Score columns of the interactive score charts, with their reference
lines. Coverage of each interval is drawn against its nominal level."""


def score_chart_data(
    table: pd.DataFrame,
    x: str,
    metrics: Mapping[str, tuple[str, float | None]],
    models: Sequence[str],
    default_on: Sequence[str] | None = None,
    x_label: str | None = None,
    kind: str = "line",
    colours: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Return the chart data for one summary table.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per model and `x`, with columns `model`, `x`, `n` and each
        metric.
    x : str
        The column along the horizontal axis, such as `"horizon"`.
    metrics : mapping of str to (str, float or None)
        Metric column to its tab label and a reference line, such as
        `{"coverage_90": ("90% coverage", 0.9)}`. The first is shown first.
    models : sequence of str
        Models in the order and colours of the static figures. Models in
        `table` but not here are left out.
    default_on : sequence of str, optional
        Models shown at first. Defaults to all.
    x_label : str, optional
        Axis label. Defaults to `x` capitalised.
    kind : str
        `"line"` for an ordered `x`, or `"bar"` for groups.
    colours : mapping of str to (str, str), optional
        Model to its light and dark colour. Defaults to `model_colour`
        over `models`, the colours of the static figures.

    Returns
    -------
    dict
        `kind`, `x_label`, `x` (the values in order), `models` (name,
        light and dark colour, shown at first), `metrics` (key, label,
        reference) and `rows`, one per model and `x`: the model index, the
        `x` index, `n` and the metric values.
    """
    shown = [m for m in models if m in set(table["model"])]
    colours = colours or {
        m: (model_colour(m, models, "light"), model_colour(m, models, "dark"))
        for m in shown
    }
    on = set(shown if default_on is None else default_on)
    xs = (
        sorted(table[x].unique())
        if kind == "line"
        else list(dict.fromkeys(table[x]))
    )
    keys = list(metrics)
    rows = [
        [
            shown.index(r["model"]),
            xs.index(r[x]),
            int(r["n"]),
            *(None if pd.isna(r[k]) else round(float(r[k]), 4) for k in keys),
        ]
        for _, r in table.loc[table["model"].isin(shown)].iterrows()
    ]
    return {
        "kind": kind,
        "x_label": x_label or x.capitalize(),
        "x": [v.item() if hasattr(v, "item") else v for v in xs],
        "models": [
            {
                "name": m,
                "light": colours[m][0],
                "dark": colours[m][1],
                "on": m in on,
            }
            for m in shown
        ],
        "metrics": [
            {"key": k, "label": metrics[k][0], "ref": metrics[k][1]}
            for k in keys
        ],
        "rows": rows,
    }


def write_score_chart(path: Path, data: Mapping[str, object]) -> None:
    """Write the chart data as compact JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def score_chart_html(src: str, title: str, static: str | None = None) -> str:
    """Return the page element the score chart script fills.

    Parameters
    ----------
    src : str
        URL of the JSON file, relative to the page.
    title : str
        Accessible name of the chart.
    static : str, optional
        Path of a static SVG, relative to the page, shown without
        JavaScript. Its `-dark` twin is shown in the dark theme.
    """
    fallback = "The interactive chart needs JavaScript."
    if static is not None:
        dark = static.removesuffix(".svg") + "-dark.svg"
        fallback = (
            f'<img src="{static}#only-light" alt="{title}">'
            f'<img src="{dark}#only-dark" alt="{title}">'
        )
    return (
        f'<div class="scorechart" data-src="{src}" role="group" '
        f'aria-label="{title}">\n'
        f"<noscript>{fallback}</noscript>\n"
        "</div>\n"
    )
