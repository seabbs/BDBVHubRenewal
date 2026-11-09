# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Per-forecast scores for the in-browser summaries.

The page embeds one row per model, origin, release, place and week, and
`docs/javascripts/summarise.js` summarises any selection of them as
`relative_wis` does: over the forecasts the baseline also made, the mean
of each metric and, for relative metrics, the model's mean over the
baseline's. A new metric is a score column and a label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from bdbvhubrenewal.forecasters import NATIONAL

__all__ = ["ENERGY_SUMMARY", "SUMMARY_METRICS", "summary_html", "summary_rows"]

SUMMARY_METRICS: tuple[tuple[str, str, float | None, bool], ...] = (
    ("wis", "WIS", None, True),
    ("log_wis", "WIS on the log scale", None, True),
    ("coverage_50", "50% coverage", 0.5, False),
    ("coverage_90", "90% coverage", 0.9, False),
    ("bias", "Bias", 0.0, False),
)
"""Score column, label, reference line and whether it is also shown
relative to the baseline, whose reference is then 1."""

ENERGY_SUMMARY: tuple[tuple[str, str], ...] = (
    ("energy", "Energy score"),
    ("log_energy", "Energy score on the log scale"),
)
"""Energy score columns and labels. There is one score per model, origin,
release and pool, so they show in the overall view only."""

_POOLS = {
    "national": "National",
    "provinces": "Provinces",
    "zones": "Health zones",
}


def _pool(location: str) -> str:
    if location == NATIONAL:
        return "national"
    return "provinces" if location.count("-") == 1 else "zones"


def summary_rows(
    scores: pd.DataFrame,
    models: Sequence[str],
    colours: Mapping[str, tuple[str, str, str]],
    names: Mapping[str, str],
    baseline: str = "randomwalk",
    metrics: Sequence[tuple[str, str, float | None, bool]] = SUMMARY_METRICS,
    default_on: Sequence[str] | None = None,
    energy: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Return the per-forecast rows and their labels for one target.

    Parameters
    ----------
    scores : pandas.DataFrame
        One row per forecast scored against one release, with `model`,
        `origin`, `truth_vintage`, `location`, `target_end_date`,
        `horizon` (weeks from the reference date) and the metric columns.
    models : sequence of str
        Models in order. Rows of other models are left out.
    colours : mapping of str to (str, str, str)
        Model to its light colour, dark colour and dash.
    names : mapping of str to str
        Place names by id. Zones not named show their code.
    baseline : str
        The model relative metrics divide by.
    metrics : sequence of (str, str, float or None, bool)
        As `SUMMARY_METRICS`.
    default_on : sequence of str, optional
        Models ticked at first. Defaults to all.
    energy : pandas.DataFrame, optional
        Energy scores (`energy_scores`) with `model`, `origin`,
        `truth_vintage`, `scale` and the `ENERGY_SUMMARY` columns.

    Returns
    -------
    dict
        `models` (name, light and dark colour, dash, ticked at first),
        `baseline` (its index), `metrics` (key, label, ref, relative),
        `origins`, `releases`, `weeks`, `pools` (id and label, in order),
        `places` (id, name, pool) and `rows`: model, origin, release,
        place and week indices, then each metric, and `horizons`, each
        row's horizon. With `energy`, also `energy`: `metrics` (key and
        label) and `rows` of model, origin, release and pool indices, then
        each energy score.
    """
    shown = [m for m in models if m in set(scores["model"])]
    s = scores.loc[scores["model"].isin(shown)]
    origins = sorted(set(s["origin"].astype(str)))
    releases = sorted(set(s["truth_vintage"].astype(str)))
    weeks = sorted(set(s["target_end_date"].astype(str)))
    pools = [p for p in _POOLS if p in set(s["location"].map(_pool))]
    places = sorted(
        set(s["location"]),
        key=lambda x: (pools.index(_pool(x)), names.get(x, x)),
    )
    on = set(shown if default_on is None else default_on)
    keys = [m[0] for m in metrics]
    idx = {
        "model": {m: i for i, m in enumerate(shown)},
        "origin": {o: i for i, o in enumerate(origins)},
        "truth_vintage": {r: i for i, r in enumerate(releases)},
        "location": {p: i for i, p in enumerate(places)},
        "target_end_date": {w: i for i, w in enumerate(weeks)},
    }
    cols = s[[*idx, *keys]].astype(dict.fromkeys(idx, str))
    horizons = [int(h) for h in s["horizon"]]
    rows = [
        [
            *(idx[c][v] for c, v in zip(idx, r[: len(idx)], strict=True)),
            *(round(float(v), 6) for v in r[len(idx) :]),
        ]
        for r in cols.itertuples(index=False)
    ]
    out: dict[str, object] = {
        "models": [
            {
                "name": m,
                "light": colours[m][0],
                "dark": colours[m][1],
                "dash": colours[m][2],
                "on": m in on,
            }
            for m in shown
        ],
        "baseline": shown.index(baseline),
        "metrics": [
            {"key": k, "label": label, "ref": ref, "relative": rel}
            for k, label, ref, rel in metrics
        ],
        "origins": origins,
        "releases": releases,
        "weeks": weeks,
        "pools": [{"id": p, "label": _POOLS[p]} for p in pools],
        "places": [
            {"id": p, "name": names.get(p, p), "pool": _pool(p)} for p in places
        ],
        "rows": rows,
        "horizons": horizons,
    }
    if energy is not None:
        out["energy"] = _energy_rows(energy, idx, pools)
    return out


def _energy_rows(
    energy: pd.DataFrame,
    idx: Mapping[str, Mapping[str, int]],
    pools: Sequence[str],
) -> dict[str, object]:
    """Index the energy scores into the summary's models, rounds and pools."""
    e = energy.loc[
        energy["model"].isin(set(idx["model"]))
        & energy["origin"].astype(str).isin(set(idx["origin"]))
        & energy["truth_vintage"].astype(str).isin(set(idx["truth_vintage"]))
        & energy["scale"].isin(set(pools))
    ]
    keys = [k for k, _ in ENERGY_SUMMARY]
    return {
        "metrics": [{"key": k, "label": label} for k, label in ENERGY_SUMMARY],
        "rows": [
            [
                idx["model"][m],
                idx["origin"][str(o)],
                idx["truth_vintage"][str(r)],
                pools.index(p),
                *(round(float(v), 6) for v in vals),
            ]
            for m, o, r, p, *vals in e[
                ["model", "origin", "truth_vintage", "scale", *keys]
            ].itertuples(index=False)
        ],
    }


def summary_html(
    src: str,
    title: str,
    view: str = "overall",
    static: str | None = None,
    map_src: str | None = None,
) -> str:
    """Return the page element `docs/javascripts/scoresummary.js` fills.

    Parameters
    ----------
    src : str
        URL of the `summary_rows` JSON, relative to the page.
    title : str
        Accessible name of the summary.
    view : str
        `"overall"` for one group of bars for the chosen scale, `"place"`
        for one group per place and a place picker, or `"horizon"` for a
        line per model by horizon.
    static : str, optional
        Path of a static SVG, relative to the page, shown without
        JavaScript. Its `-dark` twin is shown in the dark theme.
    map_src : str, optional
        URL of the health zone GeoJSON, relative to the page. The place
        view then offers a map of the metric shown.
    """
    fallback = "The interactive summary needs JavaScript."
    if static is not None:
        dark = static.removesuffix(".svg") + "-dark.svg"
        fallback = (
            f'<img src="{static}#only-light" alt="{title}">'
            f'<img src="{dark}#only-dark" alt="{title}">'
        )
    shown = "" if map_src is None else f' data-map="{map_src}"'
    return (
        f'<div class="scoresummary" data-src="{src}" data-view="{view}"{shown} '
        f'role="group" aria-label="{title}">\n'
        f"<noscript>{fallback}</noscript>\n"
        "</div>\n"
    )
