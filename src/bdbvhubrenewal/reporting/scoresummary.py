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

__all__ = ["SUMMARY_METRICS", "summary_rows"]

SUMMARY_METRICS: tuple[tuple[str, str, float | None, bool], ...] = (
    ("wis", "WIS", None, True),
    ("log_wis", "WIS on the log scale", None, True),
    ("coverage_50", "50% coverage", 0.5, False),
    ("coverage_90", "90% coverage", 0.9, False),
    ("bias", "Bias", 0.0, False),
)
"""Score column, label, reference line and whether it is also shown
relative to the baseline, whose reference is then 1."""

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
    colours: Mapping[str, tuple[str, str]],
    names: Mapping[str, str],
    baseline: str = "randomwalk",
    metrics: Sequence[tuple[str, str, float | None, bool]] = SUMMARY_METRICS,
    default_on: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return the per-forecast rows and their labels for one target.

    Parameters
    ----------
    scores : pandas.DataFrame
        One row per forecast scored against one release, with `model`,
        `origin`, `truth_vintage`, `location`, `target_end_date` and the
        metric columns.
    models : sequence of str
        Models in order. Rows of other models are left out.
    colours : mapping of str to (str, str)
        Model to its light and dark colour.
    names : mapping of str to str
        Place names by id. Zones not named show their code.
    baseline : str
        The model relative metrics divide by.
    metrics : sequence of (str, str, float or None, bool)
        As `SUMMARY_METRICS`.
    default_on : sequence of str, optional
        Models ticked at first. Defaults to all.

    Returns
    -------
    dict
        `models` (name, light and dark colour, ticked at first),
        `baseline` (its index), `metrics` (key, label, ref, relative),
        `origins`, `releases`, `weeks`, `pools` (id and label, in order),
        `places` (id, name, pool) and `rows`: model, origin, release,
        place and week indices, then each metric.
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
    rows = [
        [
            *(idx[c][v] for c, v in zip(idx, r[: len(idx)], strict=True)),
            *(round(float(v), 6) for v in r[len(idx) :]),
        ]
        for r in cols.itertuples(index=False)
    ]
    return {
        "models": [
            {
                "name": m,
                "light": colours[m][0],
                "dark": colours[m][1],
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
    }
