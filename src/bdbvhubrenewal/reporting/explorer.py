"""Data for the interactive forecast explorer on the docs site.

The explorer is a small script (`docs/javascripts/explorer.js`) that draws
SVG from a JSON file, so it works in a static site and follows the theme.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.forecasters import NATIONAL

__all__ = ["explorer_data", "explorer_html", "write_explorer"]

QUANTILE_COLUMNS = ("q05", "q25", "q50", "q75", "q95")
"""Interval columns of the forecast quantiles."""

SCORE_COLUMNS = ("wis", "coverage_50", "coverage_90", "bias")
"""Score columns shown in the explorer's table."""


def _level(location: str, provinces: set[str]) -> str:
    if location == NATIONAL:
        return "National"
    return "Province" if location in provinces else "Zone"


def explorer_data(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    names: Mapping[str, str],
    complete_after_weeks: int,
    baseline: str | None = None,
    scores: pd.DataFrame | None = None,
    truth_vintage: str | None = None,
) -> dict[str, object]:
    """Collect what the explorer draws.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        Columns `model` (the label shown), `ours`, `reference_date`,
        `location`, `target_end_date` and the `QUANTILE_COLUMNS`.
    target : pandas.DataFrame
        Target rows. The truth vintage is drawn as the data.
    names : mapping
        Location names by id.
    complete_after_weeks : int
        Weeks before the truth vintage after which a week is complete.
    baseline : str, optional
        Model label that relative WIS divides by.
    scores : pandas.DataFrame, optional
        Complete-week scores with `model` (the label shown),
        `reference_date`, `location`, `target_end_date` and the
        `SCORE_COLUMNS`.
    truth_vintage : str, optional
        Defaults to the latest real vintage.

    Returns
    -------
    dict
        JSON-ready lists. Rows of `q` and `scores` index into `models`
        and `rounds`.
    """
    real = target.loc[target["kind"] == "real"]
    truth_v = truth_vintage or str(real["vintage"].max())
    provinces = set(real["location"]) - {NATIONAL}
    models = list(dict.fromkeys(quantiles["model"]))
    ours = set(quantiles.loc[quantiles["ours"], "model"])
    rounds = sorted(set(quantiles["reference_date"].astype(str)))
    locs = [
        loc
        for loc in dict.fromkeys(quantiles["location"])
        if loc in set(quantiles["location"])
    ]
    order = {"National": 0, "Province": 1, "Zone": 2}
    locs.sort(key=lambda x: (order[_level(x, provinces)], names.get(x, x)))
    latest = real.loc[real["vintage"] == truth_v]
    data = {
        loc: [
            [str(d), int(t)]
            for d, t in zip(
                grp["target_end_date"], grp["total"], strict=True
            )
        ]
        for loc, grp in latest.loc[latest["location"].isin(locs)]
        .sort_values("target_end_date")
        .groupby("location")
    }
    cut = pd.Timestamp(truth_v) - pd.Timedelta(weeks=complete_after_weeks)
    q = quantiles.assign(
        m=quantiles["model"].map(models.index),
        r=quantiles["reference_date"].astype(str).map(rounds.index),
    )
    rows = [
        [int(m), int(r), loc, str(d), *(round(float(v), 1) for v in vals)]
        for m, r, loc, d, *vals in q[
            ["m", "r", "location", "target_end_date", *QUANTILE_COLUMNS]
        ].itertuples(index=False)
    ]
    out: dict[str, object] = {
        "truth_vintage": truth_v,
        "complete_through": cut.date().isoformat(),
        "baseline": models.index(baseline) if baseline in models else None,
        "locations": [
            {
                "id": loc,
                "name": names.get(loc, loc),
                "level": _level(loc, provinces),
            }
            for loc in locs
        ],
        "rounds": rounds,
        "models": [{"label": m, "ours": m in ours} for m in models],
        "data": data,
        "q": rows,
        "score_columns": list(SCORE_COLUMNS),
        "scores": [],
    }
    if scores is not None and not scores.empty:
        s = scores.loc[scores["model"].isin(models)]
        s = s.assign(
            m=s["model"].map(models.index),
            r=s["reference_date"].astype(str).map(rounds.index),
        ).dropna(subset=["r"])
        out["scores"] = [
            [int(m), int(r), loc, str(d), *(round(float(v), 3) for v in vals)]
            for m, r, loc, d, *vals in s[
                ["m", "r", "location", "target_end_date", *SCORE_COLUMNS]
            ].itertuples(index=False)
        ]
    return out


def write_explorer(path: Path, data: Mapping[str, object]) -> None:
    """Write the explorer data as compact JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def explorer_html(src: str, title: str) -> str:
    """Return the page element the explorer script fills.

    Parameters
    ----------
    src : str
        URL of the JSON file, relative to the page.
    title : str
        Accessible name of the explorer.
    """
    return (
        f'<div class="explorer" data-src="{src}" role="group" '
        f'aria-label="{title}">\n'
        "<noscript>The explorer needs JavaScript.</noscript>\n"
        "</div>\n"
    )
