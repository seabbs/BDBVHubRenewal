"""Data for the interactive forecast explorer on the docs site.

The explorer is a small script (`docs/javascripts/explorer.js`) that draws
SVG from a JSON file, so it works in a static site and follows the theme.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.forecasters import NATIONAL

__all__ = ["explorer_data", "explorer_html", "write_explorer"]

QUANTILE_COLUMNS = ("q05", "q25", "q50", "q75", "q95")
"""Interval columns of the forecast quantiles."""


def _level(location: str, provinces: set[str]) -> str:
    if location == NATIONAL:
        return "National"
    return "Province" if location in provinces else "Zone"


def explorer_data(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    names: Mapping[str, str],
    complete_after_weeks: int,
    colours: Mapping[str, tuple[str, str]],
    truth_vintage: str | None = None,
    hidden: Collection[str] = (),
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
    colours : mapping of str to (str, str)
        Model label to its light and dark colour, as in the static
        figures.
    truth_vintage : str, optional
        Defaults to the latest real vintage.
    hidden : collection of str
        Model labels switched off until the reader switches them on.

    Returns
    -------
    dict
        JSON-ready lists. Rows of `q` index into `models` and `rounds`.
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
            for d, t in zip(grp["target_end_date"], grp["total"], strict=True)
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
        "locations": [
            {
                "id": loc,
                "name": names.get(loc, loc),
                "level": _level(loc, provinces),
            }
            for loc in locs
        ],
        "rounds": rounds,
        "models": [
            {
                "name": m,
                "ours": m in ours,
                "on": m not in hidden,
                "light": colours[m][0],
                "dark": colours[m][1],
            }
            for m in models
        ],
        "data": data,
        "q": rows,
    }
    return out


def write_explorer(path: Path, data: Mapping[str, object]) -> None:
    """Write the explorer data as compact JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def explorer_html(
    src: str, title: str, all_rounds: bool = False, summary: str | None = None
) -> str:
    """Return the page element the explorer script fills.

    Parameters
    ----------
    src : str
        URL of the JSON file, relative to the page.
    title : str
        Accessible name of the explorer.
    all_rounds : bool
        Start with all rounds pooled, rather than the latest round.
    summary : str, optional
        URL of the page's `summary_rows` JSON, whose rounds are the
        explorer's. The table summarises it for the selection.
    """
    start = ' data-round="all"' if all_rounds else ""
    if summary is not None:
        start += f' data-summary="{summary}"'
    return (
        f'<div class="explorer" data-src="{src}"{start} role="group" '
        f'aria-label="{title}">\n'
        "<noscript>The explorer needs JavaScript.</noscript>\n"
        "</div>\n"
    )
