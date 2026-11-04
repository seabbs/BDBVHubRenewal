"""Data for the interactive forecast explorer on the docs site.

The explorer is a small script (`docs/javascripts/explorer.js`) that draws
SVG from a JSON file, so it works in a static site and follows the theme.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.reporting.legend import starts_on

__all__ = [
    "TRAJECTORY_DRAWS",
    "compact",
    "draw_subset",
    "explorer_data",
    "explorer_html",
    "trajectories",
    "write_explorer",
]

QUANTILE_COLUMNS = ("q05", "q25", "q50", "q75", "q95")
"""Interval columns of the forecast quantiles."""


TRAJECTORY_DRAWS = 50
"""Sample trajectories embedded per model, round and place."""


def compact(value: float) -> float | int:
    """Round a sample value to 0.1, as a whole number where it is one."""
    v = round(float(value), 1)
    return int(v) if v.is_integer() else v


def draw_subset(
    samples: pd.DataFrame, n: int = TRAJECTORY_DRAWS
) -> pd.DataFrame:
    """Keep each model's `n` lowest sample ids, in every round and place.

    Where a model's draws are joint across places (ours), a kept draw
    stays joint. Hub draws are joint across weeks within a place only.

    Parameters
    ----------
    samples : pandas.DataFrame
        Samples with `model` and `sample`.
    n : int
        Draws kept per model.

    Returns
    -------
    pandas.DataFrame
        The rows of the kept draws, with `draw`, 0 to n - 1.
    """
    ids = samples.groupby("model")["sample"].apply(lambda x: sorted(set(x))[:n])
    rank = {(m, s): k for m, keep in ids.items() for k, s in enumerate(keep)}
    draw = [
        rank.get((m, s), -1)
        for m, s in zip(samples["model"], samples["sample"], strict=True)
    ]
    out = samples.assign(draw=draw)
    return out.loc[out["draw"] >= 0]


def trajectories(
    samples: pd.DataFrame, n: int = TRAJECTORY_DRAWS
) -> pd.DataFrame:
    """Keep a fixed subset of each model's sample trajectories.

    Each model keeps its `n` lowest sample ids, the same ids in every
    round and place, so where a model's draws are joint across places
    (ours) a kept draw stays joint. Hub draws are joint across weeks
    within a place only, and are not linked across places here either.

    Parameters
    ----------
    samples : pandas.DataFrame
        Columns `model`, `reference_date`, `location`, `target_end_date`,
        `sample` and `value`.
    n : int
        Draws kept per model.

    Returns
    -------
    pandas.DataFrame
        One row per model, round, place and kept draw, with `draw` (0 to
        n - 1), `start` (the first week) and `values`, the draw's weekly
        values in week order.
    """
    keep = draw_subset(samples, n)
    keep = keep.assign(
        reference_date=keep["reference_date"].astype(str),
        target_end_date=keep["target_end_date"].astype(str),
    ).sort_values("target_end_date")
    keys = ["model", "reference_date", "location", "draw"]
    return (
        keep.groupby(keys)
        .agg(start=("target_end_date", "first"), values=("value", list))
        .reset_index()
    )


def _level(location: str) -> str:
    """National, Province or Zone, from the depth of the hub code."""
    return ("National", "Province", "Zone")[location.count("-")]


def explorer_data(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    names: Mapping[str, str],
    complete_after_weeks: int,
    colours: Mapping[str, tuple[str, str]],
    truth_vintage: str | None = None,
    hidden: Collection[str] = (),
    samples: pd.DataFrame | None = None,
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
    samples : pandas.DataFrame, optional
        Sample draws with `model` (the label shown), as `trajectories`
        takes them. Models without samples are drawn as bands only.

    Returns
    -------
    dict
        JSON-ready lists. Rows of `q` and `traj` (model, round, place,
        draw, first week, then the weekly values) index into `models` and
        `rounds`; each model says whether it has trajectories.
    """
    real = target.loc[target["kind"] == "real"]
    truth_v = truth_vintage or str(real["vintage"].max())
    models = list(dict.fromkeys(quantiles["model"]))
    ours = set(quantiles.loc[quantiles["ours"], "model"])
    rounds = sorted(set(quantiles["reference_date"].astype(str)))
    locs = [
        loc
        for loc in dict.fromkeys(quantiles["location"])
        if loc in set(quantiles["location"])
    ]
    order = {"National": 0, "Province": 1, "Zone": 2}
    locs.sort(key=lambda x: (order[_level(x)], names.get(x, x)))
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
    traj: list[list[object]] = []
    if samples is not None:
        t = trajectories(samples.loc[samples["model"].isin(models)])
        t = t.loc[t["reference_date"].isin(rounds) & t["location"].isin(locs)]
        traj = [
            [
                models.index(m),
                rounds.index(r),
                loc,
                int(k),
                start,
                *(compact(v) for v in vals),
            ]
            for m, r, loc, k, start, vals in t.itertuples(index=False)
        ]
    have = {r[0] for r in traj}
    on = set(models) - set(hidden)
    return {
        "truth_vintage": truth_v,
        "complete_through": cut.date().isoformat(),
        "locations": [
            {
                "id": loc,
                "name": names.get(loc, loc),
                "level": _level(loc),
            }
            for loc in locs
        ],
        "rounds": rounds,
        "models": [
            {
                "name": m,
                "ours": m in ours,
                "on": starts_on(m, on),
                "light": colours[m][0],
                "dark": colours[m][1],
                "trajectories": i in have,
            }
            for i, m in enumerate(models)
        ],
        "data": data,
        "q": rows,
        "traj": traj,
    }


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
