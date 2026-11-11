# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The German nowcast benchmark page.

`germany_page` writes the page's tables and the score summaries
(`scoresummary`) from `bdbvhubrenewal germany-score`'s results: our models
in their colours and the hub teams in greys, relative to `randomwalk`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pandas as pd

from bdbvhubrenewal.data.germany import ALL_AGES, PLACE_NAMES
from bdbvhubrenewal.evaluation.germany import summarise_germany
from bdbvhubrenewal.forecasters import list_models
from bdbvhubrenewal.reporting.plots import comparison_colours
from bdbvhubrenewal.reporting.scorechart import write_score_chart
from bdbvhubrenewal.reporting.scoresummary import summary_html, summary_rows

__all__ = ["LEVELS", "germany_page"]

LEVELS: Final = {
    "national": "National",
    "states": "States",
    "age": "Age groups",
}
"""The paper's levels, in order, and their labels."""

METRICS: Final = (("wis", "WIS", None, True),)
"""Score column, label, reference line and whether it is also relative, of
the summaries. Coverage is in the table only, as some teams' forecasts
have none (`team_scores`)."""

BASELINE: Final = "randomwalk"
"""The model relative scores divide by."""

TRUTH_DATE: Final = "2022-08-08"
"""The data version the paper scores against."""

_PAGES: Final = (
    "germany-scores.md",
    "germany-horizon.md",
    "germany-place.md",
    "germany-sources.md",
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _place(scores: pd.DataFrame) -> pd.Series:
    """Return each row's place: its age group, or else its location."""
    return scores["location"].where(
        scores["age_group"] == ALL_AGES, scores["age_group"]
    )


def _table(scores: pd.DataFrame, models: list[str]) -> str:
    """Return mean WIS and coverage by model and level, as Markdown."""
    summary = summarise_germany(scores)
    wide = summary.pivot(
        index="model", columns="level", values=["wis", "coverage_50"]
    )
    cov95 = summary.pivot(index="model", columns="level", values="coverage_95")
    cols = {}
    for level, label in LEVELS.items():
        if level not in cov95:
            continue
        cols[f"{label} WIS"] = wide[("wis", level)].round(1)
        cols[f"{label} 50%"] = wide[("coverage_50", level)].round(2)
        cols[f"{label} 95%"] = cov95[level].round(2)
    table = pd.DataFrame(cols).reindex(models)
    table = table.sort_values(next(iter(cols)))
    table.index.name = "Model"
    short = summary.loc[summary["n_coverage"] < summary["n"]]
    notes = [
        f"{r.model}'s coverage at the {LEVELS[str(r.level)].lower()} level is "
        f"over the {r.n_coverage} of its {r.n} forecasts with published "
        "quantiles.\n"
        for r in short.itertuples(index=False)
    ]
    return "\n".join([table.to_markdown() + "\n", *notes])


def _sources(sources: Mapping[str, object]) -> str:
    files = sources["files"]
    assert isinstance(files, list)
    lines = [
        f"- [`{f['path']}`](https://github.com/{f['repo']}/blob/"
        f"{f['commit']}/{f['path']}) in {f['repo']} at `{f['commit'][:7]}`"
        for f in files
    ]
    return "\n".join(lines) + "\n"


def germany_page(fig: Path, gen: Path, results: Path) -> None:
    """Write the German nowcast benchmark page's tables and summaries.

    Parameters
    ----------
    fig : pathlib.Path
        Where the summary JSON goes.
    gen : pathlib.Path
        Where the Markdown snippets go.
    results : pathlib.Path
        `germany-score` output: `scores.csv.gz` and `sources.json`. If
        missing, the page says it has not been run.
    """
    path = results / "scores.csv.gz"
    if not path.exists():
        for name in _PAGES:
            _write(gen / name, "Not run yet: `task germany`.\n")
        return
    scores = pd.read_csv(path)
    registry = [s.name for s in list_models()]
    ours = [m for m in registry if m in set(scores["model"])]
    teams = sorted(set(scores.loc[scores["source"] == "hub", "model"]))
    models = [*ours, *teams]
    colours = comparison_colours({m: m for m in ours}, {}, teams)
    rows = scores.assign(
        origin=scores["forecast_date"],
        truth_vintage=TRUTH_DATE,
        location=_place(scores),
        wis=scores["wis"].round(2),
    )
    pools = dict(zip(rows["location"], rows["level"], strict=True))
    write_score_chart(
        fig / "scoresummary-germany-latest.json",
        summary_rows(
            rows,
            models,
            colours,
            PLACE_NAMES,
            baseline=BASELINE,
            metrics=METRICS,
            pool_of=pools.__getitem__,
            pool_labels=LEVELS,
        ),
    )
    srcs = {"latest": "../figures/scoresummary-germany-latest.json"}
    _write(
        gen / "germany-scores.md",
        _table(scores, models)
        + "\n"
        + summary_html(srcs, "Overall scores by level"),
    )
    _write(
        gen / "germany-horizon.md",
        summary_html(srcs, "Scores by horizon", view="horizon"),
    )
    _write(
        gen / "germany-place.md",
        summary_html(srcs, "Scores by place", view="place"),
    )
    _write(
        gen / "germany-sources.md",
        _sources(json.loads((results / "sources.json").read_text())),
    )
