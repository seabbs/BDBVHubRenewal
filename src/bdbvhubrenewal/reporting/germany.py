# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The German nowcast benchmark page.

`germany_page` writes the page's table and score summaries from the
scores of `bdbvhubrenewal germany-backtest` (our models) and
`bdbvhubrenewal germany-teams` (the hub teams): `relative_scores` by level
and the `scoresummary` views, relative to `BASELINE`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from bdbvhubrenewal.data.germany import PLACE_NAMES, level
from bdbvhubrenewal.evaluation.germany import (
    BASELINE,
    TEAM_NAMES,
    GermanyConfig,
    relative_scores,
)
from bdbvhubrenewal.reporting.plots import model_style
from bdbvhubrenewal.reporting.scorechart import write_score_chart
from bdbvhubrenewal.reporting.scoresummary import summary_html, summary_rows

__all__ = ["LEVELS", "METRICS", "germany_page"]

LEVELS: Final = {
    "national": "National",
    "states": "States",
    "age": "Age groups",
}
"""The paper's levels, in order, and their labels."""

METRICS: Final = (
    ("wis", "WIS", None, True),
    ("ae_median", "Absolute error of the median", None, True),
    ("coverage_50", "50% coverage", 0.5, False),
    ("coverage_95", "95% coverage", 0.95, False),
)
"""The summaries' metrics, as `scoresummary.SUMMARY_METRICS`: the
paper's."""

_PAGES: Final = (
    "germany-scores.md",
    "germany-horizon.md",
    "germany-place.md",
    "germany-sources.md",
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def germany_page(
    fig: Path, gen: Path, results: Path, config: GermanyConfig
) -> None:
    """Write the German nowcast benchmark page's table and summaries.

    Parameters
    ----------
    fig : pathlib.Path
        Where the summary JSON goes.
    gen : pathlib.Path
        Where the Markdown snippets go.
    results : pathlib.Path
        `germany-backtest` output, `scores.csv.gz`, and `germany-teams`
        output, `teams.csv.gz`, if it has been run. If the first is
        missing, the page says the benchmark has not been run.
    config : GermanyConfig
        The benchmark's source, which the page credits.
    """
    url = f"https://github.com/{config.repo}/blob/{config.commit}"
    teams_url = (
        f"https://github.com/{config.teams_repo}/blob/{config.teams_commit}"
    )
    files = ", ".join(f"[`{p}`]({teams_url}/{p})" for p in config.submissions)
    _write(
        gen / "germany-sources.md",
        f"- The reporting triangle: [`{config.triangle}`]"
        f"({url}/{config.triangle}) in {config.repo} at "
        f"`{config.commit[:7]}`.\n"
        f"- The hub teams' nowcasts: {files} in {config.teams_repo} at "
        f"`{config.teams_commit[:7]}` (MIT).\n",
    )
    path = results / "scores.csv.gz"
    if not path.exists():
        for name in _PAGES[:-1]:
            _write(gen / name, "Not run yet: `task germany`.\n")
        return
    dtype = {"origin": str, "truth_vintage": str}
    frames = [pd.read_csv(path, dtype=dtype)]
    teams = results / "teams.csv.gz"
    if teams.exists():
        frames.append(pd.read_csv(teams, dtype=dtype))
    scores = pd.concat(frames, ignore_index=True)
    order = [*config.models, *TEAM_NAMES.values()]
    models = [m for m in order if m in set(scores["model"])]
    write_score_chart(
        fig / "scoresummary-germany-latest.json",
        summary_rows(
            scores,
            models,
            {m: model_style(m, models) for m in models},
            PLACE_NAMES,
            baseline=BASELINE,
            metrics=METRICS,
            pool_of=level,
            pool_labels=LEVELS,
        ),
    )
    srcs = {"latest": "../figures/scoresummary-germany-latest.json"}
    table = relative_scores(scores, ("pool",))
    table["pool"] = pd.Categorical(table["pool"], list(LEVELS))
    # The summary above gives the absolute scores, so the table fits the
    # page with the relative ones.
    table = table.sort_values(["pool", "rel_wis"]).drop(
        columns=["wis", "ae_median"]
    )
    _write(
        gen / "germany-scores.md",
        summary_html(srcs, "Overall scores by level")
        + "\n"
        + table.round(3).to_markdown(index=False)
        + "\n",
    )
    _write(
        gen / "germany-horizon.md",
        summary_html(srcs, "Scores by horizon", view="horizon"),
    )
    _write(
        gen / "germany-place.md",
        summary_html(srcs, "Scores by place", view="place"),
    )
