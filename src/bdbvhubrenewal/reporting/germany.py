# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The German nowcast benchmark page.

`germany_page` writes the page's table and score summaries from
`bdbvhubrenewal germany-backtest`'s scores, as the Backtest page shows its
own: `relative_wis` by level and the `scoresummary` views, relative to
`randomwalk`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from bdbvhubrenewal.data.germany import PLACE_NAMES, level
from bdbvhubrenewal.evaluation.germany import GermanyConfig
from bdbvhubrenewal.evaluation.nowcast import relative_wis
from bdbvhubrenewal.reporting.plots import model_style
from bdbvhubrenewal.reporting.scorechart import write_score_chart
from bdbvhubrenewal.reporting.scoresummary import summary_html, summary_rows

__all__ = ["LEVELS", "germany_page"]

LEVELS: Final = {
    "national": "National",
    "states": "States",
    "age": "Age groups",
}
"""The paper's levels, in order, and their labels."""

BASELINE: Final = "randomwalk"
"""The model relative scores divide by."""

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
        `germany-backtest` output, `scores.csv.gz`. If missing, the page
        says it has not been run.
    config : GermanyConfig
        The benchmark's source, which the page credits.
    """
    url = f"https://github.com/{config.repo}/blob/{config.commit}"
    _write(
        gen / "germany-sources.md",
        f"[`{config.triangle}`]({url}/{config.triangle}) in "
        f"{config.repo} at `{config.commit[:7]}`.\n",
    )
    path = results / "scores.csv.gz"
    if not path.exists():
        for name in _PAGES[:-1]:
            _write(gen / name, "Not run yet: `task germany`.\n")
        return
    scores = pd.read_csv(path, dtype={"origin": str, "truth_vintage": str})
    models = [m for m in config.models if m in set(scores["model"])]
    write_score_chart(
        fig / "scoresummary-germany-latest.json",
        summary_rows(
            scores,
            models,
            {m: model_style(m, models) for m in models},
            PLACE_NAMES,
            baseline=BASELINE,
            pool_of=level,
            pool_labels=LEVELS,
        ),
    )
    srcs = {"latest": "../figures/scoresummary-germany-latest.json"}
    table = relative_wis(scores, ("pool",), baseline=BASELINE)
    table["pool"] = pd.Categorical(table["pool"], list(LEVELS))
    table = table.sort_values(["pool", "rel_wis"])
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
