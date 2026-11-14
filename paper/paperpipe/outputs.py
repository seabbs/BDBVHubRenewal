"""Figures and score tables for the main text and the SI, per study.

`FIGURES` names each study's figures. The `results` panel goes in the main
text and the rest in the SI. `score_tables` writes the relative scores
overall (main text) and by horizon and location (SI), split by half where
the study has a held-out split, and for the synthetic study the grid of
scenario factors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from paperfigs.checkpoints import drop_checkpoints
from paperfigs.periods import add_period
from paperfigs.plots import (
    results_panel,
    si_scores_by_factor,
    si_scores_by_finetune_size,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.scores import relative_scores, summarise_scores
from paperfigs.style import theme
from paperpipe.meta import DUMMY_LABEL

__all__ = [
    "FIGURES",
    "MAIN_TABLES",
    "SI_TABLES",
    "figure_names",
    "score_tables",
    "to_markdown",
    "write_figures",
]

FIGURES: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "synthetic": {
        "main": ("results",),
        "si": (
            "si-factors",
            "si-factors-log",
            "si-wis-over-time",
            "si-finetune-size",
        ),
    },
    "germany": {
        "main": ("results",),
        "si": (
            "si-location-horizon",
            "si-location-horizon-log",
            "si-wis-over-time",
        ),
    },
    "ebola": {
        "main": ("results",),
        "si": (
            "si-location-horizon",
            "si-location-horizon-log",
            "si-wis-over-time",
        ),
    },
}
"""Figure names of each study, for the main text and the SI."""

MAIN_TABLES: Final = ("scores",)
"""Main-text tables of every study; the synthetic study adds `grid`."""

SI_TABLES: Final = ("scores-by-horizon", "scores-by-location")
"""SI tables of every study."""


def figure_names(study: str, part: str) -> tuple[str, ...]:
    """Return the figure names of `study` for `part`, `main` or `si`."""
    return FIGURES[study][part]


def _factors(locations: pd.DataFrame) -> pd.DataFrame:
    return locations.drop(columns="delay_days")


def _figures(
    study: str,
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    scores: pd.DataFrame,
    locations: pd.DataFrame,
    split: pd.Timestamp | None,
) -> dict[str, Figure]:
    if study == "synthetic":
        # Outbreaks are many, so scores break down by scenario factor.
        # The fine-tune's checkpoints appear only in their own figure.
        factors = _factors(locations)
        main = drop_checkpoints(scores)
        return {
            "results": results_panel(
                drop_checkpoints(forecasts),
                observations,
                main,
                factors=factors,
            ),
            "si-factors": si_scores_by_factor(main, factors),
            "si-factors-log": si_scores_by_factor(
                main, factors, col="rel_log_score"
            ),
            "si-wis-over-time": si_wis_over_time(main),
            "si-finetune-size": si_scores_by_finetune_size(scores),
        }
    return {
        "results": results_panel(forecasts, observations, scores, split=split),
        "si-location-horizon": si_scores_by_location_horizon(
            scores, split=split
        ),
        "si-location-horizon-log": si_scores_by_location_horizon(
            scores, col="rel_log_score", split=split
        ),
        "si-wis-over-time": si_wis_over_time(scores, split=split),
    }


def write_figures(
    study: str,
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    scores: pd.DataFrame,
    locations: pd.DataFrame,
    main_dir: Path,
    si_dir: Path,
    split: pd.Timestamp | None = None,
    dummy: bool = False,
) -> list[Path]:
    """Write a study's figures as `<study>-<name>.png` and return the paths.

    Parameters
    ----------
    study : str
        A key of `FIGURES`.
    forecasts, observations, scores : pandas.DataFrame
        The study's forecasts, observations and scores.
    locations : pandas.DataFrame
        The per-location table from `paperpipe.studies.load_data`.
    main_dir, si_dir : pathlib.Path
        Folders for the main-text and SI figures.
    split : pandas.Timestamp, optional
        First held-out origin date, marked on the figures.
    dummy : bool
        Whether the study ran on dummy data, which labels every figure.

    Returns
    -------
    list of pathlib.Path
        The figures written.
    """
    dirs = {"main": main_dir, "si": si_dir}
    written = []
    with theme():
        figs = _figures(
            study, forecasts, observations, scores, locations, split
        )
        for part, folder in dirs.items():
            folder.mkdir(parents=True, exist_ok=True)
            for name in figure_names(study, part):
                path = folder / f"{study}-{name}.png"
                if dummy:
                    _mark_dummy(figs[name])
                figs[name].savefig(path, dpi=200)
                plt.close(figs[name])
                written.append(path)
    return written


def _mark_dummy(fig: Figure) -> None:
    fig.suptitle(DUMMY_LABEL, color="#b3261e", fontweight="bold")


def _table(scores: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rel = relative_scores(scores, by=by).drop(columns="n")
    mean = summarise_scores(scores, by=by)
    return mean.merge(rel, on=["study", "model", *by], how="left")


def _grid(locations: pd.DataFrame) -> pd.DataFrame:
    factors = _factors(locations).set_index("location")
    return pd.DataFrame(
        {
            "factor": factors.columns,
            "min": factors.min().to_numpy(),
            "median": factors.median().to_numpy(),
            "max": factors.max().to_numpy(),
            "n_locations": len(factors),
        }
    )


def score_tables(
    study: str,
    scores: pd.DataFrame,
    locations: pd.DataFrame,
    main_dir: Path,
    si_dir: Path,
    split: pd.Timestamp | None = None,
) -> list[Path]:
    """Write a study's score tables as CSV and return the paths.

    Each table has the mean scores, coverage and number of forecasts of
    each model, with `rel_score` and `rel_log_score` relative to the
    study's baseline. With a `split`, every table is split by `period`.

    Parameters
    ----------
    study : str
        `synthetic`, `germany` or `ebola`.
    scores : pandas.DataFrame
        The study's scores.
    locations : pandas.DataFrame
        The per-location table, for the synthetic grid.
    main_dir, si_dir : pathlib.Path
        Folders for the main-text and SI tables.
    split : pandas.Timestamp, optional
        First held-out origin date.

    Returns
    -------
    list of pathlib.Path
        The tables written.
    """
    scores = add_period(drop_checkpoints(scores), split)
    period = ["period"] if split is not None else []
    tables = {
        main_dir / f"{study}-scores.csv": _table(scores, period),
        si_dir / f"{study}-scores-by-horizon.csv": _table(
            scores, [*period, "horizon"]
        ),
        si_dir / f"{study}-scores-by-location.csv": _table(
            scores, [*period, "location", "stratum"]
        ),
    }
    if study == "synthetic":
        tables[main_dir / f"{study}-grid.csv"] = _grid(locations)
    for path, table in tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        path.with_suffix(".md").write_text(to_markdown(table))
    return list(tables)


def to_markdown(table: pd.DataFrame, digits: int = 3) -> str:
    """Return a table as a Markdown pipe table, numbers to `digits`."""

    def cell(v: object) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{digits}g}"
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    rows = [
        "| " + " | ".join(cell(v) for v in r) + " |"
        for r in table.itertuples(index=False)
    ]
    return "\n".join([head, rule, *rows]) + "\n"
