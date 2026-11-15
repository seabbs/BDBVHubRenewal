"""Figures and score tables for the main text and the SI, per study.

`FIGURES` names each study's figures. The `results` panel goes in the main
text and the rest in the SI. `score_tables` writes the relative scores
overall (main text) and by horizon and location (SI), split by half where
the study has a held-out split, and for the synthetic study the grid of
scenario factors.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from paperfigs.checkpoints import drop_checkpoints, drop_german_trained
from paperfigs.factors import factor_label
from paperfigs.joint import relative_joint_scores
from paperfigs.nowcasts import (
    PLACE_RULE,
    other_places,
    pick_origins,
    pick_outbreaks,
    pick_places,
    place_models,
)
from paperfigs.periods import add_period
from paperfigs.plots import (
    nowcast_panels,
    results_panel,
    si_place_dots,
    si_scores_by_factor,
    si_scores_by_finetune_size,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.scores import (
    STUDY_BASELINES,
    relative_scores,
    summarise_scores,
)
from paperfigs.style import theme
from paperpipe.meta import DUMMY_LABEL

__all__ = [
    "FIGURES",
    "JOINT_STUDIES",
    "MAIN_TABLES",
    "SI_TABLES",
    "STUDY_SI_TABLES",
    "figure_names",
    "joint_table",
    "real_finetune_table",
    "score_tables",
    "to_markdown",
    "write_figures",
]

FIGURES: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "synthetic": {
        "main": ("nowcasts", "results"),
        "si": (
            "si-nowcasts",
            "si-factors",
            "si-factors-log",
            "si-wis-over-time",
            "si-finetune-size",
        ),
    },
    "germany": {
        "main": ("nowcasts", "results"),
        "si": (
            "si-nowcasts",
            "si-places",
            "si-location-horizon",
            "si-location-horizon-log",
            "si-wis-over-time",
        ),
    },
    "ebola": {
        "main": ("nowcasts", "results"),
        "si": (
            "si-nowcasts",
            "si-places",
            "si-location-horizon",
            "si-location-horizon-log",
            "si-wis-over-time",
        ),
    },
}
"""Figure names of each study, for the main text and the SI."""

JOINT_STUDIES: Final = ("synthetic", "ebola")
"""Studies scored on sample trajectories, so also scored jointly."""

MAIN_TABLES: Final = ("scores",)
"""Main-text tables of every study; the synthetic study adds `grid`."""

SI_TABLES: Final = ("scores-by-horizon", "scores-by-location")

STUDY_SI_TABLES: Final[dict[str, tuple[str, ...]]] = {
    "germany": ("real-finetune",)
}
"""Further SI tables of a study: for Germany, Chronos-2 fine-tuned on the
German data against the simulation-trained model (`real_finetune_table`)."""

REAL_FINETUNE_MODELS: Final = tuple(
    f"fm_chronos_{source}_ragged{variant}"
    for source in ("daily", "weekly")
    for variant in ("", "_ft", "_de")
)
"""The Chronos-2 models of the German fine-tuning comparison: zero-shot,
fine-tuned on simulations and fine-tuned on the German data."""
"""SI tables of every study."""


def figure_names(study: str, part: str) -> tuple[str, ...]:
    """Return the figure names of `study` for `part`, `main` or `si`."""
    return FIGURES[study][part]


def _factors(locations: pd.DataFrame) -> pd.DataFrame:
    return locations.drop(columns="delay_days")


def _national(study: str, forecasts: pd.DataFrame, nowcasts: Mapping) -> str:
    """Return the national location, else the one with most forecast."""
    national = nowcasts.get("national", {}).get(study)
    if national in set(forecasts["location"]):
        return str(national)
    return str(forecasts.groupby("location")["value"].sum().idxmax())


def _place_rows(
    truth: pd.DataFrame,
    places: list[tuple[str, str, str]],
    origins: list[pd.Timestamp],
    nowcasts: Mapping,
) -> list[tuple[pd.Timestamp, str, str, str]]:
    """Return the SI nowcast figure's rows: places picked by the rule.

    Each is drawn at the last origin shown in the main text
    (`paperfigs.nowcasts.pick_places`).
    """
    origin = origins[-1]
    picked = pick_places(
        truth,
        places,
        origin,
        nowcasts.get("place_rule", PLACE_RULE),
        int(nowcasts["seed"]),
    )
    return [(origin, loc, st, title) for loc, st, title in picked]


def _figures(
    study: str,
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    scores: pd.DataFrame,
    locations: pd.DataFrame,
    split: pd.Timestamp | None,
    truth: pd.DataFrame,
    nowcasts: Mapping,
    baseline: str,
) -> dict[str, Figure]:
    seed = int(nowcasts["seed"])
    models = place_models(
        forecasts["model"].unique(), nowcasts["models"][study]
    )
    # Origins every shown model forecast from, where they share any.
    shown_fc = forecasts.loc[forecasts["model"].isin(models)]
    by_model = shown_fc.groupby("model")["origin_date"].agg(set)
    common = set.intersection(*by_model) if len(by_model) else set()
    pool = (
        shown_fc.loc[shown_fc["origin_date"].isin(common)]
        if common
        else shown_fc
    )
    origins = pick_origins(pool, int(nowcasts["origins"]), seed)
    shown = {"baseline": baseline, "nowcast": False}
    reference = REFERENCE if REFERENCE in set(scores["model"]) else baseline
    rel = (
        relative_scores(drop_checkpoints(scores), baseline=reference)
        .set_index("model")["rel_score"]
        .to_dict()
    )

    def by_origin(location: str, stratum: str = "all") -> Figure:
        rows = [(o, location, stratum, f"origin {o:%Y-%m-%d}") for o in origins]
        return nowcast_panels(forecasts, observations, truth, models, rows, rel)

    def by_place(places: list[tuple[str, str, str]]) -> Figure:
        rows = _place_rows(truth, places, origins, nowcasts)
        return nowcast_panels(forecasts, observations, truth, models, rows, rel)

    if study == "synthetic":
        # Outbreaks are many, so scores break down by scenario factor.
        # The fine-tune's checkpoints appear only in their own figure.
        factors = _factors(locations)
        main = drop_checkpoints(scores)
        panels_ = pick_outbreaks(
            locations, nowcasts["factors"], int(nowcasts["outbreaks"]), seed
        )
        return {
            "nowcasts": by_origin(panels_[0][0]),
            "results": results_panel(
                drop_checkpoints(forecasts),
                observations,
                main,
                factors=factors,
                **shown,
            ),
            "si-nowcasts": by_place(panels_),
            "si-factors": si_scores_by_factor(main, factors, baseline),
            "si-factors-log": si_scores_by_factor(
                main, factors, baseline, col="rel_log_score"
            ),
            "si-wis-over-time": si_wis_over_time(main),
            "si-finetune-size": si_scores_by_finetune_size(scores, baseline),
        }
    national = _national(study, forecasts, nowcasts)
    return {
        "nowcasts": by_origin(national),
        "results": results_panel(
            forecasts,
            observations,
            scores,
            split=split,
            **shown,
        ),
        "si-nowcasts": by_place(other_places(forecasts, national)),
        "si-places": si_place_dots(scores, models, baseline),
        "si-location-horizon": si_scores_by_location_horizon(
            scores, baseline, split=split
        ),
        "si-location-horizon-log": si_scores_by_location_horizon(
            scores, baseline, col="rel_log_score", split=split
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
    *,
    truth: pd.DataFrame,
    nowcasts: Mapping,
    baseline: str,
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
    truth : pandas.DataFrame
        The rolling truth, drawn in the nowcast panels.
    nowcasts : mapping
        Which series, origins and models the nowcast panels show: the
        `nowcasts` block of the workflow config.
    baseline : str
        The model scores are relative to.

    Returns
    -------
    list of pathlib.Path
        The figures written.
    """
    dirs = {"main": main_dir, "si": si_dir}
    written = []
    # The German-trained models appear in their own SI table only.
    forecasts = drop_german_trained(forecasts)
    scores = drop_german_trained(scores)
    with theme():
        figs = _figures(
            study,
            forecasts,
            observations,
            scores,
            locations,
            split,
            truth,
            nowcasts,
            baseline,
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


REFERENCE: Final = "randomwalknowcast"
"""A second reference for the score tables when the baseline is another
model, such as a hub's point-forecast baseline."""


def _table(scores: pd.DataFrame, by: list[str], baseline: str) -> pd.DataFrame:
    keys = ["study", "model", *by]
    rel = relative_scores(scores, by=by, baseline=baseline).drop(columns="n")
    out = summarise_scores(scores, by=by).merge(rel, on=keys, how="left")
    if baseline != REFERENCE and REFERENCE in set(scores["model"]):
        ref = relative_scores(scores, by=by, baseline=REFERENCE)
        ref = ref.drop(columns="n").rename(
            columns={
                "rel_score": f"rel_score_{REFERENCE}",
                "rel_log_score": f"rel_log_score_{REFERENCE}",
            }
        )
        out = out.merge(ref, on=keys, how="left")
    return out


def _grid(locations: pd.DataFrame) -> pd.DataFrame:
    """Each grid factor's levels and their shares of the outbreaks.

    The factors are the non-numeric columns: each simulation's cell.
    """
    cells = _factors(locations).set_index("location")
    names = [
        c for c in cells.columns if not pd.api.types.is_numeric_dtype(cells[c])
    ]
    parts = [
        cells[name]
        .value_counts(normalize=True)
        .rename_axis("level")
        .reset_index(name="share")
        .assign(factor=factor_label(name))
        for name in names
    ]
    return pd.concat(parts, ignore_index=True)[["factor", "level", "share"]]


def _level(scores: pd.DataFrame) -> pd.Series:
    """Return the German level of each score row."""
    national = scores["location"] == "DE"
    ages = scores["stratum"].fillna("all") != "all"
    return pd.Series(
        [
            "age groups" if a else "national" if n else "states"
            for n, a in zip(national, ages, strict=True)
        ],
        index=scores.index,
    )


def real_finetune_table(
    scores: pd.DataFrame, split: pd.Timestamp | None
) -> pd.DataFrame:
    """Compare Chronos-2 fine-tuned on German data with the other models.

    The German-trained models saw the development half, so only the
    held-out half is scored: each model's mean scores and coverage
    overall (`level` `all`) and by level (national, states, age groups),
    relative to `REFERENCE`.

    Parameters
    ----------
    scores : pandas.DataFrame
        The German scores, every model.
    split : pandas.Timestamp or None
        First held-out origin date. Without one there is no held-out half
        and the table is empty.

    Returns
    -------
    pandas.DataFrame
    """
    keep = scores["model"].isin([REFERENCE, *REAL_FINETUNE_MODELS])
    rows = scores.loc[keep]
    if split is None:
        rows = rows.iloc[0:0]
    else:
        rows = rows.loc[rows["origin_date"] >= pd.Timestamp(split)]
    if rows.empty or REFERENCE not in set(rows["model"]):
        return pd.DataFrame(columns=["study", "model", "level"])
    rows = rows.assign(level=_level(rows))
    return pd.concat(
        [
            _table(rows, [], REFERENCE).assign(level="all"),
            _table(rows, ["level"], REFERENCE),
        ],
        ignore_index=True,
    )


def score_tables(
    study: str,
    scores: pd.DataFrame,
    locations: pd.DataFrame,
    main_dir: Path,
    si_dir: Path,
    split: pd.Timestamp | None = None,
    baseline: str | None = None,
) -> list[Path]:
    """Write a study's score tables as CSV and return the paths.

    Each table has the mean scores, coverage and number of forecasts of
    each model, with `rel_score` and `rel_log_score` relative to the
    study's baseline, and to `REFERENCE` as well when the baseline is
    another model. With a `split`, every table is split by `period`.

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
    baseline : str, optional
        The model scores are relative to. Defaults to the study's entry in
        `STUDY_BASELINES`.

    Returns
    -------
    list of pathlib.Path
        The tables written.
    """
    baseline = baseline or STUDY_BASELINES[study]
    every = scores
    scores = add_period(drop_checkpoints(drop_german_trained(scores)), split)
    period = ["period"] if split is not None else []
    tables = {
        main_dir / f"{study}-scores.csv": _table(scores, period, baseline),
        si_dir / f"{study}-scores-by-horizon.csv": _table(
            scores, [*period, "horizon"], baseline
        ),
        si_dir / f"{study}-scores-by-location.csv": _table(
            scores, [*period, "location", "stratum"], baseline
        ),
    }
    if study == "synthetic":
        tables[main_dir / f"{study}-grid.csv"] = _grid(locations)
    if study == "germany":
        tables[si_dir / "germany-real-finetune.csv"] = real_finetune_table(
            every, split
        )
    for path, table in tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        text = to_markdown(table)
        if table.empty:
            text = "*No held-out forecasts in this run.*\n"
        path.with_suffix(".md").write_text(text)
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


def joint_table(
    scores: pd.DataFrame, path: Path, baseline: str | None = None
) -> Path:
    """Write the mean joint scores relative to the baseline, as Markdown.

    Parameters
    ----------
    scores : pandas.DataFrame
        `paperfigs.joint.joint_scores` output for one study.
    path : pathlib.Path
        The table, `<study>-joint.md`.

    Returns
    -------
    pathlib.Path
        The table written.
    """
    if scores.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("*No sample trajectories to score jointly.*\n")
        return path
    table = relative_joint_scores(scores, baseline=baseline)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(table))
    return path
