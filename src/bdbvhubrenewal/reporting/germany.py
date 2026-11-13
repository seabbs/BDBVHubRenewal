# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The German nowcast benchmark page.

`germany_page` writes the page's table and score summaries from the
scores of `bdbvhubrenewal germany-backtest` (our models) and
`bdbvhubrenewal germany-teams` (the hub teams): `relative_scores` by level
and the `scoresummary` views, relative to the configured baseline.
`germany_explorer` draws the nowcasts at every `plot_every`-th
development date in the forecast explorer, against the 40-day truth and
the data each date saw (`bdbvhubrenewal germany-asof`).
Only the development half, nowcast dates before the split date, is shown.
The configured hidden models start switched off.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from bdbvhubrenewal.data.germany import PLACE_NAMES, level
from bdbvhubrenewal.evaluation.germany import (
    TEAM_NAMES,
    GermanyConfig,
    nowcast_dates,
    relative_scores,
)
from bdbvhubrenewal.reporting.explorer import (
    QUANTILE_COLUMNS,
    explorer_data,
    explorer_html,
    write_explorer,
)
from bdbvhubrenewal.reporting.plots import (
    DASHES,
    MODEL_COLOURS,
    MODEL_STYLE,
    model_style,
)
from bdbvhubrenewal.reporting.scorechart import write_score_chart
from bdbvhubrenewal.reporting.scoresummary import summary_html, summary_rows

__all__ = [
    "LEVELS",
    "METRICS",
    "germany_explorer",
    "germany_page",
    "model_styles",
]

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

_BANDS: Final = dict(
    zip(QUANTILE_COLUMNS, (0.025, 0.25, 0.5, 0.75, 0.975), strict=True)
)
"""The explorer's interval columns and the hub's levels they show."""

_PAGES: Final = (
    "germany-nowcasts.md",
    "germany-scores.md",
    "germany-horizon.md",
    "germany-place.md",
    "germany-sources.md",
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def model_styles(
    models: Sequence[str], ours: Collection[str]
) -> dict[str, tuple[str, str, str]]:
    """Return each model's light colour, dark colour and dash.

    Our models keep their own (`model_style`). The hub teams take the
    colours ours leave free in turn, solid and then dashed, so no two
    models look the same.

    Parameters
    ----------
    models : sequence of str
        Every model shown, in order.
    ours : collection of str
        Our models.

    Returns
    -------
    dict
        Model to its light colour, dark colour and dash.
    """
    out = {m: model_style(m, models) for m in models if m in ours}
    used = {MODEL_STYLE[m][0] for m in out if m in MODEL_STYLE}
    free = [i for i in range(len(MODEL_COLOURS["light"])) if i not in used]
    dashes = list(DASHES.values())
    teams = [m for m in models if m not in ours]
    for k, m in enumerate(teams):
        slot = free[k % len(free)]
        out[m] = (
            MODEL_COLOURS["light"][slot],
            MODEL_COLOURS["dark"][slot],
            dashes[(k // len(free)) % len(dashes)],
        )
    return {m: out[m] for m in models}


def _read(results: Path, names: tuple[str, ...]) -> pd.DataFrame | None:
    """Stack the result files that exist, or None if none does."""
    dtype = {"origin": str, "truth_vintage": str, "target_end_date": str}
    frames = [
        pd.read_csv(results / n, dtype=dtype)
        for n in names
        if (results / n).exists()
    ]
    return pd.concat(frames, ignore_index=True) if frames else None


def germany_explorer(
    forecasts: pd.DataFrame,
    asof: pd.DataFrame | None,
    config: GermanyConfig,
    models: list[str],
) -> dict[str, object]:
    """Return the forecast explorer's data for the German nowcasts.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        `germany-backtest` and `germany-teams` forecasts, with truth.
    asof : pandas.DataFrame, optional
        `germany-asof` output.
    config : GermanyConfig
        Gives the dates, the plot spacing and the hidden models.
    models : list of str
        Models in order, which sets their colours (`model_styles`).

    Returns
    -------
    dict
        `explorer_data` output, at every `plot_every`-th development date.
    """
    rounds = set(nowcast_dates(config)[:: config.plot_every])
    rows = forecasts.loc[
        forecasts["origin"].isin(rounds) & forecasts["model"].isin(models)
    ]
    wide = (
        rows.assign(quantile=rows["quantile"].round(6))
        .pivot_table(
            index=["model", "origin", "location", "target_end_date"],
            columns="quantile",
            values="value",
            aggfunc="first",
        )[list(_BANDS.values())]
        .set_axis(list(_BANDS), axis=1)
        .reset_index()
    )
    styles = model_styles(models, config.models)
    shown = [m for m in models if m in set(wide["model"])]
    wide["model"] = pd.Categorical(wide["model"], shown)
    wide = wide.sort_values(["model", "origin", "location"])
    wide = wide.assign(
        model=wide["model"].astype(str),
        ours=wide["model"].isin(set(config.models)),
        reference_date=wide["origin"],
    )
    truth = forecasts[["location", "target_end_date", "truth"]].dropna()
    truth = truth.drop_duplicates(["location", "target_end_date"])
    target = truth.rename(columns={"truth": "total"}).assign(
        kind="real", vintage="truth"
    )
    seen = None
    if asof is not None:
        seen = asof.rename(columns={"origin": "reference_date"})
    return explorer_data(
        wide,
        target,
        PLACE_NAMES,
        0,
        {m: styles[m] for m in shown},
        truth_vintage="truth",
        hidden=config.hidden,
        level_of=lambda x: LEVELS[level(x)],
        levels=tuple(LEVELS.values()),
        outer="95%",
        unit="hospitalisations",
        complete_through=str(truth["target_end_date"].max()),
        data_label=f"reported {config.truth_delay} days on",
        asof=seen,
    )


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
        `germany-backtest` output (`scores.csv.gz`, `forecasts.csv.gz`),
        and if they have been run `germany-teams` output (`teams.csv.gz`,
        `teams-forecasts.csv.gz`) and `germany-asof` output
        (`asof.csv.gz`). If the scores are missing, the page says the
        benchmark has not been run.
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
    if not (results / "scores.csv.gz").exists():
        for name in _PAGES[:-1]:
            _write(gen / name, "Not run yet: `task germany`.\n")
        return
    scores = _read(results, ("scores.csv.gz", "teams.csv.gz"))
    assert scores is not None
    scores = scores.loc[scores["origin"] < config.split_date]
    forecasts = _read(results, ("forecasts.csv.gz", "teams-forecasts.csv.gz"))
    run = set(scores["model"])
    if forecasts is not None:
        run |= set(forecasts["model"])
    # One list for every view, so each model keeps its colour.
    models = [m for m in [*config.models, *TEAM_NAMES.values()] if m in run]
    write_score_chart(
        fig / "scoresummary-germany-latest.json",
        summary_rows(
            scores,
            models,
            model_styles(models, config.models),
            PLACE_NAMES,
            baseline=config.baseline,
            metrics=METRICS,
            default_on=[m for m in models if m not in config.hidden],
            pool_of=level,
            pool_labels=LEVELS,
        ),
    )
    srcs = {"latest": "../figures/scoresummary-germany-latest.json"}
    table = relative_scores(scores, ("pool",), config.baseline)
    table["pool"] = pd.Categorical(table["pool"], list(LEVELS))
    # The summary above gives the absolute scores, so the table fits the
    # page with the relative ones.
    table = table.sort_values(["pool", "rel_wis"]).drop(
        columns=["wis", "ae_median"]
    )
    _write(
        gen / "germany-scores.md",
        f"Nowcast dates before {config.split_date} only; the later half "
        "is held out for the paper.\n\n"
        + summary_html(srcs, "Overall scores by level")
        + "\n"
        + table.round(3).to_markdown(index=False)
        + "\n",
    )
    if forecasts is None:
        _write(gen / "germany-nowcasts.md", "Not run yet: `task germany`.\n")
    else:
        forecasts = forecasts.loc[forecasts["origin"] < config.split_date]
        write_explorer(
            fig / "explorer-germany.json",
            germany_explorer(
                forecasts, _read(results, ("asof.csv.gz",)), config, models
            ),
        )
        _write(
            gen / "germany-nowcasts.md",
            explorer_html(
                "../figures/explorer-germany.json",
                "German nowcasts",
                summary=srcs,
            ),
        )
    _write(
        gen / "germany-horizon.md",
        summary_html(srcs, "Scores by horizon", view="horizon"),
    )
    _write(
        gen / "germany-place.md",
        summary_html(srcs, "Scores by place", view="place"),
    )
