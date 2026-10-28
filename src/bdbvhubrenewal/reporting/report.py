# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Build the docs figures and tables from the data and backtest results.

`make_report` writes light and dark SVGs to `docs/figures/` and Markdown
tables to `docs/generated/`, which the docs pages include.
Nothing it writes is committed.
Models come from the registry, so a newly registered model appears on the
Backtest and Nowcast pages without changes here.
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bdbvhubrenewal.data import (
    load_delays,
    load_target,
    national_coherence,
    newly_missing,
)
from bdbvhubrenewal.data.completeness import reported_share
from bdbvhubrenewal.data.dates import reference_date_for_vintage
from bdbvhubrenewal.data.delays import delay_summary
from bdbvhubrenewal.data.reconstruct import (
    compare_reconstruction,
    reconstruct_vintages,
)
from bdbvhubrenewal.data.simulate import simulate_outbreak
from bdbvhubrenewal.data.tidy import COHERENCE_STREAMS
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    load_config,
    make_tasks,
    run_forecasts,
    truth_changes,
)
from bdbvhubrenewal.evaluation.heldout import load_model_choices
from bdbvhubrenewal.evaluation.hubcompare import (
    attribution,
    summarise_comparison,
)
from bdbvhubrenewal.evaluation.scoring import relative_skill, summarise_scores
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastTask,
    get_model,
    list_models,
)
from bdbvhubrenewal.reporting.plots import (
    comparison_label,
    plot_delay_distribution,
    plot_delays_over_time,
    plot_forecasts,
    plot_hub_round,
    plot_imputed_share,
    plot_newly_missing,
    plot_nowcast,
    plot_reconstruction,
    plot_relative_skill,
    plot_revisions,
    plot_scenarios,
    plot_scores,
    plot_scores_by_truth,
    plot_simulation,
    plot_target,
    save_light_dark,
)

__all__ = ["make_report"]


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.round(digits).to_markdown(index=False) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_report(
    figures: Path | str = "docs/figures",
    generated: Path | str = "docs/generated",
    data_dir: Path | str = "data",
    results: Path | str = "results",
    output: Path | str = "results",
    config_path: Path | str = "config/backtest.toml",
    hub_comparison: Path | str = "results/hub-comparison",
) -> None:
    """Write every docs figure and table.

    Parameters
    ----------
    figures : pathlib.Path or str
        Where SVGs go.
    generated : pathlib.Path or str
        Where Markdown tables go.
    data_dir : pathlib.Path or str
        The tidy data.
    results : pathlib.Path or str
        Where the backtest score tables are.
    output : pathlib.Path or str
        Where the backtest forecasts are.
    config_path : pathlib.Path or str
        Backtest settings.
    hub_comparison : pathlib.Path or str
        Output of `bdbvhubrenewal compare-hub`, if it has been run.
    """
    fig, gen = Path(figures), Path(generated)
    res, out = Path(results), Path(output)
    config = load_config(config_path)
    target = load_target(data_dir)
    delays = load_delays(data_dir)
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    latest_delays = delays.loc[delays["vintage"] == delays["vintage"].max()]

    # Data page.
    save_light_dark(
        lambda mode: plot_target(real, mode=mode), fig / "target.svg"
    )
    save_light_dark(
        lambda mode: plot_revisions(real, mode=mode), fig / "revisions.svg"
    )
    save_light_dark(
        lambda mode: plot_imputed_share(real, mode=mode),
        fig / "imputed-share.svg",
    )
    save_light_dark(
        lambda mode: plot_newly_missing(real, mode=mode),
        fig / "newly-missing.svg",
    )
    save_light_dark(
        lambda mode: plot_delay_distribution(latest_delays, mode=mode),
        fig / "delay-distribution.svg",
    )
    save_light_dark(
        lambda mode: plot_delays_over_time(delays, mode=mode),
        fig / "delays-time.svg",
    )
    real_dates = sorted(real["vintage"].unique())[:-1]
    rec = reconstruct_vintages(real, delays, real_dates)
    comparison = compare_reconstruction(real, rec)
    save_light_dark(
        lambda mode: plot_reconstruction(comparison, mode=mode),
        fig / "reconstruction.svg",
    )
    recent = comparison.assign(
        rank=comparison.groupby(["vintage", "location"])[
            "target_end_date"
        ].rank(ascending=False)
    )
    recent = recent.loc[
        (recent["rank"] <= 4) & (recent["location"] == NATIONAL)
    ]
    _write(
        gen / "reconstruction.md",
        _table(
            recent[
                [
                    "vintage",
                    "target_end_date",
                    "total_real",
                    "total_reconstructed",
                    "total_difference",
                ]
            ]
        ),
    )
    summary = real.groupby("vintage").agg(
        weeks=("target_end_date", "nunique"),
        first_week=("target_end_date", "min"),
        last_week=("target_end_date", "max"),
    )
    nat = (
        real.loc[real["location"] == NATIONAL]
        .groupby("vintage")[["observed", "imputed", "total"]]
        .sum()
    )
    prov = real.loc[(real["location"] != NATIONAL) & (real["total"] > 0)]
    summary["provinces_with_cases"] = prov.groupby("vintage")[
        "location"
    ].nunique()
    summary = summary.join(nat).reset_index()
    summary["delay_rows"] = summary["vintage"].map(
        delays.groupby("vintage").size()
    )
    _write(gen / "vintages.md", _table(summary))
    coherence = national_coherence(real)
    by_vintage = (
        coherence.assign(with_gap=coherence["gap"] != 0)
        .pivot_table(
            index="vintage",
            columns="stream",
            values="with_gap",
            aggfunc="sum",
        )[list(COHERENCE_STREAMS)]
        .rename(columns=lambda s: f"{s} weeks with a gap")
        .reset_index()
    )
    _write(gen / "national-coherence.md", _table(by_vintage))
    dsum = delay_summary(latest_delays, by=("location_name",))
    _write(gen / "delays.md", _table(dsum.sort_values("n", ascending=False)))
    new = newly_missing(real)
    new = new.loc[new["location"] == NATIONAL]
    _write(gen / "newly-missing.md", _table(new.drop(columns=["kind"])))
    outbreak = simulate_outbreak()
    save_light_dark(
        lambda mode: plot_simulation(outbreak.parameters, mode=mode),
        fig / "simulation.svg",
    )

    # Backtest, held-out and comparison pages.
    names = [s.name for s in list_models()]
    _backtest_page(fig, gen, res, out, config, target, delays, names)
    _heldout_page(fig, gen, res, target, names, config)
    _comparison_page(fig, gen, Path(hub_comparison), target)

    # Nowcast page: latest real vintage, every registered model.
    rows = real.loc[real["vintage"] == latest]
    tasks = make_tasks(rows, delays, config, history=real)
    forecasts_now = _nowcast_forecasts(tasks, latest, res)
    shown = [n for n in names if n in set(forecasts_now["model"])]
    locs = [NATIONAL, *tasks[0].locations]
    for loc in locs:
        save_light_dark(
            partial(
                plot_nowcast,
                loc,
                vintage=latest,
                models=shown,
                target=target,
                forecasts=forecasts_now,
            ),
            fig / f"nowcast-{loc.lower()}.svg",
        )
    nowcast_md = []
    loc_names = dict(zip(rows["location"], rows["location_name"], strict=False))
    for loc in locs:
        slug = loc.lower()
        label = f"{loc_names.get(loc, loc)} ({loc})"
        nowcast_md.append(
            f'=== "{label}"\n\n'
            f"    ![Nowcast and forecast for {label}]"
            f"(figures/nowcast-{slug}.svg#only-light)\n"
            f"    ![Nowcast and forecast for {label}]"
            f"(figures/nowcast-{slug}-dark.svg#only-dark)\n"
        )
    _write(gen / "nowcast-tabs.md", "\n".join(nowcast_md))
    ref = tasks[0].reference_date.isoformat()
    _write(
        gen / "nowcast-meta.md",
        f"Vintage **{latest}**, reference date **{ref}**, models "
        f"{', '.join(f'`{n}`' for n in shown)}.\n",
    )


def _nowcast_forecasts(
    tasks: list[ForecastTask], latest: str, res: Path
) -> pd.DataFrame:
    """Forecasts from the latest vintage for the Nowcast page.

    Models that need no fit are run here. Fitted models are slow, so
    their forecasts are taken from the held-out run, which forecasts the
    latest vintage with every model, when it has them.
    """
    specs = list_models()
    quick = [get_model(s.name) for s in specs if not s.fitted]
    frames = [run_forecasts(quick, tasks)]
    path = res / "heldout-forecasts.csv.gz"
    if path.exists():
        held = pd.read_csv(path)
        fitted = {s.name for s in specs if s.fitted}
        keep = held.loc[
            (held["vintage"] == latest) & held["model"].isin(fitted)
        ].drop(columns="chosen_model", errors="ignore")
        frames.append(keep)
    return pd.concat(frames, ignore_index=True)


_LADDER_SECTIONS = (
    (
        "Rungs",
        "What each rung adds, and what was decided.",
        ("step", "rung", "description", "decision"),
    ),
    (
        "Sampler",
        "Sampler diagnostics of the fit to the real vintage.",
        (
            "real_seconds",
            "real_max_rhat",
            "real_min_ess",
            "real_min_ess_per_s",
            "real_divergent",
            "real_depth_hits",
        ),
    ),
    (
        "Recovery",
        "Fits to a simulated outbreak with known parameters.",
        (
            "sim_max_rhat",
            "sim_rt_mae",
            "sim_turn_error",
            "sim_ppc90",
            "sim_crps_national",
            "sim_crps_province",
            "sim_cov90_national_ahead",
            "sim_cov90_province_ahead",
        ),
    ),
    (
        "Scores",
        "Forecasts from the real vintage, scored against a later one.",
        (
            "real_ppc90",
            "real_crps_national",
            "real_crps_province",
            "real_cov90_national_now",
            "real_cov90_national_ahead",
            "real_cov90_province_now",
            "real_cov90_province_ahead",
            "prior_wild",
        ),
    ),
)
"""Sections of the Model building page, with the ladder columns each shows."""

_LADDER_LABELS = {
    "step": "Step",
    "rung": "Rung",
    "description": "Adds",
    "decision": "Decision",
    "seconds": "Seconds",
    "max_rhat": "Max R-hat",
    "min_ess": "Min ESS",
    "min_ess_per_s": "Min ESS per second",
    "divergent": "Divergent share",
    "depth_hits": "Depth cap share",
    "rt_mae": "Rt MAE",
    "turn_error": "Days off the Rt = 1 crossing",
    "ppc90": "PPC 90% coverage",
    "crps_national": "National CRPS",
    "crps_province": "Province CRPS",
    "cov90_national_now": "National 90% coverage, nowcast",
    "cov90_national_ahead": "National 90% coverage, ahead",
    "cov90_province_now": "Province 90% coverage, nowcast",
    "cov90_province_ahead": "Province 90% coverage, ahead",
    "prior_wild": "Prior draws past 10x the peak",
}
"""Column headers of the Model building tables, without the sim_ or real_."""


def _evidence_link(evidence: object) -> str:
    """Render a log row's evidence as a link."""
    if not isinstance(evidence, str) or not evidence:
        return ""
    if evidence.startswith("http"):
        label = evidence.rstrip("/").rsplit("/", 1)[-1]
        return f"[{label}]({evidence})"
    url = f"https://github.com/seabbs/BDBVHubRenewal/blob/main/{evidence}"
    return f"[`{evidence}`]({url})"


def _development_log(path: Path) -> str:
    """Return the development log as a table, newest first."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    log = pd.read_csv(path, dtype=str).fillna("")
    if log.empty:
        return ""
    log = log.iloc[::-1].sort_values("date", ascending=False, kind="stable")
    log["evidence"] = log["evidence"].map(_evidence_link)
    log.columns = [c.capitalize() for c in log.columns]
    return log.to_markdown(index=False) + "\n"


def _ladder_label(column: str) -> str:
    """Return the header of a ladder column."""
    key = column.removeprefix("sim_").removeprefix("real_")
    return _LADDER_LABELS.get(key, column)


def _ladder_page(path: Path) -> str:
    """Return the Model building tables from the ladder results."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    ladder = pd.read_csv(path)
    if "step" in ladder:
        ladder = ladder.sort_values("step")
    parts = []
    for title, lead, columns in _LADDER_SECTIONS:
        cols = [c for c in columns if c in ladder]
        if not cols:
            continue
        if "rung" not in cols:
            cols = ["rung", *cols]
        metrics = [c for c in cols if c not in ("step", "rung")]
        shown = ladder.dropna(subset=metrics, how="all")[cols].round(3)
        # tabulate blanks None but prints NaN, so cells go to objects first.
        shown = shown.astype(object).where(shown.notna(), None)
        table = shown.rename(columns=_ladder_label).to_markdown(
            index=False, missingval=""
        )
        parts.append(f"## {title}\n\n{lead}\n\n{table}\n")
    return "\n".join(parts)


_TRUTH_LABELS = {
    "final_vintage": "final vintage",
    "true_onset": "true onset week",
}
_VERSION_LABELS = {
    "fresh": "fresh draw of the final count",
    "anchored": "anchored to the vintage",
    "recentred": "known anchored, imputed around the current count",
}


def _nowcast_recovery_lead(rows: pd.DataFrame) -> str:
    """Sentences on the error and coverage of each version, from the rows."""
    final = rows.loc[rows["truth"] == "final vintage"]
    lines = []
    for vintage, by in final.groupby("vintage", sort=True):
        v = by.set_index("version")
        if not {"fresh", "recentred"} <= set(v.index):
            continue
        lines.append(
            f"On the {vintage} vintage, the absolute error of the median "
            f"is {v.loc['fresh', 'mae_median']:.1f} with a fresh draw and "
            f"{v.loc['recentred', 'mae_median']:.1f} with the reported "
            "counts kept, and 90% coverage is "
            f"{v.loc['fresh', 'coverage_90']:.2f} and "
            f"{v.loc['recentred', 'coverage_90']:.2f}."
        )
    if not lines:
        return ""
    n = int(rows["n_cells"].max()) if "n_cells" in rows else 0
    tail = (
        f"Each vintage has {n} province and national weeks, so coverage "
        f"moves in steps of {1 / n:.2f}.\n\n"
        if n
        else "\n"
    )
    return "\n".join(lines) + "\n" + tail


def _nowcast_recovery_section(path: Path) -> str:
    """Return the nowcast recovery section, or nothing without results."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    rows = pd.read_csv(path)
    if rows.empty:
        return ""
    rows = rows.assign(
        truth=rows["truth"].map(_TRUTH_LABELS).fillna(rows["truth"]),
        how=rows["version"].map(_VERSION_LABELS).fillna(""),
    )
    cols = [
        "vintage",
        "truth",
        "version",
        "how",
        "coverage_50",
        "coverage_90",
        "mae_median",
        "share_below_reported",
        "divergent_share",
        "max_rhat",
    ]
    table = (
        rows[[c for c in cols if c in rows]]
        .round(3)
        .to_markdown(index=False, missingval="")
    )
    return (
        "### Nowcasts of partly reported weeks\n\n"
        "Nowcasts of the weeks before the reference date (horizons -4 to "
        "-1, by province and nationally) on simulated vintages, by how the "
        "counts still to be reported are drawn.\n"
        "They are scored against the final simulated vintage, the hub-style "
        "target, and against cases by true onset week.\n"
        "`share_below_reported` is the mean share of draws below the counts "
        "already reported.\n\n" + _nowcast_recovery_lead(rows) + table + "\n"
    )


def _lopo_section(path: Path) -> str:
    """Return the leave-one-province-out section, or nothing without results."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    scores = pd.read_csv(path, low_memory=False)
    if scores.empty:
        return ""
    table = summarise_scores(scores, by=("model", "masked"))
    cols = ["n", "wis", "log_wis", "coverage_50", "coverage_90", "bias"]
    return (
        "## Leave one province out\n\n"
        "Each province with cases is forecast with its own counts left out "
        "of the likelihood (see "
        "[Evaluation](methods/evaluation.md#leave-one-province-out)).\n"
        "The scores are for the left-out province only, against the latest "
        "real vintage.\n\n" + _table(table[["model", "masked", *cols]])
    )


def _score_views(scores: pd.DataFrame, k: int) -> dict[str, pd.DataFrame]:
    """Split scores into complete weeks, all weeks and vintage-matched.

    Results written before the `scoring` and `complete` columns existed
    are final counts as reported, and complete weeks follow `k`.
    """
    if "scoring" not in scores:
        scores = scores.assign(scoring="as reported")
    if "complete" not in scores:
        age = pd.to_datetime(scores["truth_vintage"]) - pd.to_datetime(
            scores["target_end_date"]
        )
        scores = scores.assign(complete=age.dt.days >= 7 * k)
    reported = scores.loc[scores["scoring"] == "as reported"]
    complete = reported.loc[reported["complete"].astype(bool)]
    views = {
        "complete weeks": complete,
        "all weeks as reported": reported,
        "vintage-matched": scores.loc[scores["scoring"] == "vintage-matched"],
    }
    if "lag_prior" in scores and scores["lag_prior"].fillna(False).any():
        own = ~complete["lag_prior"].fillna(False).astype(bool)
        views["complete weeks, own lag estimate"] = complete.loc[own]
    return views


def _lag_prior_note(scores: pd.DataFrame) -> str:
    """Name the origins whose publication lag comes from the prior."""
    if "lag_prior" not in scores:
        return ""
    marked = scores.loc[scores["lag_prior"].fillna(False).astype(bool)]
    if marked.empty:
        return ""
    origins = (
        marked[["kind", "vintage"]].drop_duplicates().sort_values("vintage")
    )
    names = ", ".join(f"{v} ({k})" for k, v in origins.to_numpy())
    which = "origin" if len(origins) == 1 else "origins"
    have = "has" if len(origins) == 1 else "have"
    return (
        f"The {which} {names} {have} no earlier pair of delay files, so the "
        "renewal models take their publication lag from a prior whose shape "
        "was set from later delay files.\n"
        'The scores marked "own lag estimate" leave them out.\n'
    )


def _overall_scores(views: dict[str, pd.DataFrame]) -> str:
    """Mean scores by model and vintage kind for each view with rows."""
    frames = [
        summarise_scores(rows, by=("model", "kind")).assign(scores=name)
        for name, rows in views.items()
        if not rows.empty
    ]
    if not frames:
        return ""
    table = pd.concat(frames, ignore_index=True)
    first = ["scores", "model", "kind"]
    return _table(table[[*first, *[c for c in table if c not in first]]])


def _completeness_note(k: int, delays: pd.DataFrame) -> str:
    """Say which weeks count as complete and why."""
    latest = str(delays["vintage"].max())
    share = reported_share(delays, latest, [7 * k])[0]
    link = "reference/data.md#bdbvhubrenewal.data.completeness.complete_weeks"
    return (
        f"Complete weeks are those that ended at least {k} weeks before the "
        f"truth vintage.\nBy then {share:.0%} of a week's final count is "
        f"published, from the {latest} delay data "
        f"([`complete_weeks`]({link})).\n"
        "Scores on complete weeks are the primary scores.\n"
        "Vintage-matched scores compare each model's prediction of the "
        "counts as reported by the truth vintage with that vintage, so "
        "they cover every week.\n"
    )


BACKTEST_LOCATIONS: tuple[str, ...] = (NATIONAL, "CD-IT", "CD-NK")
"""Locations with forecast plots on the Backtest page."""


def _backtest_findings(primary: pd.DataFrame) -> str:
    """One or two sentences of results on complete weeks of real origins."""
    real = primary.loc[primary["kind"] == "real"]
    if real["model"].nunique() < 2 or "persistence" not in set(real["model"]):
        return ""
    rel = relative_skill(real).set_index("model")["relative_wis"]
    cov = real.groupby("model")["coverage_90"].mean()
    others = [m for m in rel.index if m != "persistence"]
    wis = ", ".join(f"{m} {rel[m]:.2f}" for m in others)
    cover = ", ".join(f"{m} {cov[m]:.2f}" for m in [*others, "persistence"])
    origins = real["vintage"].nunique()
    return (
        f"On complete weeks of the {origins} real origins, mean WIS relative "
        f"to persistence is {wis}.\n"
        f"90% interval coverage is {cover}.\n"
    )


def _forecast_tabs(names: Sequence[str], labels: Mapping[str, str]) -> str:
    """Tabs of each model's forecasts from the real origins, by location."""
    if not names:
        return ""
    tabs = []
    for m in names:
        tabs.append(f'=== "{m}"\n')
        for loc, label in labels.items():
            slug = f"forecasts-{m}-{loc.lower()}"
            alt = f"{m} forecasts against later data, {label}."
            tabs.append(
                f'    === "{label}"\n\n'
                f"        ![{alt}](figures/{slug}.svg#only-light)\n"
                f"        ![{alt}](figures/{slug}-dark.svg#only-dark)\n"
            )
    return (
        "## Forecasts against later data\n\n"
        "Forecasts from the real origins: 100 sample trajectories, 50% and "
        "90% intervals and the median, with the data at the time (dashed) "
        "and the latest data (solid).\n\n" + "\n".join(tabs)
    )


def _recovery_section(path: Path) -> str:
    """Return the recovery section, or nothing without results."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    if pd.read_csv(path).empty:
        return ""
    alt = "Mean WIS by horizon on simulated vintages."
    return (
        "## Recovery on simulated vintages\n\n"
        "Forecasts from vintages of a [simulated outbreak]"
        "(methods/data.md#simulated-vintages), scored against the known "
        "final counts.\n\n"
        f"![{alt}](figures/recovery-wis-horizon.svg#only-light)\n"
        f"![{alt}](figures/recovery-wis-horizon-dark.svg#only-dark)\n"
    )


def _scenario_scores(path: Path) -> str:
    """Return mean scores on the simulated scenarios, or nothing."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    scores = pd.read_csv(path, low_memory=False)
    if scores.empty:
        return ""
    if "level" not in scores:
        scores["level"] = scores["location"].map(
            lambda x: "national" if x == "CD" else "province"
        )
    views = {
        "final data, complete weeks": (scores["truth_view"] == "final")
        & scores["complete"],
        "as of 2 weeks later": scores["truth_view"] != "final",
    }
    tables = []
    for label, keep in views.items():
        table = (
            scores.loc[keep]
            .groupby(["scenario", "level", "model"], sort=False)
            .agg(
                wis=("wis", "mean"),
                coverage_50=("coverage_50", "mean"),
                coverage_90=("coverage_90", "mean"),
                bias=("bias", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    "wis": "mean WIS",
                    "coverage_50": "50% coverage",
                    "coverage_90": "90% coverage",
                    "bias": "mean bias",
                }
            )
        )
        tables.append(f"**Scored against {label}**\n\n" + _table(table))
    return "\n".join(tables)


def _scenario_figure(path: Path) -> str:
    """Return the scenario figure links, or nothing without results."""
    if not _scenario_scores(path):
        return ""
    alt = "Mean WIS by horizon on the simulated scenarios."
    return (
        f"![{alt}](figures/scenario-wis-horizon.svg#only-light)\n"
        f"![{alt}](figures/scenario-wis-horizon-dark.svg#only-dark)\n"
    )


def _relative_section(scores: pd.DataFrame) -> str:
    if scores["model"].nunique() < 2:
        return ""
    rel = relative_skill(scores, by=("kind",))
    tabs = []
    for by, label in (("horizon", "By horizon"), ("location", "By province")):
        alt = f"WIS relative to persistence {label.lower()}."
        tabs.append(
            f'=== "{label}"\n\n'
            f"    ![{alt}](figures/relative-wis-{by}.svg#only-light)\n"
            f"    ![{alt}](figures/relative-wis-{by}-dark.svg#only-dark)\n"
        )
    return (
        "## Relative skill\n\n"
        "Mean scores on complete weeks divided by those of persistence over "
        "the forecasts both made.\nValues below 1 beat persistence.\n\n"
        + _table(rel)
        + "\n"
        + "\n".join(tabs)
    )


def _other_relative(views: dict[str, pd.DataFrame]) -> str:
    """Relative scores of the other score sets that two models have."""
    parts = []
    for name in ("complete weeks, own lag estimate", "vintage-matched"):
        rows = views.get(name, pd.DataFrame())
        if rows.empty or rows["model"].nunique() < 2:
            continue
        parts.append(
            f"\n{name.capitalize()}:\n\n"
            + _table(relative_skill(rows, by=("kind",)))
            + "\n"
        )
    return "".join(parts)


def _submission_sampler(path: Path = Path("ci/policy.toml")) -> dict[str, Any]:
    """Return the main candidate's sampler settings from the CI policy."""
    policy = tomllib.loads(path.read_text())
    main = next(c for c in policy["candidates"] if c["id"] == "main")
    return dict(main["sampler"])


def _settings_text(chains: int, warmup: int, draws: int) -> str:
    return f"{chains} chains of {warmup} warm-up and {draws} draws"


def _sampler_section(fits: pd.DataFrame, submission: Mapping[str, Any]) -> str:
    """Say how long the backtest fits were and how well they converged."""
    if fits.empty:
        return ""
    keys = ["num_chains", "num_warmup", "num_samples"]
    used = fits[keys].drop_duplicates().astype(int).to_numpy()
    runs = " or ".join(_settings_text(*row) for row in used)
    target = tuple(int(submission[k]) for k in keys)
    sub = _settings_text(*target)
    if len(used) == 1 and tuple(used[0]) == target:
        compare = f"This is the same as the submission fit, {sub}.\n"
    else:
        compare = (
            f"The submission fit uses {sub}, so backtest fits are shorter "
            "than the submission fit.\n"
        )
    by = fits.groupby("model")
    table = pd.DataFrame(
        {
            "fits": by.size(),
            "median max R-hat": by["max_rhat"].median(),
            "worst max R-hat": by["max_rhat"].max(),
            "share with R-hat over 1.05": by["max_rhat"].apply(
                lambda x: float((x > 1.05).mean())
            ),
            "depth cap share": by["depth_share"].mean(),
            "divergent share": by["divergent_share"].mean(),
        }
    ).reset_index()
    return (
        "## Sampler\n\n"
        f"Backtest fits use {runs}.\n"
        + compare
        + "The table gives each model's fits and their convergence.\n"
        "A per-location model has one fit per location with enough cases.\n"
        "R-hat is the largest over a fit's parameters, and the depth cap "
        "share is the share of draws at the maximum tree depth.\n\n"
        + _table(table, digits=3)
        + "\n"
    )


def _backtest_page(
    fig: Path,
    gen: Path,
    res: Path,
    out: Path,
    config: BacktestConfig,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    names: list[str],
) -> None:
    dev_path = res / "backtest-scores.csv"
    k = config.complete_after_weeks
    if dev_path.exists():
        views = _score_views(pd.read_csv(dev_path), k)
        scores = views["all weeks as reported"]
        primary = views["complete weeks"]
        for by in ("horizon", "location"):
            for metric in ("wis", "log_wis", "coverage_90", "bias"):
                save_light_dark(
                    partial(
                        plot_scores, scores, by=by, metric=metric, models=names
                    ),
                    fig / f"scores-{metric.replace('_', '-')}-{by}.svg",
                )
            save_light_dark(
                partial(plot_relative_skill, primary, by=by, models=names),
                fig / f"relative-wis-{by}.svg",
            )
        truth_v = ", ".join(sorted(set(scores["truth_vintage"].astype(str))))
        _write(
            gen / "backtest-truth.md",
            f"Scores are against the **{truth_v}** vintage, as reported "
            "then.\n"
            + _completeness_note(k, delays)
            + _lag_prior_note(views["all weeks as reported"]),
        )
        _write(gen / "scores-overall.md", _overall_scores(views))
        _write(
            gen / "relative-section.md",
            _relative_section(primary) + _other_relative(views),
        )
        _write(gen / "backtest-findings.md", _backtest_findings(primary))
        by_loc = summarise_scores(primary, by=("model", "kind", "location"))
        _write(
            gen / "scores-location.md",
            _table(
                by_loc[
                    [
                        "model",
                        "kind",
                        "location",
                        "n",
                        "wis",
                        "log_wis",
                        "coverage_50",
                        "coverage_90",
                        "bias",
                    ]
                ]
            ),
        )
    by_truth_path = res / "backtest-scores-by-truth.csv"
    if by_truth_path.exists():
        truth_views = _score_views(pd.read_csv(by_truth_path), k)
        by_truth = truth_views["all weeks as reported"]
        save_light_dark(
            partial(plot_scores_by_truth, by_truth, models=names),
            fig / "scores-by-truth.svg",
        )
        table = pd.concat(
            [
                summarise_scores(
                    rows, by=("model", "vintage", "truth_vintage")
                ).assign(scores=name)
                for name, rows in truth_views.items()
                if not rows.empty
            ],
            ignore_index=True,
        )
        _write(
            gen / "scores-by-truth.md",
            _table(
                table[
                    [
                        "scores",
                        "model",
                        "vintage",
                        "truth_vintage",
                        "n",
                        "wis",
                        "log_wis",
                        "coverage_90",
                        "bias",
                    ]
                ]
            ),
        )
        weeks = sorted(set(by_truth["target_end_date"].astype(str)))
        changes = truth_changes(target, locations=(NATIONAL,))
        changes = changes.loc[changes["target_end_date"].isin(weeks)]
        wide = changes.pivot_table(
            index="target_end_date", columns="truth_vintage", values="truth"
        )
        wide.columns = [f"as reported {c}" for c in wide.columns]
        wide = wide.astype("Int64").astype(str).replace("<NA>", "")
        _write(gen / "truth-changes.md", _table(wide.reset_index()))
    fc_path = out / "forecasts.csv.gz"
    shown: list[str] = []
    if fc_path.exists():
        forecasts = pd.read_csv(fc_path, dtype={"as_of": object})
        if "as_of" in forecasts:
            forecasts = forecasts.loc[forecasts["as_of"].isna()]
        real_fc = forecasts.loc[forecasts["kind"] == "real"]
        shown = [m for m in names if m in set(real_fc["model"])]
        for m in shown:
            for loc in BACKTEST_LOCATIONS:
                save_light_dark(
                    partial(
                        plot_forecasts,
                        real_fc,
                        target,
                        loc,
                        models=[m],
                        kind="real",
                    ),
                    fig / f"forecasts-{m}-{loc.lower()}.svg",
                )
    labels = _location_names(target)
    _write(
        gen / "backtest-forecasts.md",
        _forecast_tabs(
            shown,
            {
                loc: f"{labels.get(loc, loc)} ({loc})"
                for loc in BACKTEST_LOCATIONS
            },
        ),
    )
    _write(gen / "lopo-section.md", _lopo_section(res / "lopo-scores.csv"))
    _write(gen / "ladder.md", _ladder_page(res / "ladder.csv"))
    _write(
        gen / "development-log.md",
        _development_log(res / "development-log.csv"),
    )
    _write(
        gen / "nowcast-recovery.md",
        _nowcast_recovery_section(res / "nowcast-recovery.csv"),
    )
    fits_path = res / "backtest-fits.csv"
    _write(
        gen / "backtest-sampler.md",
        _sampler_section(pd.read_csv(fits_path), _submission_sampler())
        if fits_path.exists()
        else "",
    )
    rec_path = res / "recovery-scores.csv"
    _write(gen / "recovery-section.md", _recovery_section(rec_path))
    if rec_path.exists():
        rscores = pd.read_csv(rec_path)
        save_light_dark(
            lambda mode: plot_scores(
                rscores, by="horizon", models=names, mode=mode
            ),
            fig / "recovery-wis-horizon.svg",
        )
    scen_path = res / "scenario-scores.csv.gz"
    _write(gen / "scenario-scores.md", _scenario_scores(scen_path))
    _write(gen / "scenario-figure.md", _scenario_figure(scen_path))
    if scen_path.exists() and scen_path.stat().st_size > 1:
        sscores = pd.read_csv(scen_path, low_memory=False)
        if not sscores.empty:
            save_light_dark(
                partial(
                    plot_scenarios,
                    sscores,
                    models=[
                        *names,
                        *(
                            m
                            for m in sscores["model"].unique()
                            if m not in names
                        ),
                    ],
                ),
                fig / "scenario-wis-horizon.svg",
            )


_HELDOUT_SECTIONS = (
    ("heldout-choices.md", "Model choices"),
    ("heldout-status.md", "Status"),
    ("heldout-scores.md", "Scores"),
    ("heldout-forecasts.md", "Forecasts against later data"),
)
"""Held-out snippets and their headings, shown only with content."""


def _heldout_page(
    fig: Path,
    gen: Path,
    res: Path,
    target: pd.DataFrame,
    names: list[str],
    config: BacktestConfig,
) -> None:
    _heldout_snippets(fig, gen, res, target, names, config)
    for name, title in _HELDOUT_SECTIONS:
        path = gen / name
        body = path.read_text() if path.exists() else ""
        _write(path, f"## {title}\n\n{body}" if body.strip() else "")


def _heldout_snippets(
    fig: Path,
    gen: Path,
    res: Path,
    target: pd.DataFrame,
    names: list[str],
    config: BacktestConfig,
) -> None:
    choices = load_model_choices(config.model_choice)
    if choices:
        table = pd.DataFrame(
            {
                "date": [c.date for c in choices],
                "model": [f"`{c.model}`" for c in choices],
                "why": [c.summary for c in choices],
            }
        )
        held = [
            v
            for v in sorted(
                set(target.loc[target["kind"] == "real", "vintage"])
            )
            if v > choices[0].date
        ]
        _write(
            gen / "heldout-choices.md",
            _table(table)
            + (
                f"\nVintages after {choices[0].date} are held out: "
                + ", ".join(held)
                + ".\n"
                if held
                else ""
            ),
        )
    else:
        _write(gen / "heldout-choices.md", "")
    run_path = res / "heldout-run.yml"
    if not run_path.exists():
        for name in ("heldout-status.md", "heldout-scores.md"):
            _write(gen / name, "")
        _write(gen / "heldout-forecasts.md", "")
        return
    run = yaml.safe_load(run_path.read_text())
    scores_path = res / "heldout-scores.csv"
    scores = (
        pd.read_csv(scores_path)
        if scores_path.stat().st_size > 1
        else pd.DataFrame()
    )
    rows = []
    for entry in [*run["scored"], *run["pending"]]:
        v = str(entry["vintage"])
        truths = (
            sorted(set(scores.loc[scores["vintage"] == v, "truth_vintage"]))
            if not scores.empty
            else []
        )
        rows.append(
            {
                "vintage": v,
                "reference_date": reference_date_for_vintage(v).isoformat(),
                "model chosen": entry["model"],
                "scored against": ", ".join(map(str, truths))
                or "with the next vintage",
            }
        )
    status = pd.DataFrame(rows).sort_values("vintage")
    _write(
        gen / "heldout-status.md",
        f"Last run {run['run_at']} on vintages up to "
        f"**{run['latest_vintage']}**.\n\n" + _table(status),
    )
    if scores.empty:
        _write(gen / "heldout-scores.md", "")
    else:
        views = _score_views(scores, config.complete_after_weeks)
        overall = pd.concat(
            [
                summarise_scores(
                    rows,
                    by=("chosen_model", "model", "vintage", "truth_vintage"),
                ).assign(scores=name)
                for name, rows in views.items()
                if not rows.empty
            ],
            ignore_index=True,
        )
        horizon = summarise_scores(
            views["all weeks as reported"],
            by=("model", "vintage", "truth_vintage", "horizon"),
        )
        cols = ["n", "wis", "log_wis", "coverage_50", "coverage_90", "bias"]
        _write(
            gen / "heldout-scores.md",
            _table(
                overall[
                    [
                        "scores",
                        "chosen_model",
                        "model",
                        "vintage",
                        "truth_vintage",
                        *cols,
                    ]
                ]
            )
            + "\nBy horizon, over provinces and the national total:\n\n"
            + _table(
                horizon[["model", "vintage", "truth_vintage", "horizon", *cols]]
            ),
        )
    fc_path = res / "heldout-forecasts.csv.gz"
    if not fc_path.exists():
        _write(gen / "heldout-forecasts.md", "")
        return
    forecasts = pd.read_csv(fc_path)
    models = [n for n in names if n in set(forecasts["model"])]
    tabs = []
    loc_names = _location_names(target)
    for loc in (NATIONAL, "CD-IT", "CD-NK"):
        slug = loc.lower()
        save_light_dark(
            partial(plot_forecasts, forecasts, target, loc, models=models),
            fig / f"heldout-{slug}.svg",
        )
        label = f"{loc_names.get(loc, loc)} ({loc})"
        alt = f"Held-out forecasts against later data, {label}."
        tabs.append(
            f'=== "{label}"\n\n'
            f"    ![{alt}](figures/heldout-{slug}.svg#only-light)\n"
            f"    ![{alt}](figures/heldout-{slug}-dark.svg#only-dark)\n"
        )
    _write(gen / "heldout-forecasts.md", "\n".join(tabs))


def _location_names(target: pd.DataFrame) -> dict[str, str]:
    pairs = target[["location", "location_name"]].drop_duplicates()
    return dict(zip(pairs["location"], pairs["location_name"], strict=False))


def _comparison_page(
    fig: Path,
    gen: Path,
    cmp_dir: Path,
    target: pd.DataFrame,
) -> None:
    summary_path = cmp_dir / "summary.json"
    if not summary_path.exists():
        missing = "Not run yet: `bdbvhubrenewal compare-hub --fetch`.\n"
        for name in (
            "comparison-lead.md",
            "comparison-scores.md",
            "comparison-teams.md",
            "comparison-figures.md",
        ):
            _write(gen / name, missing)
        _write(
            gen / "comparison-attribution.md",
            f"> {attribution(dt.date.today().isoformat())}\n",
        )
        return
    summary = json.loads(summary_path.read_text())
    _write(
        gen / "comparison-attribution.md",
        f"> {attribution(summary['accessed'])}\n",
    )
    truth, k = summary["truth_vintage"], summary["complete_after_weeks"]
    baseline = summary["baseline"]
    hub_ids = summary.get("hub_ids", {})
    complete_by = (
        (pd.Timestamp(truth) - pd.Timedelta(weeks=k)).date().isoformat()
    )
    note = ""
    if baseline != summary["preferred_baseline"]:
        note = f" (`{summary['preferred_baseline']}` once it is registered)"
    _write(
        gen / "comparison-lead.md",
        f"Scores use the **{truth}** data and the complete weeks only: "
        f"weeks ending by {complete_by} ($K = {k}$). "
        f"Relative WIS divides by `{baseline}`{note} on the weeks both "
        "forecast.\n",
    )
    scores_path = cmp_dir / "scores.csv"
    scores = (
        pd.read_csv(scores_path)
        if scores_path.stat().st_size > 1
        else pd.DataFrame()
    )
    pending = pd.read_csv(cmp_dir / "pending.csv")
    parts = []
    if not scores.empty:
        table = summarise_comparison(scores, baseline)
        table["model"] = [
            comparison_label(m, s, hub_ids)
            for m, s in zip(table["model"], table["source"], strict=True)
        ]
        parts.append(_table(table) if not table.empty else "")
    for row in pending.itertuples(index=False):
        parts.append(
            f"The {row.reference_date} round was made from the "
            f"{row.vintage} data, so it is scored once later data are "
            "published.\n"
        )
    _write(
        gen / "comparison-scores.md",
        "\n".join(p for p in parts if p) or "No round has a score yet.\n",
    )
    teams = pd.read_csv(cmp_dir / "teams.csv")
    _write(
        gen / "comparison-teams.md",
        _table(teams[["model", "team_name", "model_name", "license"]]),
    )
    quantiles = pd.read_csv(cmp_dir / "quantiles.csv.gz")
    tabs = []
    for ref in sorted(set(quantiles["reference_date"].astype(str))):
        stem = f"comparison-{ref}"
        save_light_dark(
            partial(
                plot_hub_round,
                quantiles,
                target,
                ref,
                locations=tuple(summary["locations"]),
                complete_by=complete_by,
                hub_ids=hub_ids,
            ),
            fig / f"{stem}.svg",
        )
        alt = f"Forecasts for the {ref} round against the {truth} data."
        tabs.append(
            f'=== "{ref}"\n\n'
            f"    ![{alt}](figures/{stem}.svg#only-light)\n"
            f"    ![{alt}](figures/{stem}-dark.svg#only-dark)\n"
        )
    _write(gen / "comparison-figures.md", "\n".join(tabs))
