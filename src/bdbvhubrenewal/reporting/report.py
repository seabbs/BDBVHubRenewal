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
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml
from matplotlib.figure import Figure

from bdbvhubrenewal.data import (
    downward_summary,
    load_daily,
    load_daily_zones,
    load_delays,
    load_health_zones,
    load_target,
    national_coherence,
    newly_missing,
    release_updates,
)
from bdbvhubrenewal.data.dates import reference_date_for_vintage
from bdbvhubrenewal.data.delays import delay_summary
from bdbvhubrenewal.data.simulate import simulate_outbreak
from bdbvhubrenewal.data.tidy import COHERENCE_STREAMS
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    check_distinct,
    load_config,
    make_tasks,
    run_forecasts,
    truth_changes,
)
from bdbvhubrenewal.evaluation.heldout import load_model_choices
from bdbvhubrenewal.evaluation.hubcompare import (
    attribution,
    forecast_quantiles,
    load_comparison_config,
    overall_comparison,
    stored_forecasts,
    summarise_comparison,
)
from bdbvhubrenewal.evaluation.nowcast import BASELINE, relative_wis
from bdbvhubrenewal.evaluation.scoring import summarise_scores
from bdbvhubrenewal.evaluation.zones import zone_relative_skill
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastTask,
    get_model,
    list_models,
)
from bdbvhubrenewal.reporting.explorer import (
    explorer_data,
    explorer_html,
    write_explorer,
)
from bdbvhubrenewal.reporting.nowcast_plots import (
    plot_relative_overall,
    plot_release_nowcasts,
)
from bdbvhubrenewal.reporting.plots import (
    comparison_colours,
    comparison_label,
    model_colour,
    plot_delay_check,
    plot_delay_distribution,
    plot_delays_over_time,
    plot_downward_updates,
    plot_forecasts,
    plot_hub_round,
    plot_imputed_by_release,
    plot_imputed_share,
    plot_location_grid,
    plot_newly_missing,
    plot_revisions,
    plot_scenarios,
    plot_score_summary,
    plot_scores,
    plot_scores_by_truth,
    plot_simulation,
    plot_target,
    save_light_dark,
)
from bdbvhubrenewal.reporting.scorechart import (
    OVERALL_METRICS,
    SCORE_METRICS,
    score_chart_data,
    score_chart_html,
    write_score_chart,
)
from bdbvhubrenewal.reporting.tables import filter_table
from bdbvhubrenewal.reporting.zone_nowcast import (
    ZONE_MODEL,
    zone_chart_data,
    zone_counts,
    zone_nowcast,
    zone_table,
)

__all__ = ["make_report"]


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.round(digits).to_markdown(index=False) + "\n"


def _dropdown(body: str, title: str = "Tables") -> str:
    """Wrap Markdown in a collapsed admonition, or return nothing if empty."""
    if not body.strip():
        return ""
    indented = "\n".join(
        f"    {line}" if line.strip() else "" for line in body.splitlines()
    )
    return f'??? note "{title}"\n\n{indented}\n'


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
        lambda mode: plot_imputed_by_release(real, mode=mode),
        fig / "imputed-by-release.svg",
    )
    save_light_dark(
        lambda mode: plot_newly_missing(real, mode=mode),
        fig / "newly-missing.svg",
    )
    zones = load_health_zones(data_dir)
    save_light_dark(
        lambda mode: plot_downward_updates(real, zones, mode=mode),
        fig / "downward-updates.svg",
    )
    _write(gen / "downward-updates.md", _downward_table(real, zones))
    save_light_dark(
        lambda mode: plot_delay_distribution(latest_delays, mode=mode),
        fig / "delay-distribution.svg",
    )
    save_light_dark(
        lambda mode: plot_delays_over_time(delays, mode=mode),
        fig / "delays-time.svg",
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
    _release_section(fig, gen, res, real, names)
    _backtest_page(fig, gen, res, out, config, target, names)
    _heldout_page(fig, gen, res, target, names, config)
    _comparison_page(fig, gen, Path(hub_comparison), target)

    # Nowcast page: latest real vintage, every registered model.
    rows = real.loc[real["vintage"] == latest]
    daily = load_daily(data_dir)
    tasks = make_tasks(rows, delays, config, history=real, daily=daily)
    forecasts_now = _nowcast_forecasts(tasks, latest, res)
    _nowcast_page(fig, gen, target, forecasts_now, names, config, latest)
    _zone_nowcast_section(
        fig, gen, res, data_dir, target, tasks[0], forecasts_now, config, names
    )
    groups = _nowcast_groups(target, sorted(set(forecasts_now["location"])))
    save_light_dark(
        partial(
            plot_delay_check, daily, latest, [*groups["main"], *groups["other"]]
        ),
        fig / "delay-check.svg",
    )


_RELEASE_PLACES: Final = ("CD", "CD-IT", "CD-NK")
_RELEASE_MODELS: Final = (
    "randomwalk",
    "randomwalknowcast",
    "renewal",
    "renewalsp",
    "renewalbvd",
)
_TARGET_LABELS: Final = {
    "reported": "Reported by each later release",
    "complete": "Complete weeks",
}


def _downward_table(real: pd.DataFrame, zones: pd.DataFrame) -> str:
    """Tabulate falls between consecutive releases, by place."""
    nat = real["location"] == NATIONAL
    names = _location_names(real)
    provinces = release_updates(real.loc[~nat])
    each = provinces.assign(
        place=[f"{names.get(p, p)} ({p})" for p in provinces["location"]]
    ).sort_values("place")
    updates = pd.concat(
        [
            release_updates(real.loc[nat]).assign(place="Country"),
            provinces.assign(place="Provinces"),
            each,
            release_updates(zones).assign(place="Health zones"),
        ]
    )
    table = downward_summary(updates, ("place",)).set_index("place")
    table = table.loc[list(dict.fromkeys(updates["place"]))]
    table[["share_fell", "downward_share"]] = table[
        ["share_fell", "downward_share"]
    ].round(3)
    table = table.astype(object).where(table.notna(), "")
    return _table(
        table.reset_index().rename(
            columns={
                "place": "Place",
                "weeks": "Place-weeks",
                "fell": "Fell",
                "share_fell": "Share that fell",
                "cases_down": "Cases removed",
                "cases_up": "Cases added",
                "downward_share": "Removed share of all change",
                "largest_fall": "Largest fall",
            }
        )
    )


def _ours(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Keep our models and the as-reported floor, without the hub teams."""
    return frame.loc[frame["model"].isin([*names, BASELINE])]


def _release_section(
    fig: Path, gen: Path, res: Path, real: pd.DataFrame, names: Sequence[str]
) -> None:
    """Write the real-release evaluation tables and figure.

    The tables hold the number of forecasts, $n$, in every row, for our
    models only.
    """
    paths = {
        n: res / f"nowcast-{n}.csv" for n in ("scores", "quantiles", "origins")
    }
    scores = _ours(pd.read_csv(paths["scores"]), names)
    quantiles = _ours(pd.read_csv(paths["quantiles"]), names)
    origins = _ours(pd.read_csv(paths["origins"]).fillna(""), names)
    national = scores["location"] == NATIONAL
    scores["pool"] = national.map({True: "national", False: "provinces"})
    zones = pd.read_csv(res / "zone-scores.csv", low_memory=False)
    zones = zones.loc[zones["model"].isin(set(scores["model"]))]
    pooled = pd.concat([scores, zones.assign(pool="zones")])
    pooled["target"] = pooled["target"].map(_TARGET_LABELS)
    overall = relative_wis(pooled, ("target", "pool"))
    by_pool = relative_wis(pooled, ("target", "pool", "origin"))
    scores["target"] = scores["target"].map(_TARGET_LABELS)
    by_place = relative_wis(scores, ("target", "origin", "location"))
    reported = overall.loc[overall["target"] == _TARGET_LABELS["reported"]]
    save_light_dark(
        partial(
            plot_relative_overall,
            reported,
            [n for n in names if n in set(reported["model"])],
        ),
        fig / "relative-overall.svg",
    )
    save_light_dark(
        partial(
            plot_release_nowcasts,
            quantiles,
            real,
            _RELEASE_PLACES,
            _RELEASE_MODELS,
        ),
        fig / "release-nowcasts.svg",
    )
    _write(
        gen / "release-headline.md",
        "## Overall\n\n"
        "Scores against what each later release reported "
        "([how scoring works](methods/evaluation.md#real-releases), "
        "[metrics](methods/evaluation.md#scores)).\n\n"
        "![Relative WIS, relative log WIS, 50% and 90% coverage and bias "
        "of each model on the counts each later release reported, by "
        "pool.](figures/relative-overall.svg#only-light)\n"
        "![Relative WIS, relative log WIS, 50% and 90% coverage and bias "
        "of each model on the counts each later release reported, by "
        "pool.](figures/relative-overall-dark.svg#only-dark)\n\n"
        + _dropdown(
            filter_table(_table(overall), ["target", "pool", "model"]),
            "Scores by target and pool, over every origin",
        ),
    )
    _write(
        gen / "release-nowcasts.md",
        "## Nowcasts from the real releases\n\n"
        + _dropdown(
            "![Final forecasts of the recent weeks by origin, with the "
            "counts each release reports.](figures/release-nowcasts.svg"
            "#only-light)\n"
            "![Final forecasts of the recent weeks by origin, with the "
            "counts each release reports.](figures/release-nowcasts-dark.svg"
            "#only-dark)\n",
            "Figure",
        )
        + "\n"
        + _dropdown(
            "Scores by target, pool and origin.\n\n"
            + _table(by_pool)
            + "\nScores by target, origin and place.\n\n"
            + _table(by_place)
            + "\nOrigins.\n\n"
            + _table(origins),
            "Tables",
        ),
    )


_NOWCAST_GROUPS: Final = (
    ("main", "National and main provinces"),
    ("other", "Other provinces"),
    ("zones", "Zones"),
)
"""Location groups of the Nowcast page, as figure suffix and tab label."""


def _nowcast_groups(
    target: pd.DataFrame, locations: Sequence[str]
) -> dict[str, list[str]]:
    """Group locations by level, with the main provinces by size."""
    real = target.loc[target["kind"] == "real"]
    latest = real.loc[real["vintage"] == real["vintage"].max()]
    size = latest.groupby("location")["total"].sum()
    provinces = sorted(
        (loc for loc in set(latest["location"]) - {NATIONAL}),
        key=lambda x: -size[x],
    )
    have = set(locations)
    main = [loc for loc in provinces[:3] if loc in have]
    return {
        "main": [loc for loc in (NATIONAL, *main) if loc in have],
        "other": [loc for loc in provinces[3:] if loc in have],
        "zones": sorted(have - set(provinces) - {NATIONAL}),
    }


def _nowcast_page(
    fig: Path,
    gen: Path,
    target: pd.DataFrame,
    forecasts: pd.DataFrame,
    names: Sequence[str],
    config: BacktestConfig,
    latest: str,
) -> None:
    """Write the Nowcast page figures, a tab per model and location group.

    Each model is shown from its latest origin among `forecasts`.
    """
    shown = [n for n in names if n in set(forecasts["model"])]
    origin = {
        m: str(forecasts.loc[forecasts["model"] == m, "vintage"].max())
        for m in shown
    }
    kinds = (
        ("nowcast-only", "Nowcast", -1),
        ("nowcast", "Nowcast and forecast", None),
    )
    pages: dict[str, list[str]] = {stem: [] for stem, _, _ in kinds}
    for m in shown:
        own = forecasts.loc[
            (forecasts["model"] == m) & (forecasts["vintage"] == origin[m])
        ]
        groups = _nowcast_groups(target, sorted(set(own["location"])))
        note = (
            f"    From the {origin[m]} vintage.\n\n"
            if origin[m] != latest
            else ""
        )
        for stem, title, horizon in kinds:
            inner = []
            for key, label in _NOWCAST_GROUPS:
                locs = groups[key]
                if not locs:
                    continue
                path = fig / f"{stem}-{m}-{key}.svg"
                save_light_dark(
                    partial(
                        plot_location_grid,
                        own,
                        target,
                        locs,
                        models=[m],
                        truth_vintage=latest,
                        max_horizon=horizon,
                        weeks_before=12,
                        complete_after_weeks=config.complete_after_weeks,
                    ),
                    path,
                )
                alt = f"{title} from {m}: {label.lower()}."
                inner.append(
                    f'    === "{label}"\n\n'
                    f"        ![{alt}](figures/{path.name}#only-light)\n"
                    f"        ![{alt}](figures/{path.stem}-dark.svg"
                    "#only-dark)\n"
                )
            pages[stem].append(f'=== "{m}"\n\n{note}' + "\n".join(inner))
    for stem, _, _ in kinds:
        _write(gen / f"{stem}-tabs.md", "\n".join(pages[stem]))
    _write(
        gen / "nowcast-meta.md",
        f"Latest vintage **{latest}**, models "
        f"{', '.join(f'`{n}`' for n in shown)}.\n",
    )


def _zone_nowcast_section(
    fig: Path,
    gen: Path,
    res: Path,
    data_dir: Path | str,
    target: pd.DataFrame,
    task: ForecastTask,
    forecasts: pd.DataFrame,
    config: BacktestConfig,
    names: list[str],
) -> None:
    """Write the Nowcast page's health-zone section at the latest release.

    The random walk models are run on the zones here. The fitted models'
    zone samples are taken from the stored forecasts of the comparison, as
    for the hub rounds, where there are some for the release.
    """
    daily_zones = load_daily_zones(data_dir)
    zones = hub_zones(Path(data_dir) / "hub-config" / "tasks.json")
    quick = [s.name for s in list_models() if not s.fitted]
    prov = forecasts.loc[
        (forecasts["vintage"] == task.vintage)
        & (forecasts["location"] != NATIONAL)
    ]
    ran = [
        zone_nowcast(m, task, prov.loc[prov["model"] == m], daily_zones, zones)
        for m in quick
    ]
    ref = str(task.reference_date)
    stored = stored_forecasts(
        load_comparison_config().stored, [ref], exclude=quick
    )
    stored = stored.loc[stored["location"].isin(zones)]
    stored = stored.assign(
        vintage=task.vintage,
        horizon=(
            pd.to_datetime(stored["target_end_date"]) - pd.Timestamp(ref)
        ).dt.days
        // 7,
    )
    every = pd.concat([*ran, stored], ignore_index=True)
    samples = every.loc[every["model"] == ZONE_MODEL]
    labels = _location_names(target)
    start = pd.Timestamp(task.reference_date) - pd.Timedelta(weeks=12)
    counts = zone_counts(
        daily_zones, task.vintage, start.strftime("%Y-%m-%d"), labels
    )
    scores = pd.read_csv(res / "zone-scores.csv", low_memory=False)
    tabs = []
    by_prov = counts.groupby("province")["total"].sum()
    for key in by_prov.sort_values(ascending=False, kind="stable").index:
        grp = counts.loc[counts["province"] == key]
        prov_id = str(key)
        locs = sorted(set(grp["location"]))
        path = fig / f"zone-nowcast-{prov_id.lower()}.svg"
        save_light_dark(
            partial(
                plot_location_grid,
                samples,
                counts,
                locs,
                models=[ZONE_MODEL],
                truth_vintage=task.vintage,
                max_horizon=-1,
                weeks_before=12,
                ncol=4,
                complete_after_weeks=config.complete_after_weeks,
            ),
            path,
        )
        label = labels.get(prov_id, prov_id)
        alt = f"Zone nowcasts in {label} at {task.vintage}."
        tabs.append(
            f'=== "{label}"\n\n'
            f"    ![{alt}](figures/{path.name}#only-light)\n"
            f"    ![{alt}](figures/{path.stem}-dark.svg#only-dark)\n"
        )
    write_explorer(
        fig / "zonechart.json",
        zone_chart_data(
            every,
            counts,
            scores,
            labels,
            {
                m: (
                    model_colour(m, names, "light"),
                    model_colour(m, names, "dark"),
                )
                for m in names
            },
        ),
    )
    _write(
        gen / "zone-nowcast.md",
        "## Health zones\n\n"
        f"Zone nowcasts and forecasts of each model at the {task.vintage} "
        "release, for zones with cases in the 12 weeks before the reference "
        "date "
        "([Health zones](methods/evaluation.md#health-zones)).\n\n"
        "Pick a province and a zone to see its nowcast and forecast "
        "with its scores by origin.\n\n"
        '<div class="zonechart" data-src="../figures/zonechart.json" '
        'role="group" aria-label="Zone nowcast and forecast">\n'
        "<noscript>The zone chart needs JavaScript.</noscript>\n"
        "</div>\n\n"
        f"`{ZONE_MODEL}` by province.\n\n"
        + "\n".join(tabs)
        + "\n"
        + _dropdown(
            _table(zone_table(samples, counts, scores)),
            "Newest weeks by zone",
        ),
    )


def _nowcast_forecasts(
    tasks: list[ForecastTask], latest: str, res: Path
) -> pd.DataFrame:
    """Forecasts for the Nowcast page, from every source that has them.

    Models that need no fit are run on the latest vintage here. Fitted
    models are slow, so they are taken from the held-out run when it
    forecast the latest vintage, and otherwise from the backtest, at the
    latest real origin it has.
    """
    specs = list_models()
    quick = [get_model(s.name) for s in specs if not s.fitted]
    ran = run_forecasts(quick, tasks)
    silent = {m.name for m in quick} - set(ran["model"])
    if silent:
        raise RuntimeError(
            f"No forecast from {sorted(silent)} on the latest vintage "
            f"{latest}, so the Nowcast page would show another model's."
        )
    check_distinct(ran)
    frames = [ran]
    have = {s.name for s in specs if not s.fitted}
    path = res / "heldout-forecasts.csv.gz"
    try:
        held = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        held = pd.DataFrame()
    if not held.empty:
        fitted = {s.name for s in specs if s.fitted}
        keep = held.loc[
            (held["vintage"] == latest) & held["model"].isin(fitted)
        ].drop(columns="chosen_model", errors="ignore")
        frames.append(keep)
        have |= set(keep["model"])
    backtest = res / "forecasts.csv.gz"
    rows = pd.read_csv(backtest, dtype={"as_of": object})
    if "as_of" in rows:
        rows = rows.loc[rows["as_of"].isna()]
    rows = rows.loc[(rows["kind"] == "real") & ~rows["model"].isin(have)]
    for _, grp in rows.groupby("model"):
        frames.append(grp.loc[grp["vintage"] == grp["vintage"].max()])
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
        "Forecasts from the real vintage, scored against a later release.",
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
    log = pd.read_csv(path, dtype=str).fillna("")
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
        "release.\n\n" + _table(table[["model", "masked", *cols]])
    )


def _zone_section(path: Path) -> str:
    """Return the health-zone section."""
    scores = pd.read_csv(path, low_memory=False)
    cols = ["n", "wis", "log_wis", "coverage_50", "coverage_90", "bias"]
    table = summarise_scores(scores, by=("model",)).merge(
        zone_relative_skill(scores)[["model", "relative_wis"]], on="model"
    )
    by = zone_relative_skill(scores, by=("horizon",)).pivot(
        index="model", columns="horizon", values="relative_wis"
    )
    by.columns = [f"week {int(h)}" for h in by.columns]
    return (
        "## Health zones\n\n"
        "`randomwalk` and `randomwalknowcast` on each zone with cases, as "
        "run and summed to provinces "
        "([Health zones](methods/evaluation.md#health-zones)).\n\n"
        + _table(table[["model", *cols, "relative_wis"]])
        + "\n\nRelative WIS by week, with 0 the week of the reference "
        "date.\n\n" + _table(by.reset_index())
    )


BACKTEST_LOCATIONS: tuple[str, ...] = (NATIONAL, "CD-IT", "CD-NK")
"""Locations with forecast plots on the Backtest page."""


def _release_scores(res: Path) -> pd.DataFrame:
    """Read the release scores, as `_with_horizon` gives them."""
    return _with_horizon(pd.read_csv(res / "nowcast-scores.csv"))


def _with_horizon(scores: pd.DataFrame) -> pd.DataFrame:
    """Add each origin as a real vintage, and the horizon.

    `horizon` is in weeks from the reference date, a week after the origin.
    """
    weeks = (
        pd.to_datetime(scores["target_end_date"])
        - pd.to_datetime(scores["origin"])
    ).dt.days // 7
    return scores.assign(
        kind="real", vintage=scores["origin"], horizon=weeks - 1
    )


def _backtest_findings(scores: pd.DataFrame) -> str:
    """One or two sentences on the reported target over every origin."""
    rel = relative_wis(scores.loc[scores["target"] == "reported"])
    if rel["model"].nunique() < 2:
        return ""
    rel = rel.set_index("model")
    others = [m for m in rel.index if m != "randomwalk"]
    wis = ", ".join(f"{m} {rel.loc[m, 'rel_wis']:.2f}" for m in others)
    cover = ", ".join(
        f"{m} {rel.loc[m, 'coverage_90']:.2f}" for m in [*others, "randomwalk"]
    )
    origins = scores["origin"].nunique()
    return (
        f"On the counts each later release reported, from {origins} "
        f"origins, mean WIS relative to random walk is {wis}.\n"
        f"90% interval coverage is {cover}.\n"
    )


_METRIC_TABS: Final = (("wis", "WIS", ""), ("log_wis", "Log WIS", "-log-wis"))
"""Score metrics shown as tabs, with their tab label and file suffix."""


def _save_metric_figures(
    fig: Path, stem: str, make: Callable[..., Figure]
) -> None:
    """Save `make(metric=...)` in light and dark for each metric tab."""
    for metric, _, suffix in _METRIC_TABS:
        save_light_dark(
            partial(make, metric=metric), fig / f"{stem}{suffix}.svg"
        )


def _metric_tabs(stem: str, alt: str) -> str:
    """Tabs of one figure per metric, as `_save_metric_figures` saves."""
    tabs = []
    for _, label, suffix in _METRIC_TABS:
        what = alt.replace("WIS", label)
        tabs.append(
            f'=== "{label}"\n\n'
            f"    ![{what}](figures/{stem}{suffix}.svg#only-light)\n"
            f"    ![{what}](figures/{stem}{suffix}-dark.svg#only-dark)\n"
        )
    return "\n".join(tabs)


_RELEASE_MATCH: Final = [
    "origin",
    "truth_vintage",
    "location",
    "target_end_date",
]
"""Columns that identify one forecast scored against one release."""


def _score_chart(
    fig: Path,
    gen: Path,
    scores: pd.DataFrame,
    by: str,
    names: Sequence[str],
) -> None:
    """Write an interactive chart of mean scores by `by`, and its fallback.

    The static WIS figure is shown without JavaScript.
    """
    grp = scores.groupby(["model", by])
    table = grp[list(SCORE_METRICS)].mean()
    table.insert(0, "n", grp.size())
    table = table.reset_index()
    if by == "location":
        size = scores.groupby("location")["truth"].sum()
        order = list(size.sort_values(ascending=False).index)
        table = table.sort_values(
            "location", key=lambda col: col.map(order.index)
        )
    stem = f"scores-wis-{by}"
    save_light_dark(
        partial(plot_scores, scores, by=by, metric="wis", models=names),
        fig / f"{stem}.svg",
    )
    write_score_chart(
        fig / f"scorechart-{by}.json",
        score_chart_data(
            table,
            by,
            SCORE_METRICS,
            names,
            x_label="Horizon" if by == "horizon" else "Place",
            kind="line" if by == "horizon" else "bar",
        ),
    )
    _write(
        gen / f"scorechart-{by}.md",
        score_chart_html(
            f"../figures/scorechart-{by}.json",
            f"Mean scores by {'horizon' if by == 'horizon' else 'place'}",
            static=f"../figures/{stem}.svg",
        ),
    )


def _horizon_table(
    gen: Path,
    reported: pd.DataFrame,
    names: Sequence[str],
) -> None:
    """Write the relative WIS and coverage by horizon table, reported target.

    Only the forecasts random walk also made are shown.
    """
    ours = reported.loc[reported["model"].isin(names)]
    base = ours.loc[ours["model"] == "randomwalk", _RELEASE_MATCH]
    both = ours.merge(base, on=_RELEASE_MATCH)
    if both["model"].nunique() < 2:
        _write(gen / "backtest-summary.md", "")
        return
    table = relative_wis(both, ("horizon",))
    table["model"] = pd.Categorical(table["model"], list(names), ordered=True)
    table = table.sort_values(["model", "horizon"]).reset_index(drop=True)
    origins = ", ".join(sorted(set(both["origin"].astype(str))))
    _write(
        gen / "backtest-summary.md",
        _dropdown(
            "Mean WIS divided by that of random walk, and interval "
            "coverage, on the counts each later release reported, over the "
            f"forecasts both made from the origins {origins}.\n"
            "Horizons after the latest release have no score yet.\n\n"
            + _table(table),
            "Summary table",
        ),
    )


def _forecast_tabs(
    names: Sequence[str],
    labels: Mapping[str, str],
    explorer_src: str | None = None,
) -> str:
    """Forecasts from the real origins, as an explorer and static tabs."""
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
    static = _dropdown(
        "Forecasts from the real origins "
        "([reading the figures](methods/evaluation.md#reading-the-figures))."
        "\n\n" + "\n".join(tabs),
        "Sample trajectories",
    )
    if explorer_src is None:
        return "## Forecasts against later data\n\n" + static
    return (
        "## Forecasts against later data\n\n"
        + explorer_html(explorer_src, "Backtest forecasts", all_rounds=True)
        + "\n"
        + static
    )


def _recovery_section() -> str:
    """Return the recovery section."""
    alt = "Mean WIS by horizon on simulated vintages."
    return (
        "## Recovery on simulated vintages\n\n"
        "Forecasts from vintages of a [simulated outbreak]"
        "(methods/data.md#simulated-vintages), scored against the known "
        "final counts.\n\n"
        f"![{alt}](figures/recovery-wis-horizon.svg#only-light)\n"
        f"![{alt}](figures/recovery-wis-horizon-dark.svg#only-dark)\n"
    )


def _without_baseline(
    rel: pd.DataFrame, baseline: str = "randomwalk"
) -> pd.DataFrame:
    """Drop the baseline's own rows, whose relative scores are all 1."""
    return rel.loc[rel["model"] != baseline].reset_index(drop=True)


def _scenario_scores(path: Path) -> str:
    """Return mean scores on the simulated scenarios."""
    scores = pd.read_csv(path, low_memory=False)
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
                log_wis=("log_wis", "mean"),
                coverage_50=("coverage_50", "mean"),
                coverage_90=("coverage_90", "mean"),
                bias=("bias", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    "wis": "mean WIS",
                    "log_wis": "mean log WIS",
                    "coverage_50": "50% coverage",
                    "coverage_90": "90% coverage",
                    "bias": "mean bias",
                }
            )
        )
        tables.append(f"**Scored against {label}**\n\n" + _table(table))
    return "\n".join(tables)


def _scenario_figure() -> str:
    """Return the scenario figure links."""
    alt = "Mean WIS by horizon on the simulated scenarios."
    return _metric_tabs("scenario-wis-horizon", alt)


def _submission_samplers(
    path: Path = Path("ci/policy.toml"),
) -> dict[str, dict[str, Any]]:
    """Return each fitted candidate's sampler settings from the CI policy."""
    policy = tomllib.loads(path.read_text())
    return {
        c["model"]: dict(c["sampler"])
        for c in policy["candidates"]
        if "sampler" in c
    }


def _backtest_sampler(
    path: Path = Path("config/backtest.toml"),
) -> dict[str, Any]:
    """Return the backtest's sampler settings."""
    return dict(tomllib.loads(path.read_text())["sampler"])


def _settings_text(sampler: Mapping[str, Any]) -> str:
    return (
        f"{int(sampler['num_chains'])} chains of "
        f"{int(sampler['num_warmup'])} warm-up and "
        f"{int(sampler['num_samples'])} draws at a target acceptance of "
        f"{sampler['target_accept']}"
    )


def _sampler_section(
    fits: pd.DataFrame,
    backtest: Mapping[str, Any],
    submission: Mapping[str, Mapping[str, Any]],
) -> str:
    """Give the backtest and submission sampler settings and convergence."""
    keys = ["num_chains", "num_warmup", "num_samples"]
    fits = (
        fits.dropna(subset=keys)
        if set(keys) <= set(fits.columns)
        else fits.iloc[:0]
    )
    if fits.empty:
        return ""
    sub = "".join(
        f"- `{m}`: {_settings_text(s)}\n" for m, s in submission.items()
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
        f"Backtest fits use {_settings_text(backtest)}.\n"
        "Submission fits use\n\n"
        + sub
        + "\n"
        + _dropdown(
            "Columns are as on the "
            "[Implementation](methods/implementation.md#sampling) page.\n\n"
            + _table(table, digits=3),
            "Convergence by model",
        )
    )


def _backtest_page(
    fig: Path,
    gen: Path,
    res: Path,
    out: Path,
    config: BacktestConfig,
    target: pd.DataFrame,
    names: list[str],
) -> None:
    k = config.complete_after_weeks
    scores = _ours(_release_scores(res), names)
    reported = scores.loc[scores["target"] == "reported"]
    for by in ("horizon", "location"):
        _score_chart(fig, gen, reported, by, names)
    _horizon_table(gen, reported, names)
    _write(gen / "backtest-findings.md", _backtest_findings(scores))
    _write(
        gen / "scores-location.md",
        _table(relative_wis(scores, ("target", "location"))),
    )
    save_light_dark(
        partial(plot_scores_by_truth, reported, models=names),
        fig / "scores-by-truth.svg",
    )
    _write(
        gen / "scores-by-truth.md",
        _table(relative_wis(scores, ("target", "origin", "truth_vintage"))),
    )
    weeks = sorted(set(reported["target_end_date"].astype(str)))
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
                        complete_after_weeks=k,
                    ),
                    fig / f"forecasts-{m}-{loc.lower()}.svg",
                )
    labels = _location_names(target)
    explorer_src = None
    if shown:
        quantiles = forecast_quantiles(real_fc.assign(source="ours"))
        quantiles = quantiles.assign(ours=True)
        rounds = real_fc[["vintage", "reference_date"]].drop_duplicates()
        complete = scores.loc[scores["target"] == "complete"]
        write_explorer(
            fig / "explorer-backtest.json",
            explorer_data(
                quantiles,
                target,
                labels,
                k,
                {
                    m: (
                        model_colour(m, names, "light"),
                        model_colour(m, names, "dark"),
                    )
                    for m in shown
                },
                baseline="randomwalk",
                scores=complete.merge(rounds, on="vintage"),
            ),
        )
        explorer_src = "../figures/explorer-backtest.json"
    _write(
        gen / "backtest-forecasts.md",
        _forecast_tabs(
            shown,
            {
                loc: f"{labels.get(loc, loc)} ({loc})"
                for loc in BACKTEST_LOCATIONS
            },
            explorer_src,
        ),
    )
    _write(gen / "lopo-section.md", _lopo_section(res / "lopo-scores.csv"))
    _write(gen / "zone-section.md", _zone_section(res / "zone-scores.csv"))
    _write(gen / "ladder.md", _ladder_page(res / "ladder.csv"))
    _write(
        gen / "development-log.md",
        _development_log(res / "development-log.csv"),
    )
    fits_path = res / "backtest-fits.csv"
    _write(
        gen / "backtest-sampler.md",
        _sampler_section(
            pd.read_csv(fits_path), _backtest_sampler(), _submission_samplers()
        ),
    )
    rec_path = res / "recovery-scores.csv"
    _write(gen / "recovery-section.md", _recovery_section())
    rscores = pd.read_csv(rec_path)
    save_light_dark(
        lambda mode: plot_scores(
            rscores, by="horizon", models=names, mode=mode
        ),
        fig / "recovery-wis-horizon.svg",
    )
    scen_path = res / "scenario-scores.csv.gz"
    _write(gen / "scenario-scores.md", _scenario_scores(scen_path))
    _write(gen / "scenario-figure.md", _scenario_figure())
    sscores = pd.read_csv(scen_path, low_memory=False)
    _save_metric_figures(
        fig,
        "scenario-wis-horizon",
        partial(
            plot_scenarios,
            sscores,
            models=[
                *names,
                *(m for m in sscores["model"].unique() if m not in names),
            ],
        ),
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
    if not [*run.get("scored", []), *run.get("pending", [])]:
        # No vintage after the choice yet: nothing is held out.
        for name in ("heldout-status.md", "heldout-scores.md"):
            _write(gen / name, "")
        _write(gen / "heldout-forecasts.md", "")
        return
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
            sorted(set(scores.loc[scores["origin"] == v, "truth_vintage"]))
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
        f"**{run['latest_vintage']}**.\n\n"
        + _dropdown(_table(status), "Vintages"),
    )
    if scores.empty:
        _write(gen / "heldout-scores.md", "")
    else:
        scores = _with_horizon(scores)
        reported = scores.loc[scores["target"] == "reported"]
        tables = (
            _table(
                relative_wis(
                    scores,
                    ("target", "chosen_model", "origin", "truth_vintage"),
                )
            )
            + "\nBy horizon, over provinces and the national total:\n\n"
            + _table(relative_wis(scores, ("target", "origin", "horizon")))
        )
        alt = "Held-out WIS by horizon."
        make = partial(plot_score_summary, reported, models=names)
        _save_metric_figures(fig, "heldout-scores", make)
        _write(
            gen / "heldout-scores.md",
            _metric_tabs("heldout-scores", alt) + "\n" + _dropdown(tables),
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
            partial(
                plot_forecasts,
                forecasts,
                target,
                loc,
                models=models,
                complete_after_weeks=config.complete_after_weeks,
            ),
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


HUB_METADATA_URL: Final = (
    "https://github.com/InsightNet-US/BDBV-Modeling-Hub/blob/main/"
    "model-metadata/{model}.yml"
)
"""Where the hub keeps each model's metadata file."""


def _teams_table(teams: pd.DataFrame, hub_ids: Mapping[str, str]) -> str:
    """Table of the hub's other models, with a link to each one's metadata.

    Parameters
    ----------
    teams : pandas.DataFrame
        Output of `load_team_metadata`.
    hub_ids : mapping
        Hub ids of our own models, which are left out.
    """
    teams = teams.loc[~teams["model"].isin(set(hub_ids.values()))]
    if teams.empty:
        return "No other team has submitted.\n"
    table = pd.DataFrame(
        {
            "Team": teams["team_name"],
            "Model": teams["model_name"],
            "Hub model id": [f"`{m}`" for m in teams["model"]],
            "Licence": teams["license"],
            "Metadata": [
                f"[{m}.yml]({HUB_METADATA_URL.format(model=m)})"
                for m in teams["model"]
            ],
        }
    )
    return table.to_markdown(index=False) + "\n"


def _overall_table(
    fig: Path,
    scores: pd.DataFrame,
    baseline: str,
    hub_ids: Mapping[str, str],
    colours: Mapping[str, tuple[str, str]],
) -> str:
    """Chart and tabulate every model over all scored rounds, by scale.

    The chart starts with the hub teams, our submitted models and the
    baseline. Our other models can be switched on. `colours` are those of
    the round figures and the explorer, so a model has one colour in all.
    """
    overall = overall_comparison(scores, baseline)
    raw = list(overall["model"])
    overall["model"] = [
        comparison_label(m, "hub" if hub else "real", hub_ids)
        for m, hub in zip(overall["model"], overall["hub"], strict=True)
    ]
    hub = overall.loc[overall["hub"], "model"].drop_duplicates()
    ours = dict(
        zip(
            overall.loc[~overall["hub"], "model"],
            [n for n, h in zip(raw, overall["hub"], strict=True) if not h],
            strict=True,
        )
    )
    registry = [spec.name for spec in list_models()]
    ours = dict(sorted(ours.items(), key=lambda lm: registry.index(lm[1])))
    shown = {comparison_label(m, "real", hub_ids) for m in [*hub_ids, baseline]}
    write_score_chart(
        fig / "scorechart-comparison.json",
        score_chart_data(
            overall,
            "scale",
            OVERALL_METRICS,
            [*ours, *sorted(hub)],
            default_on=[*sorted(hub), *(m for m in ours if m in shown)],
            x_label="Scale",
            kind="bar",
            colours=colours,
        ),
    )
    chart = score_chart_html(
        "../figures/scorechart-comparison.json", "Overall comparison scores"
    )
    have = set(overall["scale"])
    scales = [s for s in ("national", "provinces", "zones") if s in have]
    names = {
        "rel_wis": "rel. WIS",
        "rel_log_wis": "rel. log WIS",
        "n": "n",
        "coverage_50": "50%",
        "coverage_90": "90%",
        "bias": "bias",
    }
    long = overall.melt(
        id_vars=["model", "scale"], value_vars=list(names), var_name="metric"
    )
    long["column"] = long["scale"] + " " + long["metric"].map(names)
    wide = long.pivot(index="model", columns="column", values="value")
    order = [f"{s} {n}" for s in scales for n in names.values()]
    wide = wide[order].sort_values(order[0]).round(2).reset_index()
    wide = wide.astype(object).where(wide.notna(), "")
    missing = sorted({"national", "provinces", "zones"} - have)
    gap = f" There are no scores for {' or '.join(missing)}." if missing else ""
    return (
        chart
        + "\n"
        + "Over every scored round, on the count each later release reported "
        f"and on the forecasts `{baseline}` also made, by scale.{gap}\n"
        "Our other models can be switched on in the key.\n\n" + _table(wide)
    )


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
            "comparison-summary.md",
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
    quantiles = pd.read_csv(cmp_dir / "quantiles.csv.gz")
    hub = quantiles["source"] == "hub"
    colours = comparison_colours(
        {
            comparison_label(m, "real", hub_ids): m
            for m in quantiles.loc[~hub, "model"].unique()
        },
        sorted(set(quantiles.loc[hub, "model"])),
    )
    sent = quantiles.loc[quantiles["source"] == "submitted", "reference_date"]
    rounds = ", ".join(sorted(set(sent.astype(str)))) or "none yet"
    _write(
        gen / "comparison-lead.md",
        "Scores against what each later release reported, from the "
        "release serving each round "
        "([how scoring works]"
        "(methods/evaluation.md#comparison-with-hub-forecasts)).\n"
        "A model shown with a hub id is what we submitted under that id "
        f"for the rounds we submitted ({rounds}), and the model's own "
        "forecast for earlier rounds.\n",
    )
    scores_path = cmp_dir / "scores.csv"
    scores = (
        pd.read_csv(scores_path)
        if scores_path.stat().st_size > 1
        else pd.DataFrame()
    )
    pending = pd.read_csv(cmp_dir / "pending.csv")
    _write(gen / "comparison-summary.md", "")
    parts = []
    if not scores.empty:
        parts.append(
            _overall_table(
                fig,
                scores,
                baseline,
                hub_ids,
                colours,
            )
        )
        table = summarise_comparison(scores, baseline)
        table = _without_baseline(table, baseline)
        table["target"] = table["target"].map(_TARGET_LABELS)
        table["model"] = [
            comparison_label(m, s, hub_ids)
            for m, s in zip(table["model"], table["source"], strict=True)
        ]
        parts.append(
            _dropdown(_table(table), "Scores by round")
            if not table.empty
            else ""
        )
        _comparison_summary(fig, gen, scores, baseline, hub_ids)
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
        _dropdown(_teams_table(teams, hub_ids), "Teams"),
    )
    labelled = quantiles.assign(
        model=[
            comparison_label(m, src, hub_ids)
            for m, src in zip(
                quantiles["model"], quantiles["source"], strict=True
            )
        ],
        ours=quantiles["source"] != "hub",
    )
    hidden = set(
        labelled.loc[
            labelled["ours"] & ~quantiles["model"].isin([*hub_ids, baseline]),
            "model",
        ]
    )
    explorer_scores = None
    if not scores.empty:
        done = scores.loc[scores["target"] == "complete"]
        explorer_scores = done.assign(
            model=[
                comparison_label(m, src, hub_ids)
                for m, src in zip(done["model"], done["source"], strict=True)
            ]
        )
    write_explorer(
        fig / "explorer-comparison.json",
        explorer_data(
            labelled,
            target,
            _location_names(target),
            k,
            colours,
            baseline=_baseline_label(baseline, labelled),
            scores=explorer_scores,
            truth_vintage=truth,
            hidden=hidden,
        ),
    )
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
                complete_after_weeks=k,
                hub_ids=hub_ids,
                shown=[*hub_ids, baseline],
            ),
            fig / f"{stem}.svg",
        )
        alt = f"Forecasts for the {ref} round against the {truth} data."
        tabs.append(
            f'=== "{ref}"\n\n'
            f"    ![{alt}](figures/{stem}.svg#only-light)\n"
            f"    ![{alt}](figures/{stem}-dark.svg#only-dark)\n"
        )
    _write(
        gen / "comparison-figures.md",
        explorer_html("../figures/explorer-comparison.json", "Hub forecasts")
        + "\n"
        + _dropdown("\n".join(tabs), "Static figures by round"),
    )


def _baseline_label(baseline: str, quantiles: pd.DataFrame) -> str:
    """Return the label our baseline has in the comparison."""
    return next(
        (
            m
            for m in quantiles["model"].unique()
            if m.startswith(f"{baseline} (")
        ),
        baseline,
    )


def _comparison_summary(
    fig: Path,
    gen: Path,
    scores: pd.DataFrame,
    baseline: str,
    hub_ids: Mapping[str, str],
) -> None:
    """Write relative WIS and coverage on the counts later releases reported."""
    done = scores.loc[scores["target"] == "reported"]
    labelled = done.assign(
        model=[
            comparison_label(m, src, hub_ids)
            for m, src in zip(done["model"], done["source"], strict=True)
        ]
    )
    base = _baseline_label(baseline, labelled)
    if base not in set(labelled["model"]):
        return
    _save_metric_figures(
        fig,
        "comparison-summary",
        partial(
            plot_score_summary,
            labelled,
            baseline=base,
            models=list(
                dict.fromkeys(
                    comparison_label(m, src, hub_ids)
                    for m, src in zip(
                        scores["model"], scores["source"], strict=True
                    )
                )
            ),
        ),
    )
    alt = f"WIS relative to {baseline} and 90% coverage, by horizon."
    _write(
        gen / "comparison-summary.md", _metric_tabs("comparison-summary", alt)
    )
