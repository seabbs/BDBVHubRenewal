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
import textwrap
import tomllib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Final

import pandas as pd
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
from bdbvhubrenewal.data.bvd import BVD_REPO, bvd_for_vintage
from bdbvhubrenewal.data.dates import reference_date_for_vintage
from bdbvhubrenewal.data.delays import delay_summary
from bdbvhubrenewal.data.simulate import simulate_outbreak
from bdbvhubrenewal.data.tidy import COHERENCE_STREAMS
from bdbvhubrenewal.data.zone_map import BOUNDARIES_PATH, zone_boundaries
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    check_distinct,
    load_config,
    make_tasks,
    run_forecasts,
    truth_changes,
)
from bdbvhubrenewal.evaluation.hubcompare import (
    attribution,
    forecast_quantiles,
    load_comparison_config,
    stored_forecasts,
    summarise_comparison,
)
from bdbvhubrenewal.evaluation.nowcast import (
    BASELINE,
    relative_energy,
    relative_wis,
)
from bdbvhubrenewal.evaluation.scoring import summarise_scores
from bdbvhubrenewal.evaluation.zones import SUMMED, zone_relative_skill
from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ForecastTask,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.renewal import DELAY_SUMMARY_FILE
from bdbvhubrenewal.reporting.explorer import (
    explorer_data,
    explorer_html,
    write_explorer,
)
from bdbvhubrenewal.reporting.nowcast_plots import (
    plot_place_summary,
    plot_relative_overall,
    plot_release_nowcasts,
    plot_truth_changes,
)
from bdbvhubrenewal.reporting.plots import (
    baseline_share,
    comparison_colours,
    comparison_label,
    model_style,
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
    plot_renewal_delay,
    plot_revisions,
    plot_scenarios,
    plot_scores,
    plot_scores_by_truth,
    plot_simulation,
    plot_target,
    save_light_dark,
)
from bdbvhubrenewal.reporting.scorechart import (
    ENERGY_METRICS,
    OVERALL_METRICS,
    write_score_chart,
)
from bdbvhubrenewal.reporting.scoresummary import summary_html, summary_rows
from bdbvhubrenewal.reporting.store_note import write_store_note
from bdbvhubrenewal.reporting.tables import filter_table
from bdbvhubrenewal.reporting.zone_nowcast import (
    ZONE_MODEL,
    place_chart_data,
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
    write_store_note(res, gen)

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

    # Backtest and comparison pages.
    names = [s.name for s in list_models()]
    _release_section(fig, gen, res, real, names)
    _backtest_page(fig, gen, res, out, config, target, names, data_dir)
    _comparison_page(fig, gen, Path(hub_comparison), target)

    # Nowcast page: latest real vintage, every registered model.
    rows = real.loc[real["vintage"] == latest]
    daily = load_daily(data_dir)
    tasks = make_tasks(rows, delays, config, history=real, daily=daily)
    forecasts_now = _nowcast_forecasts(tasks, latest, res)
    _nowcast_page(fig, gen, target, forecasts_now, names, config, latest)
    _zone_nowcast_section(
        fig, gen, data_dir, target, tasks[0], forecasts_now, config, names
    )
    groups = _nowcast_groups(target, sorted(set(forecasts_now["location"])))
    save_light_dark(
        partial(
            plot_delay_check, daily, latest, [*groups["main"], *groups["other"]]
        ),
        fig / "delay-check.svg",
    )
    # Written by `fit-store load` from the stored fits.
    learnt = pd.read_csv(res / DELAY_SUMMARY_FILE)
    fitted = sorted(set(learnt["location"]) - {"pooled"})
    baseline = baseline_share(
        daily, str(learnt["vintage"].max()), [NATIONAL, *fitted]
    )
    save_light_dark(
        partial(plot_renewal_delay, learnt, baseline, _location_names(real)),
        fig / "renewal-delay.svg",
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


def _model_colours(models: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    """Each model's light colour, dark colour and dash, from `model_style`."""
    return {m: model_style(m, models) for m in models}


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
    scores = _ours(_with_horizon(pd.read_csv(paths["scores"])), names)
    quantiles = _ours(pd.read_csv(paths["quantiles"]), names)
    origins = _ours(pd.read_csv(paths["origins"]).fillna(""), names)
    national = scores["location"] == NATIONAL
    scores["pool"] = national.map({True: "national", False: "provinces"})
    zones = _as_submitted(
        pd.read_csv(res / "zone-scores.csv", low_memory=False)
    )
    pooled = pd.concat([scores, zones.assign(pool="zones")])
    pooled["target"] = pooled["target"].map(_TARGET_LABELS)
    energy = _ours(
        pd.concat(
            [
                pd.read_csv(res / "nowcast-energy.csv"),
                pd.read_csv(res / "zone-energy.csv"),
            ]
        ),
        names,
    )
    energy["target"] = energy["target"].map(_TARGET_LABELS)
    joint = relative_energy(energy).rename(columns={"scale": "pool"})
    overall = relative_wis(pooled, ("target", "pool")).merge(
        joint, on=["target", "pool", "model"], how="left"
    )
    by_pool = relative_wis(pooled, ("target", "pool", "origin"))
    scores["target"] = scores["target"].map(_TARGET_LABELS)
    by_place = relative_wis(scores, ("target", "origin", "location"))
    reported = overall.loc[overall["target"] == _TARGET_LABELS["reported"]]
    ours = [*names, BASELINE]
    colours = _model_colours(ours)
    rep = pooled.loc[pooled["target"] == _TARGET_LABELS["reported"]]
    # Rounds are named by their reference date, as in the explorer.
    rep = rep.assign(
        origin=rep["origin"].map(lambda v: str(reference_date_for_vintage(v)))
    )
    write_score_chart(
        fig / "scoresummary-backtest.json",
        summary_rows(
            rep,
            ours,
            colours,
            _location_names(real),
            default_on=[m for m in ours if m != "randomwalk"],
            energy=energy.loc[energy["target"] == _TARGET_LABELS["reported"]],
        ),
    )
    save_light_dark(
        partial(
            plot_place_summary,
            relative_wis(rep, ("pool", "location")),
            ours,
            colours,
            _location_names(real),
        ),
        fig / "scores-place.svg",
    )
    _write(
        gen / "scoresummary-horizon.md",
        summary_html(
            "../figures/scoresummary-backtest.json",
            "Scores by horizon",
            view="horizon",
            static="../figures/scores-horizon.svg",
        ),
    )
    _write(
        gen / "scores-location.md",
        filter_table(
            _table(relative_wis(pooled, ("target", "pool", "location"))),
            ["target", "pool", "model"],
        ),
    )
    _write(
        gen / "scoresummary-place.md",
        summary_html(
            "../figures/scoresummary-backtest.json",
            "Scores by place",
            view="place",
            static="../figures/scores-place.svg",
        ),
    )
    save_light_dark(
        partial(
            plot_relative_overall,
            reported,
            [n for n in names if n in set(reported["model"])],
            metrics={**OVERALL_METRICS, **ENERGY_METRICS},
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
        "[metrics](methods/evaluation.md#scores)).\n"
        + _scored_on(rep)
        + "The [energy score](methods/evaluation.md#energy-score) scores each "
        "scale's places and weeks as one forecast per origin and release, "
        "so its $n$ counts those.\n\n"
        + summary_html(
            "../figures/scoresummary-backtest.json",
            "Overall scores by scale",
            static="../figures/relative-overall.svg",
        )
        + "\n"
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


def _scored_on(reported: pd.DataFrame, baseline: str = "randomwalk") -> str:
    """Say how many origin and release pairs and forecasts are scored."""
    pairs = reported[["origin", "truth_vintage"]].drop_duplicates()
    n = reported.loc[reported["model"] == baseline].groupby("pool").size()
    scales = {
        "national": "national",
        "provinces": "province",
        "zones": "health zone",
    }
    counts = [f"{n[p]} {label}" for p, label in scales.items() if p in n]
    listed = ", ".join(counts[:-1]) + " and " if len(counts) > 1 else ""
    return (
        f"Scored on {len(pairs)} origin and release pairs.\n"
        f"{baseline} has {listed}{counts[-1]} forecasts.\n"
        "The renewal models are not scored at health zones.\n"
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


def _zone_names(data_dir: Path | str) -> dict[str, str]:
    """Each health zone's name, as the hub's time series gives it."""
    zones = load_health_zones(data_dir)
    return dict(zip(zones["location"], zones["location_name"], strict=True))


def _zone_nowcast_section(
    fig: Path,
    gen: Path,
    data_dir: Path | str,
    target: pd.DataFrame,
    task: ForecastTask,
    forecasts: pd.DataFrame,
    config: BacktestConfig,
    names: list[str],
) -> None:
    """Write the Nowcast page's picker and health-zone section.

    The picker covers the country, provinces and zones at the latest
    release. The random walk models are run on the zones here. The fitted
    models' samples are taken from the stored forecasts of the comparison,
    as for the hub rounds, where there are some for the release.
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
    stored = stored.assign(
        vintage=task.vintage,
        horizon=(
            pd.to_datetime(stored["target_end_date"]) - pd.Timestamp(ref)
        ).dt.days
        // 7,
    )
    at_release = prov.loc[prov["model"].isin(quick)]
    national = forecasts.loc[
        (forecasts["vintage"] == task.vintage)
        & (forecasts["location"] == NATIONAL)
        & forecasts["model"].isin(quick)
    ]
    every = pd.concat([national, at_release, *ran, stored], ignore_index=True)
    if "as_of" in every:
        every = every.loc[every["as_of"].isna()]
    zone_rows = every["location"].isin(zones)
    samples = every.loc[zone_rows & (every["model"] == ZONE_MODEL)]
    labels = _location_names(target)
    start = pd.Timestamp(task.reference_date) - pd.Timedelta(weeks=12)
    counts = zone_counts(
        daily_zones, task.vintage, start.strftime("%Y-%m-%d"), labels
    )
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
    real = target.loc[
        (target["kind"] == "real") & (target["vintage"] == task.vintage)
    ]
    above = real.loc[
        pd.to_datetime(real["target_end_date"]) >= start,
        ["location", "target_end_date", "total"],
    ]
    cases = above.groupby("location")["total"].transform("sum")
    above = above.loc[(cases > 0) | (above["location"] == NATIONAL)]
    write_explorer(
        fig / "nowcastchart.json",
        place_chart_data(
            every,
            pd.concat(
                [above, counts[["location", "target_end_date", "total"]]],
                ignore_index=True,
            ),
            {**labels, **_zone_names(data_dir)},
            _model_colours(names),
            config.complete_after_weeks,
        ),
    )
    _write(
        gen / "nowcast-picker.md",
        "## By place\n\n"
        f"Each model's nowcast and forecast at the {task.vintage} release, "
        "for the country and every province and health zone with a "
        "forecast or with cases in the 12 weeks before the reference date "
        "([Health zones](methods/evaluation.md#health-zones)).\n"
        "The latest nowcast has no score yet.\n"
        "Scores by origin are on the [Backtest](backtest.md) page.\n\n"
        '<div class="nowcastchart" data-src="../figures/nowcastchart.json" '
        'role="group" aria-label="Nowcast and forecast by place">\n'
        "<noscript>The nowcast chart needs JavaScript.</noscript>\n"
        "</div>\n",
    )
    release = bvd_for_vintage(task.vintage, Path(data_dir) / "bvd")
    if release is None or release.choice is None:
        raise ValueError(f"No BVDOutbreakSize release for {task.vintage}.")
    sources = json.loads((Path(data_dir) / "sources.json").read_text())
    write_explorer(
        fig / "zone-map.geojson",
        zone_boundaries(release.choice.tag, sources["commit"]),
    )
    _write(
        gen / "zone-review.md",
        "## Zone review\n\n"
        "One row per place with a forecast from the chosen model.\n"
        "Recent is the mean count of the last two complete weeks.\n"
        "Each week column gives the median and 90% interval.\n"
        "Ratio is log((median + 1) / (recent + 1)) for the chosen week.\n"
        "A ratio of 0.69 is double the recent count and -0.69 is half.\n"
        "Click a place to show it in the By place chart.\n"
        "The map colours each zone by the same ratio.\n"
        "Zone boundaries are from [BVDOutbreakSize]"
        f"(https://github.com/{BVD_REPO}/blob/{release.choice.tag}/"
        f"{BOUNDARIES_PATH}) at {release.choice.tag}, from the Ministry of "
        "Health boundaries on the [Humanitarian Data Exchange]"
        "(https://data.humdata.org/dataset/drc-health-data).\n\n"
        '<div class="zonereview" data-src="../figures/nowcastchart.json" '
        'data-map="../figures/zone-map.geojson" '
        'role="group" aria-label="Zone review">\n'
        "<noscript>The zone review needs JavaScript.</noscript>\n"
        "</div>\n",
    )
    _write(
        gen / "zone-nowcast.md",
        "## Health zones\n\n"
        + _dropdown(
            f"`{ZONE_MODEL}` by province.\n\n" + "\n".join(tabs),
            "Zone nowcasts by province",
        )
        + "\n"
        + _dropdown(
            _table(zone_table(samples, counts)),
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


def _zone_truth_changes(
    fig: Path, data_dir: Path | str, target: pd.DataFrame
) -> str:
    """Chart each province's zone counts by release, a tab per province.

    Each release's zone counts are taken over the latest release's last
    16 weeks, with 0 for a week with no onsets.
    """
    daily_zones = load_daily_zones(data_dir)
    real = target.loc[target["kind"] == "real"]
    names = _location_names(target)
    end = pd.Timestamp(real["target_end_date"].max())
    start = (end - pd.Timedelta(weeks=16)).strftime("%Y-%m-%d")
    zones = pd.concat(
        [
            zone_counts(daily_zones, v, start, names)
            for v in sorted(set(real["vintage"].astype(str)))
            if (daily_zones["vintage"].astype(str) == v).any()
        ],
        ignore_index=True,
    )
    tabs = []
    by_prov = zones.loc[zones["vintage"] == zones["vintage"].max()]
    order = by_prov.groupby("province")["total"].sum()
    for prov in order.sort_values(ascending=False, kind="stable").index:
        path = fig / f"truth-changes-{str(prov).lower()}.svg"
        save_light_dark(
            partial(
                plot_truth_changes,
                zones.loc[zones["province"] == prov],
                {},
            ),
            path,
        )
        label = names.get(str(prov), str(prov))
        alt = f"Zone counts in {label} by onset week in each release."
        tabs.append(
            f'=== "{label}"\n\n'
            f"    ![{alt}](figures/{path.name}#only-light)\n"
            f"    ![{alt}](figures/{path.stem}-dark.svg#only-dark)\n"
        )
    return "\n".join(tabs)


def _as_submitted(zones: pd.DataFrame) -> pd.DataFrame:
    """Keep the zone rows made to sum to the provinces, as submitted.

    They are named by their model, so `randomwalk` is the summed
    baseline.
    """
    summed = zones.loc[zones["model"].str.endswith(SUMMED)]
    return summed.assign(model=summed["model"].str.removesuffix(SUMMED))


def _zone_section(path: Path, energy_path: Path) -> str:
    """Return the health-zone table, closed, in the submitted form."""
    scores = _as_submitted(pd.read_csv(path, low_memory=False))
    joint = relative_energy(
        _as_submitted(pd.read_csv(energy_path)), by=("scale",)
    )
    cols = ["n", "wis", "log_wis", "coverage_50", "coverage_90", "bias"]
    energy = ["n_energy", "rel_energy", "rel_log_energy"]
    table = (
        summarise_scores(scores, by=("model",))
        .merge(
            zone_relative_skill(scores)[["model", "relative_wis"]], on="model"
        )
        .merge(joint[["model", *energy]], on="model")
    )
    return _dropdown(
        _table(table[["model", *cols, "relative_wis", *energy]]),
        "Health zones, over every origin",
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


_METRIC_TABS: Final = (
    ("wis", "WIS", ""),
    ("log_wis", "WIS on the log scale", "-log-wis"),
)
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
        return "## Forecasts by place\n\n" + static
    return (
        "## Forecasts by place\n\n"
        + explorer_html(
            explorer_src,
            "Backtest forecasts",
            all_rounds=True,
            summary="../figures/scoresummary-backtest.json",
        )
        + "\n"
        + static
    )


def _energy_means(energy: pd.DataFrame, by: Sequence[str]) -> str:
    """Tabulate the mean energy scores, with $n$ the vectors scored."""
    table = (
        energy.groupby([*by, "model"], sort=False)
        .agg(
            n=("energy", "size"),
            energy=("energy", "mean"),
            log_energy=("log_energy", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "energy": "mean energy score",
                "log_energy": "mean energy score on the log scale",
            }
        )
    )
    return (
        "Mean [energy score](methods/evaluation.md#energy-score), each "
        "scale's places and weeks scored as one forecast per vintage.\n\n"
        + _table(table)
    )


def _recovery_section(src: str, energy: pd.DataFrame) -> str:
    """Return the recovery section, with the summaries of `src`."""
    alt = "Mean WIS by horizon on simulated vintages."
    views = _summary_views(src, "(simulated vintages)")
    return (
        "## Recovery on simulated vintages\n\n"
        "Forecasts from vintages of a [simulated outbreak]"
        "(methods/data.md#simulated-vintages), scored against the known "
        "final counts.\n\n"
        + "### Overall\n\n"
        + views["overall"]
        + "\n### By horizon\n\n"
        + views["horizon"]
        + "\n### By place\n\n"
        + views["place"]
        + "\n"
        + _dropdown(
            f"![{alt}](figures/recovery-wis-horizon.svg#only-light)\n"
            f"![{alt}](figures/recovery-wis-horizon-dark.svg#only-dark)\n\n"
            + _energy_means(energy, ["scale"]),
            "Static figure and energy scores",
        )
    )


def _without_baseline(
    rel: pd.DataFrame, baseline: str = "randomwalk"
) -> pd.DataFrame:
    """Drop the baseline's own rows, whose relative scores are all 1."""
    return rel.loc[rel["model"] != baseline].reset_index(drop=True)


def _scenario_scores(path: Path, energy_path: Path) -> str:
    """Return mean scores on the simulated scenarios."""
    scores = pd.read_csv(path, low_memory=False)
    energy = pd.read_csv(energy_path)
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
                    "log_wis": "mean WIS on the log scale",
                    "coverage_50": "50% coverage",
                    "coverage_90": "90% coverage",
                    "bias": "mean bias",
                }
            )
        )
        tables.append(f"**Scored against {label}**\n\n" + _table(table))
    final = energy["truth_view"] == "final"
    for label, keep in (
        ("final data, complete weeks", final),
        ("the data as of 2 weeks later", ~final),
    ):
        tables.append(
            f"**Energy score against {label}**\n\n"
            + _energy_means(energy.loc[keep], ["scenario", "scale"])
        )
    return "\n".join(tables)


def _scenario_figure() -> str:
    """Return the scenario figure links."""
    alt = "Mean WIS by horizon on the simulated scenarios."
    return _metric_tabs("scenario-wis-horizon", alt)


_SCENARIOS: Final = {
    "calibrated_falling": "Rt falls",
    "calibrated_flat": "Rt holds",
    "calibrated_rising": "Rt rises",
}
"""Simulated scenarios, as their tabs are labelled."""


def _simulated_summary(
    fig: Path,
    stem: str,
    scores: pd.DataFrame,
    energy: pd.DataFrame,
    names: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    """Write the score summary of simulated forecasts and return its URL.

    Rounds are named by their reference date, as on the other pages.
    """
    models = [m for m in names if m in set(scores["model"])]
    write_score_chart(
        fig / f"scoresummary-{stem}.json",
        summary_rows(
            scores.assign(origin=scores["reference_date"]),
            models,
            _model_colours(models),
            labels,
            default_on=[m for m in models if m != "randomwalk"],
            energy=energy.assign(origin=energy["reference_date"]),
        ),
    )
    return f"../figures/scoresummary-{stem}.json"


def _summary_views(src: str, what: str) -> dict[str, str]:
    """Return the Overall, By horizon and By place summaries of `src`."""
    return {
        view: summary_html(src, f"{label} {what}", view=view)
        for view, label in (
            ("overall", "Overall scores"),
            ("horizon", "Scores by horizon"),
            ("place", "Scores by place"),
        )
    }


def _scenario_section(
    fig: Path,
    res: Path,
    names: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    """Return the scenario summaries and explorer, a tab per scenario.

    Forecasts are scored against the final simulated counts of complete
    weeks, which are also the explorer's data, so every week is complete.
    """
    scores = pd.read_csv(res / "scenario-scores.csv.gz", low_memory=False)
    scores = scores.loc[(scores["truth_view"] == "final") & scores["complete"]]
    energy = pd.read_csv(res / "scenario-energy.csv")
    energy = energy.loc[energy["truth_view"] == "final"]
    samples = pd.read_csv(
        res / "scenario-forecasts.csv.gz", dtype={"as_of": object}
    )
    sections: dict[str, list[str]] = {
        "Overall": [],
        "By horizon": [],
        "By place": [],
        "Forecasts by place": [],
    }
    for key, title in _SCENARIOS.items():
        own = scores.loc[scores["scenario"] == key]
        src = _simulated_summary(
            fig,
            f"scenario-{key}",
            own,
            energy.loc[energy["scenario"] == key],
            names,
            labels,
        )
        views = _summary_views(src, f"({title})")
        drawn = samples.loc[samples["scenario"] == key]
        shown = [m for m in names if m in set(drawn["model"])]
        truth = own[["truth_vintage", "location", "target_end_date", "truth"]]
        target = (
            truth.drop_duplicates()
            .rename(columns={"truth_vintage": "vintage", "truth": "total"})
            .assign(kind="real")
        )
        write_explorer(
            fig / f"explorer-scenario-{key}.json",
            explorer_data(
                forecast_quantiles(drawn.assign(source="ours")).assign(
                    ours=True
                ),
                target,
                labels,
                0,
                _model_colours(shown),
                samples=drawn,
            ),
        )
        explorer = explorer_html(
            f"../figures/explorer-scenario-{key}.json",
            f"Forecasts, {title}",
            all_rounds=True,
            summary=src,
        )
        for section, html in zip(
            sections, (*views.values(), explorer), strict=True
        ):
            body = textwrap.indent(html, "    ")
            sections[section].append(f'=== "{title}"\n\n{body}')
    return "\n".join(
        f"### {section}\n\n" + "\n".join(tabs)
        for section, tabs in sections.items()
    )


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
    data_dir: Path | str,
) -> None:
    k = config.complete_after_weeks
    scores = _ours(_release_scores(res), names)
    reported = scores.loc[scores["target"] == "reported"]
    save_light_dark(
        partial(
            plot_scores, reported, by="horizon", metric="wis", models=names
        ),
        fig / "scores-horizon.svg",
    )
    _horizon_table(gen, reported, names)
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
    save_light_dark(
        partial(
            plot_truth_changes,
            target,
            _location_names(target),
        ),
        fig / "truth-changes.svg",
    )
    alt = (
        "Weekly counts by onset week in each release, and the change from "
        "the previous release, by place."
    )
    _write(
        gen / "truth-changes.md",
        f"![{alt}](figures/truth-changes.svg#only-light)\n"
        f"![{alt}](figures/truth-changes-dark.svg#only-dark)\n\n"
        "Health zones, by province.\n\n"
        + _zone_truth_changes(fig, data_dir, target)
        + "\n"
        + _dropdown(_table(wide.reset_index()), "National counts by release"),
    )
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
        # The zone backtest's forecasts, as submitted, join the explorer.
        zone_q = pd.read_csv(res / "zone-quantiles.csv")
        zone_draws = pd.read_csv(res / "zone-draws.csv.gz")
        quantiles = pd.concat(
            [
                forecast_quantiles(real_fc.assign(source="ours")),
                zone_q.loc[zone_q["model"].isin(shown)],
            ],
            ignore_index=True,
        ).assign(ours=True)
        # The latest release's zone counts are the zones' data.
        latest = str(target.loc[target["kind"] == "real", "vintage"].max())
        zone_data = zone_counts(
            load_daily_zones(data_dir),
            latest,
            str(zone_q["target_end_date"].min()),
            labels,
        )
        write_explorer(
            fig / "explorer-backtest.json",
            explorer_data(
                quantiles,
                pd.concat([target, zone_data], ignore_index=True),
                {
                    **labels,
                    **dict(
                        zip(
                            zone_data["location"],
                            zone_data["location_name"],
                            strict=True,
                        )
                    ),
                },
                k,
                {m: model_style(m, names) for m in shown},
                samples=pd.concat(
                    [real_fc, zone_draws.loc[zone_draws["model"].isin(shown)]],
                    ignore_index=True,
                ),
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
    _write(
        gen / "zone-section.md",
        _zone_section(res / "zone-scores.csv", res / "zone-energy.csv"),
    )
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
    rscores = pd.read_csv(rec_path)
    renergy = pd.read_csv(res / "recovery-energy.csv")
    _write(
        gen / "recovery-section.md",
        _recovery_section(
            _simulated_summary(
                fig, "recovery", rscores, renergy, names, labels
            ),
            renergy,
        ),
    )
    save_light_dark(
        lambda mode: plot_scores(
            rscores, by="horizon", models=names, mode=mode
        ),
        fig / "recovery-wis-horizon.svg",
    )
    scen_path = res / "scenario-scores.csv.gz"
    _write(
        gen / "scenario-scores.md",
        _scenario_scores(scen_path, res / "scenario-energy.csv"),
    )
    _write(gen / "scenario-figure.md", _scenario_figure())
    _write(
        gen / "scenario-summaries.md",
        _scenario_section(fig, res, names, labels),
    )
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
            "comparison-horizon.md",
            "comparison-place.md",
            "comparison-rounds.md",
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
    quantiles = pd.read_csv(cmp_dir / "quantiles.csv.gz")
    source = quantiles["source"]
    registry = [spec.name for spec in list_models()]
    built = sorted(
        set(quantiles.loc[~source.isin(["hub", "submitted"]), "model"]),
        key=registry.index,
    )
    submitted = sorted(set(quantiles.loc[source == "submitted", "model"]))
    submits = {hub_id: m for m, hub_id in hub_ids.items()}
    colours = comparison_colours(
        {comparison_label(m, "real"): m for m in built},
        {comparison_label(h, "submitted"): submits[h] for h in submitted},
        sorted(set(quantiles.loc[source == "hub", "model"])),
    )
    on = [
        *(comparison_label(m, "submitted") for m in submitted),
        *(comparison_label(m, "real") for m in hub_ids if m in built),
    ]
    _write(
        gen / "comparison-lead.md",
        "Scores on the weeks the latest release reports in full, from "
        "the release serving each round "
        "([how scoring works]"
        "(methods/evaluation.md#comparison-with-hub-forecasts)).\n"
        "A series marked submitted is the file we sent to a round, and one "
        "marked main build is the current main branch's forecast for every "
        "round, so a round we did not submit to shows only the main "
        "build.\n",
    )
    scores_path = cmp_dir / "scores.csv"
    scores = (
        pd.read_csv(scores_path)
        if scores_path.stat().st_size > 1
        else pd.DataFrame()
    )
    pending = pd.read_csv(cmp_dir / "pending.csv")
    _comparison_summary(
        fig,
        gen,
        scores,
        pd.read_csv(cmp_dir / "energy.csv"),
        baseline,
        on,
        colours,
        _location_names(target),
    )
    parts = []
    if not scores.empty:
        table = summarise_comparison(scores, baseline)
        table = _without_baseline(table, baseline)
        table["target"] = table["target"].map(_TARGET_LABELS)
        table["model"] = [
            comparison_label(m, s)
            for m, s in zip(table["model"], table["source"], strict=True)
        ]
        parts.append(
            _dropdown(_table(table), "Scores by round")
            if not table.empty
            else ""
        )
    parts += _missing_builds(quantiles, hub_ids)
    for row in pending.itertuples(index=False):
        parts.append(
            f"The {row.reference_date} round was made from the "
            f"{row.vintage} data, so it is scored once later data are "
            "published.\n"
        )
    _write(
        gen / "comparison-rounds.md",
        "\n".join(p for p in parts if p) or "No round has a score yet.\n",
    )
    teams = pd.read_csv(cmp_dir / "teams.csv")
    _write(
        gen / "comparison-teams.md",
        _dropdown(_teams_table(teams, hub_ids), "Teams"),
    )
    labelled = _label_comparison(quantiles).assign(ours=source != "hub")
    hidden = set(labelled.loc[labelled["ours"], "model"]) - set(on)
    write_explorer(
        fig / "explorer-comparison.json",
        explorer_data(
            labelled,
            target,
            _location_names(target),
            k,
            colours,
            truth_vintage=truth,
            hidden=hidden,
            samples=_label_comparison(pd.read_csv(cmp_dir / "samples.csv.gz")),
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
                colours=colours,
                shown=[*on, comparison_label(baseline, "real")],
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
        explorer_html(
            "../figures/explorer-comparison.json",
            "Hub forecasts",
            all_rounds=True,
            summary="../figures/scoresummary-comparison.json",
        )
        + "\n"
        + _dropdown("\n".join(tabs), "Static figures by round"),
    )


def _comparison_summary(
    fig: Path,
    gen: Path,
    scores: pd.DataFrame,
    energy: pd.DataFrame,
    baseline: str,
    on: Sequence[str],
    colours: Mapping[str, tuple[str, str, str]],
    names: Mapping[str, str],
) -> None:
    """Write the Overall, By horizon and By place summaries.

    They are the Backtest page's summaries, on the complete weeks of the
    hub rounds. The hub teams and our series in `on` are ticked at first,
    and randomwalk is the reference for relative scores. Our series come
    in the order of `colours`.
    """
    if scores.empty:
        for name in (
            "comparison-scores.md",
            "comparison-horizon.md",
            "comparison-place.md",
        ):
            _write(gen / name, "No round has a score yet.\n")
        return
    done = _label_comparison(scores.loc[scores["target"] == "complete"])
    rows = done.assign(
        kind="real",
        vintage=done["origin"],
        origin=done["reference_date"],
        pool=done["location"]
        .str.count("-")
        .map({0: "national", 1: "provinces", 2: "zones"}),
    )
    base = _baseline_label(baseline, rows)
    hub = sorted(set(rows.loc[rows["source"] == "hub", "model"]))
    have = set(rows.loc[rows["source"] != "hub", "model"])
    ours = [label for label in colours if label in have]
    models = [*ours, *hub]
    write_score_chart(
        fig / "scoresummary-comparison.json",
        summary_rows(
            rows,
            models,
            colours,
            names,
            baseline=base,
            default_on=[*hub, *(m for m in ours if m in on)],
            energy=_label_comparison(
                energy.loc[energy["target"] == "complete"]
            ).assign(origin=lambda e: e["reference_date"]),
        ),
    )
    overall = relative_wis(rows, ("pool",), baseline=base)
    save_light_dark(
        partial(
            plot_relative_overall,
            overall,
            models,
            base,
            colours=colours,
        ),
        fig / "comparison-overall.svg",
    )
    _write(
        gen / "comparison-scores.md",
        "Scores on the complete weeks "
        "([metrics](methods/evaluation.md#scores)).\n\n"
        + summary_html(
            "../figures/scoresummary-comparison.json",
            "Overall scores by scale",
            static="../figures/comparison-overall.svg",
        )
        + "\n"
        + _dropdown(
            filter_table(_table(overall), ["pool", "model"]),
            "Scores by scale, over every round",
        ),
    )
    save_light_dark(
        partial(
            plot_scores,
            rows,
            by="horizon",
            metric="wis",
            models=models,
            colours=colours,
        ),
        fig / "comparison-horizon.svg",
    )
    _write(
        gen / "comparison-horizon.md",
        summary_html(
            "../figures/scoresummary-comparison.json",
            "Scores by horizon",
            view="horizon",
            static="../figures/comparison-horizon.svg",
        ),
    )
    save_light_dark(
        partial(
            plot_place_summary,
            relative_wis(rows, ("pool", "location"), baseline=base),
            models,
            colours,
            names,
        ),
        fig / "comparison-place.svg",
    )
    _write(
        gen / "comparison-place.md",
        summary_html(
            "../figures/scoresummary-comparison.json",
            "Scores by place",
            view="place",
            static="../figures/comparison-place.svg",
        ),
    )


def _label_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Name each row's model as the comparison shows it."""
    return frame.assign(
        model=[
            comparison_label(m, src)
            for m, src in zip(frame["model"], frame["source"], strict=True)
        ]
    )


def _missing_builds(
    quantiles: pd.DataFrame, hub_ids: Mapping[str, str]
) -> list[str]:
    """Say which submitted rounds have no main build of the same model yet.

    Their main build's n leaves those rounds out.
    """
    lines = []
    for name, hub_id in hub_ids.items():
        rounds = quantiles.loc[quantiles["model"] == hub_id, "reference_date"]
        built = quantiles.loc[
            (quantiles["model"] == name) & (quantiles["source"] != "hub"),
            "reference_date",
        ]
        for ref in sorted(set(rounds.astype(str)) - set(built.astype(str))):
            lines.append(
                f"`{comparison_label(name, 'real')}` has no forecast for the "
                f"{ref} round yet, so its n leaves that round out until a "
                "backtest or held-out run covers it.\n"
            )
    return lines


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
