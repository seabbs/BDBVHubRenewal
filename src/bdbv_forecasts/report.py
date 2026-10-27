# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Build the docs figures and tables from the data and backtest results.

`make_report` writes light and dark SVGs to `docs/figures/` and Markdown
tables to `docs/generated/`, which the docs pages include.
Nothing it writes is committed.
Models come from the registry, so a newly registered model appears on the
Backtest and Nowcast pages without changes here.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from .backtest import load_config, make_tasks, run_forecasts
from .data import load_delays, load_target, newly_missing
from .delays import delay_summary
from .models.base import NATIONAL, get_model, list_models
from .plots import (
    plot_delay_distribution,
    plot_delays_over_time,
    plot_forecasts,
    plot_imputed_share,
    plot_newly_missing,
    plot_nowcast,
    plot_reconstruction,
    plot_relative_skill,
    plot_revisions,
    plot_scores,
    plot_simulation,
    plot_target,
    save_light_dark,
)
from .scoring import relative_skill, summarise_scores
from .simulate import simulate_outbreak
from .vintages import compare_reconstruction, reconstruct_vintages

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
    output: Path | str = "output/backtest",
    config_path: Path | str = "config/backtest.toml",
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

    # Backtest page.
    names = [s.name for s in list_models()]
    dev_path = res / "backtest-scores.csv"
    if dev_path.exists():
        scores = pd.read_csv(dev_path)
        for by in ("horizon", "location"):
            for metric in ("wis", "log_wis", "coverage_90", "bias"):
                save_light_dark(
                    partial(
                        plot_scores, scores, by=by, metric=metric, models=names
                    ),
                    fig / f"scores-{metric.replace('_', '-')}-{by}.svg",
                )
            save_light_dark(
                partial(plot_relative_skill, scores, by=by, models=names),
                fig / f"relative-wis-{by}.svg",
            )
        overall = summarise_scores(scores, by=("model", "kind"))
        _write(gen / "scores-overall.md", _table(overall))
        rel = relative_skill(scores, by=("kind",))
        _write(gen / "relative-skill.md", _table(rel))
        by_loc = summarise_scores(scores, by=("model", "kind", "location"))
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
    fc_path = out / "forecasts.csv.gz"
    if fc_path.exists():
        forecasts = pd.read_csv(fc_path)
        rec_all = reconstruct_vintages(
            real, delays, list(config.reconstructed_dates)
        )
        both = pd.concat([target, rec_all], ignore_index=True)
        for loc in (NATIONAL, "CD-IT", "CD-NK"):
            slug = loc.lower()
            save_light_dark(
                partial(
                    plot_forecasts,
                    forecasts,
                    both,
                    loc,
                    models=names,
                    kind="real",
                ),
                fig / f"forecasts-real-{slug}.svg",
            )
            save_light_dark(
                partial(
                    plot_forecasts,
                    forecasts,
                    both,
                    loc,
                    models=names,
                    kind="reconstructed",
                ),
                fig / f"forecasts-reconstructed-{slug}.svg",
            )
    rec_path = res / "recovery-scores.csv"
    if rec_path.exists():
        rscores = pd.read_csv(rec_path)
        save_light_dark(
            lambda mode: plot_scores(
                rscores, by="horizon", models=names, mode=mode
            ),
            fig / "recovery-wis-horizon.svg",
        )
    heldout = res / "heldout-scores.csv"
    if heldout.exists():
        hscores = pd.read_csv(heldout)
        _write(
            gen / "heldout.md",
            _table(summarise_scores(hscores, by=("model", "horizon"))),
        )
    else:
        _write(gen / "heldout.md", "No held-out evaluation has been run yet.\n")

    # Nowcast page: latest real vintage, every registered model.
    rows = real.loc[real["vintage"] == latest]
    tasks = make_tasks(rows, delays, config)
    forecasts_now = run_forecasts([get_model(n) for n in names], tasks)
    locs = [NATIONAL, *tasks[0].locations]
    for loc in locs:
        save_light_dark(
            partial(
                plot_nowcast,
                loc,
                vintage=latest,
                models=names,
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
        f"{', '.join(f'`{n}`' for n in names)}.\n",
    )
