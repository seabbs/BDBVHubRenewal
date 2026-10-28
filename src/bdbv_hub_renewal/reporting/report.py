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
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from bdbv_hub_renewal.data import (
    load_delays,
    load_target,
    national_coherence,
    newly_missing,
)
from bdbv_hub_renewal.data.dates import reference_date_for_vintage
from bdbv_hub_renewal.data.delays import delay_summary
from bdbv_hub_renewal.data.reconstruct import (
    compare_reconstruction,
    reconstruct_vintages,
)
from bdbv_hub_renewal.data.simulate import simulate_outbreak
from bdbv_hub_renewal.data.tidy import COHERENCE_STREAMS
from bdbv_hub_renewal.evaluation.backtest import (
    BacktestConfig,
    load_config,
    make_tasks,
    run_forecasts,
    truth_changes,
)
from bdbv_hub_renewal.evaluation.heldout import load_model_choices
from bdbv_hub_renewal.evaluation.hubcompare import (
    attribution,
    display_names,
    load_comparison_config,
)
from bdbv_hub_renewal.evaluation.scoring import relative_skill, summarise_scores
from bdbv_hub_renewal.forecasters.base import (
    NATIONAL,
    ForecastTask,
    get_model,
    list_models,
)
from bdbv_hub_renewal.reporting.plots import (
    plot_comparison,
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
    hub_comparison: Path | str = "output/hub-comparison",
    comparison_config: Path | str = "config/hub-comparison.toml",
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
        Output of `bdbv-hub-renewal compare-hub`, if it has been run.
    comparison_config : pathlib.Path or str
        Hub comparison settings, for the display mode.
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
    _comparison_page(
        fig / "comparison",
        gen,
        Path(hub_comparison),
        load_comparison_config(comparison_config).display,
        target,
    )

    # Nowcast page: latest real vintage, every registered model.
    rows = real.loc[real["vintage"] == latest]
    tasks = make_tasks(rows, delays, config)
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


def _persistence_bias(scores: pd.DataFrame) -> str:
    pers = scores.loc[scores["model"] == "persistence"]
    real = pers.loc[pers["kind"] == "real"]
    if real.empty:
        return ""
    rec = pers.loc[pers["kind"] == "reconstructed"]
    by_h = real.groupby("horizon")[["bias", "coverage_90"]].mean()
    worst = by_h.sort_values("bias").head(3)
    hs = sorted(int(h) for h in worst.index)
    lo, hi = worst["coverage_90"].min(), worst["coverage_90"].max()
    rec_text = (
        f" ({rec['bias'].mean():.2f} on reconstructed vintages)"
        if not rec.empty
        else ""
    )
    return (
        f"On real vintages persistence has a mean bias of "
        f"{real['bias'].mean():.2f}{rec_text}.\n"
        f"It is most negative at horizons "
        f"{', '.join(str(h) for h in hs[:-1])} and {hs[-1]}, where 90% "
        f"coverage is {lo:.2f} to {hi:.2f}.\n"
    )


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
        table = ladder[cols].round(3).to_markdown(index=False, missingval="")
        parts.append(f"## {title}\n\n{lead}\n\n{table}\n")
    return "\n".join(parts)


def _lopo_section(path: Path) -> str:
    """Return the leave-one-province-out section, or nothing without results."""
    if not path.exists() or path.stat().st_size <= 1:
        return ""
    scores = pd.read_csv(path)
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


def _relative_section(scores: pd.DataFrame) -> str:
    if scores["model"].nunique() < 2:
        return (
            "The scores hold no model other than persistence, so there are "
            "no relative scores.\n"
        )
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
        "Mean scores divided by those of persistence over the forecasts "
        "both made.\nValues below 1 beat persistence.\n\n"
        + _table(rel)
        + "\n"
        + "\n".join(tabs)
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
    real = target.loc[target["kind"] == "real"]
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
        truth_v = ", ".join(sorted(set(scores["truth_vintage"].astype(str))))
        _write(
            gen / "backtest-truth.md",
            f"Scores are against the **{truth_v}** vintage, as reported "
            "then.\n",
        )
        overall = summarise_scores(scores, by=("model", "kind"))
        _write(gen / "scores-overall.md", _table(overall))
        _write(gen / "relative-section.md", _relative_section(scores))
        _write(gen / "persistence-bias.md", _persistence_bias(scores))
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
    by_truth_path = res / "backtest-scores-by-truth.csv"
    if by_truth_path.exists():
        by_truth = pd.read_csv(by_truth_path)
        save_light_dark(
            partial(plot_scores_by_truth, by_truth, models=names),
            fig / "scores-by-truth.svg",
        )
        table = summarise_scores(
            by_truth, by=("model", "vintage", "truth_vintage")
        )
        _write(
            gen / "scores-by-truth.md",
            _table(
                table[
                    [
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
    if fc_path.exists():
        forecasts = pd.read_csv(fc_path)
        rec_all = reconstruct_vintages(
            real, delays, list(config.reconstructed_dates)
        )
        both = pd.concat([target, rec_all], ignore_index=True)
        for loc in (NATIONAL, "CD-IT", "CD-NK"):
            slug = loc.lower()
            for kind in ("real", "reconstructed"):
                save_light_dark(
                    partial(
                        plot_forecasts,
                        forecasts,
                        both,
                        loc,
                        models=names,
                        kind=kind,
                    ),
                    fig / f"forecasts-{kind}-{slug}.svg",
                )
    _write(gen / "lopo-section.md", _lopo_section(res / "lopo-scores.csv"))
    _write(gen / "ladder.md", _ladder_page(res / "ladder.csv"))
    rec_path = res / "recovery-scores.csv"
    if rec_path.exists():
        rscores = pd.read_csv(rec_path)
        save_light_dark(
            lambda mode: plot_scores(
                rscores, by="horizon", models=names, mode=mode
            ),
            fig / "recovery-wis-horizon.svg",
        )


def _heldout_page(
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
        overall = summarise_scores(
            scores, by=("chosen_model", "model", "vintage", "truth_vintage")
        )
        horizon = summarise_scores(
            scores, by=("model", "vintage", "truth_vintage", "horizon")
        )
        cols = ["n", "wis", "log_wis", "coverage_50", "coverage_90", "bias"]
        _write(
            gen / "heldout-scores.md",
            _table(
                overall[
                    ["chosen_model", "model", "vintage", "truth_vintage", *cols]
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
    display: str,
    target: pd.DataFrame,
) -> None:
    names_out = (
        "comparison-teams.md",
        "comparison-scores.md",
        "comparison-pending.md",
        "comparison-other.md",
        "comparison-figures.md",
    )
    accessed = dt.date.today().isoformat()
    manifest = cmp_dir / "manifest.json"
    if manifest.exists():
        accessed = str(json.loads(manifest.read_text())["accessed"])
    _write(gen / "comparison-attribution.md", f"> {attribution(accessed)}\n")
    _write(gen / "comparison-display.md", f"Display mode: **{display}**.\n")
    if not (cmp_dir / "scores.csv").exists():
        for name in names_out:
            _write(gen / name, "")
        return
    teams = pd.read_csv(cmp_dir / "teams.csv")
    forecasts = pd.read_csv(cmp_dir / "forecasts.csv.gz")
    models = sorted(set(teams["model"]) | set(forecasts["model"]))
    names = display_names(models, display)
    shown = teams.loc[teams["model"].isin(list(names))].assign(
        model=lambda d: d["model"].map(names)
    )
    if display != "named":
        shown = shown.drop(columns=["team_name", "model_name", "website_url"])
    _write(gen / "comparison-teams.md", _table(shown.sort_values("model")))

    def _relabel(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df["model"].isin(list(names))].copy()
        df["model"] = df["model"].map(names)
        return df

    scores = (
        pd.read_csv(cmp_dir / "scores.csv")
        if (cmp_dir / "scores.csv").stat().st_size > 1
        else pd.DataFrame()
    )
    if scores.empty:
        _write(
            gen / "comparison-scores.md",
            "",
        )
    else:
        scores = _relabel(scores)
        _write(
            gen / "comparison-scores.md",
            _table(
                summarise_scores(
                    scores, by=("reference_date", "model", "source")
                )[
                    [
                        "reference_date",
                        "model",
                        "source",
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
    pending = pd.read_csv(cmp_dir / "pending.csv")
    pending["models"] = pending["models"].map(
        lambda ms: ", ".join(
            names[m] for m in str(ms).split(", ") if m in names
        )
    )
    _write(
        gen / "comparison-pending.md",
        _table(pending) if not pending.empty else "None.\n",
    )
    other = _relabel(pd.read_csv(cmp_dir / "other-rounds.csv"))
    summary = other.groupby(["model", "target"], as_index=False).agg(
        files=("reference_date", "size"),
        first=("reference_date", "min"),
        last=("reference_date", "max"),
    )
    _write(
        gen / "comparison-other.md",
        _table(summary) if not summary.empty else "None.\n",
    )
    tabs = []
    for loc in (NATIONAL, "CD-IT", "CD-NK"):
        if loc not in set(forecasts["location"]):
            continue
        slug = loc.lower()
        save_light_dark(
            partial(plot_comparison, forecasts, target, loc, names=names),
            fig / f"comparison-{slug}.svg",
        )
        alt = f"Hub and our forecasts for {loc}."
        src = f"figures/comparison/comparison-{slug}"
        tabs.append(
            f'=== "{loc}"\n\n'
            f"    ![{alt}]({src}.svg#only-light)\n"
            f"    ![{alt}]({src}-dark.svg#only-dark)\n"
        )
    _write(gen / "comparison-figures.md", "\n".join(tabs))
