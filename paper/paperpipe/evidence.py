"""SI tables and figures from results kept on evidence branches.

The copula, the aggregation options and the fine-tuning runs write
their results as rows to evidence branches (see
`workflow/config.yaml`, block `evidence`). This module turns those rows
into the SI's tables, figures and quoted numbers, so nothing is typed into
the paper. A result whose rows are not set yet gets a note saying what it
waits on in place of its table, and `None` in place of its numbers.

- `copula_figure`: the rolling copula against no pairing, by study and
  level.
- `rho_medians`: the median rolling rho_p of each walk model.
- `aggregation_figure`: WIS of each aggregation option relative to
  `randomwalknowcast` on the same targets, by study and level.
- `heldout_figure` and `realism_figure`: the fine-tune's held-out loss
  and the simulated outbreaks against the data and published values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from bdbvhubrenewal.evaluation.labels import MODEL_STYLES, model_label
from bdbvhubrenewal.reporting.live_checkpoint import (
    LIVE_RUN,
    checkpoint_table,
    newest,
    summary_table,
    tidy,
)
from bdbvhubrenewal.reporting.live_checkpoint import (
    label as checkpoint_label,
)
from matplotlib.figure import Figure

from paperpipe.outputs import to_markdown

__all__ = [
    "EVIDENCE",
    "aggregation_figure",
    "aggregation_numbers",
    "branch_head",
    "checkpoint_figure",
    "checkpoint_numbers",
    "checkpoint_rows",
    "comparison_figure",
    "copula_figure",
    "diseases_table",
    "drop_finetuned",
    "heldout_figure",
    "heldout_rows",
    "lag_numbers",
    "live_numbers",
    "realism_figure",
    "realism_numbers",
    "rho_medians",
    "scale_floor_numbers",
    "slots_rows",
    "slots_summary",
    "waiting",
    "write_evidence",
]

OPTIONS: Final = {
    "joint": "(a)",
    "level": "(b)",
    "zones_summed": "(c)",
    "zones_copula": "(d)",
    "weekly_row": "(e) weekly row",
    "days_copula": "(e) days, copula",
    "days_summed": "(e) days, random",
}
"""Aggregation options as the SI names them, in figure order: (a) to (d)
sum over places, and (e) over time, comparing the daily model's in-block
weekly row with its days summed after pairing by the rolling copula or at
random."""


def waiting(what: str) -> str:
    """Return the note a table shows until its rows exist."""
    return f"*Not yet available: waits on {what}.*\n"


def _pair(a: pd.Series, b: pd.Series) -> list[str]:
    return [
        "" if pd.isna(x) or pd.isna(y) else f"{x:.2f} / {y:.2f}"
        for x, y in zip(a, b, strict=True)
    ]


def aggregation_figure(
    scores: pd.DataFrame, reference: str = "randomwalknowcast"
) -> Figure:
    """Plot each option's WIS relative to the reference, by study and level.

    One panel per study and level, never pooled across levels, with the
    options (`OPTIONS`) on the x axis and one marker per model, on the
    targets each shares with `reference`.

    Parameters
    ----------
    scores : pandas.DataFrame
        Per-target rows: `study`, `model`, `option`, `level`, `location`,
        `origin_date`, `target_date` and `wis`, with the reference
        model's rows.
    reference : str
        The model scores are relative to.
    """
    keys = ["study", "level", "location", "origin_date", "target_date"]
    ref = scores.loc[scores["model"] == reference, [*keys, "wis"]]
    ref = ref.drop_duplicates(keys)
    paired = scores.loc[scores["model"] != reference].merge(
        ref, on=keys, suffixes=("", "_ref")
    )
    sums = paired.groupby(["study", "level", "model", "option"])[
        ["wis", "wis_ref"]
    ].sum()
    rel = (sums["wis"] / sums["wis_ref"]).rename("rel").reset_index()
    panels = list(dict.fromkeys(zip(rel["study"], rel["level"], strict=True)))
    studies = list(dict.fromkeys(s for s, _ in panels))
    ncol = max(sum(s == st for s, _ in panels) for st in studies)
    fig, axes = plt.subplots(
        len(studies),
        ncol,
        figsize=(3.6 * ncol, 2.8 * len(studies)),
        layout="constrained",
        squeeze=False,
    )
    options = [o for o in OPTIONS if o in set(rel["option"])]
    models = sorted(rel["model"].unique())
    markers = dict(zip(models, "osD^v<>", strict=False))
    for i, study in enumerate(studies):
        levels = [lv for s, lv in panels if s == study]
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(levels):
                ax.set_visible(False)
                continue
            one = rel.loc[(rel["study"] == study) & (rel["level"] == levels[j])]
            for model, g in one.groupby("model"):
                ax.plot(
                    [options.index(o) for o in g["option"]],
                    g["rel"],
                    ls="none",
                    marker=markers[model],
                    label=model_label(model),
                )
            ax.axhline(1, color="#888888", lw=1)
            ax.set_xticks(
                range(len(options)),
                [OPTIONS[o] for o in options],
                fontsize=7,
                rotation=30,
                ha="right",
            )
            ax.set_title(f"{study}, {levels[j]}", fontsize=9)
            ax.set_ylabel("Relative WIS")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
    return fig


def aggregation_numbers(
    scores: pd.DataFrame, reference: str = "randomwalknowcast"
) -> dict:
    """Return each option's relative WIS, as the paper quotes them.

    Keys are study, level, model and option (`OPTIONS`), and values the
    WIS relative to `reference` on shared targets, to two decimals.
    """
    keys = ["study", "level", "location", "origin_date", "target_date"]
    ref = scores.loc[scores["model"] == reference, [*keys, "wis"]]
    paired = scores.loc[scores["model"] != reference].merge(
        ref, on=keys, suffixes=("", "_ref")
    )
    sums = paired.groupby(["study", "level", "model", "option"])[
        ["wis", "wis_ref"]
    ].sum()
    rel = (sums["wis"] / sums["wis_ref"]).round(2)
    out: dict = {}
    for (study, level, model, option), value in rel.items():
        node = out.setdefault(study, {}).setdefault(level, {})
        node.setdefault(model, {})[option] = float(value)
    return out


def lag_numbers(
    germany: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame
) -> dict:
    """Return each model's relative WIS with the capped and longer lags.

    Germany's national rows name the longer horizon `<model>_fix`; the
    BDBV release backtest gives each horizon its own table, of which the
    `all` nowcast rows of the complete target are compared. Values are
    WIS relative to `randomwalknowcast`.
    """
    nat = germany.loc[germany["pool"] == "national"].set_index("model")
    rel = nat["rel_wis_randomwalknowcast"]
    de = {
        m: {
            "capped": round(float(rel[m]), 2),
            "longer": round(float(rel[f"{m}_fix"]), 2),
        }
        for m in rel.index
        if f"{m}_fix" in rel.index
    }

    def pick(t: pd.DataFrame) -> pd.Series:
        keep = (
            (t["scale"] == "all")
            & (t["target"] == "complete")
            & (t["weeks"] == "nowcast")
        )
        return t.loc[keep].set_index("model")["rel_wis"]

    b, a = pick(before), pick(after)
    bdbv = {
        m: {"capped": round(float(b[m]), 2), "longer": round(float(a[m]), 2)}
        for m in b.index
        if m in a.index and m.startswith("fm_chronos")
    }
    return {"germany": de, "bdbv": bdbv}


CHECKPOINT_RULE: Final = "own-quantile-wis_crps-1000-common"
"""The scoring rule of the checkpoint rows the paper shows: WIS on each
model's own quantiles for Germany, and CRPS on 1,000 draws with common
random numbers for BDBV."""


def _case_study_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the case studies' rows of `CHECKPOINT_RULE`, one score column.

    BDBV's relative CRPS fills `rel_wis` where that is empty.
    """
    out = scores
    if "rule" in out:
        out = out.loc[out["rule"] == CHECKPOINT_RULE]
    out = out.loc[out["study"].isin(["germany", "ebola"])]
    if "rel_crps" in out:
        out = out.assign(rel_wis=out["rel_wis"].fillna(out["rel_crps"]))
        if "rel_log_crps" in out and "rel_log_wis" in out:
            out = out.assign(
                rel_log_wis=out["rel_log_wis"].fillna(out["rel_log_crps"])
            )
    return out


def checkpoint_rows(scores: pd.DataFrame, run: str = LIVE_RUN) -> pd.DataFrame:
    """Return zero-shot and the live run's checkpoint rows, by level.

    Rows of `CHECKPOINT_RULE` and the hub target on the case studies'
    development data (`live_checkpoint.tidy`), at every level, each for
    its own panel. BDBV's relative CRPS fills `rel_wis`, so one column
    holds each study's relative score.
    """
    out = tidy(scores)
    out = out.loc[out["study"].isin(["germany", "ebola"])]
    out = out.assign(rel_wis=out["rel"], rel_log_wis=out["rel_log"])
    live = out["run"] == run if "run" in out else False
    keep = (out["checkpoint"] == "zero-shot") | live
    return out.loc[keep].reset_index(drop=True)


HELDOUT_LEVELS: Final = ("national", "intermediate", "leaf")
"""Levels of the trainer's held-out loss, each reported on its own."""

WITHOUT_BIASES: Final = "no-slots"
"""The run fine-tuned without attention biases, by the scores' `run`."""


def slots_rows(
    scores: pd.DataFrame, run: str = LIVE_RUN, without: str = WITHOUT_BIASES
) -> pd.DataFrame:
    """Return the runs with and without attention biases, by checkpoint.

    Sets are, one per level, the trainer's held-out loss and the relative
    scores of each model on the German first half and the BDBV release backtest,
    in the layout of `comparison_figure` (`set`, `run`, `outbreaks`,
    `score`).
    """
    rows = tidy(scores)
    names = {run: "attention biases", without: "no biases"}
    rows = rows.loc[rows["run"].isin(names) & (rows["outbreaks"] > 0)]
    loss = rows.loc[
        (rows["study"] == "held_out") & rows["level"].isin(HELDOUT_LEVELS)
    ]
    out = [
        pd.DataFrame(
            {
                "set": "Held-out loss, " + loss["level"],
                "run": loss["run"].map(names),
                "outbreaks": loss["outbreaks"],
                "score": loss["loss"],
            }
        )
    ]
    case = rows.loc[rows["study"].isin(["germany", "ebola"])]
    study = case["study"].map({"germany": "German", "ebola": "BDBV"})
    source = case["model"].str.extract(r"_(daily|weekly)_")[0]
    out.append(
        pd.DataFrame(
            {
                "set": study + " " + case["level"] + ", " + source,
                "run": case["run"].map(names),
                "outbreaks": case["outbreaks"],
                "score": case["rel"],
            }
        )
    )
    return pd.concat(out, ignore_index=True).dropna(subset=["score"])


def slots_summary(
    scores: pd.DataFrame, run: str = LIVE_RUN, without: str = WITHOUT_BIASES
) -> pd.DataFrame:
    """Return zero-shot and both runs at the largest size both reached.

    Per study, each run keeps only checkpoints up to the largest number
    of simulations both runs were scored at, so the rows compare like
    with like; a study with no shared size is blank.
    """
    rows = tidy(scores)
    both = rows.loc[rows["run"].isin([run, without])]
    keep = pd.Series(True, index=scores.index)
    for study, g in both.groupby("study"):
        sizes = [set(g.loc[g["run"] == r, "outbreaks"]) for r in (run, without)]
        shared = sizes[0] & sizes[1]
        top = max(shared) if shared else -1
        drop = (
            (scores["study"] == study)
            & scores["run"].isin([run, without])
            & (scores["outbreaks"] > top)
        )
        keep &= ~drop
    matched = scores.loc[keep]
    live = summary_table(matched, run)
    none = summary_table(matched, without)
    out = pd.concat([live, none.iloc[[1]]], ignore_index=True)
    out = out.rename(columns={"Simulations": "Run"})
    out["Run"] = ["Zero-shot", "attention biases", "no biases"]
    return out


def heldout_rows(scores: pd.DataFrame, run: str = LIVE_RUN) -> pd.DataFrame:
    """Return the trainer's held-out loss of the live run by checkpoint.

    In the layout of `heldout_figure`, on the trainer's held-out set, one
    line per level (`HELDOUT_LEVELS`).
    """
    rows = tidy(scores)
    rows = rows.loc[
        (rows["study"] == "held_out")
        & rows["level"].isin(HELDOUT_LEVELS)
        & (rows["run"] == run)
    ]
    return pd.DataFrame(
        {
            "run": run,
            "blocks": rows["level"],
            "outbreaks_seen": rows["outbreaks"],
            "heldout_loss": rows["loss"],
        }
    )


def live_numbers(scores: pd.DataFrame | None, run: str = LIVE_RUN) -> dict:
    """Return each study's newest checkpoint of the live run and its label.

    `label` is the newest over the studies, for captions; it says the
    scores are not yet available before any checkpoint is scored.
    """
    top = {} if scores is None else newest(scores, run)
    out: dict = {
        study: {"checkpoint": ck, "outbreaks": n, "label": checkpoint_label(n)}
        for study, (ck, n) in top.items()
    }
    out["label"] = (
        checkpoint_label(max(n for _, n in top.values()))
        if top
        else "Fine-tuned checkpoint scores are not yet available."
    )
    return out


FINETUNED_EVIDENCE: Final = ("aggregation_scores",)
"""Evidence of both zero-shot and fine-tuned models, whose fine-tuned rows
come from earlier fine-tunes than the paper's and are left out."""


def drop_finetuned(rows: pd.DataFrame) -> pd.DataFrame:
    """Leave out the fine-tuned models' rows (`_ft`, `_ft_n<k>`)."""
    fine = rows["model"].astype(str).str.contains(r"_ft(_|$)")
    return rows.loc[~fine].reset_index(drop=True)


MIN_CELLS: Final = 5
"""Panels of the checkpoint figure with fewer matched cells are greyed."""


def checkpoint_numbers(scores: pd.DataFrame) -> dict:
    """Return each panel's cells and each model's relative WIS by size.

    Keys are study and level, then `n` (the fewest matched cells of a
    point), `n_origins` where the rows give it, and each model with its
    `rel_wis` by outbreaks (`zero-shot` for 0), to two decimals.
    """
    out: dict = {}
    for (study, level), one in scores.groupby(["study", "level"]):
        node = out.setdefault(study, {}).setdefault(level, {})
        node["n"] = int(one["n"].min())
        if "n_origins" in one:
            node["n_origins"] = int(one["n_origins"].min())
        for model, g in one.groupby("model"):
            node[model] = {
                ("zero-shot" if k == 0 else f"n{int(k)}"): round(float(v), 2)
                for k, v in zip(g["outbreaks"], g["rel_wis"], strict=True)
            }
    return out


PLANNED_CHECKPOINTS: Final = (1000, 2000, 4000, 8400)
"""The final run's checkpoints, which set the axis before any lands."""


def _source_of(model: str) -> str:
    """Return the model's source (daily or weekly), which sets its colour."""
    for source in ("daily", "weekly"):
        if f"_{source}_" in model:
            return source
    return model


def checkpoint_figure(scores: pd.DataFrame) -> Figure:
    """Plot each model's relative WIS against the fine-tune's outbreaks.

    One panel per study and level, titled with its n of matched cells and
    greyed when n is under `MIN_CELLS`. Each fine-tuned model is a line
    over its checkpoints on a log axis of outbreaks. The models with no
    fine-tuning are horizontal lines in the same colour: zero-shot
    Chronos-2 on the ragged layout (dashed) and on complete data only
    (dotted, the `_complete` models), with `randomwalknowcast` the line at
    1. `scores` has one row per study, level, model and checkpoint, with
    `outbreaks` (0 for zero-shot) and `rel_wis` against `randomwalknowcast`.
    """
    sc = scores.loc[scores["model"] != "randomwalknowcast"]
    sizes = sorted(sc.loc[sc["outbreaks"] > 0, "outbreaks"].unique())
    ticks = sizes or list(PLANNED_CHECKPOINTS)
    # One colour per source across panels, so the shared legend holds.
    sources = sorted({_source_of(m) for m in sc["model"]})
    colours = {s: f"C{i}" for i, s in enumerate(sources)}
    # Models with a shared colour (`evaluation.labels`) keep it.
    colours |= {
        _source_of(m): MODEL_STYLES[m][0]
        for m in sc["model"].unique()
        if m in MODEL_STYLES
    }
    panels = list(sc.groupby(["study", "level"], sort=False).groups)
    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 2.8 * nrow),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (study, level) in zip(axes, panels, strict=False):
        one = sc.loc[(sc["study"] == study) & (sc["level"] == level)]
        n = int(one["n"].min())
        thin = n < MIN_CELLS
        k = int(one["n_origins"].min()) if "n_origins" in one else None
        origins = "" if k is None else f", {k} origin{'s' if k != 1 else ''}"
        for model, g in one.groupby("model"):
            colour = "#bbbbbb" if thin else colours[_source_of(model)]
            tuned = g.loc[g["outbreaks"] > 0].sort_values("outbreaks")
            if not tuned.empty:
                ax.plot(
                    tuned["outbreaks"],
                    tuned["rel_wis"],
                    marker="o",
                    ms=3,
                    color=colour,
                    label=model_label(f"{model}_ft"),
                )
            complete = "_complete" in model
            for value in g.loc[g["outbreaks"] == 0, "rel_wis"]:
                ax.axhline(
                    value,
                    color=colour,
                    ls=":" if complete else "--",
                    lw=1.2,
                    label=model_label(model),
                )
        if thin:
            ax.text(
                0.5,
                0.5,
                f"too few cells (n < {MIN_CELLS})",
                transform=ax.transAxes,
                ha="center",
                color="#888888",
            )
        ax.set_xscale("log")
        ax.set_xlim(ticks[0] / 1.5, ticks[-1] * 1.5)
        ax.set_xticks(ticks, [f"{t:,.0f}" for t in ticks], fontsize=7)
        ax.minorticks_off()
        reference = model_label("randomwalknowcast")
        ax.axhline(1, color="#888888", ls="-", lw=1, label=reference)
        ax.set_title(f"{study}, {level} (n = {n}{origins})", fontsize=9)
        ax.set_xlabel("Fine-tuning outbreaks")
        ax.set_ylabel("Relative WIS")
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    # The legend from the panels drawn in colour, one entry per line.
    shown = [ax for ax in axes if ax.get_visible() and ax.texts == []]
    entries = {}
    for ax in shown or axes[:1]:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, h)
    labels = sorted(entries)
    handles = [entries[label] for label in labels]
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
    return fig


def scale_floor_numbers(rows: pd.DataFrame) -> dict:
    """Return the scale-floor test without and with the floor.

    Held-out simulation rows give, by model and series size (the first
    word of `size`), the median loss and the percentage of 99% quantiles
    over 100 times the truth; BDBV rows give relative WIS by level,
    source (daily or weekly) and model. Each value is keyed `none` (no
    floor) or `floor`.
    """
    arm = {0: "none", 1: "floor"}
    sims = rows.loc[rows["source"].isna()]
    out: dict = {"sims": {}, "bdbv": {}}
    for r in sims.itertuples():
        size = str(r.size).split()[0]
        node = out["sims"].setdefault(r.model, {}).setdefault(size, {})
        node.setdefault("median_loss", {})[arm[r.floor]] = round(
            float(r.median_loss), 1
        )
        node.setdefault("q99_over_100x", {})[arm[r.floor]] = round(
            100 * float(r.share_q99_over_100x)
        )
    for r in rows.loc[rows["source"].notna()].itertuples():
        node = (
            out["bdbv"]
            .setdefault(r.size, {})
            .setdefault(r.source, {})
            .setdefault(r.model, {})
        )
        node[arm[r.floor]] = round(float(r.relative_wis), 2)
    return out


def heldout_figure(curve: pd.DataFrame) -> Figure:
    """Plot held-out loss against the outbreaks seen.

    One line per run and set of held-out blocks (`blocks`), as losses on
    different blocks do not compare.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for (run, blocks), g in curve.groupby(["run", "blocks"]):
        g = g.sort_values("outbreaks_seen")
        ax.plot(
            g["outbreaks_seen"],
            g["heldout_loss"],
            marker="o",
            ms=3,
            label=f"{run}, {blocks}",
        )
    ax.set_xlabel("Simulated outbreaks seen")
    ax.set_ylabel("Held-out loss")
    ax.legend(frameon=False, fontsize=8)
    return fig


def realism_figure(
    sims: pd.DataFrame,
    reference: pd.DataFrame,
    summaries: Sequence[str] | None = None,
    colours: Mapping[str, str] | None = None,
) -> Figure:
    """Plot each summary of the simulated outbreaks against the data.

    Parameters
    ----------
    sims : pandas.DataFrame
        One row per simulated outbreak, one column per summary.
    reference : pandas.DataFrame
        One row per summary (the index), one column per data set (such
        as `BDBV`), its value for that data set; other columns are
        ignored when they start with `sim`.
    summaries : sequence of str, optional
        Summaries to show. Defaults to those in both tables.
    colours : mapping, optional
        Line colour of each data set.

    Returns
    -------
    matplotlib.figure.Figure
        A histogram per summary with a line per data set.
    """
    data = [c for c in reference.columns if not str(c).startswith("sim")]
    names = list(summaries or [s for s in reference.index if s in sims])
    colours = colours or dict(
        zip(data, ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"], strict=False)
    )
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names, strict=False):
        ax.hist(sims[name].dropna(), bins=30, color="#9e9e9e")
        for d in data:
            ax.axvline(reference.loc[name, d], color=colours[d], lw=1.5)
        ax.set_title(name.replace("_", " "), fontsize=9)
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=colours[d], label=d) for d in data]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(data))
    return fig


def realism_numbers(reference: pd.DataFrame) -> dict:
    """Return each summary's simulated quantiles and data values.

    Keys are the summaries, then the columns with spaces as underscores
    (`sim_q95`, `BDBV`, `Germany_states`), rounded to three figures.
    """
    return {
        str(name): {
            str(col).replace(" ", "_"): float(f"{value:.3g}")
            for col, value in row.items()
            if pd.notna(value)
        }
        for name, row in reference.iterrows()
    }


def diseases_table(diseases: pd.DataFrame) -> pd.DataFrame:
    """Return the simulated ranges against published values, with DOIs."""
    sims = [
        f"{a:.3g} [{b:.3g}, {c:.3g}]"
        for a, b, c in zip(
            diseases["sim_q50"],
            diseases["sim_q5"],
            diseases["sim_q95"],
            strict=True,
        )
    ]
    sources = [
        "; ".join(f"[{d}](https://doi.org/{d})" for d in str(s).split("; "))
        for s in diseases["sources"]
    ]
    return pd.DataFrame(
        {
            "Summary": diseases["summary"],
            "Simulated median [5%, 95%]": sims,
            "Published": diseases["published"],
            "Sources": sources,
        }
    )


EVIDENCE: Final = {
    "si_copula": "fm-copula's walk scores, no pairing against rolling rho_p",
    "copula_rho": "the walk models' rho_p by origin (copula.json rows)",
    "aggregation_scores": "fm-agg's per-target aggregation scores",
    "finetune_heldout": "fm-finetune3's held-out loss curve",
    "sim_summaries": "fm-finetune3's per-outbreak r3 pilot summaries",
    "sim_realism": "fm-finetune3's data reference values",
    "sim_diseases": "fm-finetune3's published disease values",
    "checkpoint_scores": "germany-chronos2's scores of each r3 checkpoint",
    "scale_floor": "fm-finetune3's scale-floor test",
    "lag_germany": "germany-chronos2's German lag-horizon scores",
    "lag_bdbv_before": "the BDBV scores with the 4-week lag horizon",
    "lag_bdbv_after": "the BDBV scores with lags to the truth delay",
    "compare_slots": "the slots and no-slots runs' scores by checkpoint",
    "compare_delays": "the full and no-delay runs' scores by checkpoint",
    "compare_realdata": "the real-data and synthetic fine-tunes' scores",
    "compare_daily": "our models' daily and hub-target scores by checkpoint",
    "block_combinations": (
        "fm-finetune3's realised shares of block layouts and options"
    ),
}
"""Each evidence input and what it waits on until set.

The `compare_*` files share one layout: `set` (a held-out source, or a
case study and level), `run`, `outbreaks` (the run's checkpoint) and
`score` (held-out loss, or relative WIS or CRPS against
`randomwalknowcast`)."""


NOT_YET: Final = "(not yet available)"
"""What the paper quotes in place of a number whose rows are not set."""

COPULA_MODELS: Final = ("randomwalk", "randomwalknowcast")
"""The models that pair their province draws by rho_p."""

QUOTED: Final = {
    "rho": {
        m: {"rho_p": NOT_YET, "origins": NOT_YET, "fallback": NOT_YET}
        for m in COPULA_MODELS
    },
    "realism": {
        "zero_share": {"sim_q95": NOT_YET, "BDBV": NOT_YET},
        "delay_q99": {"sim_q95": NOT_YET, "Germany_states": NOT_YET},
    },
}
"""The numbers the paper quotes, as they read until their rows exist."""


COPULA_STUDIES: Final = {
    "germany": "Germany",
    "bdbv": "BDBV",
    "synthetic": "Held-out synthetic",
}
"""The copula figure's studies, in its row order, all on development data."""

COPULA_LEVELS: Final = ("national", "provinces", "states")
"""The levels the copula changes: the country, which is the sum of the
provinces (or states), and the provinces themselves. Zones are not
paired."""

COPULA_METRICS: Final = {
    "cell_crps": "CRPS",
    "cell_cover90": "90% coverage",
    "joint_log_energy": "ES (log)",
}
"""The copula figure's scores: CRPS and 90% coverage of each cell, which
at the national level is the sum of the provinces, and the energy score
on log counts of the level's places and weeks as one vector."""


def rho_medians(rho: pd.DataFrame) -> dict:
    """Return each walk model's median rolling rho_p over origins.

    `rho` has the walk models' `copula.json` rows: `model`, `origin`,
    `n_scores`, `rho_provinces`, `n_pairs` and `fallback`. Origins that
    fell back to the prior for want of pairs are counted, not in the
    median.
    """
    fallback = rho["fallback"].astype(str).str.lower() == "true"
    out: dict = {}
    for model, g in rho.groupby("model", sort=False):
        own = fallback.loc[g.index]
        est = g.loc[~own, "rho_provinces"]
        out[model] = {
            "rho_p": round(float(est.median()), 2) if len(est) else NOT_YET,
            "origins": int(g["origin"].nunique()),
            "fallback": int(own.sum()),
        }
    return out


def copula_figure(rows: pd.DataFrame) -> Figure:
    """Plot each score of the rolling copula relative to no pairing.

    One panel per study and level in `COPULA_LEVELS`, never pooled across
    levels. Each panel gives the ratio rolling / off of every score in
    `COPULA_METRICS`, one marker per model, with 1 no pairing. `rows` are
    `evaluation.pairing.summarise_pairing`'s: `study`, `level`, `model`,
    `variant` (`off` or `rolling`), `target`, `metric`, `value` and `n`.
    """
    rows = rows.assign(metric=rows["target"] + "_" + rows["metric"])
    rows = rows.loc[
        rows["metric"].isin(list(COPULA_METRICS))
        & rows["level"].isin(COPULA_LEVELS)
    ]
    wide = rows.pivot_table(
        index=["study", "level", "model", "metric"],
        columns="variant",
        values="value",
    ).reset_index()
    wide["ratio"] = wide["rolling"] / wide["off"]
    n = rows.groupby(["study", "level"])["n"].min()
    studies = [s for s in COPULA_STUDIES if s in set(wide["study"])]
    levels = {
        s: list(dict.fromkeys(rows.loc[rows["study"] == s, "level"]))
        for s in studies
    }
    ncol = max(len(v) for v in levels.values())
    fig, axes = plt.subplots(
        len(studies),
        ncol,
        figsize=(3.4 * ncol, 2.6 * len(studies)),
        layout="constrained",
        squeeze=False,
    )
    metrics = list(COPULA_METRICS)
    models = sorted(wide["model"].unique())
    markers = dict(zip(models, "osD^v", strict=False))
    for i, study in enumerate(studies):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(levels[study]):
                ax.set_visible(False)
                continue
            level = levels[study][j]
            one = wide.loc[(wide["study"] == study) & (wide["level"] == level)]
            for model, g in one.groupby("model"):
                x = [metrics.index(m) for m in g["metric"]]
                ax.plot(
                    x,
                    g["ratio"],
                    ls="none",
                    marker=markers[model],
                    label=model_label(model),
                )
            ax.axhline(1, color="#888888", lw=1)
            ax.set_xticks(
                range(len(metrics)),
                [COPULA_METRICS[m] for m in metrics],
                fontsize=7,
            )
            ax.set_title(
                f"{COPULA_STUDIES[study]}, {level} "
                f"(n = {int(n.loc[(study, level)])})",
                fontsize=9,
            )
            ax.set_ylabel("Rolling / no pairing")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(models))
    return fig


def comparison_figure(rows: pd.DataFrame) -> Figure:
    """Plot a fine-tuning comparison: each run's score by checkpoint.

    One panel per set, in the order the rows give, one line per run on a
    log axis of outbreaks.
    """
    sets = list(dict.fromkeys(rows["set"]))
    runs = sorted(rows["run"].unique())
    colours = {r: f"C{i}" for i, r in enumerate(runs)}
    ncol = min(3, len(sets))
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 2.6 * nrow),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, name in zip(axes, sets, strict=False):
        one = rows.loc[rows["set"] == name]
        for run, g in one.groupby("run"):
            g = g.sort_values("outbreaks")
            ax.plot(
                g["outbreaks"],
                g["score"],
                marker="o",
                ms=3,
                color=colours[run],
                label=run,
            )
        ax.set_xscale("log")
        ticks = sorted(one["outbreaks"].unique())
        ax.set_xticks(ticks, [f"{t:,.0f}" for t in ticks], fontsize=7)
        ax.minorticks_off()
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Fine-tuning outbreaks")
        ax.set_ylabel("Score")
    for ax in axes[len(sets) :]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(runs))
    return fig


def branch_head(branch: str, repo: Path | str = ".") -> str:
    """Return the commit at the head of an evidence branch on `origin`.

    The branch is fetched first, so each workflow run reads its newest
    rows; an evidence file named by this commit is extracted anew when
    the branch moves. Offline, the branch as last fetched is used.
    """
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        git("fetch", "-q", "origin", branch)
        return git("rev-parse", "FETCH_HEAD")
    except subprocess.CalledProcessError:
        # Offline: the branch as last fetched.
        return git("rev-parse", f"origin/{branch}")


def _read(paths: Mapping[str, Path | None], name: str) -> pd.DataFrame | None:
    path = paths.get(name)
    if path is None:
        return None
    return pd.read_csv(path, index_col=0 if name == "sim_realism" else None)


def write_evidence(
    paths: Mapping[str, Path | None],
    si_dir: Path,
    fig_dir: Path,
    out_yml: Path,
    run: str = LIVE_RUN,
) -> dict:
    """Write every evidence table and figure, and the numbers they quote.

    Parameters
    ----------
    paths : mapping
        `EVIDENCE` name to its file, or None while it is not set.
    si_dir, fig_dir : pathlib.Path
        Folders for the SI tables and figures.
    out_yml : pathlib.Path
        The quoted numbers, as Quarto metadata under `evidence`.
    run : str
        The live fine-tuning run, whose scored checkpoints the curve, the
        tables and the comparisons show as preliminary.

    Returns
    -------
    dict
        The quoted numbers.
    """
    si_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = {name: _read(paths, name) for name in EVIDENCE}
    for name in FINETUNED_EVIDENCE:
        if data.get(name) is not None and "model" in data[name]:
            data[name] = drop_finetuned(data[name])

    def table(name: str, frame: pd.DataFrame | None, needs: list[str]) -> None:
        # Rows for another study only still leave this one waiting.
        text = (
            to_markdown(frame)
            if frame is not None and not frame.empty
            else waiting(", ".join(EVIDENCE[n] for n in needs))
        )
        (si_dir / f"{name}.md").write_text(text)

    def figure(name: str, fig: Figure | None, needs: list[str]) -> None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                waiting(EVIDENCE[needs[0]]).strip("*\n"),
                ha="center",
                va="center",
            )
        fig.savefig(fig_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    agg = data["aggregation_scores"]
    figure(
        "aggregation",
        None if agg is None else aggregation_figure(agg),
        ["aggregation_scores"],
    )
    cop = data["si_copula"]
    figure(
        "copula-impact",
        None if cop is None else copula_figure(cop),
        ["si_copula"],
    )
    dis = data["sim_diseases"]
    table(
        "sim-diseases",
        None if dis is None else diseases_table(dis),
        ["sim_diseases"],
    )
    raw = data["checkpoint_scores"]
    ckpt = None if raw is None else checkpoint_rows(raw, run)
    figure(
        "checkpoint-curve",
        None if ckpt is None else checkpoint_figure(ckpt),
        ["checkpoint_scores"],
    )
    table(
        "checkpoint-table",
        None if raw is None else checkpoint_table(raw, run),
        ["checkpoint_scores"],
    )
    table(
        "live-summary",
        None if raw is None else summary_table(raw, run),
        ["checkpoint_scores"],
    )
    table(
        "slots-summary",
        None if raw is None else slots_summary(raw, run),
        ["checkpoint_scores"],
    )
    table(
        "block-combinations",
        data["block_combinations"],
        ["block_combinations"],
    )
    if data["compare_slots"] is None and raw is not None:
        data["compare_slots"] = slots_rows(raw, run)
    for name in ("slots", "delays", "realdata", "daily"):
        rows = data[f"compare_{name}"]
        figure(
            f"compare-{name}",
            None if rows is None else comparison_figure(rows),
            [f"compare_{name}"],
        )
    curve = data["finetune_heldout"]
    if curve is None and raw is not None:
        curve = heldout_rows(raw, run)
    figure(
        "finetune-heldout",
        None if curve is None or curve.empty else heldout_figure(curve),
        ["finetune_heldout"],
    )
    sims, ref = data["sim_summaries"], data["sim_realism"]
    figure(
        "sim-realism",
        None if sims is None or ref is None else realism_figure(sims, ref),
        ["sim_summaries", "sim_realism"],
    )
    rho = data["copula_rho"]
    numbers = {
        "evidence": {
            "rho": QUOTED["rho"] if rho is None else rho_medians(rho),
            "lag": (
                None
                if any(
                    data[n] is None
                    for n in (
                        "lag_germany",
                        "lag_bdbv_before",
                        "lag_bdbv_after",
                    )
                )
                else lag_numbers(
                    data["lag_germany"],
                    data["lag_bdbv_before"],
                    data["lag_bdbv_after"],
                )
            ),
            "scale_floor": (
                None
                if data["scale_floor"] is None
                else scale_floor_numbers(data["scale_floor"])
            ),
            "checkpoints": (None if ckpt is None else checkpoint_numbers(ckpt)),
            "live": live_numbers(raw, run),
            "aggregation": (None if agg is None else aggregation_numbers(agg)),
            "realism": (
                QUOTED["realism"] if ref is None else realism_numbers(ref)
            ),
            "waiting": [EVIDENCE[n] for n, d in data.items() if d is None],
        }
    }
    out_yml.write_text(yaml.safe_dump(numbers, sort_keys=False))
    return numbers
