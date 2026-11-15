"""SI tables and figures from results of code not yet on main.

The copula (#221), the aggregation options (#222) and the fine-tuning
runs (#192) write their results as rows to evidence branches (see
`workflow/config.yaml`, block `evidence`). This module turns those rows
into the SI's tables, figures and quoted numbers, so nothing is typed into
the paper. A result whose rows are not set yet gets a note saying what it
waits on in place of its table, and `None` in place of its numbers.

- `copula_table`: mean joint scores, coverage of sums and WIS of each
  model and pairing arm, by strata.
- `rho_medians`: median rolling copula correlations over origins and places.
- `aggregation_table`: WIS of each aggregation option relative to
  `randomwalknowcast` on the same targets, with coverage and n per level.
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
from matplotlib.figure import Figure

from paperpipe.outputs import to_markdown

__all__ = [
    "ARMS",
    "EVIDENCE",
    "aggregation_numbers",
    "aggregation_table",
    "checkpoint_figure",
    "checkpoint_numbers",
    "copula_table",
    "cost_table",
    "diseases_table",
    "heldout_figure",
    "lag_numbers",
    "realism_figure",
    "realism_numbers",
    "rho_medians",
    "scale_floor_numbers",
    "waiting",
    "write_evidence",
]

ARMS: Final = {
    "unpaired": "Unpaired",
    "fixed": "Fixed",
    "rolling": "Rolling",
    "rolling_w25": "Rolling, w = 25",
}
"""Pairing arms of the copula comparison, in table order."""

OPTIONS: Final = {
    "joint": "Joint block",
    "level": "Level by level",
    "zones_summed": "Zones summed",
    "zones_copula": "Zones summed after pairing",
}
"""Aggregation options of #222, in table order."""


def waiting(what: str) -> str:
    """Return the note a table shows until its rows exist."""
    return f"*Not yet available: waits on {what}.*\n"


def _pair(a: pd.Series, b: pd.Series) -> list[str]:
    return [
        "" if pd.isna(x) else f"{x:.2f} / {y:.2f}"
        for x, y in zip(a, b, strict=True)
    ]


def copula_table(scores: pd.DataFrame, study: str) -> pd.DataFrame:
    """Return the copula comparison of one study, one row per model and arm.

    Parameters
    ----------
    scores : pandas.DataFrame
        Per-forecast rows: `study`, `strata`, `model`, `arm`,
        `energy_log`, `variogram_log`, `cover50_weeks`, `cover90_weeks`,
        `cover50_places`, `cover90_places`, `wis_strata`, `wis_national`.
    study : str
        `ebola` or `germany`.

    Returns
    -------
    pandas.DataFrame
        Display columns, means over forecasts.
    """
    sc = scores.loc[scores["study"] == study]
    cols = [
        "energy_log",
        "variogram_log",
        "cover50_weeks",
        "cover90_weeks",
        "cover50_places",
        "cover90_places",
        "wis_strata",
        "wis_national",
    ]
    m = sc.groupby(["strata", "model", "arm"], sort=False)[cols].mean()
    m = m.reset_index()
    m["arm"] = pd.Categorical(m["arm"], categories=list(ARMS), ordered=True)
    m = m.sort_values(["strata", "model", "arm"])
    return pd.DataFrame(
        {
            "Strata": m["strata"].str.capitalize(),
            "Model": m["model"],
            "Pairing": m["arm"].map(ARMS).astype(str),
            "ES (log)": m["energy_log"].round(3),
            "VS (log)": m["variogram_log"].round(3),
            "Weeks 50% / 90%": _pair(m["cover50_weeks"], m["cover90_weeks"]),
            "Places 50% / 90%": _pair(m["cover50_places"], m["cover90_places"]),
            "WIS strata": m["wis_strata"].round(2),
            "WIS national": m["wis_national"].round(2),
        }
    )


def rho_medians(rho: pd.DataFrame) -> dict:
    """Return the median rolling correlations by study, strata and model.

    Rows of other pairing arms, where `rho` has an `arm` column, are left
    out.
    """
    if "arm" in rho.columns:
        rho = rho.loc[rho["arm"] == "rolling"]
    out: dict = {}
    for (study, strata, model), g in rho.groupby(["study", "strata", "model"]):
        out.setdefault(study, {}).setdefault(strata, {})[model] = {
            name: round(float(g[name].median()), 2)
            for name in ("rho_t", "rho_p", "rho_n")
            if name in g and g[name].notna().any()
        }
    return out


def aggregation_table(
    scores: pd.DataFrame, study: str, reference: str = "randomwalknowcast"
) -> pd.DataFrame:
    """Return each option's WIS relative to the reference, by level.

    Parameters
    ----------
    scores : pandas.DataFrame
        Per-target rows: `study`, `model`, `option`, `level`, `location`,
        `origin_date`, `target_date`, `wis`, `log_wis`, `cover50`,
        `cover95`, with the reference model's rows.
    study : str
        `ebola` or `germany`.
    reference : str
        The model scores are relative to.

    Returns
    -------
    pandas.DataFrame
        One row per model, option and level: relative WIS with the log
        scale in brackets, 50% and 95% coverage and n, the number of
        targets shared with the reference.
    """
    sc = scores.loc[scores["study"] == study]
    keys = ["level", "location", "origin_date", "target_date"]
    ref = sc.loc[sc["model"] == reference, [*keys, "wis", "log_wis"]]
    paired = sc.loc[sc["model"] != reference].merge(
        ref, on=keys, suffixes=("", "_ref")
    )
    g = paired.groupby(["model", "option", "level"], sort=False)
    sums = g[["wis", "wis_ref", "log_wis", "log_wis_ref"]].sum()
    rel = sums["wis"] / sums["wis_ref"]
    rel_log = sums["log_wis"] / sums["log_wis_ref"]
    cover = g[["cover50", "cover95"]].mean()
    out = pd.DataFrame(
        {
            "Relative WIS (log)": [
                f"{a:.2f} ({b:.2f})" for a, b in zip(rel, rel_log, strict=True)
            ],
            "50% / 95%": _pair(cover["cover50"], cover["cover95"]),
            "n": g.size(),
        },
        index=sums.index,
    ).reset_index()
    out["option"] = pd.Categorical(
        out["option"], categories=list(OPTIONS), ordered=True
    )
    out = out.sort_values(["model", "option", "level"])
    out["option"] = out["option"].map(OPTIONS).astype(str)
    return out.rename(
        columns={"model": "Model", "option": "Option", "level": "Level"}
    )


def cost_table(cost: pd.DataFrame) -> pd.DataFrame:
    """Return each option's cost, with the options named as in the SI."""
    out = cost.assign(option=cost["option"].map(OPTIONS).fillna(cost["option"]))
    return out.rename(
        columns={
            "study": "Study",
            "model": "Model",
            "option": "Option",
            "series": "Series per block",
            "memory_gb": "Memory (GB)",
            "seconds": "Time (s)",
        }
    )


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


def checkpoint_figure(scores: pd.DataFrame) -> Figure:
    """Plot each model's relative WIS against the fine-tune's outbreaks.

    One panel per study and level, titled with its n of matched cells and
    greyed when n is under `MIN_CELLS`, one line per model, on a log axis
    of outbreaks with the zero-shot model left of the first checkpoint.
    `scores` has one row per study, level, model and checkpoint, with
    `outbreaks` and `rel_wis` against `randomwalknowcast`.
    """
    sc = scores.loc[scores["model"] != "randomwalknowcast"]
    sizes = sorted(sc.loc[sc["outbreaks"] > 0, "outbreaks"].unique())
    # Zero-shot sits at half the first checkpoint on the log axis.
    zero = sizes[0] / 2 if sizes else 500
    ticks = [zero, *sizes]
    labels = ["zero-shot", *(f"{n:,.0f}" for n in sizes)]
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
        for i, (model, g) in enumerate(one.groupby("model")):
            g = g.sort_values("outbreaks")
            x = g["outbreaks"].where(g["outbreaks"] > 0, zero)
            colour = "#bbbbbb" if thin else f"C{i}"
            ax.plot(
                x, g["rel_wis"], marker="o", ms=3, color=colour, label=model
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
        ax.set_xticks(ticks, labels, rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.axhline(1, color="#888888", ls="--", lw=1)
        ax.set_title(f"{study}, {level} (n = {n}{origins})", fontsize=9)
        ax.set_xlabel("Fine-tuning outbreaks")
        ax.set_ylabel("Relative WIS")
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    # The legend from a panel drawn in colour.
    shown = [ax for ax in axes if ax.get_visible() and ax.texts == []]
    handles, labels = (shown or axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
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
    "copula_scores": "fm-copula's per-forecast copula scores (#221)",
    "copula_rho": "fm-copula's rolling correlation estimates (#221)",
    "aggregation_scores": "fm-agg's per-target aggregation scores (#222)",
    "aggregation_cost": "fm-agg's cost of each option (#222)",
    "finetune_heldout": "fm-finetune3's held-out loss curve (#192)",
    "sim_summaries": "fm-finetune3's per-outbreak r3 pilot summaries",
    "sim_realism": "fm-finetune3's data reference values",
    "sim_diseases": "fm-finetune3's published disease values",
    "checkpoint_scores": "germany-chronos2's scores of each r3 checkpoint",
    "scale_floor": "fm-finetune3's scale-floor test",
    "lag_germany": "germany-chronos2's German lag-horizon scores",
    "lag_bdbv_before": "the BDBV scores with the 4-week lag horizon",
    "lag_bdbv_after": "the BDBV scores with lags to the truth delay",
}
"""Each evidence input and what it waits on until set."""


NOT_YET: Final = "(not yet available)"
"""What the paper quotes in place of a number whose rows are not set."""

QUOTED: Final = {
    "rho": {
        "germany": {
            strata: {"randomwalknowcast": {"rho_t": NOT_YET, "rho_p": NOT_YET}}
            for strata in ("states", "ages")
        }
    },
    "realism": {
        "zero_share": {"sim_q95": NOT_YET, "BDBV": NOT_YET},
        "delay_q99": {"sim_q95": NOT_YET, "Germany_states": NOT_YET},
    },
}
"""The numbers the paper quotes, as they read until their rows exist."""


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

    Returns
    -------
    dict
        The quoted numbers.
    """
    si_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = {name: _read(paths, name) for name in EVIDENCE}

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

    cop = data["copula_scores"]
    agg = data["aggregation_scores"]
    for study in ("ebola", "germany"):
        table(
            f"copula-{study}",
            None if cop is None else copula_table(cop, study),
            ["copula_scores"],
        )
        table(
            f"aggregation-{study}",
            None if agg is None else aggregation_table(agg, study),
            ["aggregation_scores"],
        )
    cost = data["aggregation_cost"]
    table(
        "aggregation-cost",
        None if cost is None else cost_table(cost),
        ["aggregation_cost"],
    )
    dis = data["sim_diseases"]
    table(
        "sim-diseases",
        None if dis is None else diseases_table(dis),
        ["sim_diseases"],
    )
    ckpt = data["checkpoint_scores"]
    figure(
        "checkpoint-curve",
        None if ckpt is None else checkpoint_figure(ckpt),
        ["checkpoint_scores"],
    )
    curve = data["finetune_heldout"]
    figure(
        "finetune-heldout",
        None if curve is None else heldout_figure(curve),
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
            "aggregation": (None if agg is None else aggregation_numbers(agg)),
            "realism": (
                QUOTED["realism"] if ref is None else realism_numbers(ref)
            ),
            "waiting": [EVIDENCE[n] for n, d in data.items() if d is None],
        }
    }
    out_yml.write_text(yaml.safe_dump(numbers, sort_keys=False))
    return numbers
