"""Score the Chronos-2 models on the German benchmark's development half.

Reads `germany-backtest` and `germany-teams` output for the development
half (nowcast dates before the split date) and writes, to OUT,
`scores.csv` (relative WIS and log WIS against two baselines, and
coverage, by level on the cells every model of a level made) and
`overlay.png` (national and two age groups at three dates).

Usage: python scripts/germany_chronos_dev.py RESULTS OUT
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bdbvhubrenewal.data.germany import hub_target, hub_triangle, nowcast_tasks
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.forecasters import foundation as fm

BASELINES = ("randomwalknowcast", "KIT-frozen_baseline")
CELL = ["origin", "location", "target_end_date"]
LEVELS = ("national", "states", "age")
PLOT_PLACES = ("DE", "35-59", "80+")
PLOT_DATES = ("2021-11-29", "2021-12-27", "2022-01-31")
SOURCE = "daily"
PLOT_MODELS = {
    f"fm_chronos_{SOURCE}_ragged_ft": "#1b9e77",
    f"fm_chronos_{SOURCE}_ragged": "#d95f02",
    f"fm_chronos_{SOURCE}_complete": "#7570b3",
    "randomwalknowcast": "#e7298a",
    "MeanEnsemble": "#66a61e",
}


def scores(forecasts: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Score on the natural and the log scale (log WIS on log1p)."""
    nat = de.score_germany(forecasts, delay)
    logged = forecasts.assign(
        value=np.log1p(forecasts["value"].clip(lower=0)),
        truth=np.log1p(forecasts["truth"]),
    )
    log = de.score_germany(logged, delay)[["model", *CELL, "wis"]]
    return nat.merge(
        log.rename(columns={"wis": "log_wis"}), on=["model", *CELL]
    )


def matched(sc: pd.DataFrame) -> pd.DataFrame:
    """Keep, in each level, the cells every model of that level made.

    A model that covers under half a level's cells (ILM's states, RKI's
    ages) is left out of that level.
    """
    out = []
    for _, rows in sc.groupby("pool"):
        cells = rows.groupby("model").size()
        keep = cells.index[cells >= 0.5 * cells.max()]
        rows = rows.loc[rows["model"].isin(keep)]
        n = rows.groupby(CELL)["model"].nunique()
        full = n.index[n == len(keep)]
        out.append(rows.set_index(CELL).loc[full].reset_index())
    return pd.concat(out, ignore_index=True)


def table(sc: pd.DataFrame) -> pd.DataFrame:
    """Return mean scores and relative scores by level and model."""
    mean = sc.groupby(["pool", "model"])[
        ["wis", "log_wis", "coverage_50", "coverage_95"]
    ].mean()
    mean["n"] = sc.groupby(["pool", "model"]).size()
    for base in BASELINES:
        ref = mean.xs(base, level="model")
        for m in ("wis", "log_wis"):
            mean[f"rel_{m}_{base}"] = (
                mean[m]
                / ref[m].reindex(mean.index.get_level_values("pool")).to_numpy()
            )
    mean = mean.reset_index()
    mean["pool"] = pd.Categorical(mean["pool"], list(LEVELS))
    return mean.sort_values(["pool", "rel_wis_randomwalknowcast"])


def context_end(task, model: str) -> str:
    """Return the last context step Chronos had for a task (task days)."""
    source = "weekly" if "_weekly_" in model else "daily"
    rows = task.target_history if source == "weekly" else task.daily_history
    tri = fm.hub_triangle(
        rows, source, ("DE", *task.locations), task.target_end_dates[-1]
    )
    if model.endswith("_complete"):
        return fm._walk_start(task)
    return tri.times[tri.n_context - 1]


def overlay(fc: pd.DataFrame, triangle: pd.DataFrame, cfg, out: Path) -> None:
    """Draw the nowcasts of `PLOT_MODELS` at `PLOT_DATES`."""
    fig, axes = plt.subplots(
        len(PLOT_PLACES), len(PLOT_DATES), figsize=(13, 10), sharex="col"
    )
    asof = de.asof_sums(triangle, list(PLOT_DATES))
    ends = sorted(set(asof["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), cfg.truth_delay)
    for j, date in enumerate(PLOT_DATES):
        tasks = nowcast_tasks(triangle, date, 10, cfg.seed)
        for i, place in enumerate(PLOT_PLACES):
            ax = axes[i, j]
            task = tasks[1] if place != "DE" else tasks[0]
            seen = asof.loc[(asof.origin == date) & (asof.location == place)]
            tr = truth.loc[
                (truth.location == place)
                & truth.target_end_date.isin(seen.target_end_date)
            ].sort_values("target_end_date")
            ax.plot(
                pd.to_datetime(tr.target_end_date),
                tr.truth,
                color="black",
                lw=1.2,
                label="40-day truth",
            )
            seen = seen.sort_values("target_end_date")
            ax.plot(
                pd.to_datetime(seen.target_end_date),
                seen.total,
                color="grey",
                ls="--",
                lw=1,
                label="data as of date",
            )
            for model, col in PLOT_MODELS.items():
                q = fc.loc[
                    (fc.model == model)
                    & (fc.origin == date)
                    & (fc.location == place)
                ].pivot_table(
                    index="target_end_date", columns="quantile", values="value"
                )
                if q.empty:
                    continue
                # Forecasts are on the nowcast date's own days; the task's
                # are moved to the next Saturday.
                shift = pd.Timestamp(task.reference_date) - pd.Timestamp(date)
                x = pd.to_datetime(q.index)
                start = None
                if model.startswith("fm_chronos"):
                    start = pd.Timestamp(context_end(task, model)) - shift
                    keep = x >= start
                    q, x = q.loc[keep], x[keep]
                    anchor = seen.loc[
                        pd.to_datetime(seen.target_end_date) == start, "total"
                    ]
                    if len(anchor) and (not len(x) or start < x[0]):
                        q = pd.concat(
                            [
                                pd.DataFrame(
                                    {c: [anchor.iloc[0]] for c in q.columns},
                                    index=[start.date().isoformat()],
                                ),
                                q,
                            ]
                        )
                        x = x.insert(0, start)
                ax.fill_between(x, q[0.25], q[0.75], color=col, alpha=0.15)
                ax.plot(x, q[0.5], color=col, lw=1.6, marker=".", label=model)
            ax.set_title(f"{place}, nowcast {date}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(
        "German 7-day hospitalisations: median and 50% interval; each "
        "Chronos line starts at its last context step"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=130)


def main(results: Path, out: Path) -> None:
    """Score the development half and draw the overlay."""
    cfg = de.load_germany_config("config/germany.toml")
    read = {"dtype": {"origin": str, "target_end_date": str}}
    fc = pd.concat(
        [
            pd.read_csv(results / "forecasts.csv.gz", **read),
            pd.read_csv(results / "teams-forecasts.csv.gz", **read),
        ],
        ignore_index=True,
    )
    fc = fc.loc[fc["origin"] < cfg.split_date]
    out.mkdir(parents=True, exist_ok=True)
    sc = matched(scores(fc, cfg.truth_delay))
    tab = table(sc)
    tab.round(3).to_csv(out / "scores.csv", index=False)
    print(tab.round(3).to_string(index=False))
    triangle = hub_triangle(Path("data/raw/germany") / Path(cfg.triangle).name)
    overlay(fc, triangle, cfg, out / "overlay.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
