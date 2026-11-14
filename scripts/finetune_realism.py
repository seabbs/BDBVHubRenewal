"""Compare simulated outbreaks with real data before fine-tuning.

Summaries of a sample of outbreaks from `scripts/finetune_simulate.R`
against the same summaries of the BDBV releases (latest release) and the
development half of the German hospitalisation data (report dates up to
2022-02-09, the middle of the German hub's nowcast dates; the second half
is held out and never read here). Run from the repo root:

    uv run python scripts/finetune_realism.py OUT_PNG OUT_CSV GERMANY_CSV \
        SIM_DIR...
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bdbvhubrenewal.data import load_delays, load_target

GERMANY_LAST = "2022-02-09"
"""Last report date of the German development half."""

SAMPLE = 200
"""Simulated outbreaks compared."""

LAGS = (0, 7, 14, 28)
"""Days by which the share reported is summarised."""


def weekly_summary(weekly: pd.DataFrame) -> dict[str, float]:
    """Return summaries of weekly counts by stratum.

    `weekly` has `location`, `week` and `count`, and the total is the sum
    over strata.
    """
    total = weekly.groupby("week")["count"].sum().sort_index()
    active = total.index[total > 0]
    total = total.loc[active.min() : active.max()]
    w = total.to_numpy(dtype=float)
    ok = (w[:-1] >= 5) & (w[1:] >= 5)
    growth = np.log(w[1:][ok] / w[:-1][ok])
    smooth = np.convolve(w, np.ones(3) / 3, mode="same")
    inner = slice(1, len(w) - 1)
    keep = smooth[inner] > 0
    dispersion = np.mean(
        (w[inner][keep] - smooth[inner][keep]) ** 2 / smooth[inner][keep]
    )
    by = weekly.pivot_table(
        index="week", columns="location", values="count", aggfunc="sum"
    ).reindex(total.index)
    return {
        "size": w.sum(),
        "peak_weekly": w.max(),
        "growth_median": np.median(growth) if len(growth) else np.nan,
        "growth_sd": np.std(growth) if len(growth) > 1 else np.nan,
        "zero_share": float(by.fillna(0).eq(0).to_numpy().mean())
        if by.shape[1] > 1
        else float((w == 0).mean()),
        "dispersion": dispersion,
    }


def phases(weekly: pd.Series, last: pd.Timestamp | None = None) -> dict:
    """Return the shares of origins at which the total rises, is flat, falls.

    At an origin, the log ratio of the week to two weeks earlier is above
    0.2 (rising), below -0.2 (falling) or between (flat). Origins are the
    weeks from the third after the first case to `last`.
    """
    w = weekly.sort_index()
    w = w.loc[w.index >= w.index[w.to_numpy() > 0][0]]
    if last is not None:
        w = w.loc[w.index <= last]
    x = np.log(w.to_numpy(dtype=float) + 1)
    if len(x) < 3:
        return {}
    ratio = x[2:] - x[:-2]
    return {
        "origin_rising": float((ratio > 0.2).mean()),
        "origin_flat": float((np.abs(ratio) <= 0.2).mean()),
        "origin_falling": float((ratio < -0.2).mean()),
    }


def delay_summary(delays: np.ndarray, weights: np.ndarray) -> dict:
    """Mean, 99% point and share reported by `LAGS` of a delay sample."""
    order = np.argsort(delays)
    d, w = delays[order], weights[order] / weights.sum()
    cdf = np.cumsum(w)
    out = {
        "delay_mean": float(np.sum(d * w)),
        "delay_q99": float(d[np.searchsorted(cdf, 0.99)]),
    }
    for lag in LAGS:
        out[f"reported_by_{lag}d"] = float(w[d <= lag].sum())
    return out


def report_weekdays(report_dates: pd.Series, weights: np.ndarray) -> dict:
    """Share of reports by report weekday."""
    days = pd.to_datetime(report_dates).dt.day_of_week.to_numpy()
    share = np.bincount(days, weights=weights, minlength=7) / weights.sum()
    return {f"report_dow_{i}": share[i] for i in range(7)}


def simulated(sim_dirs: list[Path], rng: np.random.Generator) -> pd.DataFrame:
    """Summaries of a sample of simulated outbreaks."""
    files = sorted(p for d in sim_dirs for p in d.glob("counts-*.csv.gz"))
    counts = pd.concat([pd.read_csv(p) for p in files])
    ids = counts["outbreak"].unique()
    ids = rng.choice(ids, size=min(SAMPLE, len(ids)), replace=False)
    rows = []
    for outbreak in ids:
        c = counts.loc[counts["outbreak"] == outbreak]
        ref = pd.to_datetime(c["reference_date"])
        rep = pd.to_datetime(c["report_date"])
        week = (ref - pd.to_timedelta(ref.dt.day_of_week, unit="D")).dt.date
        weekly = (
            c.assign(week=week)
            .groupby(["location", "week"], as_index=False)["count"]
            .sum()
        )
        delays = (rep - ref).dt.days.to_numpy()
        w = c["count"].to_numpy(dtype=float)
        total = weekly.groupby("week")["count"].sum()
        total.index = pd.to_datetime(total.index)
        # Blocks are cut at least the longest delay plus five weeks before
        # the last report.
        last = rep.max() - pd.Timedelta(weeks=int(delays.max()) // 7 + 5)
        rows.append(
            {
                "source": "simulated",
                "id": outbreak,
                **weekly_summary(weekly),
                **delay_summary(delays, w),
                **report_weekdays(c["report_date"], w),
                **phases(total, last),
            }
        )
    return pd.DataFrame(rows)


def bdbv() -> dict:
    """Summaries of the latest BDBV release, by province."""
    t = load_target("data")
    t = t.loc[(t["vintage"] == t["vintage"].max()) & (t["location"] != "CD")]
    weekly = t.rename(columns={"target_end_date": "week", "total": "count"})
    total = weekly.groupby("week")["count"].sum()
    total.index = pd.to_datetime(total.index)
    d = load_delays("data")
    d = d.loc[d["vintage"] == d["vintage"].max(), "delay"].to_numpy()
    d = d[(d >= 0) & (d <= 120)].astype(float)
    return {
        "source": "BDBV",
        "id": "BDBV",
        **weekly_summary(weekly[["location", "week", "count"]]),
        **delay_summary(d, np.ones_like(d)),
        **phases(total, total.index.max() - pd.Timedelta(weeks=4)),
    }


def germany(path: Path) -> list[dict]:
    """Summaries of the German development half, by state and by age."""
    wide = pd.read_csv(path)
    changes = [c for c in wide.columns if c.startswith("value_")]
    long = wide.melt(
        id_vars=["date", "location", "age_group"],
        value_vars=changes,
        var_name="delay",
        value_name="count",
    ).dropna(subset=["count"])
    long["delay"] = long["delay"].map({c: i for i, c in enumerate(changes)})
    ref = pd.to_datetime(long["date"])
    long["report_date"] = ref + pd.to_timedelta(long["delay"], unit="D")
    long = long.loc[long["report_date"] <= GERMANY_LAST]
    long = long.loc[long["count"] > 0]
    out = []
    for name, rows in (
        ("Germany states", long.loc[long["age_group"] == "00+"]),
        (
            "Germany ages",
            long.loc[(long["location"] == "DE") & (long["age_group"] != "00+")],
        ),
    ):
        rows = rows.loc[rows["location"] != "DE"] if "states" in name else rows
        rows = rows.assign(
            location=rows["location"] + rows["age_group"],
            week=pd.to_datetime(rows["date"]).dt.to_period("W").dt.start_time,
        )
        weekly = rows.groupby(["location", "week"], as_index=False)[
            "count"
        ].sum()
        # Delays only of dates with every report up to 80 days seen.
        last = pd.Timestamp(GERMANY_LAST) - pd.Timedelta(days=81)
        seen = rows.loc[pd.to_datetime(rows["date"]) <= last]
        w = seen["count"].to_numpy(dtype=float)
        total = weekly.groupby("week")["count"].sum()
        # Origins are the German hub's development nowcast dates.
        total = total.loc[total.index >= "2021-11-15"]
        out.append(
            {
                "source": name,
                "id": name,
                **phases(total, pd.Timestamp(GERMANY_LAST)),
                **weekly_summary(weekly),
                **delay_summary(seen["delay"].to_numpy(dtype=float), w),
                **report_weekdays(seen["report_date"], w),
            }
        )
    return out


def main() -> None:
    """Write the figure and the table."""
    out_png, out_csv, germany_csv = map(Path, sys.argv[1:4])
    sim_dirs = [Path(p) for p in sys.argv[4:]]
    sims = simulated(sim_dirs, np.random.default_rng(1))
    real = pd.DataFrame([bdbv(), *germany(germany_csv)])
    stats = [
        ("size", True),
        ("peak_weekly", True),
        ("growth_median", False),
        ("growth_sd", False),
        ("zero_share", False),
        ("dispersion", True),
        ("delay_mean", True),
        ("delay_q99", True),
        *[(f"reported_by_{lag}d", False) for lag in LAGS],
        ("origin_rising", False),
        ("origin_falling", False),
    ]
    fig, axes = plt.subplots(3, 6, figsize=(21, 10))
    colours = {"BDBV": "C3", "Germany states": "C2", "Germany ages": "C1"}
    for ax, (stat, log) in zip(axes.flat, stats, strict=False):
        x = sims[stat].dropna().to_numpy()
        if log:
            x = x[x > 0]
            bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 30)
            ax.set_xscale("log")
        else:
            bins = 30
        ax.hist(x, bins=bins, color="grey", alpha=0.7)
        for _, r in real.iterrows():
            if pd.notna(r[stat]):
                ax.axvline(r[stat], color=colours[r["source"]], lw=2)
        ax.set_title(stat, fontsize=9)
    ax = axes.flat[len(stats)]
    dow = [f"report_dow_{i}" for i in range(7)]
    q = sims[dow].quantile([0.1, 0.5, 0.9])
    ax.fill_between(range(7), q.loc[0.1], q.loc[0.9], color="grey", alpha=0.4)
    ax.plot(range(7), q.loc[0.5], color="grey")
    for _, r in real.iterrows():
        if pd.notna(r.get("report_dow_0")):
            share = r[dow].to_numpy(float)
            ax.plot(range(7), share, color=colours[r["source"]])
    ax.set_xticks(range(7), list("MTWTFSS"))
    ax.set_title("share reported by weekday", fontsize=9)
    for ax in axes.flat[len(stats) + 1 :]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in colours.values()]
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color="grey"), *handles],
        [f"simulated ({len(sims)})", *colours],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    cols = [s for s, _ in stats]
    table = pd.concat(
        [
            sims[cols]
            .quantile([0.05, 0.5, 0.95])
            .T.rename(columns=lambda q: f"sim q{int(q * 100)}"),
            real.set_index("source")[cols].T,
        ],
        axis=1,
    )
    table.to_csv(out_csv)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
