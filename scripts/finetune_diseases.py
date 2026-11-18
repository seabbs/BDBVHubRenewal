"""Simulated outbreaks against published values for infectious diseases.

Compares a sample of simulations from `scripts/finetune_simulate.R` with
published generation times or serial intervals, doubling and halving
times, reporting delays and outbreak sizes (`DISEASES`, a CSV with
columns disease, quantity, value, unit, source_doi, note), with BDBV and
Germany's development half as two further points. Run from the repo root:

    uv run python scripts/finetune_diseases.py SIM_DIR DISEASES OUT_PNG \
        OUT_CSV
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASE_STUDIES = {
    "BDBV": {"outbreak_size": 10878, "delay_mean": 6.9, "delay_q99": 36},
    "Germany (dev)": {"outbreak_size": 195661, "delay_mean": 9.6},
}
"""Values of the case studies from `scripts/finetune_realism.py`."""


def first_number(text: str) -> float:
    """Return the first number in a published value."""
    return float(re.search(r"[\d.]+", str(text)).group())


def simulated(sim_dir: Path) -> pd.DataFrame:
    """Summaries of each simulation's first draw."""
    settings = pd.concat(
        pd.read_csv(p) for p in sim_dir.glob("settings-*.csv")
    ).set_index("outbreak")
    rows = []
    for path in sim_dir.glob("counts-*.csv.gz"):
        counts = pd.read_csv(path)
        counts = counts[counts["outbreak"].str.endswith("-1")]
        for outbreak, c in counts.groupby("outbreak"):
            simulation = outbreak.rsplit("-", 1)[0]
            ref = pd.to_datetime(c["reference_date"])
            delay = (pd.to_datetime(c["report_date"]) - ref).dt.days
            w = c["count"].to_numpy(float)
            order = np.argsort(delay.to_numpy())
            cdf = np.cumsum(w[order]) / w.sum()
            weekly = c.groupby(ref.dt.to_period("W"))["count"].sum()
            x = np.log(weekly.to_numpy(float) + 1)
            growth = np.diff(x) / 7
            settled = growth[np.abs(growth) > 0.01]
            s = settings.loc[simulation]
            rows.append(
                {
                    "simulation": simulation,
                    "trajectory": s["trajectory"],
                    "gt_mean": s["gt_mean"],
                    "outbreak_size": w.sum(),
                    "delay_mean": float(np.sum(delay.to_numpy() * w) / w.sum()),
                    "delay_q99": float(
                        delay.to_numpy()[order][np.searchsorted(cdf, 0.99)]
                    ),
                    "doubling_time": np.log(2) / np.median(settled[settled > 0])
                    if (settled > 0).any()
                    else np.nan,
                    "halving_time": np.log(2) / -np.median(settled[settled < 0])
                    if (settled < 0).any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def published(path: Path) -> pd.DataFrame:
    """Return the published values, mapped to the summaries compared."""
    p = pd.read_csv(path)
    names = {
        "generation_time_mean": "gt_mean",
        "serial_interval_mean": "gt_mean",
        "doubling_time": "doubling_time",
        "halving_time": "halving_time",
        "onset_to_confirmation_median": "delay_median",
        "onset_to_confirmation_q95": "delay_q95",
        "onset_to_hospitalisation_mean": "delay_mean",
        "outbreak_size": "outbreak_size",
    }
    p = p[p["quantity"].isin(names)].copy()
    p["summary"] = p["quantity"].map(names)
    p["number"] = p["value"].map(first_number)
    return p


def main() -> None:
    """Write the figure and the table."""
    sim_dir, diseases, out_png, out_csv = map(Path, sys.argv[1:5])
    sims = simulated(sim_dir)
    pub = published(diseases)
    panels = [
        ("gt_mean", "generation time or serial interval mean (days)", False),
        ("doubling_time", "doubling time (days)", True),
        ("halving_time", "halving time (days)", True),
        ("delay_mean", "reporting delay mean (days)", True),
        ("delay_q99", "reporting delay 99% point (days)", True),
        ("outbreak_size", "outbreak size (cases)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    diseases_seen = sorted(pub["disease"].unique())
    colours = dict(zip(diseases_seen, plt.cm.tab10.colors, strict=False))
    for ax, (stat, label, log) in zip(axes.flat, panels, strict=True):
        x = sims[stat].dropna().to_numpy()
        x = x[x > 0] if log else x
        bins = (
            np.logspace(np.log10(x.min()), np.log10(x.max()), 30) if log else 30
        )
        ax.hist(x, bins=bins, color="grey", alpha=0.6)
        if log:
            ax.set_xscale("log")
        for _, r in pub[pub["summary"] == stat].iterrows():
            ax.axvline(r["number"], color=colours[r["disease"]], lw=2)
        for name, values in CASE_STUDIES.items():
            if stat in values:
                ax.axvline(values[stat], color="black", ls="--", lw=1.5)
                ax.text(values[stat], ax.get_ylim()[1] * 0.9, name, fontsize=7)
        ax.set_title(label, fontsize=9)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in colours.values()]
    fig.legend(handles, list(colours), loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=80)
    rows = []
    for stat, label, _ in panels:
        x = sims[stat].dropna()
        q = x.quantile([0.05, 0.5, 0.95]) if len(x) else pd.Series(dtype=float)
        refs = pub[pub["summary"] == stat]
        rows.append(
            {
                "summary": label,
                "sim_q5": q.get(0.05),
                "sim_q50": q.get(0.5),
                "sim_q95": q.get(0.95),
                "published": "; ".join(
                    f"{r.disease} {r.number:g}" for r in refs.itertuples()
                ),
                "sources": "; ".join(sorted(set(refs["source_doi"]))),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
