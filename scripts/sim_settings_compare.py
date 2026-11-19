"""Compare simulated summaries with real ones, as a markdown table.

Reads `scripts/sim_settings_check.R`'s and `scripts/sim_settings_real.py`'s
outputs and prints, per summary, the simulated 1%, 5%, 50%, 95% and 99%
centiles beside the range over each real source's series.

    uv run python scripts/sim_settings_compare.py <sim.csv> <real.csv>
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SUMMARIES = {
    "max_growth": ("max_growth", "max_growth", "fastest daily growth rate"),
    "doubling_days": ("doubling_days", None, "doubling time at it (days)"),
    "peak_daily": (
        "peak_per_stratum",
        "peak_daily",
        "peak daily count per series",
    ),
    "total": ("total_per_stratum", "total", "total count per series"),
    "delay_mean": ("delay_mean", "delay_mean", "mean delay (days)"),
    "delay_q95": ("delay_q95", "delay_q95", "95th centile of the delay (days)"),
    "reported_7": ("reported_7", "reported_7", "share reported by day 7"),
    "reported_14": ("reported_14", "reported_14", "share reported by day 14"),
    "reported_28": ("reported_28", "reported_28", "share reported by day 28"),
    "cv_large": (
        "cv_at_peak",
        "cv_large",
        "relative noise at large counts (500 or more a day)",
    ),
    "dow": ("onset_dow_ratio", "onset_dow_ratio", "weekday effect, max / min"),
    "shock_mult": ("shock_mult", None, "backlog size (x mean delay)"),
}


def fmt(x: float) -> str:
    """Return a number to about two significant figures."""
    if not np.isfinite(x):
        return "-"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.2g}"


def main() -> None:
    """Print the comparison table and the realised shock shares."""
    sim = pd.read_csv(sys.argv[1])
    real = pd.read_csv(sys.argv[2])
    real["doubling_days"] = np.log(2) / real["max_growth"]
    lines = [
        "| summary | simulated 1% / 5% / 50% / 95% / 99% | Germany | BDBV |",
        "| --- | --- | --- | --- |",
    ]
    for key, (s, r, label) in SUMMARIES.items():
        rows = sim
        if key == "cv_large":
            rows = sim.loc[sim["peak_per_stratum"] >= 500]
        x = rows[s].replace([np.inf, -np.inf], np.nan).dropna()
        q = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
        cells = []
        for source in ("Germany", "BDBV"):
            col = r or key
            if col not in real:
                cells.append("-")
                continue
            y = (
                real.loc[real["source"] == source, col]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            cells.append(f"{fmt(y.min())} to {fmt(y.max())}" if len(y) else "-")
        sims = " / ".join(fmt(v) for v in q)
        lines.append(f"| {label} | {sims} | {cells[0]} | {cells[1]} |")
    shares = sim["shocks"].value_counts(normalize=True)
    quiet = (
        (sim["shocks"] == "none") & (sim["delay_change"] == "static")
    ).mean()
    lines.append("")
    lines.append(
        "Shocks: "
        + ", ".join(f"{k} {v:.0%}" for k, v in shares.items())
        + f"; no shock and a static delay: {quiet:.0%}."
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
