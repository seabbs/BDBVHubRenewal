"""Interval widths relative to the final count on the German development half.

Reads the checkpoint scorer's cached German rows (<cache>/germany/<label>/
<date>.csv.gz: hub quantiles 0.025-0.975 with truth) and writes, per
checkpoint, model and horizon, the median and mean 80% and 95% interval
widths over the final count, and coverage.

    python scripts/ftpractice/german_widths.py CACHE OUT.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "reference": ("zero-shot", 0),
    "slots-cbd59b585052": ("r5", 250),
    "slots-88bf85407baf": ("r5", 500),
    "slots-d21de535479f": ("r5", 750),
    "slots-5605742212cc": ("r5", 1000),
    "slots-444ffce9213d": ("r5", 2000),
    "slots-56221d82b0ed": ("r5", 4000),
    "slots-6d3c64da9bc2": ("r6", 250),
    "slots-6bb954c7ffe7": ("r6", 500),
}
INTERVALS = {"80": (0.1, 0.9), "95": (0.025, 0.975)}
HORIZONS = (0, -7, -14, -21, -28)


def read(cache: Path, label: str) -> pd.DataFrame:
    """Return a checkpoint's cached German rows."""
    files = sorted((cache / "germany" / label).glob("*.csv.gz"))
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def widths(rows: pd.DataFrame) -> pd.DataFrame:
    """Return widths and coverage by model and horizon."""
    rows = rows[rows["horizon"].isin(HORIZONS)]
    # Zero-shot files also hold daily rows; the hub target is the 7-day sum.
    if "target" in rows.columns or rows.shape[1] > 9:
        col = "target" if "target" in rows.columns else rows.columns[9]
        rows = rows[rows[col].fillna("7day") == "7day"]
    wide = rows.pivot_table(
        index=["model", "origin", "location", "horizon"],
        columns="quantile",
        values="value",
    ).reset_index()
    truth = rows.drop_duplicates(["model", "origin", "location", "horizon"])
    wide = wide.merge(
        truth[["model", "origin", "location", "horizon", "truth"]],
        on=["model", "origin", "location", "horizon"],
    )
    out = []
    for (model, h), g in wide.groupby(["model", "horizon"]):
        rec = {"model": model, "horizon": h, "rows": len(g)}
        final = np.maximum(g["truth"].to_numpy(), 1.0)
        for name, (lo, hi) in INTERVALS.items():
            w = (g[hi] - g[lo]).to_numpy() / final
            rec[f"rel_width{name}_median"] = float(np.median(w))
            rec[f"rel_width{name}_mean"] = float(np.mean(w))
            rec[f"cover{name}"] = float(
                ((g["truth"] >= g[lo]) & (g["truth"] <= g[hi])).mean()
            )
        rec["median_over_final"] = float(np.median(g[0.5].to_numpy() / final))
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    """Write the table and print the ragged models' widths."""
    cache, out = Path(sys.argv[1]), Path(sys.argv[2])
    frames = []
    for label, (run, n) in LABELS.items():
        w = widths(read(cache, label))
        w.insert(0, "n", n)
        w.insert(0, "run", run)
        w.insert(0, "label", label)
        frames.append(w)
    table = pd.concat(frames, ignore_index=True)
    table["model"] = table["model"].str.replace("_ft", "", regex=False)
    table.to_csv(out, index=False)
    keep = table[
        table["model"].isin(
            ["fm_chronos_daily_ragged", "fm_chronos_weekly_ragged"]
        )
    ]
    print(
        keep.pivot_table(
            index=["model", "run", "n"],
            columns="horizon",
            values="rel_width95_median",
        )
        .round(2)
        .to_string()
    )
    print(
        keep.pivot_table(
            index=["model", "run", "n"], columns="horizon", values="cover95"
        )
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
