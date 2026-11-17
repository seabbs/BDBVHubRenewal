"""Rejections and coverage of a pilot of the simulation grid, by factor.

For each factor of `scripts/finetune_simulate.R`'s design and each of its
levels: the draws tried, the share rejected or failed by cause, and the
share of kept simulations against the intended share (even across
levels). Run from the repo root:

    uv run python scripts/finetune_pilot.py SIM_DIR OUT_CSV
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

FACTORS = (
    "process",
    "gt_shape",
    "trajectory",
    "delay_family",
    "delay_form",
    "delay_change",
    "reporting",
    "reference_dow",
    "observation",
    "strata",
)


def main() -> None:
    """Print and write the table."""
    sim_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    kept = pd.concat(pd.read_csv(p) for p in sim_dir.glob("settings-*.csv"))
    files = list(sim_dir.glob("failures-*.csv"))
    failed = pd.concat(pd.read_csv(p) for p in files) if files else None
    kept = kept.assign(outcome="kept")
    draws = kept
    if failed is not None:
        cause = failed["reason"].str.extract(
            r"^(too large|too small|No chains)", expand=False
        )
        failed = failed.assign(outcome=cause.fillna("other error"))
        draws = pd.concat([kept, failed], ignore_index=True)
    total = len(draws)
    print(
        f"{len(kept)} simulations kept from {total} draws; "
        + ", ".join(
            f"{k} {v / total:.1%}"
            for k, v in draws["outcome"].value_counts().items()
        )
    )
    rows = []
    for factor in FACTORS:
        levels = draws[factor].dropna().unique()
        for level in sorted(levels, key=str):
            at = draws[draws[factor] == level]
            row = {
                "factor": factor,
                "level": level,
                "draws": len(at),
                "kept_share": (kept[factor] == level).mean(),
                "intended_share": 1 / len(levels),
            }
            for outcome in ("too large", "too small", "No chains", "other"):
                hits = at["outcome"].str.startswith(outcome).sum()
                row[f"{outcome} rate"] = hits / len(at)
            rows.append(row)
    table = pd.DataFrame(rows).round(3)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
