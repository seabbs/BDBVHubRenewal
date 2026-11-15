# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Write the real validation blocks the trainer scores each checkpoint on.

The blocks are BDBV as known at pseudo-origins before the first release
the paper scores, rebuilt from the hub's public per-case delay table
(`auxiliary-data/weekly-reporting-delays-by-province.csv` of
InsightNet-US/BDBV-Modeling-Hub, its 2026-10-09 vintage, as
`bdbvhubrenewal.data.fetch_hub_data` saves it;
`finetune_chronos.real_blocks`). Every forecast week ends by the last
week before the paper's first scored week, so choosing checkpoints on
them leaks nothing the paper evaluates. Germany's development half is
not used.

    python scripts/finetune_real_validation.py data/delays.csv.gz \\
        output/finetune/real-validation.pkl
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft

VINTAGE = "2026-10-09"
"""The vintage of the hub's delay table that gives the blocks: the first
scored release, and the earliest with a delay table."""

CUTS = ["2026-07-25", "2026-08-01", "2026-08-08"]
"""Pseudo-origins: Saturdays whose forecast weeks end by `LAST_WEEK`."""

LAST_WEEK = "2026-09-05"
"""The week before the paper's first scored week: the 2026-10-09 origin's
first horizon, four weeks before its reference week."""

LAG = 4
"""Release lag $D$ in weeks, as the hub's weekly target."""


def main() -> None:
    """Build the blocks from the delay file `sys.argv[1]`."""
    delays = pd.read_csv(sys.argv[1])
    line_list = delays[delays["vintage"] == VINTAGE]
    blocks = ft.real_blocks(line_list, CUTS, LAST_WEEK, LAG)
    with Path(sys.argv[2]).open("wb") as f:
        pickle.dump(blocks, f)
    print(f"{len(blocks)} blocks from {len(line_list)} cases")


if __name__ == "__main__":
    main()
