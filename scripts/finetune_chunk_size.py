# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Blocks and steps per training chunk, from the trainer's own code.

Cuts every ready simulation in SIM_DIR as `finetune_stream.py` does with
the launcher's settings (`scripts/finetune_continuous.sh`: 32 blocks per
simulation, one draw, subset seed 1, the five layouts, univariate share
0.2, batch 256) and packs them with `plan_by_length`, so the SI's blocks
and steps per chunk of 250 can be checked before a checkpoint lands.

    nice -n 19 uv run --group chronos python scripts/finetune_chunk_size.py \
        SIM_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs


def main() -> None:
    """Print blocks and steps of SIM_DIR's simulations and per chunk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--blocks-per-simulation", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    ft.SOURCES = ("daily", "weekly", "3day", "daily+weekly", "daily+3day")
    ft.UNIVARIATE_SHARE = 0.2
    fs.DRAWS_PER_PASS, fs.SUBSET_SEED = 1, 1
    ft.MAX_SERIES = min(ft.MAX_SERIES, ft.max_series(ft.MEMORY_GB))
    seeds = sorted(
        int(p.name.split("-")[1].split(".")[0])
        for p in args.sim_dir.glob("counts-*.csv.gz")
    )
    ids = fs.ready_outbreaks(args.sim_dir, seeds)
    empty: list[str] = []
    pairs = fs.cut(args.sim_dir, ids, 1, args.blocks_per_simulation, empty)
    blocks = [b for _, b in pairs]
    sizes = np.array([int(b["context"].shape[0]) for b in blocks])
    lengths = np.array([b["prediction_length"] for b in blocks])
    plan = ft.plan_by_length(
        lengths,
        sizes,
        args.batch_size,
        np.random.default_rng([1, 0]),
        ft.MAX_SERIES,
    )
    n = len(ids)
    per = args.chunk / n
    print(
        f"simulations {n}, blocks {len(blocks)} ({len(empty)} empty skipped),"
        f" steps {len(plan)}; per chunk of {args.chunk}: blocks"
        f" {len(blocks) * per:.0f}, steps {len(plan) * per:.0f}"
    )


if __name__ == "__main__":
    main()
