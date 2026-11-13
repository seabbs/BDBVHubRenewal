"""Run the models on the held-out synthetic outbreaks (`paper.synthetic`).

Simulate the held-out outbreaks first, with seeds the fine-tune did not
use:

    for s in 9001 9002 9003 9004; do
      Rscript scripts/finetune_simulate.R 25 $s output/finetune/heldout
    done

The held-out set is kept for the final evaluation (`paper/si/heldout-set.md`).
During development, check the pipeline on the validation set instead
(seed 7001, `output/finetune/validation`). Then, from the repo root:

    uv run --group chronos python -m paper.run_synthetic \
        output/finetune/heldout output/paper/synthetic.csv --device cpu

Writes the long sample table, as parquet if `out` ends `.parquet`.
Beside it go the observations as of each release (`-observations.csv`),
each series' outbreak settings (`-factors.csv`) and each model's seconds
per outbreak (`-timings.csv`).
Models that are not registered are skipped, and so are the fine-tuned
ones until their checkpoint exists.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from paper.synthetic import (
    COLUMNS,
    MODELS,
    evaluate,
    factor_table,
    models_by_name,
    observation_table,
)


def _cpu(threads: int) -> None:
    """Run Chronos-2 on the CPU, with `threads` threads."""
    import torch

    def no_mps() -> bool:
        return False

    # torch._dynamo reads the original through `__wrapped__`.
    no_mps.__wrapped__ = torch.backends.mps.is_available  # type: ignore[attr-defined]
    torch.backends.mps.is_available = no_mps
    torch.set_num_threads(threads)


def main() -> None:
    """Forecast each held-out outbreak with each model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--outbreaks", type=int, default=None)
    parser.add_argument("--origins", type=int, default=4)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.device == "cpu":
        _cpu(args.threads)

    counts = pd.concat(
        [pd.read_csv(p) for p in sorted(args.sim_dir.glob("counts-*.csv.gz"))]
    )
    ids = sorted(counts["outbreak"].unique())[: args.outbreaks]
    models, skipped = models_by_name(args.models)
    for name in skipped:
        logging.warning("%s is not registered, so is skipped", name)

    settings = pd.concat(
        [pd.read_csv(p) for p in sorted(args.sim_dir.glob("settings-*.csv"))]
    )
    tables, timings, observed, factors = [], [], [], []
    for outbreak in ids:
        rows = counts.loc[counts["outbreak"] == outbreak]
        observed.append(observation_table(rows, outbreak, args.origins))
        factors.append(factor_table(rows, settings))
        for model in models:
            t0 = time.perf_counter()
            table = evaluate(
                rows, [model], args.origins, args.samples, args.seed
            )
            seconds = time.perf_counter() - t0
            n = table["origin_date"].nunique()
            logging.info(
                "%s %s: %d origins in %.1f s", outbreak, model.name, n, seconds
            )
            tables.append(table)
            timings.append(
                {
                    "outbreak": outbreak,
                    "model": model.name,
                    "n_places": rows["location"].nunique(),
                    "origins": n,
                    "seconds": seconds,
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(tables, ignore_index=True) if tables else None
    if out is None or out.empty:
        out = pd.DataFrame(columns=list(COLUMNS))
    if args.out.suffix == ".parquet":
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    pd.concat(observed, ignore_index=True).to_csv(
        args.out.with_name(args.out.stem + "-observations.csv"), index=False
    )
    pd.concat(factors, ignore_index=True).to_csv(
        args.out.with_name(args.out.stem + "-factors.csv"), index=False
    )
    pd.DataFrame(timings).to_csv(
        args.out.with_name(args.out.stem + "-timings.csv"), index=False
    )


if __name__ == "__main__":
    main()
