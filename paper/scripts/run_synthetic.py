"""Run the models on a set of simulated outbreaks for the synthetic study.

The workflow's `synthetic_forecasts_<mode>` rule runs it from `paper/` on
the run's seeds (`workflow/config.yaml`):

    PYTHONPATH=. python scripts/run_synthetic.py SIM_DIR OUT --seeds S...

SIM_DIR holds the simulator's `counts-<seed>.csv.gz` and
`settings-<seed>.csv`; only `--seeds` are read. OUT is the long sample
table (`schema.md`); beside it go the observations as of each release,
each series' grid cell and 98% delay, and each model's seconds per
outbreak, named as `paperdata.synthetic.outputs` reads them. A model that
is not registered stops the run, and so does a fine-tuned checkpoint that
cannot be fetched, so give the run a token, `GH_TOKEN=$(gh auth token)`.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from paperdata.synthetic import outputs
from paperdata.synthetic_runner import (
    COLUMNS,
    MODELS,
    evaluate,
    factor_table,
    first_draws,
    hub_codes,
    models_by_name,
    observation_table,
    simulation_files,
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
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--outbreaks", type=int, default=None)
    parser.add_argument("--origins", type=int, default=4)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--draw", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.device == "cpu":
        _cpu(args.threads)

    count_files, settings_files = simulation_files(args.sim_dir, args.seeds)
    counts = first_draws(
        hub_codes(pd.concat([pd.read_csv(p) for p in count_files])),
        args.draw,
    )
    ids = sorted(counts["outbreak"].unique())[: args.outbreaks]
    models, skipped = models_by_name(args.models)
    if skipped:
        raise SystemExit(f"Models not registered: {', '.join(skipped)}.")

    settings = pd.concat([pd.read_csv(p) for p in settings_files])
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
    out.to_csv(args.out, index=False)
    _, observations_file, factors_file = outputs(args.out)
    pd.concat(observed, ignore_index=True).to_csv(
        observations_file, index=False
    )
    pd.concat(factors, ignore_index=True).to_csv(factors_file, index=False)
    pd.DataFrame(timings).to_csv(
        factors_file.with_name(
            factors_file.name.replace("-factors", "-timings")
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
