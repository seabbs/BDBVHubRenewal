"""Pairing sensitivity of the walk models on simulated outbreaks.

The workflow's `pairing_synthetic_<mode>` rule runs it from `paper/` on
the run's seeds, as the synthetic runner (`scripts/run_synthetic.py`):

    PYTHONPATH=. python scripts/pairing_synthetic.py SIM_DIR OUT --seeds S...

Each outbreak's tasks are the synthetic runner's (`outbreak_tasks`, the
same origins and horizons), with the strata in the hub's codes, so the
country, the intermediate level (provinces) and the leaves (zones) are
scored apart. Each walk model forecasts each task once and its draws are
scored with every pairing (`bdbvhubrenewal.evaluation.pairing`) against
the outbreak's final counts. OUT is the `variant_scores` table, with
`study` `synthetic`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation.pairing import (
    MODELS,
    SCORE_COLUMNS,
    variant_forecasts,
    variant_scores,
)
from bdbvhubrenewal.forecasters import get_model

from paperdata.synthetic_runner import (
    first_draws,
    hub_codes,
    model_task,
    outbreak_tasks,
    simulation_files,
    truth_table,
)


def main() -> None:
    """Score each held-out outbreak with every pairing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--outbreaks", type=int, default=None)
    parser.add_argument("--origins", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--draw", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    count_files, _ = simulation_files(args.sim_dir, args.seeds)
    counts = first_draws(
        hub_codes(pd.concat([pd.read_csv(p) for p in count_files])),
        args.draw,
    )
    frames = []
    for outbreak in sorted(counts["outbreak"].unique())[: args.outbreaks]:
        rows = counts.loc[counts["outbreak"] == outbreak]
        truth = truth_table(rows)
        tasks = outbreak_tasks(
            rows, outbreak, args.origins, args.samples, args.seed
        )
        for task in tasks:
            for name in MODELS:
                # As the synthetic runner gives each model its task.
                mine = model_task(get_model(name), task)
                for label, drawn in variant_forecasts(name, mine).items():
                    frames.append(
                        variant_scores(
                            drawn,
                            truth,
                            study="synthetic",
                            model=name,
                            variant=label,
                            origin=f"{outbreak}|{task.vintage}",
                        )
                    )
        logging.info("%s: %d origins", outbreak, len(tasks))
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(SCORE_COLUMNS))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, args.out)


if __name__ == "__main__":
    main()
