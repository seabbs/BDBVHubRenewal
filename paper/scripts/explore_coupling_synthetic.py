"""Couplings of the foundation model's quantile draws on held-out outbreaks.

Run from paper/ (#356, exploration):

    PYTHONPATH=. python scripts/explore_coupling_synthetic.py SIM_DIR OUT \
        --seeds S... [--outbreaks N]

Each outbreak's tasks are the synthetic runner's (`outbreak_tasks`); each
model forecasts each task once per coupling (`scripts/explore_coupling.py`,
Chronos-2 called once per input, its quantiles cached in OUT/q/), scored
against the outbreak's final counts by `variant_scores`. OUT/scores.csv.gz
has one row per model, coupling, origin, level, target and metric.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import pandas as pd
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation.pairing import variant_scores
from bdbvhubrenewal.forecasters import get_model

from paperdata.synthetic_runner import (
    first_draws,
    hub_codes,
    model_task,
    outbreak_tasks,
    simulation_files,
    truth_table,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from explore_coupling import MODELS, VARIANTS, sampler  # noqa: E402


def main() -> None:
    """Score every coupling on held-out outbreaks."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sim_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--outbreaks", type=int, default=None)
    p.add_argument("--origins", type=int, default=4)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache = a.out / "q"
    cache.mkdir(parents=True, exist_ok=True)
    done = a.out / "outbreaks"
    done.mkdir(exist_ok=True)
    count_files, _ = simulation_files(a.sim_dir, a.seeds)
    counts = first_draws(
        hub_codes(pd.concat([pd.read_csv(f) for f in count_files]))
    )
    outbreaks = sorted(counts["outbreak"].unique())[: a.outbreaks]
    for outbreak in outbreaks:
        path = done / f"{outbreak}.csv.gz"
        if path.exists():
            continue
        rows = counts.loc[counts["outbreak"] == outbreak]
        truth = truth_table(rows)
        frames = []
        for task in outbreak_tasks(
            rows, outbreak, a.origins, a.samples, a.seed
        ):
            for name in MODELS:
                base = model_task(get_model(name), task)
                for variant in VARIANTS:
                    model = dataclasses.replace(
                        get_model(name), sample=sampler(variant, cache)
                    )
                    frames.append(
                        variant_scores(
                            model.forecast(base),
                            truth,
                            study="synthetic",
                            model=name,
                            variant=variant,
                            origin=f"{outbreak}|{task.vintage}",
                        )
                    )
        if not frames:
            # Too short for any origin: nothing to score.
            logging.info("%s: no origins", outbreak)
            continue
        write_csv(pd.concat(frames, ignore_index=True), path)
        logging.info("%s done", outbreak)
    scores = pd.concat(
        [
            pd.read_csv(done / f"{o}.csv.gz")
            for o in outbreaks
            if (done / f"{o}.csv.gz").exists()
        ],
        ignore_index=True,
    )
    write_csv(scores, a.out / "scores.csv.gz")


if __name__ == "__main__":
    main()
