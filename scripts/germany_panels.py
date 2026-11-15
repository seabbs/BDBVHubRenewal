"""German national nowcast panels, one per model (`nowcast_panels`).

Usage: python scripts/germany_panels.py RESULTS CHECKPOINTS OUT

RESULTS holds `forecasts.csv.gz` (our models, round 1 fine-tuned as
`fm_chronos_daily_ragged_ft`) and `teams-forecasts.csv.gz`. CHECKPOINTS
holds `<hash>/<date>.csv.gz` per r3 checkpoint
(`evidence/checkpoints.csv`); a checkpoint is drawn once every
development date is there. Relative WIS comes from
`results/germany-chronos-dev/scores.csv` and
`evidence/checkpoint-scores.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nowcast_panels import draw_panels

from bdbvhubrenewal.data.germany import hub_target
from bdbvhubrenewal.data.germany import hub_triangle as read_triangle
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.nowcast import rolling_truth

ORIGINS = ("2021-11-29", "2021-12-27", "2022-01-31")
PLACE = "DE"
READ = {"dtype": {"origin": str, "target_end_date": str}}


def main(results: Path, ckpt_dir: Path, out: Path) -> None:
    """Draw the German national panels."""
    config = de.load_germany_config("config/germany.toml")
    ours = pd.read_csv(results / "forecasts.csv.gz", **READ)
    teams = pd.read_csv(results / "teams-forecasts.csv.gz", **READ)
    scores = pd.read_csv("results/germany-chronos-dev/scores.csv")
    national = scores.loc[scores["pool"] == "national"].set_index("model")
    rel = national["rel_wis_randomwalknowcast"]
    curve = pd.read_csv("evidence/checkpoint-scores.csv")
    curve = curve.loc[
        (curve["study"] == "germany")
        & (curve["level"] == "national")
        & (curve["model"] == "fm_chronos_daily_ragged")
    ].set_index("checkpoint")
    models = {
        "randomwalknowcast": ours.loc[ours["model"] == "randomwalknowcast"],
        "Hub MeanEnsemble": teams.loc[teams["model"] == "MeanEnsemble"],
        "Chronos-2 zero-shot": ours.loc[
            ours["model"] == "fm_chronos_daily_ragged"
        ],
        "Chronos-2 ft round 1": ours.loc[
            ours["model"] == "fm_chronos_daily_ragged_ft"
        ],
    }
    scored = {
        "randomwalknowcast": 1.0,
        "Hub MeanEnsemble": rel["MeanEnsemble"],
        "Chronos-2 zero-shot": curve.loc["zero-shot", "rel_wis"],
        "Chronos-2 ft round 1": rel["fm_chronos_daily_ragged_ft"],
    }
    n_dates = len(de.nowcast_dates(config, held_out=False))
    checkpoints = pd.read_csv("evidence/checkpoints.csv", dtype=str)
    for ckpt, outbreaks in checkpoints.itertuples(index=False):
        files = sorted((ckpt_dir / ckpt).glob("*.csv.gz"))
        if len(files) < n_dates or ckpt not in curve.index:
            continue
        fc = pd.concat(pd.read_csv(f, **READ) for f in files)
        label = f"Chronos-2 ft r3 n{outbreaks}"
        models[label] = fc.loc[fc["model"] == "fm_chronos_daily_ragged_ft"]
        scored[label] = curve.loc[ckpt, "rel_wis"]
    triangle = read_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    asof = de.asof_sums(triangle, list(ORIGINS))
    ends = sorted(set(asof["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), config.truth_delay)
    draw_panels(
        models,
        scored,
        truth,
        asof,
        ORIGINS,
        PLACE,
        "German COVID-19 hospitalisations, national 7-day sums: nowcasts "
        "by model (daily Chronos-2 source)",
        out,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
